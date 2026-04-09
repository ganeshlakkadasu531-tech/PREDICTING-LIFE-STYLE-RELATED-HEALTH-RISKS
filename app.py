# app.py
from flask import Flask, render_template, request, redirect, url_for, make_response, send_file,session
import os
import pandas as pd
import joblib
import sqlite3
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, session
import sqlite3
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- CONFIG ---
app = Flask(__name__)  # no app.secret_key set (per your request)

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'Models'
STATIC_FOLDER = 'static'
DATASET_FOLDER = 'Dataset'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

FEATURE_COLUMNS = [
    'age',
    'gender',
    'resting_heart_rate',
    'heart_rate_variability',
    'pulse_amplitude',
    'stress_level',
    'sleep_duration_hours',
    'sleep_quality_score',
    'steps_per_day',
    'calories_burned',
    'blood_oxygen_level',
    'activity_level'
]

TARGET_COLUMN = 'lifestyle_disorder_risk'

CATEGORICAL_COLUMNS = ['gender', 'activity_level']
NUMERIC_COLUMNS = [
    'age', 'resting_heart_rate', 'heart_rate_variability',
    'pulse_amplitude', 'stress_level', 'sleep_duration_hours',
    'sleep_quality_score', 'steps_per_day',
    'calories_burned', 'blood_oxygen_level'
]

# Globals to store accuracy values (used for graph)
xgb_acc = None
rf_acc = None
dec_acc = None

xgb_acc = rf_acc = dec_acc = None


app.secret_key = '123'
DB_NAME = "database1.db"

def init_user_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            mobile TEXT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    conn.commit()
    conn.close()
def init_reports_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,

            age REAL,
            gender TEXT,
            resting_heart_rate REAL,
            heart_rate_variability REAL,
            pulse_amplitude REAL,
            stress_level REAL,
            sleep_duration_hours REAL,
            sleep_quality_score REAL,
            steps_per_day REAL,
            calories_burned REAL,
            blood_oxygen_level REAL,
            activity_level TEXT,

            prediction TEXT,
            probability TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
@app.route('/userdashboard')
def userdashboard():
    if 'username' not in session:
        return redirect(url_for('userlogin'))
    return render_template('UserApp/Home.html')

@app.route('/')
def home():
   
    return render_template('index.html')
# ---------- Admin ----------
@app.route('/adminlogin', methods=['GET','POST'])
def adminlogin():
    return render_template('AdminApp/AdminLogin.html')

@app.route('/AdminAction', methods=['POST'])
def AdminAction():
    if request.method == 'POST':
        username=request.form['username']
        password=request.form['password']

        if username=='Admin' and password=='Admin':
            return render_template("AdminApp/AdminHome.html")
        else:
            context={'msg':'Login Failed..!!'}
            return render_template("AdminApp/AdminLogin.html",**context)


@app.route('/AdminHome')
def AdminHome():
    return render_template("AdminApp/AdminHome.html")

@app.route('/Upload')
def Upload():
    return render_template("AdminApp/Upload.html")

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

global dataset,filepath
@app.route('/UploadAction', methods=['POST'])
def UploadAction():
    global dataset,filepath
    if 'dataset' not in request.files:
        return "No file part"
    file = request.files['dataset']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    dataset = pd.read_csv(filepath)
    columns = dataset.columns.tolist()
    rows = dataset.head().values.tolist()
    return render_template('AdminApp/ViewDataset.html', columns=columns, rows=rows)

global dataset, X_train, X_test, y_train, y_test

@app.route('/preprocess')
def preprocess():
    global dataset, X_train, X_test, y_train, y_test

    dataset = pd.read_csv('Dataset/lifestyle_disorder_wearable_dataset.csv')
    dataset.dropna(inplace=True)

    # ===============================
    # Encode categorical features
    # ===============================
    label_encoders = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])
        label_encoders[col] = le

    joblib.dump(label_encoders, "Models/encoders.joblib")

    # ===============================
    # Encode target
    # ===============================
    target_encoder = LabelEncoder()
    dataset[TARGET_COLUMN] = target_encoder.fit_transform(dataset[TARGET_COLUMN])

    joblib.dump(target_encoder, "Models/target_encoder.joblib")

    # ===============================
    # Scale numeric features (FIT + SAVE)
    # ===============================
    scaler = StandardScaler()
    dataset[NUMERIC_COLUMNS] = scaler.fit_transform(dataset[NUMERIC_COLUMNS])

    joblib.dump(scaler, "Models/scaler.joblib")

    # ===============================
    # Train-test split
    # ===============================
    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return render_template(
        'AdminApp/SplitStatus.html',
        total=len(X),
        train=len(X_train),
        test=len(X_test)
    )

@app.route('/trainmodels')
def trainmodels():
    global X_train, X_test, y_train, y_test
    global xgb_acc, rf_acc, dec_acc

    results = {}

    # =======================
    # 1️⃣ XGBOOST
    # =======================
    xgb_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, "Models/XGModel.joblib")

    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = round(accuracy_score(y_test, xgb_pred) * 100, 2)
    results["XGBoost"] = xgb_acc

    # =======================
    # 2️⃣ RANDOM FOREST
    # =======================
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "Models/RFModel.joblib")

    rf_pred = rf_model.predict(X_test)
    rf_acc = round(accuracy_score(y_test, rf_pred) * 100, 2)
    results["Random Forest"] = rf_acc

    # =======================
    # 3️⃣ DECISION TREE
    # =======================
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    joblib.dump(dt_model, "Models/DTModel.joblib")

    dt_pred = dt_model.predict(X_test)
    dec_acc = round(accuracy_score(y_test, dt_pred) * 100, 2)
    results["Decision Tree"] = dec_acc

    # =======================
    # SHAP
    # =======================
    shap_explainer = shap.TreeExplainer(rf_model)
    shap_values = shap_explainer.shap_values(X_test[:50])
    shap.summary_plot(shap_values, X_test[:50], show=False)
    plt.savefig("static/shap_summary.png")
    plt.close()

    # =======================
    # LIME
    # =======================
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["LOW", "MODERATE", "HIGH"],
        mode="classification"
    )

    lime_exp = lime_explainer.explain_instance(
        X_test.iloc[0].values,
        rf_model.predict_proba,
        num_features=10
    )
    lime_exp.save_to_file("static/lime_explanation.html")

    return render_template(
        "AdminApp/AlgorithmStatus.html",
        msg="All Algorithms Executed Successfully",
        results=results,
        shap_image="shap_summary.png",
        lime_file="lime_explanation.html"
    )

@app.route('/comparison')
def comparison():
    global xgb_acc, rf_acc, dec_acc

    models = ['XGBoost', 'Random Forest', 'Decision Tree']
    accuracies = [xgb_acc, rf_acc, dec_acc]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, accuracies)

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            acc + 1,
            f'{acc}%',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('static/model_accuracy.png')
    plt.close()

    return render_template('AdminApp/Grpah.html')

@app.route('/userlogin')
def userlogin():
    return render_template('UserApp/Login.html')

@app.route('/register')
def register():
    return render_template('UserApp/Register.html')

@app.route('/RegAction', methods=['POST'])
def RegAction():
    name = request.form['name']
    email = request.form['email']
    mobile = request.form['mobile']
    username = request.form['username']
    password = request.form['password']

    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    cur.execute("SELECT * FROM user WHERE username=? OR email=?", (username, email))
    data = cur.fetchone()

    if data is None:
        try:
            cur.execute("INSERT INTO user (name, email, mobile, username, password) VALUES (?, ?, ?, ?, ?)",
                        (name, email, mobile, username, password))
            con.commit()
            msg = "Successfully Registered!"
        except sqlite3.IntegrityError:
            msg = "Username already exists!"
    else:
        msg = "Username or email already exists!"

    con.close()
    return render_template('UserApp/Register.html', msg=msg)

@app.route('/UserAction', methods=['GET','POST'])
def UserAction():
    username = request.form['username']
    password = request.form['password']

    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    cur.execute("SELECT * FROM user WHERE username=? AND password=?", (username, password))
    data = cur.fetchone()
    con.close()

    if data is None:
        return render_template('UserApp/Login.html', msg="Login Failed!")
    else:
        session['username'] = data[3]
        return render_template('UserApp/Home.html', username=session['username'])


@app.route('/ManageUsers')
def ManageUsers():
    conn = sqlite3.connect(DB_NAME)   # ✅ FIXED
    cur = conn.cursor()

    cur.execute("SELECT name, email, mobile, username FROM user")
    users = cur.fetchall()

    conn.close()
    return render_template("AdminApp/ManageUsers.html", users=users)

@app.route("/delete_user/<username>")
def delete_user(username):
    conn = sqlite3.connect(DB_NAME)   # ✅ FIXED
    cur = conn.cursor()

    cur.execute("DELETE FROM user WHERE username = ?", (username,))
    conn.commit()
    conn.close()

    return redirect(url_for("ManageUsers"))  # ✅ FIXED

@app.route('/Detect')
def Detect():
    return render_template('UserApp/Detect.html')

@app.route('/UserHome')
def UserHome():
    return render_template('UserApp/Home.html')
SMTP_EMAIL = "kaleem202120@gmail.com"          # sender
SMTP_PASSWORD = "xyljzncebdxcubjq"        # gmail app password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
import smtplib
import sqlite3
import pandas as pd
import joblib

from flask import Flask, render_template, request, session
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
def get_health_suggestion(prediction):
    if prediction == "HIGH RISK":
        return "⚠️ Immediate medical consultation is strongly recommended."
    elif prediction == "MODERATE RISK":
        return "🩺 Maintain a healthy routine and monitor your health regularly."
    else:
        return "✅ You are healthy. Continue your good lifestyle habits."
def send_health_result_email(to_email, username, prediction, probability):
    print("📧 Email function triggered")

    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = to_email
        msg['Subject'] = "Health Risk Report"

        body = f"""
Hello {username},

Prediction: {prediction}
Probability: {probability}

Regards,
Health System
"""
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("✅ Email sent successfully")

    except Exception as e:
        print("❌ Email error:", e)

@app.route('/DetectAction', methods=['POST'])
def DetectAction():

    if 'username' not in session:
        return redirect(url_for('userlogin'))

    raw_input = {
        'age': request.form['age'],
        'gender': request.form['gender'],
        'resting_heart_rate': request.form['resting_heart_rate'],
        'heart_rate_variability': request.form['heart_rate_variability'],
        'pulse_amplitude': request.form['pulse_amplitude'],
        'stress_level': request.form['stress_level'],
        'sleep_duration_hours': request.form['sleep_duration_hours'],
        'sleep_quality_score': request.form['sleep_quality_score'],
        'steps_per_day': request.form['steps_per_day'],
        'calories_burned': request.form['calories_burned'],
        'blood_oxygen_level': request.form['blood_oxygen_level'],
        'activity_level': request.form['activity_level']
    }

    # Convert input to DataFrame
    test = pd.DataFrame([{
        'age': float(raw_input['age']),
        'gender': raw_input['gender'],
        'resting_heart_rate': float(raw_input['resting_heart_rate']),
        'heart_rate_variability': float(raw_input['heart_rate_variability']),
        'pulse_amplitude': float(raw_input['pulse_amplitude']),
        'stress_level': float(raw_input['stress_level']),
        'sleep_duration_hours': float(raw_input['sleep_duration_hours']),
        'sleep_quality_score': float(raw_input['sleep_quality_score']),
        'steps_per_day': float(raw_input['steps_per_day']),
        'calories_burned': float(raw_input['calories_burned']),
        'blood_oxygen_level': float(raw_input['blood_oxygen_level']),
        'activity_level': raw_input['activity_level']
    }])

    # Load encoders
    encoders = joblib.load("Models/encoders.joblib")
    for col in encoders:
        test[col] = encoders[col].transform(test[col])

    # Load scaler
    scaler = joblib.load("Models/scaler.joblib")
    test[NUMERIC_COLUMNS] = scaler.transform(test[NUMERIC_COLUMNS])

    # Load trained model
    model = joblib.load("Models/RFModel.joblib")

    pred = int(model.predict(test)[0])
    proba = model.predict_proba(test)[0][pred]

    risk_labels = {
        0: "LOW RISK",
        1: "HIGH RISK",
        2: "MODERATE RISK"
    }

    prediction = risk_labels[pred]
    probability = f"{proba * 100:.2f}%"

    # ----------------------------------------------------
    # Doctor Suggestions
    # ----------------------------------------------------

    doctor_suggestions = {

        "LOW RISK": {
            "advice": "Your health indicators appear normal.",
            "diet": "Maintain a balanced diet rich in fruits, vegetables, whole grains, lean proteins, and adequate hydration.",
            "exercise": "Perform at least 30 minutes of moderate physical activity daily such as walking, cycling, or yoga.",
            "sleep": "Ensure 7–8 hours of good quality sleep every night.",
            "checkup": "Routine health check-ups once every 6–12 months are recommended."
        },

        "MODERATE RISK": {
            "advice": "Some health indicators require attention.",
            "diet": "Reduce processed foods, salt, and sugar intake. Include heart-healthy foods like fish, nuts, oats, and leafy greens.",
            "exercise": "Engage in regular cardiovascular exercises such as brisk walking, jogging, or swimming for at least 40 minutes daily.",
            "sleep": "Maintain a consistent sleep schedule and reduce screen exposure before bedtime.",
            "checkup": "Consult a general physician or cardiologist for preventive health evaluation."
        },

        "HIGH RISK": {
            "advice": "Your health indicators suggest a high health risk.",
            "diet": "Follow a medically supervised diet plan with low sodium, low saturated fat, and high fiber foods.",
            "exercise": "Avoid heavy physical exertion until evaluated by a medical professional.",
            "sleep": "Practice stress reduction techniques such as meditation, breathing exercises, and adequate rest.",
            "checkup": "Immediate consultation with a cardiologist is strongly recommended. Diagnostic tests such as ECG, blood pressure monitoring, and blood tests may be required."
        }
    }

    suggestion = doctor_suggestions[prediction]

    username = session['username']

    # ----------------------------------------------------
    # Store Prediction in Database
    # ----------------------------------------------------

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO reports (
            username, age, gender, resting_heart_rate,
            heart_rate_variability, pulse_amplitude, stress_level,
            sleep_duration_hours, sleep_quality_score,
            steps_per_day, calories_burned, blood_oxygen_level,
            activity_level, prediction, probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        raw_input['age'], raw_input['gender'],
        raw_input['resting_heart_rate'],
        raw_input['heart_rate_variability'],
        raw_input['pulse_amplitude'],
        raw_input['stress_level'],
        raw_input['sleep_duration_hours'],
        raw_input['sleep_quality_score'],
        raw_input['steps_per_day'],
        raw_input['calories_burned'],
        raw_input['blood_oxygen_level'],
        raw_input['activity_level'],
        prediction,
        probability
    ))

    conn.commit()

    # ----------------------------------------------------
    # Send Email Result
    # ----------------------------------------------------

    cur.execute("SELECT email FROM user WHERE username = ?", (username,))
    row = cur.fetchone()

    if row:
        send_health_result_email(
            to_email=row[0],
            username=username,
            prediction=prediction,
            probability=probability,
            suggestion=suggestion
        )

    conn.close()

    # ----------------------------------------------------
    # Show Result Page
    # ----------------------------------------------------

    return render_template(
        "UserApp/Result.html",
        prediction=prediction,
        probability=probability,
        suggestion=suggestion,
        input_data=raw_input,
        show_result=True
    )
@app.route('/ViewReports')
def ViewReports():

    if 'doctor' not in session:
        return redirect(url_for('doctorlogin'))

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        SELECT
            age,
            gender,
            resting_heart_rate,
            heart_rate_variability,
            pulse_amplitude,
            stress_level,
            sleep_duration_hours,
            sleep_quality_score,
            steps_per_day,
            calories_burned,
            blood_oxygen_level,
            activity_level,
            prediction,
            probability,
            created_at
        FROM reports
        ORDER BY created_at DESC
    """)

    reports = cur.fetchall()
    conn.close()

    return render_template("DoctorApp/ViewReports.html", reports=reports)


# ---------- DOCTOR LOGIN ----------
@app.route('/doctorlogin')
def doctorlogin():
    return render_template('DoctorApp/Login.html')

@app.route('/DoctorAction', methods=['POST'])
def DoctorAction():
    username = request.form['username']
    password = request.form['password']

    if username == 'doctor' and password == 'doctor':
        session['doctor'] = 'doctor'
        return redirect(url_for('DoctorHome'))
    else:
        return render_template('DoctorApp/Login.html', msg="Invalid Doctor Credentials")
@app.route('/DoctorHome')
def DoctorHome():
    if 'doctor' not in session:
        return redirect(url_for('doctorlogin'))
    return render_template('DoctorApp/Home.html')
@app.route('/DoctorManageUsers')
def DoctorManageUsers():
    if 'doctor' not in session:
        return redirect(url_for('doctorlogin'))

    conn = sqlite3.connect(DB_NAME)   # ✅ FIXED
    cur = conn.cursor()

    cur.execute("SELECT name, email, mobile, username FROM user")
    users = cur.fetchall()

    conn.close()
    return render_template("DoctorApp/ManageUsers.html", users=users)
@app.route("/DoctorDeleteUser/<username>")
def DoctorDeleteUser(username):
    if 'doctor' not in session:
        return redirect(url_for('doctorlogin'))

    conn = sqlite3.connect(DB_NAME)   # ✅ FIXED
    cur = conn.cursor()

    cur.execute("DELETE FROM user WHERE username = ?", (username,))
    conn.commit()
    conn.close()

    return redirect(url_for("DoctorManageUsers"))

@app.route('/DoctorLogout')
def DoctorLogout():
    session.pop('doctor', None)
    return redirect(url_for('doctorlogin'))

if __name__ == '__main__':
    init_user_db()
    init_reports_db()
    app.run(debug=True)
