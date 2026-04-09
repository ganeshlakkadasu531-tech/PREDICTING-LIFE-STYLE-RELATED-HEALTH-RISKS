import sqlite3

conn = sqlite3.connect("database1.db")
cur = conn.cursor()
cur.execute("SELECT * FROM user")
print(cur.fetchall())
conn.close()
