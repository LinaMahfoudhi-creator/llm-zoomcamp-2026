'''import sqlite3
conn = sqlite3.connect("traces.db")
cur = conn.execute("SELECT DISTINCT name FROM spans")
print(cur.fetchall())'''

'''import sqlite3
import pandas as pd

conn = sqlite3.connect("traces.db")
df = pd.read_sql("SELECT * FROM spans", conn)

df["duration_ms"] = (df["end_time"] - df["start_time"]) / 1_000_000

print(df[["name", "duration_ms"]])
print()
print(df[df["name"] != "rag"].groupby("name")["duration_ms"].sum())'''


import sqlite3
import pandas as pd

conn = sqlite3.connect("traces.db")
df = pd.read_sql("SELECT * FROM spans WHERE name='llm'", conn)
print(df[["input_tokens"]])
print()
print("min:", df["input_tokens"].min())
print("max:", df["input_tokens"].max())
print("variation:", (df["input_tokens"].max() - df["input_tokens"].min()) / df["input_tokens"].min() * 100, "%")