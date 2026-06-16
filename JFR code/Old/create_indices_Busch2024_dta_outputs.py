# create_indices_Busch2024_dta_outputs.py | Created 2026-04-14
# Creates SQLite indices on Busch2024 output tables to accelerate common exploratory queries, especially cost-effectiveness bucketing and filtering operations during analysis.
"""Create indices on Busch2024_dta_outputs.sqlite to speed up ad hoc queries."""
from pathlib import Path
import sqlite3

DB_FILE = Path(__file__).resolve().parent.parent / "Output" / "Busch2024_dta_outputs.sqlite"

con = sqlite3.connect(str(DB_FILE))
cur = con.cursor()

# Partial index on mi_whrv_costeff (non-null only), matching the WHERE clause in
# the mi_whrv_costeff_bucket_50 grouping query.
cur.execute("""CREATE INDEX IF NOT EXISTS idx_Busch2024_dta_outputs_mi_whrv_costeff
	ON Busch2024_dta_outputs (mi_whrv_costeff)
	WHERE mi_whrv_costeff IS NOT NULL""")
print("Created idx_Busch2024_dta_outputs_mi_whrv_costeff.")

con.commit()
con.close()
print("Done.")
