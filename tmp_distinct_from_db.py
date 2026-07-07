import sqlite3
from pathlib import Path

db = Path(r"C:\Users\johnr\Documents\Work documents\2 Research\Global warming\Numerical Simulation\Busch2024\Output\Databases\Busch2024_to_SMDAMAGE.sqlite")
conn = sqlite3.connect(db)
cur = conn.cursor()

row_total = cur.execute("SELECT COUNT(1) FROM Undiscounted_dta_output").fetchone()[0]
by_option = cur.execute("SELECT selected_option, COUNT(1) FROM Undiscounted_dta_output GROUP BY selected_option ORDER BY selected_option").fetchall()

q = """
SELECT
    selected_option,
    plantation_genus,
    selected_A,
    selected_k,
    selected_rotation_year,
    crop_va_USD_per_ha_per_year,
    selected_establishment_cost_USD_per_ha,
    p_USD_per_tC_harvested
FROM Undiscounted_dta_output
"""

q_lean = """
SELECT
    selected_option,
    plantation_genus,
    selected_A,
    selected_k,
    selected_rotation_year
FROM Undiscounted_dta_output
"""

full_set = set()
lean_set = set()
chunk = 200000
processed = 0

for row in cur.execute(q):
    full_set.add(row)
    processed += 1
    if processed % chunk == 0:
        print(f"processed_full={processed}", flush=True)

processed = 0
for row in cur.execute(q_lean):
    lean_set.add(row)
    processed += 1
    if processed % chunk == 0:
        print(f"processed_lean={processed}", flush=True)

print("TOTAL_ROWS", row_total)
print("BY_OPTION", by_option)
print("DISTINCT_FULL_EXCL_X_Y_AREA", len(full_set))
print("DISTINCT_LEAN_EXCL_X_Y_AREA", len(lean_set))
print("EXCLUDED_FIELDS", ["x", "y", "area", "area_m2", "area_ha"])

conn.close()
