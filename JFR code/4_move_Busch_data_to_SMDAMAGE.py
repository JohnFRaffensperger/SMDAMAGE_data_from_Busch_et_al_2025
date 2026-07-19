# 4_move_Busch_data_to_SMDAMAGE.py
from __future__ import annotations
from pathlib import Path
import sqlite3

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Output"
DATABASE_DIR = OUTPUT_DIR / "Databases"
DB_FILE = DATABASE_DIR / "Busch2024_to_SMDAMAGE.sqlite"

DISCOUNT_COLUMN_BY_RATE_00 = {0: "NPV_0_pct_per_ha", 15: "NPV_015_pct_per_ha", 30: "NPV_03_pct_per_ha", 60: "NPV_06_pct_per_ha"}
DEFAULT_BUCKET_AREA_SHARE = 0.03
DEFAULT_OMIT_HIGH_COST_AREA_SHARE = 0.01

def discover_contract_years(con: sqlite3.Connection) -> list[int]:
	years = [int (row[0]) for row in con.execute ("select distinct contract_years from Pixel_bids order by contract_years")]
	if not years: raise ValueError ("Pixel_bids has no contract_years values")
	return years

def rebuild_curve_table(con: sqlite3.Connection, years: int) -> str:
	table = f"cluster_forestry_bid_curves_{years}"
	con.execute (f"drop table if exists {table}")
	con.execute (f""" create table {table} (cluster_id INTEGER NOT NULL, discount_rate_00 INTEGER NOT NULL, bid_step INTEGER NOT NULL, npv_max_per_ha REAL NOT NULL, step_area_ha REAL NOT NULL, PRIMARY KEY (cluster_id, discount_rate_00, bid_step)) """)
	con.execute (f"create index idx_{table}_cluster_rate_step on {table} (cluster_id, discount_rate_00, bid_step)")
	return table

def get_cluster_ids(con: sqlite3.Connection, years: int) -> list[int]:
	rows = con.execute (""" select distinct cluster_id from Pixel_bids where contract_years = ? and cluster_id is not null order by cluster_id """, (years,), ).fetchall ()
	return [int (row[0]) for row in rows]

def build_steps_for_cluster_rate( con: sqlite3.Connection, years: int, cluster_id: int, npv_column: str, rate_00: int, curve_table: str, bucket_area_share: float, omit_high_cost_area_share: float, ) -> int:
	total_area = con.execute (f""" select sum(area_ha) from Pixel_bids where contract_years = ? and cluster_id = ? and area_ha is not null and {npv_column} is not null """, (years, cluster_id), ).fetchone ()[0]
	if total_area is None or total_area <= 0.0: return 0
	included_area_target = total_area*(1.0 - omit_high_cost_area_share)
	if included_area_target <= 0.0: return 0
	bucket_area_target = total_area*bucket_area_share
	if bucket_area_target <= 0.0: return 0
	step_rows: list[tuple[int, int, int, float, float]] = []
	included_area = 0.0
	step_area = 0.0
	step_npv_max = None
	bid_step = 1
	cursor = con.execute (f""" select {npv_column}, area_ha from Pixel_bids where contract_years = ? and cluster_id = ? and area_ha is not null and {npv_column} is not null order by {npv_column}, pixel_id """, (years, cluster_id),)
	for npv_value, area_ha in cursor:
		if included_area >= included_area_target: break
		remaining = included_area_target - included_area
		used_area = area_ha if area_ha <= remaining else remaining
		if used_area <= 0.0: break
		step_area += used_area
		included_area += used_area
		step_npv_max = npv_value
		if step_area >= bucket_area_target or included_area >= included_area_target:
			step_rows.append((cluster_id, rate_00, bid_step, step_npv_max, step_area))
			bid_step += 1
			step_area = 0.0
			step_npv_max = None
	if step_rows: con.executemany (f"insert into {curve_table} (cluster_id, discount_rate_00, bid_step, npv_max_per_ha, step_area_ha) values (?, ?, ?, ?, ?)", step_rows,)
	return len (step_rows)

def verify_monotonicity(con: sqlite3.Connection, curve_table: str) -> None:
	violations = con.execute (f""" select count(*) from (select cluster_id, discount_rate_00, bid_step, npv_max_per_ha, lag(npv_max_per_ha) over (partition by cluster_id, discount_rate_00 order by bid_step ) as prev_npv from {curve_table}) where prev_npv is not null and npv_max_per_ha < prev_npv """ ).fetchone ()[0]
	if violations != 0: raise ValueError (f"Monotonicity check failed in {curve_table}: {violations} decreasing steps found")

def log_row_counts(con: sqlite3.Connection, years: int, curve_table: str) -> None:
	for rate_00 in sorted (DISCOUNT_COLUMN_BY_RATE_00):
		rows = con.execute (f"select count(*) from {curve_table} where discount_rate_00 = ?", (rate_00,), ).fetchone ()[0]
		print (f"contract_years={years}, discount_rate_00={rate_00}, rows={rows}", flush = True)

def run_bid_curve_build( db_file: Path = DB_FILE, bucket_area_share: float = DEFAULT_BUCKET_AREA_SHARE, omit_high_cost_area_share: float = DEFAULT_OMIT_HIGH_COST_AREA_SHARE, ) -> None:
	with sqlite3.connect (db_file) as con:
		has_pixel_bids = con.execute ("select count(*) from sqlite_master where type='table' and name='Pixel_bids'" ).fetchone ()[0]
		if has_pixel_bids != 1: raise ValueError ("Pixel_bids table is missing. Run 3_import_k_means_csv_to_sqlite.py first.")
		contract_years_list = discover_contract_years (con)
		for years in contract_years_list:
			curve_table = rebuild_curve_table (con, years)
			cluster_ids = get_cluster_ids (con, years)
			for cluster_id in cluster_ids:
				for rate_00, npv_column in DISCOUNT_COLUMN_BY_RATE_00.items():
					build_steps_for_cluster_rate (con = con, years = years, cluster_id = cluster_id, npv_column = npv_column, rate_00 = rate_00, curve_table = curve_table, bucket_area_share = bucket_area_share, omit_high_cost_area_share = omit_high_cost_area_share,)
			verify_monotonicity (con, curve_table)
			log_row_counts (con, years, curve_table)
		con.commit()
		print (f"Bid-curve build complete: {db_file}", flush = True)

if __name__ == "__main__": run_bid_curve_build ()
