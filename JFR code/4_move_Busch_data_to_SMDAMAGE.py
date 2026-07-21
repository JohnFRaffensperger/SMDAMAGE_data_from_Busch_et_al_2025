# 4_move_Busch_data_to_SMDAMAGE.py
from __future__ import annotations
from pathlib import Path
import sqlite3

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Output"
DATABASE_DIR = OUTPUT_DIR / "Databases"
DB_FILE = DATABASE_DIR / "Busch2024_to_SMDAMAGE.sqlite"

DISCOUNT_COLUMN_BY_RATE_00 = {0: "NPV_0_pct_per_ha", 15: "NPV_015_pct_per_ha", 30: "NPV_03_pct_per_ha", 60: "NPV_06_pct_per_ha"}
DEFAULT_BUCKET_AREA_SHARE = 0.03 # So each bid step is about 3% of the pixels. Aiming for around 33 or 34 bid steps.
DEFAULT_OMIT_HIGH_COST_AREA_SHARE = 0.01

def discover_contract_years(con: sqlite3.Connection) -> list[int]:
	years = [int (row[0]) for row in con.execute ("select distinct contract_years from Pixel_bids order by contract_years")]
	if not years: raise ValueError ("Pixel_bids has no contract_years values")
	return years

def apply_read_pragmas(con: sqlite3.Connection) -> None:
	con.execute ("pragma cache_size = -524288")
	con.execute ("pragma mmap_size = 4294967296")
	con.execute ("pragma temp_store = MEMORY")

def rebuild_curve_table(con: sqlite3.Connection, years: int) -> str:
	table = f"cluster_forestry_bid_curves_{years}"
	con.execute (f"drop table if exists {table}")
	con.execute (f""" create table {table} (cluster_id INTEGER NOT NULL, discount_rate_00 INTEGER NOT NULL, bid_step INTEGER NOT NULL, npv_max_per_ha REAL NOT NULL, step_area_ha REAL NOT NULL, PRIMARY KEY (cluster_id, discount_rate_00, bid_step)) """)
	return table

def create_curve_table_indexes(con: sqlite3.Connection, table: str) -> None: con.execute (f"create index if not exists idx_{table}_cluster_rate_step on {table} (cluster_id, discount_rate_00, bid_step)")

def build_all_steps_for_year_rate(con: sqlite3.Connection, years: int, npv_column: str, rate_00: int, curve_table: str, bucket_area_share: float, omit_high_cost_area_share: float) -> int:
	sql = f"""
	with all_rows as (select cluster_id, {npv_column} as npv_value, area_ha, pixel_id, sum(area_ha) over (partition by cluster_id) as cluster_total_area from Pixel_bids where contract_years = ? and cluster_id is not null and area_ha is not null and {npv_column} is not null),
	cumulative as (select cluster_id, npv_value, area_ha, cluster_total_area * (1.0 - ?) as included_area_target, cluster_total_area * ? as bucket_area_target, sum(area_ha) over (partition by cluster_id order by npv_value, pixel_id rows unbounded preceding) - area_ha as cum_before from all_rows),
	adjusted as (select cluster_id, npv_value, cum_before, bucket_area_target, case when cum_before >= included_area_target then 0.0 when cum_before + area_ha > included_area_target then included_area_target - cum_before else area_ha end as used_area from cumulative where cum_before < included_area_target),
	adj_cum as (select cluster_id, npv_value, used_area, bucket_area_target, sum(used_area) over (partition by cluster_id order by cum_before, npv_value rows unbounded preceding) - used_area as adj_cum_before from adjusted where used_area > 0),
	bucketed as (select cluster_id, cast(adj_cum_before / bucket_area_target as integer) + 1 as bid_step, max(npv_value) as npv_max_per_ha, sum(used_area) as step_area_ha from adj_cum group by cluster_id, cast(adj_cum_before / bucket_area_target as integer) + 1)
	select cluster_id, bid_step, npv_max_per_ha, step_area_ha from bucketed order by cluster_id, bid_step
	"""
	result_rows = con.execute (sql, (years, omit_high_cost_area_share, bucket_area_share)).fetchall ()
	if result_rows: con.executemany (f"insert into {curve_table} (cluster_id, discount_rate_00, bid_step, npv_max_per_ha, step_area_ha) values (?, ?, ?, ?, ?)", [(r[0], rate_00, r[1], r[2], r[3]) for r in result_rows],)
	return len (result_rows)

def verify_monotonicity(con: sqlite3.Connection, curve_table: str) -> None:
	violations = con.execute (f""" select count(*) from (select cluster_id, discount_rate_00, bid_step, npv_max_per_ha, lag(npv_max_per_ha) over (partition by cluster_id, discount_rate_00 order by bid_step ) as prev_npv from {curve_table}) where prev_npv is not null and npv_max_per_ha < prev_npv """ ).fetchone ()[0]
	if violations != 0: raise ValueError (f"Monotonicity check failed in {curve_table}: {violations} decreasing steps found")

def log_row_counts(con: sqlite3.Connection, years: int, curve_table: str) -> None:
	for rate_00 in sorted (DISCOUNT_COLUMN_BY_RATE_00):
		rows = con.execute (f"select count(*) from {curve_table} where discount_rate_00 = ?", (rate_00,), ).fetchone ()[0]
		print (f"contract_years={years}, discount_rate_00={rate_00}, rows={rows}", flush = True)

def run_bid_curve_build(db_file: Path = DB_FILE, bucket_area_share: float = DEFAULT_BUCKET_AREA_SHARE, omit_high_cost_area_share: float = DEFAULT_OMIT_HIGH_COST_AREA_SHARE) -> None:
	with sqlite3.connect (db_file) as con:
		apply_read_pragmas (con)
		has_pixel_bids = con.execute ("select count(*) from sqlite_master where type='table' and name='Pixel_bids'").fetchone ()[0]
		if has_pixel_bids != 1: raise ValueError ("Pixel_bids table is missing. Run 3_import_k_means_csv_to_sqlite.py first.")
		contract_years_list = discover_contract_years (con)
		for years in contract_years_list:
			curve_table = rebuild_curve_table (con, years)
			for rate_00, npv_column in DISCOUNT_COLUMN_BY_RATE_00.items():
				steps = build_all_steps_for_year_rate (con, years, npv_column, rate_00, curve_table, bucket_area_share, omit_high_cost_area_share)
				print (f"contract_years={years}, rate_00={rate_00}: {steps} steps inserted", flush = True)
			create_curve_table_indexes (con, curve_table)
			verify_monotonicity (con, curve_table)
			log_row_counts (con, years, curve_table)
			con.commit ()
		print (f"Bid-curve build complete: {db_file}", flush = True)

if __name__ == "__main__": run_bid_curve_build ()
