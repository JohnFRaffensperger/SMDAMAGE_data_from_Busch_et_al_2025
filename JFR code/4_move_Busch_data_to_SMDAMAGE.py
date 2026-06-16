# 4_move_Busch_data_to_SMDAMAGE.py | Drafted by Copilot, supervised by John F. Raffensperger. Created 2026-06-12
# Exports Busch forestry bidder metadata and bidder-specific bid-step curves from the k-means-linked SQLite tables.

from __future__ import annotations
from pathlib import Path
import sqlite3
# import numpy as np
# import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Output"

DATABASE_DIR = OUTPUT_DIR / "Databases"
SOURCE_DB_FILE = DATABASE_DIR / "Busch2024_to_SMDAMAGE.sqlite"
# TARGET_DB_FILE = DATABASE_DIR / "Busch2024_to_SMDAMAGE_temp.sqlite"

# EXPORT_DIR = OUTPUT_DIR / "20_exports_smdamage"
# BIDDER_METADATA_DIR = EXPORT_DIR / "bidder_metadata"
# BID_STEPS_DIR = EXPORT_DIR / "bid_steps"
# SEQUESTRATION_DIR = EXPORT_DIR / "sequestration"
# PIXEL_COSTS_DIR = EXPORT_DIR / "pixel_costs"

PIXEL_TABLE = "Undiscounted_dta_output"
YEAR_TABLE = "tC_per_h_per_year"
ROTATION_YEAR_INT_COLUMN = "selected_rotation_year_int"
BATCH_SIZE = 50_000
# METADATA_TABLE = "forestry_bidder_metadata"
# BID_STEPS_TABLE = "forestry_bid_steps_long"
# PIXEL_COSTS_TABLE = "forestry_bidder_pixel_costs"
BID_CURVE_TABLE = "forestry_bid_curves"
# DEFAULT_DISCOUNT_RATE = 0.05  # Compute for discount rates in 0%, 1.5%, 3%, and 6%.
# DEFAULT_STEP_AREA_HA = 25_000.0

DISCOUNT_RATE = 0.03
RATE_TO_NPV_COLUMN = {0.0: "NPV_0_pct_per_ha", 0.015: "NPV_015_pct_per_ha", 0.03: "NPV_03_pct_per_ha", 0.06: "NPV_06_pct_per_ha"}
OLD_TO_NEW_NPV_COLUMNS = {"NPV_0_pct": "NPV_0_pct_per_ha", "NPV_015_pct": "NPV_015_pct_per_ha", "NPV_03_pct": "NPV_03_pct_per_ha", "NPV_06_pct": "NPV_06_pct_per_ha"}

# RUN_MODE = "partial"  # Set to "full" for full dataset run.
# PARTIAL_ROTATION_YEARS = [6]
# PARTIAL_CLUSTER_INDICES = [0, 1, 2]

# def load_pixel_level_inputs(db_file: Path = SOURCE_DB_FILE, run_mode: str = RUN_MODE, rotation_years: list[int] = PARTIAL_ROTATION_YEARS, cluster_indices: list[int] = PARTIAL_CLUSTER_INDICES, ) -> pd.DataFrame:
# 	query = f"""select country, pixel_id, selected_rotation_year, cluster_index, area_ha, crop_va_USD_per_ha_per_year, selected_establishment_cost_USD_per_ha
# 		from {PIXEL_TABLE} where cluster_index is not null"""
# 	params: list[int] = []
# 	if run_mode == "partial":
# 		if len(rotation_years) > 0:
# 			rotation_placeholders = ", ".join(["?"]*len(rotation_years))
# 			query += f" and cast(selected_rotation_year as integer) in ({rotation_placeholders})"
# 			params.extend(int(year) for year in rotation_years)
# 		if len(cluster_indices) > 0:
# 			cluster_placeholders = ", ".join(["?"]*len(cluster_indices))
# 			query += f" and cluster_index in ({cluster_placeholders})"
# 			params.extend(int(cluster) for cluster in cluster_indices)
# 	with sqlite3.connect(db_file) as con:
# 		return pd.read_sql_query(query, con, params=params)

# ---------------------------------------------------------------------------------------------
# Prepare structure of database. Rename legacy NPV columns and ensure _per_ha columns exist.
def rename_npv_columns_to_per_ha(db_file: Path = SOURCE_DB_FILE) -> None:
	with sqlite3.connect(db_file) as con:
		existing_columns = {str(row[1]) for row in con.execute(f"pragma table_info('{PIXEL_TABLE}')")}
		for old_name, new_name in OLD_TO_NEW_NPV_COLUMNS.items():
			if old_name in existing_columns and new_name not in existing_columns:
				con.execute(f"alter table {PIXEL_TABLE} rename column {old_name} to {new_name}")
				existing_columns.remove(old_name)
				existing_columns.add(new_name)
		for new_name in RATE_TO_NPV_COLUMN.values():
			if new_name not in existing_columns:
				con.execute(f"alter table {PIXEL_TABLE} add column {new_name} REAL")
		con.commit()

def ensure_rotation_year_int(con: sqlite3.Connection) -> None:
	existing_columns = {str(row[1]) for row in con.execute(f"pragma table_info('{PIXEL_TABLE}')")}
	if ROTATION_YEAR_INT_COLUMN not in existing_columns:
		con.execute(f"alter table {PIXEL_TABLE} add column {ROTATION_YEAR_INT_COLUMN} INTEGER")
	con.execute(f"update {PIXEL_TABLE} set {ROTATION_YEAR_INT_COLUMN} = cast(selected_rotation_year as integer) where {ROTATION_YEAR_INT_COLUMN} is null and selected_rotation_year is not null")
	con.commit()
# with sqlite3.connect(SOURCE_DB_FILE) as con:
# 	ensure_rotation_year_int(con)

def ensure_performance_indexes(con: sqlite3.Connection) -> None:
	con.execute(f"create index if not exists idx_{PIXEL_TABLE}_cluster_country_pixel_rotation_int on {PIXEL_TABLE} (cluster_index, country, pixel_id, {ROTATION_YEAR_INT_COLUMN})")
	con.execute(f"create index if not exists idx_{YEAR_TABLE}_country_pixel_year on {YEAR_TABLE} (country, pixel_id, year)")
	for npv_column in RATE_TO_NPV_COLUMN.values():
		con.execute(f"create index if not exists idx_{PIXEL_TABLE}_cluster_{npv_column} on {PIXEL_TABLE} (cluster_index, {npv_column})")
		con.execute(f"create index if not exists idx_{PIXEL_TABLE}_rotation_cluster_{npv_column} on {PIXEL_TABLE} ({ROTATION_YEAR_INT_COLUMN}, cluster_index, {npv_column})")
		con.execute(f"create index if not exists idx_{PIXEL_TABLE}_rotation_cluster_{npv_column}_area on {PIXEL_TABLE} ({ROTATION_YEAR_INT_COLUMN}, cluster_index, {npv_column}, area_ha)")
	con.commit()
# with sqlite3.connect(SOURCE_DB_FILE) as con:
# 	ensure_performance_indexes(con)

def prepare_bid_curve_run(db_file: Path = SOURCE_DB_FILE) -> None:
	with sqlite3.connect(db_file) as con:
		con.execute(f"drop table if exists {BID_CURVE_TABLE}")
		con.execute(f"""create table {BID_CURVE_TABLE} ({ROTATION_YEAR_INT_COLUMN} INTEGER not null, cluster_index INTEGER not null, discount_rate REAL not null,
			bucket_id INTEGER not null, npv_column TEXT not null, npv_min_per_ha REAL not null, npv_max_per_ha REAL not null, area_ha_sum REAL not null,
			bucket_area_share REAL not null,
			primary key ({ROTATION_YEAR_INT_COLUMN}, cluster_index, discount_rate, bucket_id))""")
		con.execute(f"create index if not exists idx_{BID_CURVE_TABLE}_rotation_cluster_rate_bucket on {BID_CURVE_TABLE} ({ROTATION_YEAR_INT_COLUMN}, cluster_index, discount_rate, bucket_id)")
		con.commit()

# ---------------------------------------------------------------------------------------------
# DONE. Step 1. Calculating a contract's NPV with a given interest rate introduces an error on using a different interest rate in SMDAMAGE.
# Solution: pre-compute the NPV cost of each contract for selected interest rates, for each cluster index. Each contract is discounted here to the contract start date.
# I can then bring these into SMDAMAGE directly and discount the contract from its start date to the present.
# So first I have to add the columns in database Busch2024_to_SMDAMAGE.sqlite.
# I added them, but didn't like the column names, so I renamed them.
# See cluster_index_count.txt for number of rows by cluster_index. Start with fewest rows to see how long it takes.
def write_single_rate_npv_to_table(db_file: Path = SOURCE_DB_FILE, discount_rate: float = DISCOUNT_RATE, cluster_index: int = 28) -> None:
	if discount_rate not in RATE_TO_NPV_COLUMN: raise ValueError(f"Unsupported discount rate: {discount_rate}")
	target_column = RATE_TO_NPV_COLUMN[discount_rate]
	
	with sqlite3.connect(db_file) as con:
		total_rows = 0
		annuity_factor_by_year: dict[int, float] = {}
		cursor = con.execute(f"select p.rowid, p.{ROTATION_YEAR_INT_COLUMN}, p.selected_option, p.crop_va_USD_per_ha_per_year, p.selected_establishment_cost_USD_per_ha, p.p_USD_per_tC_harvested, t.tC_per_ha_per_year from {PIXEL_TABLE} p left join {YEAR_TABLE} t on t.country = p.country and t.pixel_id = p.pixel_id and t.year = p.{ROTATION_YEAR_INT_COLUMN} where p.cluster_index = ?", (cluster_index,),)
		while True:
			rows = cursor.fetchmany(BATCH_SIZE)
			if len(rows) == 0: break

			updates: list[tuple[float, int]] = []
			for rowid, selected_rotation_year_int, selected_option, crop_va_USD_per_ha_per_year, selected_establishment_cost_USD_per_ha, p_USD_per_tC_harvested, tC_per_ha_per_year in rows:
				years = int(selected_rotation_year_int)
				if years < 1: raise ValueError(f"Invalid selected_rotation_year_int at rowid={rowid}: {selected_rotation_year_int}")
				if crop_va_USD_per_ha_per_year is None or selected_establishment_cost_USD_per_ha is None or p_USD_per_tC_harvested is None:
					raise ValueError(f"Missing required NPV input at rowid={rowid}")
				if selected_option == "plantation" and tC_per_ha_per_year is None:
					raise ValueError(f"Missing harvested tC at rowid={rowid}, rotation_year_int={years}")
				
				if years not in annuity_factor_by_year: annuity_factor_by_year[years] = sum(1.0 / ((1.0 + discount_rate) ** u) for u in range(years))

				npv_value_per_ha = (selected_establishment_cost_USD_per_ha + crop_va_USD_per_ha_per_year * annuity_factor_by_year[years]
					- (tC_per_ha_per_year if selected_option == "plantation" else 0.0)*(p_USD_per_tC_harvested * 1.0 / ((1.0 + discount_rate) ** (years - 1))))
				updates.append((npv_value_per_ha, rowid))

			con.executemany(f"update {PIXEL_TABLE} set {target_column} = ? where rowid = ?", updates,)
			total_rows += len(rows)
		con.commit()
		print(f"discount_rate={discount_rate}, cluster_index={cluster_index}")

# ---------------------------------------------------------------------------------------------
# 2. Summarize the costs into buckets. A bucket is a bid step.
# If you bucket by, say, $1000s, a few buckets in the middle have the most area. 
# So I switched to area-share buckets. Current default is 3% buckets and omitting the highest-cost 1% tail by area,
# yielding 33 buckets that cover the lowest-cost 99% area for each (selected_rotation_year_int, cluster_index, discount_rate).
def write_bid_steps_for_cluster(selected_rotation_year_int: int, cluster_index: int, discount_rate: float = 0.03, db_file: Path = SOURCE_DB_FILE,
	area_share_per_bucket: float = 0.03, high_cost_area_share_to_omit: float = 0.01, con: sqlite3.Connection | None = None) -> int:
	if selected_rotation_year_int < 1: raise ValueError(f"selected_rotation_year_int must be >= 1, got {selected_rotation_year_int}")
	if area_share_per_bucket <= 0.0 or area_share_per_bucket > 1.0:
		raise ValueError(f"area_share_per_bucket must be in (0, 1], got {area_share_per_bucket}")
	if high_cost_area_share_to_omit < 0.0 or high_cost_area_share_to_omit >= 1.0:
		raise ValueError(f"high_cost_area_share_to_omit must be in [0, 1), got {high_cost_area_share_to_omit}")
	if discount_rate not in RATE_TO_NPV_COLUMN: raise ValueError(f"Unsupported discount_rate: {discount_rate}")
	npv_column = RATE_TO_NPV_COLUMN[discount_rate]
	bucket_rows: list[tuple[int, int, float, int, str, float, float, float, float]] = []
	manage_connection = con is None
	if manage_connection: con = sqlite3.connect(db_file)
	try:
		total_area_row = con.execute(f"select sum(area_ha) from {PIXEL_TABLE} where {ROTATION_YEAR_INT_COLUMN} = ? and cluster_index = ? and {npv_column} is not null and area_ha is not null", (int(selected_rotation_year_int), int(cluster_index),)).fetchone()
		total_area_ha = float(total_area_row[0] if total_area_row is not None and total_area_row[0] is not None else 0.0)
		if total_area_ha <= 0.0:
			if manage_connection: con.commit()
			return 0
		included_area_ha_target = total_area_ha*(1.0 - high_cost_area_share_to_omit)
		target_bucket_area_ha = total_area_ha*area_share_per_bucket
		bucket_id = 1
		bucket_area_ha = 0.0
		bucket_npv_min = None
		bucket_npv_max = None
		included_area_ha = 0.0
		cursor = con.execute(f"select {npv_column}, area_ha from {PIXEL_TABLE} where {ROTATION_YEAR_INT_COLUMN} = ? and cluster_index = ? and {npv_column} is not null and area_ha is not null order by {npv_column}", (int(selected_rotation_year_int), int(cluster_index),))
		for npv_per_ha, area_ha in cursor:
			if included_area_ha >= included_area_ha_target: break
			npv_per_ha_float = float(npv_per_ha)
			area_ha_float = float(area_ha)
			remaining_included_area_ha = included_area_ha_target - included_area_ha
			area_ha_used = area_ha_float if area_ha_float <= remaining_included_area_ha else remaining_included_area_ha
			if bucket_npv_min is None: bucket_npv_min = npv_per_ha_float
			bucket_npv_max = npv_per_ha_float
			bucket_area_ha += area_ha_used
			included_area_ha += area_ha_used
			is_last_included_row = included_area_ha >= included_area_ha_target
			if bucket_area_ha >= target_bucket_area_ha or is_last_included_row:
				bucket_rows.append((int(selected_rotation_year_int), int(cluster_index), float(discount_rate), bucket_id, npv_column,
					float(bucket_npv_min), float(bucket_npv_max), float(bucket_area_ha),
					float(bucket_area_ha / total_area_ha)))
				bucket_id += 1
				bucket_area_ha = 0.0
				bucket_npv_min = None
				bucket_npv_max = None
		con.executemany(f"insert into {BID_CURVE_TABLE} ({ROTATION_YEAR_INT_COLUMN}, cluster_index, discount_rate, bucket_id, npv_column, npv_min_per_ha, npv_max_per_ha, area_ha_sum, bucket_area_share) values (?, ?, ?, ?, ?, ?, ?, ?, ?)", bucket_rows)
		if manage_connection: con.commit()
	finally:
		if manage_connection: con.close()
	return len(bucket_rows)

# -------------------------------------------------------------------------------
if __name__ == "__main__":
	# 0. Rename legacy NPV columns and ensure _per_ha columns exist.
	# rename_npv_columns_to_per_ha()

	# 1. Pre-computing NPV of each pixel for each discount rate, by cluster_index for batching.
	# See cluster_index_count.txt for number of rows by cluster_index. Start with fewest rows to see how long it takes.
	# write_single_rate_npv_to_table(discount_rate=0.0, cluster_index=28)
	# write_single_rate_npv_to_table(discount_rate=0.0, cluster_index=31)
	#...
	# write_single_rate_npv_to_table(discount_rate=0.06, cluster_index=17)
	# ....
	# write_single_rate_npv_to_table(discount_rate=0.06, cluster_index=5)

	# 2. Pull the curves. 
	# If you bucket by, say, $1000s, a few buckets in the middle have the most area. A bucket is a bid step.
	# So I switched to buckets of 3% area and omit the highest-cost 1% tail by area, to get 33 buckets.
	# Going in order of fewest rows to most, to see progress.
	# prepare_bid_curve_run(SOURCE_DB_FILE) # this burns down any existing forestry bid curves.
	# with sqlite3.connect(SOURCE_DB_FILE) as con:
	# 	rotation_cluster_pairs = [(int(rotation_year_int), int(cluster_index)) for rotation_year_int, cluster_index, _ in con.execute(
	# 		f"select {ROTATION_YEAR_INT_COLUMN}, cluster_index, count(*) as row_count from {PIXEL_TABLE} where cluster_index is not null and {ROTATION_YEAR_INT_COLUMN} is not null group by {ROTATION_YEAR_INT_COLUMN}, cluster_index order by row_count")]
	# 	print("Got cluster pairs")
	# 	batch_commit_size = 32
	# 	pairs_since_commit = 0
	# 	for discount_rate in [0.0, 0.015, 0.03, 0.06]:
	# 		for selected_rotation_year_int, cluster_index in rotation_cluster_pairs:
	# 			bucket_count = write_bid_steps_for_cluster(selected_rotation_year_int=selected_rotation_year_int, cluster_index=cluster_index, discount_rate=discount_rate, con=con)
	# 			pairs_since_commit += 1
	# 			if pairs_since_commit >= batch_commit_size:
	# 				con.commit()
	# 				pairs_since_commit = 0
	# 			print(f"discount_rate={discount_rate}, selected_rotation_year_int={selected_rotation_year_int}, cluster_index={cluster_index}, buckets={bucket_count}")
	# 	if pairs_since_commit > 0: con.commit()
	pass