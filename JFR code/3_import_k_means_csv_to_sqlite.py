# 3_import_k_means_csv_to_sqlite.py
from __future__ import annotations
import csv
from pathlib import Path
import re
import sqlite3

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Output"
DATABASE_DIR = OUTPUT_DIR / "Databases"
KMEANS_DIR = OUTPUT_DIR / "Kmeans_temp_files"
DB_FILE = DATABASE_DIR / "Busch2024_to_SMDAMAGE.sqlite"

UNDISCOUNTED_PATTERN = re.compile (r"undiscounted_contracts_(?P<years>\d+)\.csv$")
KMEANS_PATTERN = re.compile (r"k_means_carbon_removal_contract_years_(?P<years>\d+)_k_(?P<k>\d+)_(?P<kind>assignments|centers|overall)\.csv$")

def discover_undiscounted_files() -> dict[int, Path]:
	by_years: dict[int, Path] = {}
	for p in DATABASE_DIR.glob ("undiscounted_contracts_*.csv"):
		match = UNDISCOUNTED_PATTERN.match (p.name)
		if not match: continue
		years = int (match.group ("years"))
		if years in by_years: raise ValueError (f"Multiple undiscounted files found for contract_years={years}")
		by_years[years] = p
	if not by_years: raise ValueError (f"No undiscounted_contracts_*.csv files found in {DATABASE_DIR}")
	return dict (sorted (by_years.items ()))

def discover_kmeans_sets() -> dict[int, dict[str, Path]]:
	by_years: dict[int, dict[str, Path]] = {}
	for p in KMEANS_DIR.glob ("k_means_carbon_removal_contract_years_*_k_*_*.csv"):
		match = KMEANS_PATTERN.match (p.name)
		if not match: continue
		years = int (match.group ("years"))
		kind = match.group ("kind")
		bucket = by_years.setdefault (years, {})
		if kind in bucket: raise ValueError (f"Multiple {kind} files found for contract_years={years}")
		bucket[kind] = p
	missing = [years for years, parts in by_years.items () if {"assignments", "centers", "overall"} - set (parts)]
	if missing: raise ValueError (f"Incomplete k-means file sets for contract_years={sorted (missing)}")
	if not by_years: raise ValueError (f"No k-means contract_years files found in {KMEANS_DIR}")
	return dict (sorted (by_years.items ()))

def rebuild_pixel_bids_table(con: sqlite3.Connection) -> None:
	con.execute ("drop table if exists Pixel_bids")
	con.execute ("""create table Pixel_bids (pixel_id INTEGER NOT NULL, cluster_id INTEGER, selected_option TEXT NOT NULL, contract_years INTEGER NOT NULL, area_ha REAL NOT NULL, NPV_0_pct_per_ha REAL NOT NULL, NPV_015_pct_per_ha REAL NOT NULL, NPV_03_pct_per_ha REAL NOT NULL, NPV_06_pct_per_ha REAL NOT NULL, PRIMARY KEY (pixel_id, contract_years)) """)

def apply_bulk_load_pragmas(con: sqlite3.Connection) -> None:
	con.execute ("pragma journal_mode = OFF")
	con.execute ("pragma synchronous = OFF")
	con.execute ("pragma temp_store = MEMORY")
	con.execute ("pragma locking_mode = EXCLUSIVE")
	con.execute ("pragma cache_size = -262144")

def load_centers_and_write_carbon_table(con: sqlite3.Connection, years: int, centers_path: Path, batch_size: int = 100000) -> int:
	table_name = f"cluster_carbon_schedules_{years}"
	con.execute (f"drop table if exists {table_name}")
	con.execute (f""" create table {table_name} (cluster_id INTEGER NOT NULL, growth_year INTEGER NOT NULL, tC_per_ha_per_year REAL NOT NULL, PRIMARY KEY (cluster_id, growth_year)) """)
	with open (centers_path, "r", newline = "", encoding = "utf-8") as handle:
		center_rows = list (csv.DictReader (handle))
	if not center_rows: return 0
	if not center_rows[0]: return 0
	year_columns = [name for name in center_rows[0] if name.startswith ("year_")]
	rows: list[tuple[int, int, float]] = []
	total_rows = 0
	max_year_cols = min (years, len(year_columns))
	for row in center_rows:
		cluster_id = int (row["cluster_id"])
		for growth_year in range (max_year_cols): rows.append ((cluster_id, growth_year + 1, float (row[year_columns[growth_year]])))
		if len(rows) >= batch_size:
			con.executemany (f"insert into {table_name} (cluster_id, growth_year, tC_per_ha_per_year) values (?, ?, ?)", rows)
			total_rows += len (rows)
			rows.clear()
	if rows:
		con.executemany (f"insert into {table_name} (cluster_id, growth_year, tC_per_ha_per_year) values (?, ?, ?)", rows)
		total_rows += len (rows)
	return total_rows

def load_overall_csv(overall_path: Path) -> dict[str, str]:
	with open (overall_path, "r", newline = "", encoding = "utf-8") as handle:
		rows = list (csv.DictReader (handle))
	if not rows: return {}
	return rows[0]

def insert_pixel_bids_for_year_from_streams(con: sqlite3.Connection, years: int, undiscounted_path: Path, assignments_path: Path, batch_size: int = 100000) -> tuple[int, int]:
	insert_sql = ("insert into Pixel_bids " "(pixel_id, cluster_id, selected_option, contract_years, area_ha, NPV_0_pct_per_ha, NPV_015_pct_per_ha, NPV_03_pct_per_ha, NPV_06_pct_per_ha) " "values (?, ?, ?, ?, ?, ?, ?, ?, ?)")
	rows: list[tuple[int, int, str, int, float, float, float, float, float]] = []
	inserted_rows = 0
	assignment_rows = 0
	with open (assignments_path, "r", newline = "", encoding = "utf-8") as handle:
		assignment_reader = csv.reader (handle)
		assignment_header = next (assignment_reader, None)
		if assignment_header is None: return 0, 0
		assignment_index = {name: idx for idx, name in enumerate (assignment_header)}
		a_pixel_id = assignment_index["pixel_id"]
		a_cluster_id = assignment_index["cluster_id"]
		with open (undiscounted_path, "r", newline = "", encoding = "utf-8") as undiscounted_handle:
			undiscounted_reader = csv.reader (undiscounted_handle)
			undiscounted_header = next (undiscounted_reader, None)
			if undiscounted_header is None: return 0, 0
			undiscounted_index = {name: idx for idx, name in enumerate (undiscounted_header)}
			u_pixel_id = undiscounted_index["pixel_id"]
			u_option = undiscounted_index["option"]
			u_area_ha = undiscounted_index["area_ha"]
			u_bid_r000 = undiscounted_index["bid_r000"]
			u_bid_r015 = undiscounted_index["bid_r015"]
			u_bid_r030 = undiscounted_index["bid_r030"]
			u_bid_r060 = undiscounted_index["bid_r060"]

			assignment_row = next (assignment_reader, None)
			undiscounted_row = next (undiscounted_reader, None)
			while assignment_row is not None and undiscounted_row is not None:
				if not assignment_row:
					assignment_row = next (assignment_reader, None)
					continue
				if not undiscounted_row:
					undiscounted_row = next (undiscounted_reader, None)
					continue
				assignment_pid = int (assignment_row[a_pixel_id])
				undiscounted_pid = int (undiscounted_row[u_pixel_id])
				if assignment_pid == undiscounted_pid:
					rows.append ((undiscounted_pid, int (assignment_row[a_cluster_id]), undiscounted_row[u_option], years, float (undiscounted_row[u_area_ha]), float (undiscounted_row[u_bid_r000]), float (undiscounted_row[u_bid_r015]), float (undiscounted_row[u_bid_r030]), float (undiscounted_row[u_bid_r060]),))
					assignment_rows += 1
					if len(rows) >= batch_size:
						con.executemany (insert_sql, rows)
						inserted_rows += len (rows)
						rows.clear()
					assignment_row = next (assignment_reader, None)
					undiscounted_row = next (undiscounted_reader, None)
					continue
				if assignment_pid < undiscounted_pid:
					assignment_row = next (assignment_reader, None)
					continue
				undiscounted_row = next (undiscounted_reader, None)
	if rows:
		con.executemany (insert_sql, rows)
		inserted_rows += len (rows)
	return assignment_rows, inserted_rows

def create_pixel_bids_indexes(con: sqlite3.Connection) -> None:
	print ("Creating Pixel_bids covering indexes (this will take a few minutes)...", flush = True)
	for suffix, col in [("00", "NPV_0_pct_per_ha"), ("015", "NPV_015_pct_per_ha"), ("03", "NPV_03_pct_per_ha"), ("06", "NPV_06_pct_per_ha")]:
		name = f"idx_pixel_bids_cover_{suffix}"
		con.execute (f"create index if not exists {name} on Pixel_bids (contract_years, cluster_id, {col}, pixel_id, area_ha)")
		print (f"  created {name}", flush = True)
	print ("Pixel_bids indexes ready.", flush = True)

def run_loader(db_file: Path = DB_FILE) -> None:
	DATABASE_DIR.mkdir (parents = True, exist_ok = True)
	kmeans_sets = discover_kmeans_sets ()
	undiscounted_files = discover_undiscounted_files ()
	print ("run_loader: insert_only=true, checks=off, temp_tables=off, indexes=off", flush = True)
	with sqlite3.connect (db_file) as con:
		apply_bulk_load_pragmas (con)
		rebuild_pixel_bids_table (con)
		for years, parts in kmeans_sets.items():
			undiscounted_path = undiscounted_files[years]
			overall = load_overall_csv (parts["overall"])
			carbon_rows = load_centers_and_write_carbon_table (con, years, parts["centers"])
			assignment_rows, inserted_rows = insert_pixel_bids_for_year_from_streams (con, years, undiscounted_path, parts["assignments"])
			overall_pixels = overall.get ("pixel_count", "")
			print (f"contract_years={years}: carbon_rows={carbon_rows}, assignment_rows={assignment_rows}, inserted_rows={inserted_rows}, overall_pixel_count={overall_pixels}", flush = True,)
			con.commit ()
		con.commit()
		row_count = con.execute ("select count(*) from Pixel_bids").fetchone ()[0]
		print (f"Pixel_bids rows: {row_count}", flush = True)
		print (f"Loader complete: {db_file}", flush = True)

if __name__ == "__main__":

	# First run_loader, then index the tables.
	with sqlite3.connect (DB_FILE) as con:
		print("Indexing...")
		create_pixel_bids_indexes (con)
		con.commit ()
