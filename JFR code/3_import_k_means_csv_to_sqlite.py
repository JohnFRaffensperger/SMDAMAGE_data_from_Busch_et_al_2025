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
KMEANS_PATTERN = re.compile (r"k_means_carbon_removal_contract_years_(?P<years>\d+)_k_(?P<k>\d+)_(?P<kind>assignments|centers|summary)\.csv$")

def discover_undiscounted_files() -> list[Path]:
	files = [p for p in DATABASE_DIR.glob ("undiscounted_contracts_*.csv") if UNDISCOUNTED_PATTERN.match (p.name)]
	if not files: raise ValueError (f"No undiscounted_contracts_*.csv files found in {DATABASE_DIR}")
	return sorted (files)

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
	missing = [years for years, parts in by_years.items () if {"assignments", "centers", "summary"} - set (parts)]
	if missing: raise ValueError (f"Incomplete k-means file sets for contract_years={sorted (missing)}")
	if not by_years: raise ValueError (f"No k-means contract_years files found in {KMEANS_DIR}")
	return dict (sorted (by_years.items ()))

def rebuild_pixel_bids_table(con: sqlite3.Connection) -> None:
	con.execute ("drop table if exists Pixel_bids")
	con.execute (""" create table Pixel_bids (pixel_id INTEGER NOT NULL, cluster_id INTEGER, selected_option TEXT NOT NULL, contract_years INTEGER NOT NULL, area_ha REAL NOT NULL, NPV_0_pct_per_ha REAL NOT NULL, NPV_015_pct_per_ha REAL NOT NULL, NPV_03_pct_per_ha REAL NOT NULL, NPV_06_pct_per_ha REAL NOT NULL, PRIMARY KEY (pixel_id, contract_years)) """)
	con.execute ("create index idx_pixel_bids_contract_cluster on Pixel_bids (contract_years, cluster_id)")

def import_undiscounted_csvs(con: sqlite3.Connection, files: list[Path], batch_size: int = 100000) -> None:
	insert_sql = ("insert into Pixel_bids " "(pixel_id, selected_option, contract_years, area_ha, NPV_0_pct_per_ha, NPV_015_pct_per_ha, NPV_03_pct_per_ha, NPV_06_pct_per_ha) " "values (?, ?, ?, ?, ?, ?, ?, ?)")
	for csv_path in files:
		rows: list[tuple[int, str, int, float, float, float, float, float]] = []
		with open (csv_path, "r", newline = "", encoding = "utf-8") as handle:
			reader = csv.DictReader (handle)
			for row in reader:
				rows.append((int (row["pixel_id"]), row["option"], int (row["best_contract_length"]), float (row["area_ha"]), float (row["bid_r000"]), float (row["bid_r015"]), float (row["bid_r030"]), float (row["bid_r060"]),))
				if len(rows) >= batch_size:
					con.executemany (insert_sql, rows)
					rows.clear()
		if rows: con.executemany (insert_sql, rows)

def rebuild_carbon_table_for_years(con: sqlite3.Connection, years: int, centers_path: Path) -> int:
	table_name = f"cluster_carbon_schedules_{years}"
	con.execute (f"drop table if exists {table_name}")
	con.execute (f""" create table {table_name} (cluster_id INTEGER NOT NULL, growth_year INTEGER NOT NULL, tC_per_ha_per_year REAL NOT NULL, PRIMARY KEY (cluster_id, growth_year)) """)
	rows: list[tuple[int, int, float]] = []
	with open (centers_path, "r", newline = "", encoding = "utf-8") as handle:
		reader = csv.DictReader (handle)
		year_columns = [name for name in reader.fieldnames or [] if name.startswith ("year_")]
		if not year_columns: raise ValueError (f"No year_* columns in {centers_path}")
		if len (year_columns) != years: raise ValueError (f"contract_years={years} expected {years} year columns, found {len (year_columns)} in {centers_path.name}")
		for row in reader:
			cluster_id = int (row["cluster_id"])
			for growth_year, col in enumerate (year_columns, start = 1): rows.append ((cluster_id, growth_year, float (row[col])))
	con.executemany (f"insert into {table_name} (cluster_id, growth_year, tC_per_ha_per_year) values (?, ?, ?)", rows,)
	con.execute (f"create index idx_{table_name}_cluster_growth on {table_name} (cluster_id, growth_year)")
	return len (rows)

def load_cluster_set_from_csv(path: Path, id_column: str = "cluster_id") -> set[int]:
	ids: set[int] = set()
	with open (path, "r", newline = "", encoding = "utf-8") as handle:
		reader = csv.DictReader (handle)
		for row in reader: ids.add (int (row[id_column]))
	return ids

def stage_assignments(con: sqlite3.Connection, assignments_path: Path, batch_size: int = 100000) -> int:
	con.execute ("drop table if exists _tmp_assignments")
	con.execute (""" create temporary table _tmp_assignments (contract_years INTEGER NOT NULL, pixel_id INTEGER NOT NULL, cluster_id INTEGER NOT NULL, PRIMARY KEY (contract_years, pixel_id)) """)
	rows: list[tuple[int, int, int]] = []
	total_rows = 0
	with open (assignments_path, "r", newline = "", encoding = "utf-8") as handle:
		reader = csv.DictReader (handle)
		for row in reader:
			rows.append ((int (row["contract_years"]), int (row["pixel_id"]), int (row["cluster_id"])))
			if len(rows) >= batch_size:
				con.executemany ("insert into _tmp_assignments (contract_years, pixel_id, cluster_id) values (?, ?, ?)", rows,)
				total_rows += len (rows)
				rows.clear()
	if rows:
		con.executemany ("insert into _tmp_assignments (contract_years, pixel_id, cluster_id) values (?, ?, ?)", rows)
		total_rows += len (rows)
	return total_rows

def verify_years_checks(con: sqlite3.Connection, years: int, centers_path: Path, summary_path: Path, assignment_rows: int) -> None:
	pixel_rows = con.execute ("select count(*) from Pixel_bids where contract_years = ?", (years,)).fetchone ()[0]
	if assignment_rows != pixel_rows: raise ValueError (f"contract_years={years}: assignments row count {assignment_rows} != Pixel_bids rows {pixel_rows}")
	centers_clusters = load_cluster_set_from_csv (centers_path)
	summary_clusters = load_cluster_set_from_csv (summary_path)
	assignment_clusters = {row[0] for row in con.execute ("select distinct cluster_id from _tmp_assignments where contract_years = ?", (years,),)}
	if not (centers_clusters == summary_clusters == assignment_clusters): raise ValueError (f"contract_years={years}: distinct cluster_id mismatch across centers/summary/assignments " f"({len(centers_clusters)}, {len(summary_clusters)}, {len(assignment_clusters)})")
	missing_pixels = con.execute (""" select count(*) from _tmp_assignments a left join Pixel_bids p on p.pixel_id = a.pixel_id and p.contract_years = a.contract_years where a.contract_years = ? and p.pixel_id is null """, (years,), ).fetchone ()[0]
	if missing_pixels != 0: raise ValueError (f"contract_years={years}: {missing_pixels} assignment pixel_id values missing from Pixel_bids")

def apply_assignments(con: sqlite3.Connection, years: int) -> int:
	con.execute (""" update Pixel_bids set cluster_id = (select a.cluster_id from _tmp_assignments a where a.contract_years = Pixel_bids.contract_years and a.pixel_id = Pixel_bids.pixel_id) where contract_years = ? """, (years,),)
	assigned = con.execute ("select count(*) from Pixel_bids where contract_years = ? and cluster_id is not null", (years,), ).fetchone ()[0]
	return assigned

def verify_global_checks(con: sqlite3.Connection) -> None:
	duplicate_rows = con.execute (""" select count(*) from (select pixel_id, contract_years, count(*) c from Pixel_bids group by pixel_id, contract_years having c > 1) """ ).fetchone ()[0]
	if duplicate_rows != 0: raise ValueError (f"Pixel_bids has {duplicate_rows} duplicate (pixel_id, contract_years) keys")
	null_clusters = con.execute ("select count(*) from Pixel_bids where cluster_id is null").fetchone ()[0]
	if null_clusters != 0: raise ValueError (f"Pixel_bids has {null_clusters} rows with null cluster_id")

def run_loader(db_file: Path = DB_FILE) -> None:
	DATABASE_DIR.mkdir (parents = True, exist_ok = True)
	kmeans_sets = discover_kmeans_sets ()
	undiscounted_files = discover_undiscounted_files ()
	with sqlite3.connect (db_file) as con:
		rebuild_pixel_bids_table (con)
		import_undiscounted_csvs (con, undiscounted_files)
		for years, parts in kmeans_sets.items():
			carbon_rows = rebuild_carbon_table_for_years (con, years, parts["centers"])
			assignment_rows = stage_assignments (con, parts["assignments"])
			verify_years_checks (con, years, parts["centers"], parts["summary"], assignment_rows)
			assigned_rows = apply_assignments (con, years)
			con.execute ("drop table if exists _tmp_assignments")
			print (f"contract_years={years}: carbon_rows={carbon_rows}, assignments={assignment_rows}, assigned_pixel_rows={assigned_rows}", flush = True,)
		verify_global_checks (con)
		con.commit()
		row_count = con.execute ("select count(*) from Pixel_bids").fetchone ()[0]
		print (f"Pixel_bids rows: {row_count}", flush = True)
		print (f"Loader complete: {db_file}", flush = True)

if __name__ == "__main__": run_loader ()
