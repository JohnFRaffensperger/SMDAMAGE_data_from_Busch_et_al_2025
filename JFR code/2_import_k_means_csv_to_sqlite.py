# 2_import_k_means_csv_to_sqlite.py | Made by Claude guided by JFR | Created 2026-04-18
# Imports completed k-means CSV outputs into Busch2024_to_SMDAMAGE.sqlite, populating carbon_removal_schedules and pixel cluster assignments.
from __future__ import annotations
import csv
from pathlib import Path
import re
import sqlite3

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Output"
DATABASE_DIR = OUTPUT_DIR / "Databases"
KMEANS_DIR = OUTPUT_DIR / "Kmeans_temp_files"
DEFAULT_DB_FILE = DATABASE_DIR / "Busch2024_to_SMDAMAGE.sqlite"
PIXEL_TABLE = "Undiscounted_dta_output"
SCHEDULE_YEAR_TABLE = "carbon_removal_schedules"
CLUSTER_INDEX_COLUMN = "cluster_index"
PIXEL_CLUSTER_INDEX = "idx_Undiscounted_dta_output_rotation_cluster_country_pixel"
SCHEDULE_YEAR_INDEX = "idx_carbon_removal_schedules_rotation_cluster_year"
DEFAULT_MAX_YEARS = 35

CSV_PATTERN = re.compile(r"k_means_carbon_removal_rotation_year_(?P<rotation>\d+)_k_(?P<k>\d+)_(?P<kind>centers|summary|assignments)\.csv$")

def get_table_column_names(con: sqlite3.Connection, table_name: str) -> set[str]:
	return {str(row[1]) for row in con.execute(f"pragma table_info('{table_name}')")}

def ensure_cluster_schema(con: sqlite3.Connection) -> None:
	if CLUSTER_INDEX_COLUMN not in get_table_column_names(con, PIXEL_TABLE):
		con.execute(f"alter table {PIXEL_TABLE} add column {CLUSTER_INDEX_COLUMN} INTEGER")
	con.execute(
		f"create table if not exists {SCHEDULE_YEAR_TABLE} ("
		"selected_rotation_year REAL NOT NULL, "
		"cluster_index INTEGER NOT NULL, "
		"year INTEGER NOT NULL, "
		"tC_per_ha_per_year REAL, "
		"pixel_count INTEGER, "
		"total_area_ha REAL)"
	)
	con.execute(f"create index if not exists {SCHEDULE_YEAR_INDEX} on {SCHEDULE_YEAR_TABLE} (selected_rotation_year, cluster_index, year)")
	con.execute(f"create index if not exists {PIXEL_CLUSTER_INDEX} on {PIXEL_TABLE} (selected_rotation_year, {CLUSTER_INDEX_COLUMN}, country, pixel_id)")

def clear_schedule_results_for_harvest_year(con: sqlite3.Connection, harvest_year: int) -> None:
	parameter = (float(harvest_year),)
	con.execute(f"delete from {SCHEDULE_YEAR_TABLE} where selected_rotation_year = ?", parameter)
	con.execute(f"update {PIXEL_TABLE} set {CLUSTER_INDEX_COLUMN} = null where selected_rotation_year = ?", parameter)

def discover_kmeans_csv_sets(output_dir: Path) -> list[dict[str, object]]:
	grouped = {}
	for path in output_dir.rglob("k_means_carbon_removal_rotation_year_*_k_*_*.csv"):
		match = CSV_PATTERN.match(path.name)
		if not match: continue
		rotation_year = int(match.group("rotation"))
		cluster_count = int(match.group("k"))
		kind = match.group("kind")
		key = (rotation_year, cluster_count)
		if key not in grouped: grouped[key] = {"rotation_year": rotation_year, "k": cluster_count}
		grouped[key][kind] = path
	required = {"centers", "summary", "assignments"}
	return [grouped[key] for key in sorted(grouped) if required.issubset(grouped[key].keys())]

def load_summary_lookup(summary_path: Path) -> dict[int, tuple[int, float]]:
	lookup = {}
	with open(summary_path, "r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			cluster_id = int(row["cluster_id"])
			lookup[cluster_id] = (int(row["pixel_count"]), float(row["total_area_ha"]))
	return lookup

def import_centers_csv_to_schedule_table(con: sqlite3.Connection, centers_path: Path, summary_lookup: dict[int, tuple[int, float]], max_years: int) -> tuple[int, int]:
	rows = []
	distinct_clusters = set()
	with open(centers_path, "r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			rotation_year = float(row["rotation_year"])
			cluster_id = int(row["cluster_id"])
			if cluster_id not in summary_lookup: raise ValueError(f"Missing summary row for cluster_id={cluster_id} in {centers_path.name}")
			pixel_count, total_area_ha = summary_lookup[cluster_id]
			distinct_clusters.add(cluster_id)
			for year in range(1, max_years + 1):
				rows.append((rotation_year, cluster_id, year, float(row[f"year_{year}"]), pixel_count, total_area_ha))
	con.executemany(
		f"insert into {SCHEDULE_YEAR_TABLE} (selected_rotation_year, cluster_index, year, tC_per_ha_per_year, pixel_count, total_area_ha) values (?, ?, ?, ?, ?, ?)",
		rows,
	)
	return len(rows), len(distinct_clusters)

def import_assignments_csv_to_pixel_table(con: sqlite3.Connection, assignments_path: Path, harvest_year: int, insert_batch_size: int = 100000) -> int:
	con.execute("create temporary table if not exists _cluster_assignments (country TEXT NOT NULL, pixel_id INTEGER NOT NULL, cluster_index INTEGER NOT NULL)")
	con.execute("create index if not exists temp.idx_cluster_assignments_country_pixel on _cluster_assignments (country, pixel_id)")
	con.execute("delete from _cluster_assignments")
	buffer = []
	inserted = 0
	with open(assignments_path, "r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			buffer.append((row["country"], int(row["pixel_id"]), int(row["cluster_id"])))
			if len(buffer) >= insert_batch_size:
				con.executemany("insert into _cluster_assignments (country, pixel_id, cluster_index) values (?, ?, ?)", buffer)
				inserted += len(buffer)
				buffer.clear()
	if buffer:
		con.executemany("insert into _cluster_assignments (country, pixel_id, cluster_index) values (?, ?, ?)", buffer)
		inserted += len(buffer)
	con.execute(
		f"""
		update {PIXEL_TABLE} as p
		set {CLUSTER_INDEX_COLUMN} = (
			select a.cluster_index from _cluster_assignments as a
			where a.country = p.country and a.pixel_id = p.pixel_id)
		where p.selected_rotation_year = ?
			and exists (
				select 1 from _cluster_assignments as a
				where a.country = p.country and a.pixel_id = p.pixel_id)
		""",
		(float(harvest_year),),
	)
	con.execute("delete from _cluster_assignments")
	return inserted

def verify_rotation_year_import(con: sqlite3.Connection, harvest_year: int, expected_clusters: int, expected_assignments: int, max_years: int) -> None:
	schedule_rows = con.execute(
		f"select count(*) from {SCHEDULE_YEAR_TABLE} where selected_rotation_year = ?",
		(float(harvest_year),),
	).fetchone()[0]
	cluster_count = con.execute(
		f"select count(distinct cluster_index) from {SCHEDULE_YEAR_TABLE} where selected_rotation_year = ?",
		(float(harvest_year),),
	).fetchone()[0]
	assigned_pixels = con.execute(
		f"select count(*) from {PIXEL_TABLE} where selected_rotation_year = ? and {CLUSTER_INDEX_COLUMN} is not null",
		(float(harvest_year),),
	).fetchone()[0]
	if schedule_rows != expected_clusters*max_years:
		raise ValueError(f"Rotation year {harvest_year}: expected {expected_clusters*max_years} schedule rows, found {schedule_rows}")
	if cluster_count != expected_clusters:
		raise ValueError(f"Rotation year {harvest_year}: expected {expected_clusters} clusters in schedule table, found {cluster_count}")
	if assigned_pixels != expected_assignments:
		raise ValueError(f"Rotation year {harvest_year}: expected {expected_assignments} assigned pixels, found {assigned_pixels}")

def import_one_csv_set(con: sqlite3.Connection, csv_set: dict[str, object], max_years: int) -> tuple[int, int, int]:
	rotation_year = int(csv_set["rotation_year"])
	clear_schedule_results_for_harvest_year(con, rotation_year)
	summary_lookup = load_summary_lookup(Path(csv_set["summary"]))
	schedule_rows, cluster_count = import_centers_csv_to_schedule_table(con, Path(csv_set["centers"]), summary_lookup, max_years)
	assignment_rows = import_assignments_csv_to_pixel_table(con, Path(csv_set["assignments"]), rotation_year)
	verify_rotation_year_import(con, rotation_year, cluster_count, assignment_rows, max_years)
	return schedule_rows, assignment_rows, cluster_count

def import_kmeans_csvs_to_db(db_file: Path = DEFAULT_DB_FILE, output_dir: Path = KMEANS_DIR, max_years: int = DEFAULT_MAX_YEARS) -> None:
	csv_sets = discover_kmeans_csv_sets(output_dir)
	if not csv_sets: raise ValueError(f"No complete k-means CSV sets found in {output_dir}")
	with sqlite3.connect(db_file) as con:
		ensure_cluster_schema(con)
		for csv_set in csv_sets:
			rotation_year = int(csv_set["rotation_year"])
			cluster_count = int(csv_set["k"])
			try:
				schedule_rows, assignment_rows, imported_clusters = import_one_csv_set(con, csv_set, max_years)
				con.commit()
				print(f"Rotation year {rotation_year}, k={cluster_count}: imported {schedule_rows} schedule rows, {assignment_rows} assignments, {imported_clusters} clusters.", flush=True)
			except Exception:
				con.rollback()
				raise

if __name__ == "__main__":
	import_kmeans_csvs_to_db()