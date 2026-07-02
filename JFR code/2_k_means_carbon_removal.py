# 2_k_means_carbon_removal.py | Made by Claude guided by JFR | Created 2026-04-15
# Loads one rotation year's pixel carbon-removal profiles, evaluates multiple full k-means schedule sizes, and optionally writes comparison outputs to CSV.
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Output"
DATABASE_DIR = OUTPUT_DIR / "Databases"
KMEANS_DIR = OUTPUT_DIR / "Kmeans_temp_files"
DEFAULT_DB_FILE = DATABASE_DIR / "Busch2024_to_SMDAMAGE.sqlite"
PIXEL_TABLE = "Undiscounted_dta_output"
YEAR_TABLE = "tC_per_h_per_year"
SCHEDULE_YEAR_TABLE = "carbon_removal_schedules"
CLUSTER_INDEX_COLUMN = "cluster_index"
PIXEL_ROTATION_INDEX = "idx_Undiscounted_dta_output_rotation_country_pixel_area"
PIXEL_ROTATION_INDEX_FALLBACK = "idx_Undiscounted_dta_output_rotation_country_pixel"
PIXEL_CLUSTER_INDEX = "idx_Undiscounted_dta_output_rotation_cluster_country_pixel"
SCHEDULE_YEAR_INDEX = "idx_carbon_removal_schedules_rotation_cluster_year"
DEFAULT_HARVEST_YEARS = [6]
DEFAULT_MAX_YEARS = 101
DEFAULT_OUTPUT_PREFIX = KMEANS_DIR / "k_means_carbon_removal"

@dataclass(slots=True)
class RunConfig:
	db_file: Path = DEFAULT_DB_FILE
	k_values: list[int] = None
	rotation_year: int = DEFAULT_HARVEST_YEARS[0]
	max_years: int = DEFAULT_MAX_YEARS
	profile_source: str = "pixel"
	load_method: str = "pivot"
	max_iter: int = 4000 # How long you want this to run.
	tol: float = 1e-6 # Optimization criterion, smaller is slower and lower error.
	seed: int = 0
	normalize: bool = False
	write_to_CSV: bool = True
	output_prefix: Path = DEFAULT_OUTPUT_PREFIX

	def __post_init__(self) -> None:
		if self.k_values is None: self.k_values = [20]

CONFIG = RunConfig()

def squared_distance_argmin(batch: np.ndarray, centers: np.ndarray) -> np.ndarray:
	batch_sq = np.sum(batch*batch, axis=1, keepdims=True)
	center_sq = np.sum(centers*centers, axis=1)
	distances = batch_sq + center_sq[np.newaxis, :] - 2.0*batch @ centers.T
	return np.argmin(distances, axis=1)

def l2_normalize_rows(data: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(data, axis=1, keepdims=True)
	norms[norms == 0] = 1.0
	return data / norms

def initialize_centers(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
	if len(data) == 0: raise ValueError("Cannot initialize centers from empty data.")
	actual_k = min(k, len(data))
	indices = rng.choice(len(data), size=actual_k, replace=False)
	return data[indices].copy()

def assign_all_labels(data: np.ndarray, centers: np.ndarray, chunk_size: int = 16384) -> np.ndarray:
	labels = np.empty(len(data), dtype=np.int32)
	for start in range(0, len(data), chunk_size):
		stop = min(start + chunk_size, len(data))
		labels[start:stop] = squared_distance_argmin(data[start:stop], centers)
	return labels

def assigned_squared_distances(data: np.ndarray, centers: np.ndarray, labels: np.ndarray, chunk_size: int = 16384) -> np.ndarray:
	distances = np.empty(len(data), dtype=np.float64)
	for start in range(0, len(data), chunk_size):
		stop = min(start + chunk_size, len(data))
		chunk = data[start:stop]
		chunk_centers = centers[labels[start:stop]]
		delta = chunk - chunk_centers
		distances[start:stop] = np.sum(delta*delta, axis=1)
	return distances

def recompute_centers(data: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	new_centers = centers.copy()
	counts = np.bincount(labels, minlength=len(centers)).astype(np.int64)
	for cluster_id in range(len(centers)):
		if counts[cluster_id] == 0: continue
		new_centers[cluster_id] = data[labels == cluster_id].mean(axis=0)
	return new_centers, counts

def run_kmeans(data: np.ndarray, k: int, max_iter: int, tol: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
	if len(data) == 0: raise ValueError("No rows available for clustering.")
	rng = np.random.default_rng(seed)
	centers = initialize_centers(data, k, rng)
	iterations_run = 0
	for iteration in range(max_iter):
		iterations_run = iteration + 1
		labels = assign_all_labels(data, centers)
		old_centers = centers.copy()
		centers, counts = recompute_centers(data, labels, centers)
		max_shift = np.max(np.linalg.norm(centers - old_centers, axis=1)) if len(centers) else 0.0
		if max_shift <= tol: break
	final_labels = assign_all_labels(data, centers)
	centers, final_counts = recompute_centers(data, final_labels, centers)
	final_labels = assign_all_labels(data, centers)
	return centers, final_labels, final_counts, iterations_run

def select_pixel_rotation_index(con: sqlite3.Connection) -> tuple[str | None, bool]:
	index_names = {row[1] for row in con.execute(f"pragma index_list('{PIXEL_TABLE}')")}
	if PIXEL_ROTATION_INDEX in index_names: return PIXEL_ROTATION_INDEX, False
	if PIXEL_ROTATION_INDEX_FALLBACK in index_names: return PIXEL_ROTATION_INDEX_FALLBACK, True
	return None, False

def get_table_column_names(con: sqlite3.Connection, table_name: str) -> set[str]:
	return {str(row[1]) for row in con.execute(f"pragma table_info('{table_name}')")}

def ensure_cluster_schema(con: sqlite3.Connection) -> None:
	if CLUSTER_INDEX_COLUMN not in get_table_column_names(con, PIXEL_TABLE):
		con.execute(f"alter table {PIXEL_TABLE} add column {CLUSTER_INDEX_COLUMN} INTEGER")
	con.execute(f"create table if not exists {SCHEDULE_YEAR_TABLE} (selected_rotation_year REAL NOT NULL,"
		"cluster_index INTEGER NOT NULL, year INTEGER NOT NULL, tC_per_ha_per_year REAL, pixel_count INTEGER, total_area_ha REAL)")
	con.execute(f"create index if not exists {SCHEDULE_YEAR_INDEX} on {SCHEDULE_YEAR_TABLE} (selected_rotation_year, cluster_index, year)")
	con.execute(f"create index if not exists {PIXEL_CLUSTER_INDEX} on {PIXEL_TABLE} (selected_rotation_year, {CLUSTER_INDEX_COLUMN}, country, pixel_id)")

def clear_schedule_results_for_harvest_year(con: sqlite3.Connection, harvest_year: int) -> None:
	parameter = (float(harvest_year),)
	con.execute(f"delete from {SCHEDULE_YEAR_TABLE} where selected_rotation_year = ?", parameter)
	con.execute(f"update {PIXEL_TABLE} set {CLUSTER_INDEX_COLUMN} = null where selected_rotation_year = ?", parameter)

def write_schedule_results_to_db(con: sqlite3.Connection, harvest_year: int, centers: np.ndarray, keys: list[tuple[str, int]], labels: np.ndarray, areas: np.ndarray, counts: np.ndarray, max_years: int) -> tuple[int, int]:
	clear_schedule_results_for_harvest_year(con, harvest_year)
	area_totals = np.bincount(labels, weights=areas, minlength=len(centers))
	schedule_rows = []
	for cluster_id, center in enumerate(centers):
		for year in range(1, max_years + 1):
			schedule_rows.append((float(harvest_year), int(cluster_id), int(year), float(center[year - 1]), int(counts[cluster_id]), float(area_totals[cluster_id]),))
	con.executemany(f"insert into {SCHEDULE_YEAR_TABLE} (selected_rotation_year, cluster_index, year, tC_per_ha_per_year, pixel_count, total_area_ha) values (?, ?, ?, ?, ?, ?)", schedule_rows, )
	con.execute("create temporary table if not exists _cluster_assignments (country TEXT NOT NULL, pixel_id INTEGER NOT NULL, cluster_index INTEGER NOT NULL)")
	con.execute("delete from _cluster_assignments")
	con.executemany("insert into _cluster_assignments (country, pixel_id, cluster_index) values (?, ?, ?)", [(country, int(pixel_id), int(cluster_id)) for (country, pixel_id), cluster_id in zip(keys, labels)], )
	con.execute(f"""update {PIXEL_TABLE} as p set {CLUSTER_INDEX_COLUMN} = (select a.cluster_index from _cluster_assignments as a
			where a.country = p.country and a.pixel_id = p.pixel_id) where p.selected_rotation_year = ?
			and exists (select 1 from _cluster_assignments as a where a.country = p.country and a.pixel_id = p.pixel_id)""",
		(float(harvest_year),),)
	con.execute("delete from _cluster_assignments")
	return len(schedule_rows), len(keys)

def iter_profiles_for_harvest_year(con: sqlite3.Connection, harvest_year: int, max_years: int, pixel_index: str | None, needs_cast: bool):
	index_clause = f" indexed by {pixel_index}" if pixel_index else ""
	where_clause = "cast(p.selected_rotation_year as integer) = ?" if needs_cast else "p.selected_rotation_year = ?"
	parameter = harvest_year if needs_cast else float(harvest_year)
	query = f"""select p.country, p.pixel_id, p.area_ha, y.year, y.tC_per_ha_per_year from {PIXEL_TABLE} as p{index_clause}
		join {YEAR_TABLE} y on y.country = p.country and y.pixel_id = p.pixel_id where {where_clause} order by p.country, p.pixel_id, y.year"""
	cursor = con.execute(query, (parameter,))
	current_key = None
	current_area = 0.0
	current_vector = np.zeros(max_years, dtype=np.float32)
	for country, pixel_id, area_ha, year, value in cursor:
		key = (str(country), int(pixel_id))
		if current_key is None:
			current_key = key
			current_area = float(area_ha or 0.0)
		elif key != current_key:
			yield current_key[0], current_key[1], current_area, current_vector
			current_key = key
			current_area = float(area_ha or 0.0)
			current_vector = np.zeros(max_years, dtype=np.float32)
		if 1 <= int(year) <= max_years: current_vector[int(year) - 1] = float(value or 0.0)
	if current_key is not None: yield current_key[0], current_key[1], current_area, current_vector

def load_profiles_for_harvest_year_rowwise(con: sqlite3.Connection, harvest_year: int, max_years: int, normalize: bool, pixel_index: str | None, needs_cast: bool) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray]:
	keys = []
	areas = []
	vectors = []
	for country, pixel_id, area_ha, vector in iter_profiles_for_harvest_year(con, harvest_year, max_years, pixel_index, needs_cast):
		keys.append((country, pixel_id))
		areas.append(area_ha)
		vectors.append(vector)
	if not vectors: return [], np.zeros((0, max_years), dtype=np.float32), np.zeros(0, dtype=np.float32)
	data = np.vstack(vectors).astype(np.float32, copy=False)
	if normalize: data = l2_normalize_rows(data).astype(np.float32, copy=False)
	return keys, data, np.asarray(areas, dtype=np.float32)

def load_profiles_for_harvest_year_pivot(con: sqlite3.Connection, harvest_year: int, max_years: int, normalize: bool, pixel_index: str | None, needs_cast: bool) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray]:
	keys = []
	areas = []
	vectors = []
	index_clause = f" indexed by {pixel_index}" if pixel_index else ""
	where_clause = "cast(p.selected_rotation_year as integer) = ?" if needs_cast else "p.selected_rotation_year = ?"
	parameter = harvest_year if needs_cast else float(harvest_year)
	year_columns = ", ".join([f"coalesce(max(case when y.year = {year} then y.tC_per_ha_per_year end), 0.0) as year_{year}"
		for year in range(1, max_years + 1)])
	query = f"""select p.country, p.pixel_id, p.area_ha, {year_columns} from {PIXEL_TABLE} as p{index_clause}
		left join {YEAR_TABLE} as y on y.country = p.country and y.pixel_id = p.pixel_id and y.year between 1 and {max_years}
		where {where_clause} group by p.country, p.pixel_id, p.area_ha order by p.country, p.pixel_id"""
	for row in con.execute(query, (parameter,)):
		country, pixel_id, area_ha = row[:3]
		keys.append((str(country), int(pixel_id)))
		areas.append(float(area_ha or 0.0))
		vectors.append(np.asarray(row[3:], dtype=np.float32))
	if not vectors: return [], np.zeros((0, max_years), dtype=np.float32), np.zeros(0, dtype=np.float32)
	data = np.vstack(vectors).astype(np.float32, copy=False)
	if normalize: data = l2_normalize_rows(data).astype(np.float32, copy=False)
	return keys, data, np.asarray(areas, dtype=np.float32)

def load_schedule_profiles_for_harvest_year(con: sqlite3.Connection, harvest_year: int, max_years: int, normalize: bool) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray]:
	keys = []
	areas = []
	vectors = []
	year_columns = ", ".join([f"coalesce(max(case when year = {year} then tC_per_ha_per_year end), 0.0) as year_{year}" for year in range(1, max_years + 1)])
	query = f"""select cluster_index, max(total_area_ha) as total_area_ha, {year_columns} from {SCHEDULE_YEAR_TABLE}
		where selected_rotation_year = ? group by cluster_index order by cluster_index"""
	for row in con.execute(query, (float(harvest_year),)):
		cluster_index, total_area_ha = row[:2]
		keys.append(("schedule", int(cluster_index)))
		areas.append(float(total_area_ha or 0.0))
		vectors.append(np.asarray(row[2:], dtype=np.float32))
	if not vectors: return [], np.zeros((0, max_years), dtype=np.float32), np.zeros(0, dtype=np.float32)
	data = np.vstack(vectors).astype(np.float32, copy=False)
	if normalize: data = l2_normalize_rows(data).astype(np.float32, copy=False)
	return keys, data, np.asarray(areas, dtype=np.float32)

def load_profiles_for_harvest_year(con: sqlite3.Connection, harvest_year: int, max_years: int, normalize: bool, load_method: str, profile_source: str) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray]:
	if profile_source == "pixel":
		pixel_index, needs_cast = select_pixel_rotation_index(con)
		if load_method == "pivot": return load_profiles_for_harvest_year_pivot(con, harvest_year, max_years, normalize, pixel_index, needs_cast)
		if load_method == "rowwise": return load_profiles_for_harvest_year_rowwise(con, harvest_year, max_years, normalize, pixel_index, needs_cast)
		raise ValueError(f"Unsupported load method: {load_method}")
	if profile_source == "schedule": return load_schedule_profiles_for_harvest_year(con, harvest_year, max_years, normalize)
	raise ValueError(f"Unsupported profile source: {profile_source}")

def write_cluster_centers_csv(path: Path, rows: list[dict[str, object]], max_years: int) -> None:
	fieldnames = ["rotation_year", "k", "cluster_id"] + [f"year_{year}" for year in range(1, max_years + 1)]
	with open(path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def write_assignments_csv(path: Path, rows: list[dict[str, object]]) -> None:
	fieldnames = ["rotation_year", "k", "country", "pixel_id", "area_ha", "cluster_id"]
	with open(path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
	fieldnames = ["rotation_year", "k", "cluster_id", "pixel_count", "total_area_ha", "iterations_run",
		"cluster_total_squared_distance", "cluster_mean_squared_distance", "cluster_rmse_per_year",
		"cluster_max_squared_distance", "cluster_max_distance",]
	with open(path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def write_overall_csv(path: Path, rows: list[dict[str, object]]) -> None:
	fieldnames = ["rotation_year", "k", "pixel_count", "cluster_count", "iterations_run", "total_squared_distance", 
		"mean_squared_distance", "rmse_per_year", "max_squared_distance", "max_distance",]
	with open(path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def validate_config(config: RunConfig) -> None:
	if not config.k_values: raise ValueError("k_values must contain at least one cluster size.")
	if any(k <= 0 for k in config.k_values): raise ValueError("All k values must be positive.")
	if config.rotation_year <= 0: raise ValueError("rotation_year must be positive.")
	if config.max_years <= 0: raise ValueError("max_years must be positive.")
	if config.max_iter <= 0: raise ValueError("max_iter must be positive.")
	if config.profile_source not in {"pixel", "schedule"}: raise ValueError("profile_source must be 'pixel' or 'schedule'.")
	if config.load_method not in {"pivot", "rowwise"}: raise ValueError("load_method must be 'pivot' or 'rowwise'.")
	if not config.db_file.exists(): raise FileNotFoundError(f"Database file not found: {config.db_file}")

def k_values_tag(k_values: list[int]) -> str: return "_".join([str(int(k)) for k in k_values])

def cluster_trees(rotation_year: int, k: list[int], write_to_CSV: bool = True) -> None:
	config = CONFIG
	config.rotation_year = int(rotation_year)
	config.k_values = [int(value) for value in k]
	config.write_to_CSV = bool(write_to_CSV)
	validate_config(config)
	config.output_prefix.parent.mkdir(parents=True, exist_ok=True)
	centers_rows = []
	assignment_rows = []
	summary_rows = []
	overall_rows = []
	t0 = time.perf_counter()
	with sqlite3.connect(config.db_file) as con:
		# ensure_cluster_schema(con)
		load_start = time.perf_counter()
		keys, data, areas = load_profiles_for_harvest_year(con, config.rotation_year, config.max_years, config.normalize, config.load_method, config.profile_source)
		if len(keys) == 0: raise ValueError(f"No pixels found for rotation_year={config.rotation_year}.")
		print(f"Rotation year {config.rotation_year}: loaded {len(keys)} pixels once in {time.perf_counter() - load_start:.2f} s.", flush=True)
		for cluster_count in config.k_values:
			cluster_start = time.perf_counter()
			centers, labels, counts, iterations_run = run_kmeans(data=data, k=cluster_count, max_iter=config.max_iter, tol=config.tol, seed=config.seed + config.rotation_year + cluster_count)
			print(f"Rotation year {config.rotation_year}: computed {len(centers)} carbon removal schedules for k={cluster_count} in {time.perf_counter() - cluster_start:.2f} s.", flush=True)
			squared_distances = assigned_squared_distances(data, centers, labels)
			total_squared_distance = float(squared_distances.sum())
			mean_squared_distance = float(squared_distances.mean())
			rmse_per_year = float(np.sqrt(total_squared_distance / (len(data)*config.rotation_year)))
			max_squared_distance = float(squared_distances.max())
			max_distance = float(np.sqrt(max_squared_distance))
			overall_rows.append({"rotation_year": config.rotation_year, "k": cluster_count, "pixel_count": int(len(data)),
				"cluster_count": int(len(centers)), "iterations_run": iterations_run, "total_squared_distance": total_squared_distance,
				"mean_squared_distance": mean_squared_distance, "rmse_per_year": rmse_per_year, "max_squared_distance": max_squared_distance, "max_distance": max_distance,})
			print(f"Rotation year {config.rotation_year}, k={cluster_count}: total squared distance {total_squared_distance:,.6f}, rmse_per_year {rmse_per_year:.6f}.", flush=True)
			for cluster_id, center in enumerate(centers):
				row = {"rotation_year": config.rotation_year, "k": cluster_count, "cluster_id": cluster_id}
				for year in range(1, config.max_years + 1): row[f"year_{year}"] = float(center[year - 1])
				centers_rows.append(row)
			area_totals = np.bincount(labels, weights=areas, minlength=len(centers))
			for cluster_id in range(len(centers)):
				cluster_mask = labels == cluster_id
				cluster_squared = squared_distances[cluster_mask]
				cluster_total_squared_distance = float(cluster_squared.sum()) if len(cluster_squared) else 0.0
				cluster_mean_squared_distance = float(cluster_squared.mean()) if len(cluster_squared) else 0.0
				cluster_rmse_per_year = float(np.sqrt(cluster_total_squared_distance / (len(cluster_squared)*config.rotation_year))) if len(cluster_squared) else 0.0
				cluster_max_squared_distance = float(cluster_squared.max()) if len(cluster_squared) else 0.0
				cluster_max_distance = float(np.sqrt(cluster_max_squared_distance)) if len(cluster_squared) else 0.0
				summary_rows.append({
					"rotation_year": config.rotation_year,
					"k": cluster_count,
					"cluster_id": cluster_id,
					"pixel_count": int(counts[cluster_id]),
					"total_area_ha": float(area_totals[cluster_id]),
					"iterations_run": iterations_run,
					"cluster_total_squared_distance": cluster_total_squared_distance,
					"cluster_mean_squared_distance": cluster_mean_squared_distance,
					"cluster_rmse_per_year": cluster_rmse_per_year,
					"cluster_max_squared_distance": cluster_max_squared_distance,
					"cluster_max_distance": cluster_max_distance,
				})
			for (country, pixel_id), area_ha, cluster_id in zip(keys, areas, labels):
				assignment_rows.append({
					"rotation_year": config.rotation_year,
					"k": cluster_count,
					"country": country,
					"pixel_id": pixel_id,
					"area_ha": float(area_ha),
					"cluster_id": int(cluster_id),
				})
			# if config.profile_source == "pixel":
			# 	schedule_row_count, assignment_count = write_schedule_results_to_db(con, config.rotation_year, centers, keys, labels, areas, counts, config.max_years)
			# 	print(f"Rotation year {config.rotation_year}, k={cluster_count}: wrote {schedule_row_count} schedule-year rows and updated {assignment_count} pixel assignments in SQLite.", flush=True)
	if config.write_to_CSV:
		k_tag = k_values_tag(config.k_values)
		centers_path = Path(f"{config.output_prefix}_rotation_year_{config.rotation_year}_k_{k_tag}_centers.csv")
		assignments_path = Path(f"{config.output_prefix}_rotation_year_{config.rotation_year}_k_{k_tag}_assignments.csv")
		summary_path = Path(f"{config.output_prefix}_rotation_year_{config.rotation_year}_k_{k_tag}_summary.csv")
		overall_path = Path(f"{config.output_prefix}_rotation_year_{config.rotation_year}_k_{k_tag}_overall.csv")
		write_cluster_centers_csv(centers_path, centers_rows, config.max_years)
		write_assignments_csv(assignments_path, assignment_rows)
		write_summary_csv(summary_path, summary_rows)
		write_overall_csv(overall_path, overall_rows)
		print(f"Wrote {centers_path}")
		print(f"Wrote {assignments_path}")
		print(f"Wrote {summary_path}")
		print(f"Wrote {overall_path}")
	else:
		print("write_to_CSV is False; skipped CSV output.", flush=True)
	print(f"Done in {time.perf_counter() - t0:.2f} s.")

if __name__ == "__main__":
	# I experimented with k-means using batching, which is faster than full k-means.
	# After the experiments, I settled on these values, ran full k-means,
	# and wrote the results to the Busch2024_to_SMDAMAGE database.
	cluster_trees(rotation_year=32, k=[12], write_to_CSV=True) 
	# cluster_trees(rotation_year=38, k=[12], write_to_CSV=True) 
	# cluster_trees(rotation_year=46, k=[12], write_to_CSV=True) 
	# cluster_trees(rotation_year=57, k=[12], write_to_CSV=True) 
	# cluster_trees(rotation_year=73, k=[12], write_to_CSV=True) 
	# cluster_trees(rotation_year=101, k=[12], write_to_CSV=True) 

	config = CONFIG
	config.rotation_year = 32
	config.max_years = 101
	config.seed = 0
	config.max_iter = 4000
	config.tol = 1e-6
	config.normalize = False
	config.load_method = "pivot"
	config.profile_source = "pixel"
	validate_config(config)

	# Golden mean algorithm.
	k_low, k_high = 4, 36
	penalty_lambda = 0.02
	phi = 0.6180339887498949
	evaluated: dict[int, dict[str, float]] = {}

	with sqlite3.connect(config.db_file) as con:
		keys, data, areas = load_profiles_for_harvest_year(con, config.rotation_year, config.max_years, config.normalize, config.load_method, config.profile_source)
		n = len(data)
		if n == 0: raise ValueError(f"No pixels found for rotation_year={config.rotation_year}.")

	while k_high - k_low > 4:
		k1 = int(round(k_high - phi*(k_high - k_low)))
		k2 = int(round(k_low + phi*(k_high - k_low)))
		if k1 == k2: k2 = min(k_high, k1 + 1)

		if k1 not in evaluated:
			centers, labels, counts, iters = run_kmeans(data=data, k=k1, max_iter=config.max_iter, tol=config.tol, seed=config.seed + config.rotation_year + k1)
			sse = float(assigned_squared_distances(data, centers, labels).sum())
			obj = np.log(sse/max(n, 1)) + penalty_lambda*k1
			evaluated[k1] = {"obj": obj, "sse": sse, "iters": float(iters)}

		if k2 not in evaluated:
			centers, labels, counts, iters = run_kmeans(data=data, k=k2, max_iter=config.max_iter, tol=config.tol, seed=config.seed + config.rotation_year + k2)
			sse = float(assigned_squared_distances(data, centers, labels).sum())
			obj = np.log(sse/max(n, 1)) + penalty_lambda*k2
			evaluated[k2] = {"obj": obj, "sse": sse, "iters": float(iters)}

		if evaluated[k1]["obj"] <= evaluated[k2]["obj"]: k_high = k2
		else: k_low = k1

	for k in range(k_low, k_high + 1):
		if k in evaluated: continue
		centers, labels, counts, iters = run_kmeans(data=data, k=k, max_iter=config.max_iter, tol=config.tol, seed=config.seed + config.rotation_year + k)
		sse = float(assigned_squared_distances(data, centers, labels).sum())
		obj = np.log(sse/max(n, 1)) + penalty_lambda*k
		evaluated[k] = {"obj": obj, "sse": sse, "iters": float(iters)}

	best_obj = min(v["obj"] for v in evaluated.values())
	candidates = sorted(k for k, v in evaluated.items() if v["obj"] <= 1.01*best_obj)
	best_k = min(candidates)

	print(f"rotation_year={config.rotation_year}, selected_k={best_k}, obj={evaluated[best_k]['obj']:.6f}, sse={evaluated[best_k]['sse']:.3f}", flush=True)

	# Final run with CSV outputs for chosen k.
	cluster_trees(rotation_year=config.rotation_year, k=[best_k], write_to_CSV=True)