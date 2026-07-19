# 2_k_means_carbon_removal.py | Made by Claude guided by JFR | 2026-07-15.

# A Busch et al pixel is a plot of land. We can think of that plot as a bidder in SMDAMAGE.
# 1a_import_Busch2024_to_SMDAMAGE.py finds that bidder's best bid for SMDAMAGE.

# But 89 million bidders is too big for SMDAMAGE, so I use a k-means to cluster pixels into 100 groups,
# based on carbon schedule, treating them as identical within the group, except for bid.
# The code writes the solution to CSV files, which I will import into Busch2024_to_SMDAMAGE.sqlite
# with the cleverly named program 3_import_k_means_csv_to_sqlite.py.

# Later in the workflow, SMDAMAGE project file create_database.py reads Busch2024_to_SMDAMAGE.sqlite
# and loads forestry bidders, one per group, into the SMDAMAGE auction.
# This code can run out of memory. It is fiddly to run. I finally got a satisfactory run 
# that took about 18 hours to run. I think the limiting factor was 8GB RAM when I wish I had 64GB RAM, 
# resulting in caching to the solid state drive. Claude was helpful in getting it to run faster and use less memory.

# Below is summary output from 1a_import_Busch2024_to_SMDAMAGE.py, stored in Busch2024_to_SMDAMAGE.Undiscounted_dta_output.
	# Count of best_contract_length values with cluster count
	# Years	Count		Area		JFR guess, actual.
	# 20	10,399,106	999,775,298		15	14
	# 30	10,815,908	994,859,185		15	11
	# 40	5,256,921	483,240,557		10	9
	# 50	5,689,848	542,955,473		11	13
	# 60	2,505,180	234,482,214		7	9
	# 70	1,623,950	151,199,734		6	8
	# 80	1,121,736	103,727,989		5	5
	# 90	862,483		78,841,709		4	5
	# 100	809,700		73,235,760		4	5
	# 110	756,737		67,899,812		4	5
	# 120	18,463,262	1,578,573,429	19	16

from __future__ import annotations
import csv
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from typing import Callable, Iterator, cast
import time
import numpy as np
import pandas as pd  # Used for fast CSV loading.
from sklearn.cluster import KMeans as SklearnKMeans

DEFAULT_contract_YEARS = [6] # not used AFAIK.

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Output"
DATABASE_DIR = OUTPUT_DIR / "Databases"
KMEANS_DIR = OUTPUT_DIR / "Kmeans_temp_files"
# Input CSV files produced by 1a_import_Busch2024_to_SMDAMAGE.py; one per contract length.
DEFAULT_CSV_DIR = DATABASE_DIR
UNDISCOUNTED_CSV_PREFIX = "undiscounted_contracts"
DEFAULT_OUTPUT_PREFIX = KMEANS_DIR / "k_means_carbon_removal"
FAST_SWEEP_CHUNK_SIZE = 25000
FAST_SWEEP_SAMPLE_PER_CHUNK = 128

@dataclass(slots=True)
class RunConfig:
	csv_dir: Path = DEFAULT_CSV_DIR
	k_values: list[int] | None = None
	contract_years: int = DEFAULT_contract_YEARS[0]
	max_iter: int = 4000 # How long you want this to run.
	tol: float = 1e-6 # Optimization criterion, smaller is slower and lower error.
	seed: int = 0
	normalize: bool = False
	sklearnex_target_offload: str = "gpu"
	allow_cpu_fallback: bool = False
	gpu_fit_sample_rows: int = 20_000
	gpu_progressive_stage_rows: tuple[int, ...] = (2_000, 5_000, 10_000, 20_000, 50_000)
	csv_chunk_size: int = 100_000
	sample_strata_count: int = 8
	sample_sketch_per_chunk: int = 256
	write_to_CSV: bool = True
	output_prefix: Path = DEFAULT_OUTPUT_PREFIX

	def __post_init__(self) -> None:
		if self.k_values is None: self.k_values = [20]

CONFIG = RunConfig()

def year_columns(contract_year: int) -> list[str]:
	return [f"year{y}" for y in range(1, contract_year + 1)]

def l2_normalize_rows(data: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(data, axis=1, keepdims=True)
	norms[norms == 0] = 1.0
	return data / norms

def assigned_squared_distances(data: np.ndarray, centers: np.ndarray, labels: np.ndarray, chunk_size: int = 131072) -> np.ndarray:
	distances = np.empty(len(data), dtype=np.float64)
	for start in range(0, len(data), chunk_size):
		stop = min(start + chunk_size, len(data))
		chunk = data[start:stop]
		chunk_centers = centers[labels[start:stop]]
		delta = chunk - chunk_centers
		distances[start:stop] = np.sum(delta*delta, axis=1)
	return distances

def assign_labels_to_centers(data: np.ndarray, centers: np.ndarray, chunk_size: int = 131072) -> np.ndarray:
	labels = np.empty(len(data), dtype=np.int32)
	center_sq = np.sum(centers*centers, axis=1)
	for start in range(0, len(data), chunk_size):
		stop = min(start + chunk_size, len(data))
		chunk = data[start:stop]
		chunk_sq = np.sum(chunk*chunk, axis=1, keepdims=True)
		d2 = chunk_sq + center_sq[np.newaxis, :] - 2.0*chunk @ centers.T
		labels[start:stop] = np.argmin(d2, axis=1)
	return labels

def iter_schedule_chunks(csv_dir: Path, contract_year: int, normalize: bool, chunk_size: int = 100_000) -> Iterator[tuple[np.ndarray, np.ndarray]]:
	path = csv_path_for_length(csv_dir, contract_year)
	cols = year_columns(contract_year)
	dtype_map = {"pixel_id": "int64", **{col: "float32" for col in cols}}
	chunk_reader = cast(Iterator[pd.DataFrame], pd.read_csv(path, usecols=["pixel_id"] + cols, dtype=dtype_map, chunksize=chunk_size))  # type: ignore[arg-type]
	for chunk in chunk_reader:
		if len(chunk) == 0: continue
		keys = chunk["pixel_id"].to_numpy(dtype=np.int64, copy=False)
		data = chunk[cols].to_numpy(dtype=np.float32, copy=False)
		if normalize: data = l2_normalize_rows(data).astype(np.float32, copy=False)
		yield keys, data

def build_stratified_training_sample(csv_dir: Path, contract_year: int, normalize: bool, target_rows: int, chunk_size: int = 100_000, strata_count: int = 8, sketch_per_chunk: int = 256, seed: int = 0) -> np.ndarray:
	if target_rows <= 0: raise ValueError("target_rows must be positive.")
	rng = np.random.default_rng(seed)
	sketch_scores: list[np.ndarray] = []
	chunk_count = 0
	for _keys, data in iter_schedule_chunks(csv_dir, contract_year, normalize, chunk_size=chunk_size):
		chunk_count += 1
		scores = np.sum(data, axis=1, dtype=np.float64)
		if len(scores) > sketch_per_chunk:
			idx = rng.choice(len(scores), size=sketch_per_chunk, replace=False)
			sketch_scores.append(scores[idx])
		else:
			sketch_scores.append(scores)
	if chunk_count == 0: return np.empty((0, contract_year), dtype=np.float32)
	sketch = np.concatenate(sketch_scores) if sketch_scores else np.empty(0, dtype=np.float64)
	if len(sketch) == 0: return np.empty((0, contract_year), dtype=np.float32)
	quantiles = np.linspace(0.0, 1.0, strata_count + 1)
	edges = np.quantile(sketch, quantiles)
	per_chunk_budget = max(1, int(np.ceil(target_rows / chunk_count)))
	per_stratum_budget = max(1, int(np.ceil(per_chunk_budget / strata_count)))
	sampled_chunks: list[np.ndarray] = []
	strata_hits = np.zeros(strata_count, dtype=np.int64)
	for _keys, data in iter_schedule_chunks(csv_dir, contract_year, normalize, chunk_size=chunk_size):
		scores = np.sum(data, axis=1, dtype=np.float64)
		bins = np.searchsorted(edges, scores, side="right") - 1
		bins = np.clip(bins, 0, strata_count - 1)
		for s in range(strata_count):
			idx = np.flatnonzero(bins == s)
			if len(idx) == 0: continue
			take = min(per_stratum_budget, len(idx))
			chosen = rng.choice(idx, size=take, replace=False)
			sampled_chunks.append(data[chosen])
			strata_hits[s] += take
	if not sampled_chunks: return np.empty((0, contract_year), dtype=np.float32)
	sample = np.vstack(sampled_chunks).astype(np.float32, copy=False)
	if len(sample) > target_rows:
		idx = rng.choice(len(sample), size=target_rows, replace=False)
		sample = sample[idx]
	print(f"Stratified sample built: rows={len(sample):,}, chunks={chunk_count}, strata={strata_count}, strata_hits={strata_hits.tolist()}.", flush=True)
	return sample

def stage_sizes_for_progressive_fit(target_rows: int, stage_rows: tuple[int, ...]) -> list[int]:
	if target_rows <= 0: return []
	stage_sizes = sorted({min(target_rows, int(v)) for v in stage_rows if int(v) > 0})
	if not stage_sizes or stage_sizes[-1] != target_rows: stage_sizes.append(target_rows)
	return stage_sizes

def should_fallback_from_sklearnex(exc: Exception, target_offload: str) -> bool:
	message = str(exc)
	if "target_offload" in message and "DPC++ backend" in message: return True
	if target_offload == "gpu" and "SyclQueue" in message: return True
	if target_offload == "gpu" and "UR_RESULT_ERROR_DEVICE_LOST" in message: return True
	if target_offload == "gpu" and "level_zero backend failed" in message: return True
	return False

def gpu_backend_error(exc: Exception) -> RuntimeError:
	return RuntimeError(f"GPU offload was requested but sklearnex/oneDAL GPU execution failed. Current interpreter: {sys.executable}. If your GPU runtime is unstable (for example UR_RESULT_ERROR_DEVICE_LOST), set allow_cpu_fallback=True to permit an automatic CPU retry.")

def fit_centers_gpu_progressive(sample_data: np.ndarray, k: int, max_iter: int, tol: float, seed: int, target_offload: str, stage_rows: tuple[int, ...], use_kmeans_plusplus: bool = True, allow_cpu_fallback: bool = False) -> tuple[np.ndarray, int]:
	if len(sample_data) == 0: raise ValueError("No rows available for center fitting.")
	accelerated_kmeans: type[SklearnKMeans] | None = None
	config_context: Callable[..., AbstractContextManager[None]] | None = None
	use_sklearnex = False
	try:
		from sklearnex import patch_sklearn, config_context as sklearnex_config_context  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
		patch_sklearn()
		from sklearn.cluster import KMeans as SklearnexKMeans
		accelerated_kmeans = SklearnexKMeans
		config_context = cast(Callable[..., AbstractContextManager[None]], sklearnex_config_context)
		use_sklearnex = True
		print(f"Using sklearnex KMeans with target_offload={target_offload}.", flush=True)
	except Exception as exc:
		print(f"sklearnex unavailable for accelerated fit ({type(exc).__name__}: {exc}). Falling back to plain sklearn CPU KMeans.", flush=True)
	actual_k = min(k, len(sample_data))
	if actual_k <= 0: raise ValueError("No clusters can be fit from empty sample.")
	rng = np.random.default_rng(seed)
	progressive_sizes = stage_sizes_for_progressive_fit(len(sample_data), stage_rows)
	centers: np.ndarray | None = None
	iterations_run = 0
	for stage_n in progressive_sizes:
		fit_n = max(actual_k, stage_n)
		if fit_n >= len(sample_data):
			stage_data = sample_data
		else:
			idx = rng.choice(len(sample_data), size=fit_n, replace=False)
			stage_data = sample_data[idx]
		if centers is None:
			init_value: str | np.ndarray = "k-means++" if use_kmeans_plusplus else "random"
		else:
			init_value = centers
		while True:
			try:
				backend_name = f"sklearnex:{target_offload}" if use_sklearnex else "sklearn:cpu"
				print(f"Progressive fit stage: backend={backend_name}, rows={len(stage_data):,}, warm_start={centers is not None}.", flush=True)
				if use_sklearnex and accelerated_kmeans is not None and config_context is not None:
					with config_context(target_offload=target_offload):
						model = accelerated_kmeans(n_clusters=actual_k, init=init_value, n_init=1, max_iter=max_iter, tol=tol, random_state=seed, algorithm="lloyd")
						model.fit(stage_data.astype(np.float32, copy=False))
				else:
					model = SklearnKMeans(n_clusters=actual_k, init=init_value, n_init=1, max_iter=max_iter, tol=tol, random_state=seed, algorithm="lloyd")
					model.fit(stage_data.astype(np.float32, copy=False))
				centers = model.cluster_centers_.astype(np.float32, copy=False)
				iterations_run += int(getattr(model, "n_iter_", max_iter))
				break
			except Exception as exc:
				print(f"Progressive stage rows={len(stage_data):,} failed: {type(exc).__name__}: {exc}", flush=True)
				if use_sklearnex and should_fallback_from_sklearnex(exc, target_offload):
					if not allow_cpu_fallback: raise gpu_backend_error(exc) from exc
					use_sklearnex = False
					print("Falling back to plain sklearn CPU KMeans and retrying the same stage.", flush=True)
					continue
				if centers is None: raise
				print("Keeping last successful centers and continuing.", flush=True)
				break
	if centers is None: raise RuntimeError("Progressive center fitting failed at all stages.")
	return centers, iterations_run

def run_kmeans(data: np.ndarray, k: int, max_iter: int, tol: float, seed: int, use_kmeans_plusplus: bool = True, target_offload: str = "gpu", fit_sample_rows: int | None = 2_000, allow_cpu_fallback: bool | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
	if len(data) == 0: raise ValueError("No rows available for clustering.")
	if allow_cpu_fallback is None: allow_cpu_fallback = CONFIG.allow_cpu_fallback
	fit_data = data
	if fit_sample_rows and len(data) > fit_sample_rows:
		rng = np.random.default_rng(seed)
		fit_idx = rng.choice(len(data), size=fit_sample_rows, replace=False)
		fit_data = data[fit_idx]
		print(f"Fitting on sample rows={len(fit_data):,} of total rows={len(data):,} to keep GPU runtime stable.", flush=True)
	centers, iterations_run = fit_centers_gpu_progressive(fit_data, k, max_iter, tol, seed, target_offload, stage_rows=(2_000, 5_000, 10_000, 20_000, 50_000), use_kmeans_plusplus=use_kmeans_plusplus, allow_cpu_fallback=allow_cpu_fallback)
	labels = assign_labels_to_centers(data, centers)
	counts = np.bincount(labels, minlength=len(centers)).astype(np.int64)
	return centers, labels, counts, iterations_run

def csv_path_for_length(csv_dir: Path, contract_year: int) -> Path:
	return csv_dir / f"{UNDISCOUNTED_CSV_PREFIX}_{contract_year:03d}.csv"

def load_schedules_from_csv(csv_dir: Path, contract_year: int, normalize: bool) -> tuple[list[int], np.ndarray]:
	# Load pixel carbon schedules from the wide CSV produced by 1a_import_Busch2024_to_SMDAMAGE.py.
	# Each row in the file is one pixel; year columns are year1..yearN where N = contract_year.
	# Returns (keys, data): keys is a list of pixel_ids; data is a float32 array of shape (n_pixels, contract_year).
	path = csv_path_for_length(csv_dir, contract_year)
	year_cols = [f"year{y}" for y in range(1, contract_year + 1)]
	dtype_map = {"pixel_id": "int64", **{col: "float32" for col in year_cols}}
	df = pd.read_csv(path, usecols=["pixel_id"] + year_cols, dtype=dtype_map)
	keys: list[int] = df["pixel_id"].astype(np.int64).tolist()
	data = df[year_cols].to_numpy(dtype=np.float32)
	if normalize: data = l2_normalize_rows(data).astype(np.float32, copy=False)
	return keys, data

def load_schedules_sampled_from_csv(csv_dir: Path, contract_year: int, normalize: bool, sample_per_chunk: int = FAST_SWEEP_SAMPLE_PER_CHUNK, chunk_size: int = FAST_SWEEP_CHUNK_SIZE, seed: int = 0) -> tuple[list[int], np.ndarray]:
	# Stream the CSV in chunks and keep only a bounded per-chunk sample.
	# This keeps the fast sweep memory-safe on very large contract files.
	path = csv_path_for_length(csv_dir, contract_year)
	year_cols = [f"year{y}" for y in range(1, contract_year + 1)]
	dtype_map = {"pixel_id": "int64", **{col: "float32" for col in year_cols}}
	keys: list[int] = []
	data_chunks: list[np.ndarray] = []
	chunk_reader = cast(Iterator[pd.DataFrame], pd.read_csv(path, usecols=["pixel_id"] + year_cols, dtype=dtype_map, chunksize=chunk_size))  # type: ignore[arg-type]
	for chunk_index, chunk in enumerate(chunk_reader):
		if len(chunk) == 0: continue
		take = min(sample_per_chunk, len(chunk))
		sampled = chunk.sample(n=take, random_state=seed + chunk_index, replace=False)
		keys.extend(sampled["pixel_id"].astype(np.int64).tolist())
		data_chunks.append(sampled[year_cols].to_numpy(dtype=np.float32, copy=False))
	if not data_chunks: return keys, np.empty((0, len(year_cols)), dtype=np.float32)
	data = np.vstack(data_chunks).astype(np.float32, copy=False)
	if normalize: data = l2_normalize_rows(data).astype(np.float32, copy=False)
	return keys, data

def write_schedule_results_to_csv(output_prefix: Path, contract_year: int, centers: np.ndarray, keys: list[int], labels: np.ndarray, counts: np.ndarray) -> None:
	# Write cluster center schedules and pixel assignments back to CSV files.
	schedule_path = Path(f"{output_prefix}_contract_years_{contract_year}_cluster_schedules.csv")
	with open(schedule_path, "w", newline="", encoding="utf-8") as fh:
		w = csv.writer(fh)
		w.writerow(["contract_years", "cluster_index", "pixel_count"] + [f"year{y}" for y in range(1, contract_year + 1)])
		for cluster_id, center in enumerate(centers):
			w.writerow([contract_year, cluster_id, int(counts[cluster_id])] + center.tolist())

def validate_config(config: RunConfig) -> None:
	if not config.k_values: raise ValueError("k_values must contain at least one cluster size.")
	if any(k <= 0 for k in config.k_values): raise ValueError("All k values must be positive.")
	if config.contract_years <= 0: raise ValueError("contract_years must be positive.")
	if config.max_iter <= 0: raise ValueError("max_iter must be positive.")
	path = csv_path_for_length(config.csv_dir, config.contract_years)
	if not path.exists(): raise FileNotFoundError(f"Input CSV not found: {path}")

def write_cluster_centers_csv(path: Path, rows: list[dict[str, object]], contract_years: int) -> None:
	fieldnames = ["contract_years", "k", "cluster_id"] + [f"year_{year}" for year in range(1, contract_years + 1)]
	with open(path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def write_assignments_csv(path: Path, rows: list[dict[str, object]]) -> None:
	fieldnames = ["contract_years", "k", "pixel_id", "cluster_id"]
	with open(path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
	fieldnames = ["contract_years", "k", "cluster_id", "pixel_count", "iterations_run", "cluster_total_squared_distance",
		"cluster_mean_squared_distance", "cluster_rmse_per_year", "cluster_max_squared_distance", "cluster_max_distance",]
	with open(path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def write_overall_csv(path: Path, rows: list[dict[str, object]]) -> None:
	fieldnames = ["contract_years", "k", "pixel_count", "cluster_count", "iterations_run", "total_squared_distance", "mean_squared_distance", "rmse_per_year", "max_squared_distance", "max_distance",]
	with open(path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def k_values_tag(k_values: list[int]) -> str: return "_".join([str(int(k)) for k in k_values])

def normalized_error_curve(data: np.ndarray, max_k: int, max_iter: int, tol: float, seed: int, min_k: int = 3, k_step: int = 3, n_restarts: int = 3) -> dict[int, float]:
	# Return normalized error values on a sparse k-grid (e.g., k=3,6,...,30).
	# Normalized error is WSS(k) / TSS, where TSS is the grand-centroid sum of squares.
	# n_restarts: number of independent k-means runs per k; best WSS is kept.
	if len(data) == 0: return {}
	grand_centroid = data.mean(axis=0, keepdims=True)
	tss = float(np.sum((data - grand_centroid) ** 2))
	upper_k = min(max_k, len(data))
	k_values = [k for k in range(min_k, upper_k + 1, k_step)]
	if tss <= 0.0: return {k: 0.0 for k in k_values}
	curve: dict[int, float] = {}
	for k in k_values:
		best_wss = float("inf")
		best_iters = 0
		for restart in range(n_restarts):
			centers, labels, _, iters = run_kmeans(data, k, max_iter, tol, seed + k * 100 + restart)
			wss = float(assigned_squared_distances(data, centers, labels).sum())
			if wss < best_wss:
				best_wss = wss
				best_iters = iters
		curve[k] = best_wss / tss
		print(f"    k={k:3d}: normalized_error={curve[k]:.6f}, best_iters={best_iters}", flush=True)
	return curve

def run_fast_sweep(csv_dir: Path, contract_list: list[int], max_k: int = 100, max_iter: int = 100, tol: float = 1e-3, seed: int = 0, normalize: bool = False, min_k: int = 3, k_step: int = 3, n_restarts: int = 3) -> tuple[dict[int, dict[int, float]], dict[int, int]]:
	# Fast pass: compute a normalized error curve for each contract length.
	# Returns ({T: {k: normalized_error}}, {T: sample_n}) — sample_n used as pixel-weight proxy.
	results: dict[int, dict[int, float]] = {}
	sample_sizes: dict[int, int] = {}
	for T in contract_list:
		path = csv_path_for_length(csv_dir, T)
		if not path.exists(): print(f"[SKIP] {path} not found.", flush=True); continue
		_keys, data = load_schedules_sampled_from_csv(csv_dir, T, normalize, seed=seed + T)
		n = len(data)
		sample_sizes[T] = n
		if n == 0: results[T] = {}; continue
		curve = normalized_error_curve(data, max_k=max_k, max_iter=max_iter, tol=tol, seed=seed + T * 1000, min_k=min_k, k_step=k_step, n_restarts=n_restarts)
		results[T] = curve
		final_error = curve[max(curve)] if curve else 0.0
		print(f"  sweep T={T:3d}: sample_n={n:>8,}, grid_points={len(curve):3d}, normalized_error@kmax={final_error:.6f}", flush=True)
	return results, sample_sizes

def run_fast_sweep_with_sampling(csv_dir: Path, contract_list: list[int], max_k: int = 100, max_iter: int = 100, tol: float = 1e-3, seed: int = 0, normalize: bool = False, min_k: int = 3, k_step: int = 3, n_restarts: int = 3, sample_per_chunk: int = FAST_SWEEP_SAMPLE_PER_CHUNK) -> tuple[dict[int, dict[int, float]], dict[int, int]]:
	# Same as run_fast_sweep but with explicit per-pass sampling control.
	results: dict[int, dict[int, float]] = {}
	sample_sizes: dict[int, int] = {}
	for T in contract_list:
		path = csv_path_for_length(csv_dir, T)
		if not path.exists(): print(f"[SKIP] {path} not found.", flush=True); continue
		_keys, data = load_schedules_sampled_from_csv(csv_dir, T, normalize, sample_per_chunk=sample_per_chunk, seed=seed + T)
		n = len(data)
		sample_sizes[T] = n
		if n == 0: results[T] = {}; continue
		curve = normalized_error_curve(data, max_k=max_k, max_iter=max_iter, tol=tol, seed=seed + T * 1000, min_k=min_k, k_step=k_step, n_restarts=n_restarts)
		results[T] = curve
		final_error = curve[max(curve)] if curve else 0.0
		print(f"  sweep T={T:3d}: sample_n={n:>8,}, grid_points={len(curve):3d}, normalized_error@kmax={final_error:.6f}", flush=True)
	return results, sample_sizes

def row_count_for_contract(csv_dir: Path, contract_year: int, normalize: bool, chunk_size: int = 100_000) -> int:
	row_count = 0
	for _keys, data in iter_schedule_chunks(csv_dir, contract_year, normalize, chunk_size=chunk_size):
		row_count += len(data)
	return row_count

def refine_centers_streaming_until_label_stable(csv_dir: Path, contract_year: int, normalize: bool, centers: np.ndarray, max_iter: int, stop_fraction: float = 1e-4, chunk_size: int = 100_000) -> tuple[np.ndarray, int, int, int]:
	# Streamed Lloyd refinement on full data with label-change stopping.
	# Stops when changed labels < stop_fraction * pixel_count.
	total_pixels = row_count_for_contract(csv_dir, contract_year, normalize, chunk_size=chunk_size)
	if total_pixels <= 0: raise ValueError(f"No rows found for contract_years={contract_year}.")
	k = len(centers)
	dim = centers.shape[1]
	stop_threshold = max(1, int(np.ceil(stop_fraction * total_pixels)))
	with tempfile.NamedTemporaryFile(prefix=f"kmeans_prev_labels_{contract_year}_", suffix=".bin", delete=False) as prev_fh, tempfile.NamedTemporaryFile(prefix=f"kmeans_curr_labels_{contract_year}_", suffix=".bin", delete=False) as curr_fh:
		prev_path = Path(prev_fh.name)
		curr_path = Path(curr_fh.name)
	prev_labels: np.memmap | None = None
	curr_labels: np.memmap | None = None
	try:
		prev_labels = np.memmap(prev_path, dtype=np.int32, mode="w+", shape=(total_pixels,))
		curr_labels = np.memmap(curr_path, dtype=np.int32, mode="w+", shape=(total_pixels,))
		changed_labels = total_pixels
		iterations_run = 0
		for iteration in range(1, max_iter + 1):
			sums = np.zeros((k, dim), dtype=np.float64)
			counts = np.zeros(k, dtype=np.int64)
			row_cursor = 0
			changed_labels = 0
			for _keys, data_chunk in iter_schedule_chunks(csv_dir, contract_year, normalize, chunk_size=chunk_size):
				labels = assign_labels_to_centers(data_chunk, centers)
				n = len(labels)
				curr_labels[row_cursor:row_cursor + n] = labels
				if iteration > 1:
					changed_labels += int(np.count_nonzero(labels != prev_labels[row_cursor:row_cursor + n]))
				np.add.at(sums, labels, data_chunk)
				counts += np.bincount(labels, minlength=k).astype(np.int64, copy=False)
				row_cursor += n
			if row_cursor != total_pixels: raise RuntimeError("Row count changed between pass-3 iterations.")
			non_empty = counts > 0
			if np.any(non_empty): centers[non_empty] = (sums[non_empty] / counts[non_empty, None]).astype(np.float32, copy=False)
			iterations_run = iteration
			if iteration > 1:
				print(f"  pass 3 refinement iter={iteration:4d}, changed_labels={changed_labels:,}, stop_threshold={stop_threshold:,}", flush=True)
				if changed_labels < stop_threshold:
					print(f"  pass 3 refinement converged at iter={iteration}: changed_labels={changed_labels:,} < {stop_threshold:,}", flush=True)
					break
			prev_labels, curr_labels = curr_labels, prev_labels
		return centers, iterations_run, changed_labels, total_pixels
	finally:
		try:
			if prev_labels is not None: prev_labels.flush()
			if curr_labels is not None: curr_labels.flush()
		except Exception:
			pass
		for path in (prev_path, curr_path):
			try:
				path.unlink(missing_ok=True)
			except Exception:
				pass

def curve_value_at(curve: dict[int, float], k: int) -> float:
	# Exact value on the evaluated k-grid, with linear interpolation for in-between k.
	if not curve: return 0.0
	if k in curve: return curve[k]
	ks = sorted(curve)
	if k <= ks[0]: return curve[ks[0]]
	if k >= ks[-1]: return curve[ks[-1]]
	lower = max(v for v in ks if v <= k)
	upper = min(v for v in ks if v >= k)
	if lower == upper: return curve[lower]
	weight = (k - lower) / (upper - lower)
	return curve[lower] + weight * (curve[upper] - curve[lower])

def allocate_k(error_curves: dict[int, dict[int, float]], sample_sizes: dict[int, int] | None = None, target_total_clusters: int = 100, min_k_per_dataset: int = 3) -> dict[int, int]:
	# Greedy single-cluster allocation weighted by pixel count (sample_sizes proxy).
	# At each step allocates to the dataset with the largest absolute marginal drop:
	#   weighted_drop = fractional_drop × sample_sizes[T]
	# This maximises reduction in total absolute squared error across all datasets.
	# Without sample_sizes, falls back to unweighted fractional drops.
	available = {t: curve for t, curve in error_curves.items() if curve}
	if not available: return {}
	allocation = {t: min_k_per_dataset for t in available}
	remaining = target_total_clusters - sum(allocation.values())
	if remaining < 0: raise ValueError("target_total_clusters must be at least min_k_per_dataset * number of datasets with data.")
	while remaining > 0:
		best_t = None
		best_drop = -1.0
		for t, curve in available.items():
			current_k = allocation[t]
			if current_k + 1 > max(curve): continue
			drop = curve_value_at(curve, current_k) - curve_value_at(curve, current_k + 1)
			if sample_sizes: drop *= sample_sizes.get(t, 1)
			if drop > best_drop:
				best_drop = drop
				best_t = t
		if best_t is None:
			break
		allocation[best_t] += 1
		remaining -= 1
	return allocation

def run_final_pass(k_allocation: dict[int, int], max_iter_other: int = 2000, max_iter_120: int = 300, tol: float = 1e-6) -> None:
	# Final high-quality run using the cluster counts from allocate_k.
	CONFIG.tol = tol
	for T, k in sorted(k_allocation.items()):
		pass3_max_iter = max_iter_120 if T == 120 else max_iter_other
		CONFIG.max_iter = pass3_max_iter
		print(f"Pass 3: contract_years={T}, k={k}, max_iter={pass3_max_iter}, tol={tol}.", flush=True)
		cluster_trees(contract_years=T, k=[k], write_to_CSV=True, pass3_max_iter=pass3_max_iter)

def cluster_trees (contract_years: int, k: list[int], write_to_CSV: bool = True, pass3_max_iter: int | None = None) -> None:
	config = CONFIG
	config.contract_years = int(contract_years)
	config.k_values = [int(value) for value in k]
	config.write_to_CSV = bool(write_to_CSV)
	validate_config(config)
	config.output_prefix.parent.mkdir(parents=True, exist_ok=True)
	t0 = time.perf_counter()

	print("Building stratified training sample from streamed CSV chunks...", flush=True)
	sample_start = time.perf_counter()
	training_data = build_stratified_training_sample(config.csv_dir, config.contract_years, config.normalize, target_rows=config.gpu_fit_sample_rows, chunk_size=config.csv_chunk_size, strata_count=config.sample_strata_count, sketch_per_chunk=config.sample_sketch_per_chunk, seed=config.seed + config.contract_years)
	if len(training_data) == 0: raise ValueError(f"No rows sampled for contract_years={config.contract_years}.")
	print(f"Contract year {config.contract_years}: built training sample with {len(training_data):,} rows in {time.perf_counter() - sample_start:.2f} s.", flush=True)

	for cluster_count in config.k_values:
		cluster_start = time.perf_counter()
		print(f"Fitting on stratified sample up to rows={config.gpu_fit_sample_rows:,} (progressive warm-start GPU stages).", flush=True)
		centers, iterations_run = fit_centers_gpu_progressive(training_data, cluster_count, config.max_iter, config.tol, config.seed + config.contract_years + cluster_count, config.sklearnex_target_offload, stage_rows=config.gpu_progressive_stage_rows, use_kmeans_plusplus=False, allow_cpu_fallback=config.allow_cpu_fallback)
		if pass3_max_iter is not None:
			print(f"Running pass 3 streamed refinement with max_iter={pass3_max_iter} and stop threshold 0.0001 * pixel_count.", flush=True)
			centers, refine_iters, changed_labels, total_pixels = refine_centers_streaming_until_label_stable(config.csv_dir, config.contract_years, config.normalize, centers, max_iter=pass3_max_iter, stop_fraction=1e-4, chunk_size=config.csv_chunk_size)
			iterations_run = refine_iters
			print(f"Pass 3 refinement done: iters={refine_iters}, last_changed_labels={changed_labels:,}, pixel_count={total_pixels:,}.", flush=True)
		print(f"contract year {config.contract_years}: computed {len(centers)} carbon removal schedules for k={cluster_count} in {time.perf_counter() - cluster_start:.2f} s.", flush=True)

		k_tag = k_values_tag([cluster_count])
		centers_path = Path(f"{config.output_prefix}_contract_years_{config.contract_years}_k_{k_tag}_centers.csv")
		assignments_path = Path(f"{config.output_prefix}_contract_years_{config.contract_years}_k_{k_tag}_assignments.csv")
		summary_path = Path(f"{config.output_prefix}_contract_years_{config.contract_years}_k_{k_tag}_summary.csv")
		overall_path = Path(f"{config.output_prefix}_contract_years_{config.contract_years}_k_{k_tag}_overall.csv")

		if config.write_to_CSV:
			with open(assignments_path, "w", newline="", encoding="utf-8") as handle:
				writer = csv.writer(handle)
				writer.writerow(["contract_years", "k", "pixel_id", "cluster_id"])

		cluster_pixel_count = np.zeros(len(centers), dtype=np.int64)
		cluster_total_squared_distance = np.zeros(len(centers), dtype=np.float64)
		cluster_max_squared_distance = np.zeros(len(centers), dtype=np.float64)
		total_pixels = 0
		total_squared_distance = 0.0
		max_squared_distance = 0.0
		pass_start = time.perf_counter()
		for chunk_index, (keys, data_chunk) in enumerate(iter_schedule_chunks(config.csv_dir, config.contract_years, config.normalize, chunk_size=config.csv_chunk_size), start=1):
			labels = assign_labels_to_centers(data_chunk, centers)
			chunk_centers = centers[labels]
			delta = data_chunk - chunk_centers
			squared_distances = np.sum(delta*delta, axis=1, dtype=np.float64)
			total_pixels += len(data_chunk)
			total_squared_distance += float(squared_distances.sum())
			if len(squared_distances): max_squared_distance = max(max_squared_distance, float(squared_distances.max()))
			for cluster_id in range(len(centers)):
				mask = labels == cluster_id
				if not np.any(mask): continue
				cluster_vals = squared_distances[mask]
				cluster_pixel_count[cluster_id] += int(mask.sum())
				cluster_total_squared_distance[cluster_id] += float(cluster_vals.sum())
				cluster_max_squared_distance[cluster_id] = max(cluster_max_squared_distance[cluster_id], float(cluster_vals.max()))
			if config.write_to_CSV:
				assign_df = pd.DataFrame({"contract_years": config.contract_years, "k": cluster_count, "pixel_id": keys, "cluster_id": labels})
				assign_df.to_csv(assignments_path, mode="a", header=False, index=False)
			if chunk_index % 10 == 0:
				print(f"  assignment pass chunk={chunk_index}, rows={total_pixels:,}", flush=True)
		print(f"assignment/stat pass finished in {time.perf_counter() - pass_start:.2f} s over rows={total_pixels:,}.", flush=True)

		mean_squared_distance = total_squared_distance / total_pixels
		rmse_per_year = float(np.sqrt(total_squared_distance / (total_pixels*config.contract_years)))
		max_distance = float(np.sqrt(max_squared_distance))
		print(f"contract year {config.contract_years}, k={cluster_count}: total squared distance {total_squared_distance:,.6f}, rmse_per_year {rmse_per_year:.6f}.", flush=True)

		centers_rows: list[dict[str, object]] = []
		for cluster_id, center in enumerate(centers):
			row: dict[str, object] = {"contract_years": config.contract_years, "k": cluster_count, "cluster_id": cluster_id}
			for year in range(1, config.contract_years + 1): row[f"year_{year}"] = float(center[year - 1])
			centers_rows.append(row)

		summary_rows: list[dict[str, object]] = []
		for cluster_id in range(len(centers)):
			count = int(cluster_pixel_count[cluster_id])
			cluster_total = float(cluster_total_squared_distance[cluster_id])
			cluster_mean = cluster_total / count if count else 0.0
			cluster_rmse = float(np.sqrt(cluster_total / (count*config.contract_years))) if count else 0.0
			cluster_max_sq = float(cluster_max_squared_distance[cluster_id])
			summary_rows.append({"contract_years": config.contract_years, "k": cluster_count, "cluster_id": cluster_id, "pixel_count": count, "iterations_run": iterations_run,
				"cluster_total_squared_distance": cluster_total, "cluster_mean_squared_distance": cluster_mean, "cluster_rmse_per_year": cluster_rmse,
				"cluster_max_squared_distance": cluster_max_sq, "cluster_max_distance": float(np.sqrt(cluster_max_sq)),})

		overall_rows = [{"contract_years": config.contract_years, "k": cluster_count, "pixel_count": total_pixels, "cluster_count": int(len(centers)), "iterations_run": iterations_run,
			"total_squared_distance": total_squared_distance, "mean_squared_distance": mean_squared_distance, "rmse_per_year": rmse_per_year,
			"max_squared_distance": max_squared_distance, "max_distance": max_distance,}]

		if config.write_to_CSV:
			write_cluster_centers_csv(centers_path, centers_rows, config.contract_years)
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
	# Below is summary output from 1a_import_Busch2024_to_SMDAMAGE.py, stored in Busch2024_to_SMDAMAGE.Undiscounted_dta_output.
	# Count of best_contract_length values with cluster count
	# Years	Count		Area		JFR guess, actual.
	# 20	10,399,106	999,775,298		15	14
	# 30	10,815,908	994,859,185		15	11
	# 40	5,256,921	483,240,557		10	9
	# 50	5,689,848	542,955,473		11	13
	# 60	2,505,180	234,482,214		7	9
	# 70	1,623,950	151,199,734		6	8
	# 80	1,121,736	103,727,989		5	5
	# 90	862,483		78,841,709		4	5
	# 100	809,700		73,235,760		4	5
	# 110	756,737		67,899,812		4	5
	# 120	18,463,262	1,578,573,429	19	16

	config = CONFIG
	config.seed = 0
	config.normalize = False
	config.sklearnex_target_offload = "gpu"
	config.allow_cpu_fallback = True
	CONTRACT_LIST = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
	TARGET_TOTAL_CLUSTERS = 100
	SWEEP_MIN_K = 5  # Based on experience.
	SWEEP_K_STEP = 1  # Step of 1: fine-grained elbow curve.
	SWEEP_MAX_K = 16  # Based on experience.
	SWEEP_MAX_ITER = 100
	SWEEP_TOL = 1e-3   # Tighter than old 1e-2; each restart converges better on the sample.
	SWEEP_N_RESTARTS = 5  # Pass 1 quality: 5 restarts.
	PASS1_SAMPLE_PER_CHUNK = FAST_SWEEP_SAMPLE_PER_CHUNK * 2  # Pass 1 quality: doubled sample size.
	PASS2_SAMPLE_PER_CHUNK = PASS1_SAMPLE_PER_CHUNK * 2  # Pass 2 quality: doubled again.

	print(f"Pass 1: fast sweep of normalized error curves for k={SWEEP_MIN_K},{SWEEP_MIN_K + SWEEP_K_STEP},...,{SWEEP_MAX_K} ({SWEEP_N_RESTARTS} restarts, sample_per_chunk={PASS1_SAMPLE_PER_CHUNK}).", flush=True)
	_ignored_sweep, _ignored_sample_sizes = run_fast_sweep_with_sampling(config.csv_dir, CONTRACT_LIST, max_k=SWEEP_MAX_K, max_iter=SWEEP_MAX_ITER, tol=SWEEP_TOL, seed=config.seed, normalize=config.normalize, min_k=SWEEP_MIN_K, k_step=SWEEP_K_STEP, n_restarts=SWEEP_N_RESTARTS, sample_per_chunk=PASS1_SAMPLE_PER_CHUNK)

	print(f"Pass 2: refined sweep plus greedy allocation of {TARGET_TOTAL_CLUSTERS} total clusters (sample_per_chunk={PASS2_SAMPLE_PER_CHUNK}).", flush=True)
	sweep, sample_sizes = run_fast_sweep_with_sampling(config.csv_dir, CONTRACT_LIST, max_k=SWEEP_MAX_K, max_iter=SWEEP_MAX_ITER, tol=SWEEP_TOL, seed=config.seed + 1_000_000, normalize=config.normalize, min_k=SWEEP_MIN_K, k_step=SWEEP_K_STEP, n_restarts=SWEEP_N_RESTARTS, sample_per_chunk=PASS2_SAMPLE_PER_CHUNK)
	k_alloc = allocate_k(sweep, sample_sizes=sample_sizes, target_total_clusters=TARGET_TOTAL_CLUSTERS, min_k_per_dataset=SWEEP_MIN_K)
	for T, k in sorted(k_alloc.items()):
		curve = sweep.get(T, {})
		start_error = curve_value_at(curve, SWEEP_MIN_K)
		end_error = curve_value_at(curve, k)
		print(f"  T={T:3d}: k={k:3d}  (sample_n={sample_sizes.get(T,0):>8,}, normalized_error {start_error:.6f} -> {end_error:.6f})", flush=True)

	FINAL_TOL = 5e-6
	print("Pass 3: full run for all contract lengths (max_iter=2000 except 120->300).", flush=True)
	run_final_pass(k_alloc, max_iter_other=2000, max_iter_120=300, tol=FINAL_TOL)