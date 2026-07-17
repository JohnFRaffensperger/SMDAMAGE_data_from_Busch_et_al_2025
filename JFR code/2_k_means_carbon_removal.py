# 2_k_means_carbon_removal.py | Made by Claude guided by JFR | 2026-07-15.

# A Busch et al pixel is a plot of land. We can think of that plot as a bidder in SMDAMAGE.
# 1a_import_Busch2024_to_SMDAMAGE.py finds that bidder's best bid for SMDAMAGE.

# But 58 million bidders is too big for SMDAMAGE, so I use a k-means to cluster similar bidders into groups,
# treating them identically within the group. So this code clusters the pixels based on bid and carbon schedule. 
# The code optionally writes to Busch2024_to_SMDAMAGE.sqlite and/or CSV.

# Later in the workflow, SMDAMAGE project file create_database.py reads Busch2024_to_SMDAMAGE.sqlite
# and loads forestry bidders, one per group, into the SMDAMAGE auction.

from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, cast
import time
import subprocess
import sys
import numpy as np
import pandas as pd  # Used for fast CSV loading.

DEFAULT_contract_YEARS = [6]

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
	k_values: list[int] = None
	contract_years: int = DEFAULT_contract_YEARS[0]
	max_iter: int = 4000 # How long you want this to run.
	tol: float = 1e-6 # Optimization criterion, smaller is slower and lower error.
	min_reassigned_frac: float = 0.0  # Stop when fewer than this fraction of pixels reassign.
	seed: int = 0
	normalize: bool = False
	kmeans_backend: str = "custom"  # custom|sklearnex
	sklearnex_target_offload: str = "gpu"  # Option 2: explicit device targeting.
	sklearnex_startup_probe: bool = False  # Optional preflight probe; disabled by default to avoid false negatives.
	allow_backend_fallback: bool = True
	write_to_CSV: bool = True
	output_prefix: Path = DEFAULT_OUTPUT_PREFIX

	def __post_init__(self) -> None:
		if self.k_values is None: self.k_values = [20]

CONFIG = RunConfig()

def squared_distance_argmin(batch: np.ndarray, centers: np.ndarray, center_sq: np.ndarray | None = None) -> np.ndarray:
	batch_sq = np.sum(batch*batch, axis=1, keepdims=True)
	if center_sq is None: center_sq = np.sum(centers*centers, axis=1)
	distances = batch_sq + center_sq[np.newaxis, :] - 2.0*batch @ centers.T
	return np.argmin(distances, axis=1)

def l2_normalize_rows(data: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(data, axis=1, keepdims=True)
	norms[norms == 0] = 1.0
	return data / norms

def initialize_centers(data: np.ndarray, k: int, rng: np.random.Generator, use_kmeans_plusplus: bool = True, chunk_size: int = 131072) -> np.ndarray:
	# k-means++ (use_kmeans_plusplus=True) or fast random selection.
	# k-means++ distance computation is chunked and stays in float32 to avoid large allocations.
	if len(data) == 0: raise ValueError("Cannot initialize centers from empty data.")
	actual_k = min(k, len(data))
	if not use_kmeans_plusplus:
		return data[rng.choice(len(data), size=actual_k, replace=False)].copy()
	centers = [data[int(rng.integers(len(data)))].copy()]
	for _ in range(1, actual_k):
		min_sq = np.full(len(data), np.inf, dtype=np.float32)
		for c in centers:
			c32 = c.astype(data.dtype)
			for start in range(0, len(data), chunk_size):
				stop = min(start + chunk_size, len(data))
				diffs = data[start:stop] - c32
				min_sq[start:stop] = np.minimum(min_sq[start:stop], np.sum(diffs * diffs, axis=1))
		probs = min_sq.astype(np.float64); probs /= probs.sum()
		centers.append(data[int(rng.choice(len(data), p=probs))].copy())
	return np.stack(centers)

def assign_all_labels(data: np.ndarray, centers: np.ndarray, center_sq: np.ndarray | None = None, chunk_size: int = 131072) -> np.ndarray:
	labels = np.empty(len(data), dtype=np.int32)
	if center_sq is None: center_sq = np.sum(centers*centers, axis=1)
	for start in range(0, len(data), chunk_size):
		stop = min(start + chunk_size, len(data))
		labels[start:stop] = squared_distance_argmin(data[start:stop], centers, center_sq)
	return labels

def assigned_squared_distances(data: np.ndarray, centers: np.ndarray, labels: np.ndarray, chunk_size: int = 131072) -> np.ndarray:
	distances = np.empty(len(data), dtype=np.float64)
	for start in range(0, len(data), chunk_size):
		stop = min(start + chunk_size, len(data))
		chunk = data[start:stop]
		chunk_centers = centers[labels[start:stop]]
		delta = chunk - chunk_centers
		distances[start:stop] = np.sum(delta*delta, axis=1)
	return distances

def recompute_centers(data: np.ndarray, labels: np.ndarray, centers: np.ndarray, chunk_size: int = 131072) -> tuple[np.ndarray, np.ndarray]:
	# Row-chunked matmul accumulation: reads data sequentially (cache-friendly for C-order arrays).
	# For each chunk, one-hot encode labels then multiply: (k×chunk) @ (chunk×T) → accumulates (k×T).
	k = len(centers); T = data.shape[1]
	counts = np.bincount(labels, minlength=k).astype(np.int64)
	new_sums = np.zeros((k, T), dtype=np.float32)
	k_arange = np.arange(k, dtype=np.int32)
	for start in range(0, len(data), chunk_size):
		stop = min(start + chunk_size, len(data))
		chunk = data[start:stop]                                        # sequential read
		one_hot = (labels[start:stop, np.newaxis] == k_arange).view(np.uint8).astype(np.float32)
		new_sums += one_hot.T @ chunk                                   # BLAS (k×chunk) @ (chunk×T)
	mask = counts > 0
	new_centers = centers.copy()
	new_centers[mask] = (new_sums[mask] / counts[mask, np.newaxis]).astype(centers.dtype)
	return new_centers, counts

def windows_visible_gpus() -> list[str]:
	# OS-level check that does not depend on Python SYCL libraries.
	if sys.platform != "win32": return []
	cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
	result = subprocess.run(cmd, capture_output=True, text=True)
	if result.returncode != 0: return []
	return [line.strip() for line in result.stdout.splitlines() if line.strip()]

def probe_sklearnex_target(target_offload: str) -> tuple[bool, str]:
	# Probe in a child process so oneDAL/SYCL crashes do not kill the main run.
	probe_code = (
		"from sklearnex import patch_sklearn, config_context\n"
		"patch_sklearn()\n"
		"from sklearn.cluster import KMeans\n"
		"import numpy as np\n"
		"X=np.random.default_rng(0).standard_normal((2000,20)).astype(np.float32)\n"
		f"with config_context(target_offload={target_offload!r}):\n"
		"    km=KMeans(n_clusters=4,n_init=1,max_iter=5,random_state=0)\n"
		"    km.fit(X)\n"
		"print('OK', float(km.inertia_))\n"
	)
	result = subprocess.run([sys.executable, "-c", probe_code], capture_output=True, text=True)
	output = (result.stdout + "\n" + result.stderr).strip()
	if result.returncode == 0 and "OK" in output: return True, output
	if output: return False, f"exit_code={result.returncode}\n{output}"
	return False, f"probe exited with code {result.returncode} and no output"

def probe_dpctl_devices() -> tuple[bool, str]:
	# Probe in a child process because SYCL/device enumeration can hard-exit the process.
	probe_code = (
		"import dpctl\n"
		"devices = dpctl.get_devices()\n"
		"print('OK', len(devices))\n"
		"for d in devices: print(str(d))\n"
	)
	result = subprocess.run([sys.executable, "-c", probe_code], capture_output=True, text=True)
	output = (result.stdout + "\n" + result.stderr).strip()
	if result.returncode != 0:
		if output: return False, f"exit_code={result.returncode}\n{output}"
		return False, f"dpctl probe exited with code {result.returncode} and no output"
	if "OK" not in output: return False, output if output else "dpctl probe returned no recognizable output"
	return True, output

def run_kmeans_sklearnex(data: np.ndarray, k: int, max_iter: int, tol: float, seed: int, use_kmeans_plusplus: bool = True, target_offload: str = "gpu") -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
	from sklearnex import patch_sklearn, config_context
	patch_sklearn()
	from sklearn.cluster import KMeans
	actual_k = min(k, len(data))
	init_method = "k-means++" if use_kmeans_plusplus else "random"
	with config_context(target_offload=target_offload):
		model = KMeans(n_clusters=actual_k, init=init_method, n_init=1, max_iter=max_iter, tol=tol, random_state=seed, algorithm="lloyd")
		model.fit(data.astype(np.float32, copy=False))
	centers = model.cluster_centers_.astype(np.float32, copy=False)
	labels = model.labels_.astype(np.int32, copy=False)
	counts = np.bincount(labels, minlength=len(centers)).astype(np.int64)
	iterations_run = int(getattr(model, "n_iter_", max_iter))
	return centers, labels, counts, iterations_run

def run_kmeans(data: np.ndarray, k: int, max_iter: int, tol: float, seed: int, use_kmeans_plusplus: bool = True, chunk_size: int = 131072, min_reassigned_frac: float = 0.0, backend: str = "custom", sklearnex_target_offload: str = "gpu") -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
	# min_reassigned_frac: stop early when the fraction of pixels that changed cluster
	# drops below this threshold (e.g. 0.001 = stop when <0.1% of pixels reassigned).
	if len(data) == 0: raise ValueError("No rows available for clustering.")
	if backend == "sklearnex":
		print(f"Using sklearnex backend with target_offload={sklearnex_target_offload}.", flush=True)
		return run_kmeans_sklearnex(data, k, max_iter, tol, seed, use_kmeans_plusplus=use_kmeans_plusplus, target_offload=sklearnex_target_offload)
	print("Using custom k-means backend.", flush=True)
	rng = np.random.default_rng(seed)
	centers = initialize_centers(data, k, rng, use_kmeans_plusplus=use_kmeans_plusplus)
	T = data.shape[1]; n = len(data)
	grand_centroid = data.mean(axis=0).astype(data.dtype)
	tss = 0.0
	for start in range(0, n, chunk_size):
		stop = min(start + chunk_size, n)
		diff = data[start:stop] - grand_centroid
		tss += float(np.sum(diff * diff, dtype=np.float64))
	labels = np.full(n, -1, dtype=np.int32)  # sentinel: no assignment yet
	min_changes = int(min_reassigned_frac * n)  # pixel threshold derived from fraction
	iterations_run = 0
	for iteration in range(max_iter):
		iterations_run = iteration + 1
		center_sq = np.sum(centers * centers, axis=1)
		new_labels = assign_all_labels(data, centers, center_sq=center_sq)
		n_changed = int(np.sum(new_labels != labels))
		labels = new_labels
		old_centers = centers.copy()
		centers, counts = recompute_centers(data, labels, centers)
		max_shift = np.max(np.linalg.norm(centers - old_centers, axis=1)) / np.sqrt(T) if len(centers) else 0.0
		if iteration % 10 == 0:
			pct = 100.0 * n_changed / n
			print(f"iteration {iteration}, label_changes={n_changed:,} ({pct:.2f}%), max_shift/yr={max_shift:.3e}", flush=True)
		if n_changed == 0: break
		if max_shift <= tol: break
		if min_changes > 0 and n_changed < min_changes: break
	center_sq = np.sum(centers * centers, axis=1)
	final_labels = assign_all_labels(data, centers, center_sq=center_sq)
	centers, final_counts = recompute_centers(data, final_labels, centers)
	center_sq = np.sum(centers * centers, axis=1)
	final_labels = assign_all_labels(data, centers, center_sq=center_sq)
	return centers, final_labels, final_counts, iterations_run

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

def run_final_pass(k_allocation: dict[int, int], max_iter: int = 4000, tol: float = 1e-6) -> None:
	# Final high-quality run using the cluster counts from allocate_k.
	CONFIG.max_iter = max_iter
	CONFIG.tol = tol
	for T, k in sorted(k_allocation.items()):
		print(f"Final pass: contract_years={T}, k={k}", flush=True)
		cluster_trees(contract_years=T, k=[k], write_to_CSV=True)

def cluster_trees (contract_years: int, k: list[int], write_to_CSV: bool = True) -> None:
	config = CONFIG
	config.contract_years = int(contract_years)
	config.k_values = [int(value) for value in k]
	config.write_to_CSV = bool(write_to_CSV)
	validate_config(config)
	config.output_prefix.parent.mkdir(parents=True, exist_ok=True)
	centers_rows = []
	assignment_rows = []
	summary_rows = []
	overall_rows = []
	t0 = time.perf_counter()

	load_start = time.perf_counter()
	print("Loading carbon schedules...", flush=True)
	keys, data = load_schedules_from_csv(config.csv_dir, config.contract_years, config.normalize)
	if len(keys) == 0: raise ValueError(f"No pixels found for contract_years={config.contract_years}.")
	print(f"Contract year {config.contract_years}: loaded {len(keys)} pixels once in {time.perf_counter() - load_start:.2f} s.", flush=True)
	active_backend = config.kmeans_backend
	if active_backend == "sklearnex":
		dpctl_ok, dpctl_msg = probe_dpctl_devices()
		if dpctl_ok:
			print("dpctl probe succeeded; SYCL devices visible.", flush=True)
			print(dpctl_msg, flush=True)
		else:
			print("dpctl probe failed.", flush=True)
			print(dpctl_msg, flush=True)
			print("Continuing: dpctl probe is informational only; sklearnex will still be attempted.", flush=True)
		if config.sklearnex_startup_probe:
			ok, msg = probe_sklearnex_target(config.sklearnex_target_offload)
			status = "ready" if ok else "failed"
			print(f"sklearnex probe ({config.sklearnex_target_offload}) {status}.", flush=True)
			if not ok:
				print(msg, flush=True)
				if config.allow_backend_fallback:
					active_backend = "custom"
					print("Falling back to custom k-means backend because startup probe failed.", flush=True)
				else:
					raise RuntimeError("sklearnex probe failed and allow_backend_fallback=False")

	for cluster_count in config.k_values:
		cluster_start = time.perf_counter()
		try:
			centers, labels, counts, iterations_run = run_kmeans(data=data, k=cluster_count, max_iter=config.max_iter, tol=config.tol, seed=config.seed + config.contract_years + cluster_count, use_kmeans_plusplus=False, min_reassigned_frac=config.min_reassigned_frac, backend=active_backend, sklearnex_target_offload=config.sklearnex_target_offload)
		except Exception as exc:
			if active_backend == "sklearnex" and config.allow_backend_fallback:
				print(f"sklearnex failed at fit time ({type(exc).__name__}: {exc}). Falling back to custom backend.", flush=True)
				active_backend = "custom"
				centers, labels, counts, iterations_run = run_kmeans(data=data, k=cluster_count, max_iter=config.max_iter, tol=config.tol, seed=config.seed + config.contract_years + cluster_count, use_kmeans_plusplus=False, min_reassigned_frac=config.min_reassigned_frac, backend=active_backend, sklearnex_target_offload=config.sklearnex_target_offload)
			else:
				raise
		print(f"contract year {config.contract_years}: computed {len(centers)} carbon removal schedules for k={cluster_count} in {time.perf_counter() - cluster_start:.2f} s.", flush=True)
		squared_distances = assigned_squared_distances(data, centers, labels)
		total_squared_distance = float(squared_distances.sum())
		mean_squared_distance = float(squared_distances.mean())
		rmse_per_year = float(np.sqrt(total_squared_distance / (len(data)*config.contract_years)))
		max_squared_distance = float(squared_distances.max())
		max_distance = float(np.sqrt(max_squared_distance))
		overall_rows.append({"contract_years": config.contract_years, "k": cluster_count, "pixel_count": int(len(data)), "cluster_count": int(len(centers)), "iterations_run": iterations_run,
			"total_squared_distance": total_squared_distance, "mean_squared_distance": mean_squared_distance, "rmse_per_year": rmse_per_year, "max_squared_distance": max_squared_distance, "max_distance": max_distance,})
		print(f"contract year {config.contract_years}, k={cluster_count}: total squared distance {total_squared_distance:,.6f}, rmse_per_year {rmse_per_year:.6f}.", flush=True)
		for cluster_id, center in enumerate(centers):
			row = {"contract_years": config.contract_years, "k": cluster_count, "cluster_id": cluster_id}
			for year in range(1, config.contract_years + 1): row[f"year_{year}"] = float(center[year - 1])
			centers_rows.append(row)
		for cluster_id in range(len(centers)):
			cluster_mask = labels == cluster_id
			cluster_squared = squared_distances[cluster_mask]
			cluster_total_squared_distance = float(cluster_squared.sum()) if len(cluster_squared) else 0.0
			cluster_mean_squared_distance = float(cluster_squared.mean()) if len(cluster_squared) else 0.0
			cluster_rmse_per_year = float(np.sqrt(cluster_total_squared_distance / (len(cluster_squared)*config.contract_years))) if len(cluster_squared) else 0.0
			cluster_max_squared_distance = float(cluster_squared.max()) if len(cluster_squared) else 0.0
			cluster_max_distance = float(np.sqrt(cluster_max_squared_distance)) if len(cluster_squared) else 0.0
			summary_rows.append({"contract_years": config.contract_years, "k": cluster_count, "cluster_id": cluster_id, "pixel_count": int(counts[cluster_id]), "iterations_run": iterations_run, "cluster_total_squared_distance": cluster_total_squared_distance,
				"cluster_mean_squared_distance": cluster_mean_squared_distance, "cluster_rmse_per_year": cluster_rmse_per_year, "cluster_max_squared_distance": cluster_max_squared_distance, "cluster_max_distance": cluster_max_distance,})
		for pixel_id, cluster_id in zip(keys, labels):
			assignment_rows.append({"contract_years": config.contract_years, "k": cluster_count, "pixel_id": pixel_id, "cluster_id": int(cluster_id),})

	if config.write_to_CSV:
		k_tag = k_values_tag(config.k_values)
		centers_path = Path(f"{config.output_prefix}_contract_years_{config.contract_years}_k_{k_tag}_centers.csv")
		assignments_path = Path(f"{config.output_prefix}_contract_years_{config.contract_years}_k_{k_tag}_assignments.csv")
		summary_path = Path(f"{config.output_prefix}_contract_years_{config.contract_years}_k_{k_tag}_summary.csv")
		overall_path = Path(f"{config.output_prefix}_contract_years_{config.contract_years}_k_{k_tag}_overall.csv")
		write_cluster_centers_csv(centers_path, centers_rows, config.contract_years)
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
	# Count of best_contract_length values in Busch2024_to_SMDAMAGE.Undiscounted_dta_output
	# Years	Count		Area		Possible target cluster count = round(count(pixels)^0.5/222), comes out to 100.
	# 20	10,399,106	999,775,298		15
	# 30	10,815,908	994,859,185		15
	# 40	5,256,921	483,240,557		10
	# 50	5,689,848	542,955,473		11
	# 60	2,505,180	234,482,214		7
	# 70	1,623,950	151,199,734		6
	# 80	1,121,736	103,727,989		5
	# 90	862,483		78,841,709		4
	# 100	809,700		73,235,760		4
	# 110	756,737		67,899,812		4
	# 120	18,463,262	1,578,573,429	19

	config = CONFIG
	config.seed = 0
	config.normalize = False
	config.kmeans_backend = "sklearnex"  # Set to "custom" to always use the in-file implementation.
	config.sklearnex_target_offload = "gpu"  # Option 2: explicit device targeting.
	config.sklearnex_startup_probe = False  # Keep False: probe may false-fail in some Windows driver stacks.
	config.allow_backend_fallback = True
	CONTRACT_LIST = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
	TARGET_TOTAL_CLUSTERS = 100
	SWEEP_MIN_K = 5  # Based on experience.
	SWEEP_K_STEP = 1  # Step of 1: fine-grained elbow curve.
	SWEEP_MAX_K = 16  # Based on experience.
	SWEEP_MAX_ITER = 100
	SWEEP_TOL = 1e-3   # Tighter than old 1e-2; each restart converges better on the sample.
	SWEEP_N_RESTARTS = 4  # Increased from 3 for better curve quality.

	# Pass 1 and Pass 2 already completed for this run; T=20..T=110 also finished in Pass 3.
	# Uncomment the block below to rerun all three passes from scratch.
	# print(f"Pass 1: fast sweep of normalized error curves for k={SWEEP_MIN_K},{SWEEP_MIN_K + SWEEP_K_STEP},...,{SWEEP_MAX_K} ({SWEEP_N_RESTARTS} restarts, k-means++).", flush=True)
	# sweep, sample_sizes = run_fast_sweep(config.csv_dir, CONTRACT_LIST, max_k=SWEEP_MAX_K, max_iter=SWEEP_MAX_ITER, tol=SWEEP_TOL, seed=config.seed, normalize=config.normalize, min_k=SWEEP_MIN_K, k_step=SWEEP_K_STEP, n_restarts=SWEEP_N_RESTARTS)
	# print(f"Pass 2: greedy single-cluster allocation of {TARGET_TOTAL_CLUSTERS} total clusters, drops weighted by sample_n.", flush=True)
	# k_alloc = allocate_k(sweep, sample_sizes=sample_sizes, target_total_clusters=TARGET_TOTAL_CLUSTERS, min_k_per_dataset=SWEEP_MIN_K)
	# for T, k in sorted(k_alloc.items()):
	# 	curve = sweep.get(T, {})
	# 	start_error = curve_value_at(curve, SWEEP_MIN_K)
	# 	end_error = curve_value_at(curve, k)
	# 	print(f"  T={T:3d}: k={k:3d}  (sample_n={sample_sizes.get(T,0):>8,}, normalized_error {start_error:.6f} -> {end_error:.6f})", flush=True)

	max_iter = 150
	tol = 5e-6
	min_reassigned_frac = 0.0001  # stop when <0.1% of pixels reassign
	CONFIG.max_iter = max_iter; CONFIG.tol = tol; CONFIG.min_reassigned_frac = min_reassigned_frac
	gpus = windows_visible_gpus()
	if gpus: print(f"Windows-visible GPUs: {gpus}", flush=True)
	else: print("Windows-visible GPUs: none detected.", flush=True)
	print(f"Pass 3 (resume): contract_years=120, k=16 (max_iter={max_iter}, tol={tol}, min_reassigned_frac={min_reassigned_frac}).", flush=True)
	cluster_trees(contract_years=120, k=[16], write_to_CSV=True)