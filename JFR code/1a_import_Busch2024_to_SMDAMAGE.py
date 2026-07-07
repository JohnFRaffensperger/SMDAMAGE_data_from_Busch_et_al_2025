# 1a_import_Busch2024_to_SMDAMAGE.py — Pandas-free refactor of 1_import_Busch2024_to_SMDAMAGE.py.
# Written by JFR and CoPilot (mostly Claude), 2026-07-07.
# Purpose: identical to 1_import_Busch2024_to_SMDAMAGE.py but uses no pandas.
# Reads .dta files via pyreadstat (output_format='dict') and stores all pixel data in
# plain dict[str, np.ndarray] throughout.  CSV files are read with Python's csv module.
# SQLite writes use executemany directly.  All other logic is identical to the pandas version.
# See 1_import_Busch2024_to_SMDAMAGE.py for full documentation of the pipeline.

from __future__ import annotations
import argparse
import csv
import functools
import time
from pathlib import Path
import sqlite3
import numpy as np
import pyreadstat

SMDAMAGE_DB_FILE = "Busch2024_to_SMDAMAGE.sqlite"
SMDAMAGE_PIXEL_TABLE = "Undiscounted_dta_output"
SMDAMAGE_YEAR_TABLE = "tC_per_h_per_year"
TARGET_PIXEL_ID = 5119
RUN_ALL_PIXELS = True
GENUS_ORDER = ["pinu", "cunn", "euca", "brde", "brev", "nede", "neev"]
GENUS_BY_TYPE = {1: "neev", 2: "brev", 3: "brde", 4: "neev", 5: "cunn", 6: "euca", 7: "nede", 8: "neev", 9: "pinu", 10: "brde", 11: "neev", 12: "brde", 13: "brev", 14: "brde", 15: "neev"}
TIME_HORIZON_YEARS = 120
DISCOUNT_RATE = 0.03
AUSTMANN_F = 1
NATURAL_SOIL_C = 0.415107735
PLANTATION_SOIL_C = 0.092749069
AGB_C_POOL = 1
BGB_C_POOL = 1
SOIL_C_POOL = 1
HARVEST_C_POOL = 1
SMALL_DTA_COUNT = 50
SKIP_SQLITE_EXPORT = False
HARVEST_YEAR_BUCKETS_CSV = Path(__file__).resolve().parent.parent / "Output" / "choose_harvest_years_mapping.csv"
RESUME_AFTER_ISO = "KAZ"

AFRI = {"DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COM", "COG", "CIV", "COD", "DJI", "EGY", "GNQ",
	"ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY", "MDG", "MWI", "MDV", "MLI", "MRT", "MUS",
	"MAR", "MOZ", "NAM", "NER", "NGA", "RWA", "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "TUN", "UGA",
	"ESH", "ZMB", "ZWE"}
AMER = {"ATG", "ARG", "BRB", "BLZ", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DMA", "DOM", "ECU", "SLV", "GUF", "GRD", "GTM",
	"GUY", "HTI", "HND", "JAM", "MEX", "NIC", "PAN", "PRY", "PER", "KNA", "LCA", "VCT", "SUR", "TTO", "URY", "VEN"}
ASIA = {"AFG", "BGD", "BTN", "KHM", "CHN", "COK", "PRK", "FJI", "IND", "IDN", "IRN", "IRQ", "JOR", "KAZ", "KIR", "KGZ", "LAO",
	"LBN", "MYS", "MHL", "FSM", "MNG", "MMR", "NRU", "NPL", "OMN", "PAK", "PLW", "PSE", "PNG", "PHL", "WSM", "SLB", "LKA", "SYR",
	"TJK", "THA", "TLS", "TKL", "TON", "TKM", "TUV", "UZB", "VUT", "VNM", "YEM", "ARM", "AZE", "GEO", "TUR"}

NR_ROOTSHOOT_RULES = {(11, "AFRI"): 0.825, (11, "AMER"): 0.221, (11, "ASIA"): 0.207, (12, "AFRI"): 0.232, (12, "AMER"): 0.2845, (12, "ASIA"): 0.323,
	(13, "AFRI"): 0.332, (13, "AMER"): 0.334, (13, "ASIA"): 0.440, (16, "AMER"): 0.348, (16, "ASIA"): 0.322, (21, "AFRI"): 0.232, (21, "AMER"): 0.175,
	(21, "ASIA"): 0.230, (22, "AMER"): 0.336, (22, "ASIA"): 0.440, (23, "AMER"): 1.338, (23, "ASIA"): 1.338}
PL_ROOTSHOOT_RULES = {(11, "AMER"): 0.170, (11, "ASIA"): 0.325, (16, "AMER"): 2.158, (23, "ASIA"): 2.158}

WOOD_PRODUCT_STORAGE_MAP = {0.188: {"ATG", "BLZ", "BRB", "CRI", "CUB", "DMA", "DOM", "GRD", "GTM", "HND", "HTI", "JAM", "KNA", "LCA", "MEX", "NIC", "PAN", "SLV", "TTO", "VCT"},
	0.268: {"ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUF", "GUY", "PER", "PRY", "SUR", "URY", "VEN"},
	0.355: {"AFG", "ARM", "AZE", "BGD", "BTN", "CHN", "GEO", "IDN", "IND", "IRN", "IRQ", "JOR", "KAZ", "KGZ", "KHM", "LAO", "LBN", "LKA", "MMR", "MNG", "MYS", "NPL", "OMN", "PAK", "PHL", "PNG", "PRK", "PSE", "SYR", "THA", "TJK", "TKM", "TLS", "TUR", "UZB", "VNM", "YEM"},
	0.392: {"AGO", "BDI", "BEN", "BFA", "BWA", "CAF", "CIV", "CMR", "COD", "COG", "COM", "CPV", "DJI", "DZA", "EGY", "ERI", "ESH", "ETH", "GAB", "GHA", "GIN", "GMB", "GNB", "GNQ", "KEN", "LBR", "LBY", "LSO", "MAR", "MDG", "MDV", "MLI", "MOZ", "MRT", "MUS", "MWI", "NAM", "NER", "NGA", "RWA", "SDN", "SEN", "SLE", "SOM", "SSD", "STP", "SWZ", "SYC", "TCD", "TGO", "TUN", "TZA", "UGA", "ZAF", "ZMB", "ZWE"},
	0.415: {"COK", "FJI", "FSM", "KIR", "MHL", "NRU", "PLW", "SLB", "TKL", "TON", "TUV", "VUT", "WSM"}}
WOOD_PRODUCT_REVENUE_MAP = {52.65: {"ATG", "BLZ", "BRB", "CRI", "CUB", "DMA", "DOM", "GRD", "GTM", "HND", "HTI", "JAM", "KNA", "LCA", "NIC", "PAN", "SLV", "TTO", "VCT"},
	59.73: {"MEX"}, 61.04: {"BOL", "BRA", "COL", "ECU", "GUF", "GUY", "PER", "SUR", "VEN"}, 67.76: {"BGD", "IDN", "KHM", "LAO", "LKA", "MMR", "MYS", "OMN", "PHL", "PNG", "THA", "TLS", "VNM", "YEM"},
	69.26: {"PRY"}, 71.24: {"AGO", "BDI", "BEN", "BFA", "CAF", "CIV", "CMR", "COD", "COG", "COM", "CPV", "DJI", "ERI", "ETH", "GAB", "GHA", "GIN", "GMB", "GNB", "GNQ", "KEN", "LBR", "MDG", "MDV", "MLI", "MOZ", "MRT", "MUS", "MWI", "NER", "NGA", "RWA", "SDN", "SEN", "SLE", "SOM", "SSD", "STP", "SYC", "TCD", "TGO", "TZA", "UGA", "ZMB", "ZWE"},
	76.08: {"COK", "FJI", "FSM", "KIR", "MHL", "NRU", "PLW", "SLB", "TKL", "TON", "TUV", "VUT", "WSM"}, 76.88: {"IND"}, 80.003: {"ARG", "CHL", "URY"}, 80.83: {"BWA", "NAM"},
	88.84: {"AFG", "ARM", "AZE", "BTN", "CHN", "GEO", "IRN", "IRQ", "JOR", "KAZ", "KGZ", "LBN", "MNG", "NPL", "PAK", "PRK", "PSE", "SYR", "TJK", "TKM", "TUR", "UZB"},
	93.40: {"DZA", "EGY", "ESH", "LBY", "LSO", "MAR", "SWZ", "TUN", "ZAF"}}
EXOTIC_SHARE_MAP = {0.74: {"AGO", "BWA", "COM", "DJI", "ERI", "SWZ", "ETH", "KEN", "LSO", "MDG", "MWI", "MDV", "MUS", "MOZ", "NAM", "SYC", "SOM", "ZAF", "SSD", "SDN", "TZA", "UGA", "ZMB", "ZWE", "BEN", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COG", "CIV", "COD", "GNQ", "GAB", "GMB", "GHA", "GIN", "GNB", "LBR", "MLI", "MRT", "NER", "NGA", "RWA", "STP", "SEN", "SLE", "TGO"},
	0.50: {"DZA", "EGY", "LBY", "MAR", "TUN", "ESH"}, 0.31: {"CHN", "PRK", "MNG"}, 0.40: {"BGD", "BTN", "KHM", "IND", "IDN", "LAO", "MYS", "MMR", "NPL", "PAK", "PHL", "LKA", "THA", "TLS", "VNM"},
	0.05: {"AFG", "IRN", "IRQ", "JOR", "KAZ", "KGZ", "LBN", "OMN", "PSE", "SYR", "TJK", "TKM", "UZB", "YEM", "ARM", "AZE", "GEO", "TUR"},
	0.78: {"COK", "FJI", "KIR", "MHL", "FSM", "NRU", "PLW", "PNG", "WSM", "SLB", "TKL", "TON", "TUV", "VUT"}, 0.32: {"ATG", "BRB", "CUB", "DMA", "DOM", "GRD", "HTI", "KNA", "LCA", "VCT", "TTO"},
	0.18: {"BLZ", "CRI", "SLV", "GTM", "HND", "JAM", "MEX", "NIC", "PAN"}, 0.97: {"ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUF", "GUY", "PRY", "PER", "SUR", "URY", "VEN"}}

def get_continent(iso: str) -> str:
	return "AFRI" if iso in AFRI else "AMER" if iso in AMER else "ASIA" if iso in ASIA else ""

def mapped_value(mapping: dict[float, set[str]], iso: str, default: float = 0.0) -> float:
	return next((value for value, isos in mapping.items() if iso in isos), default)

def iter_input_isos(dta_dir: Path) -> list[str]:
	return sorted(path.stem.upper() for path in dta_dir.glob("*.dta") if path.is_file())

def iter_smallest_input_isos(dta_dir: Path, count: int) -> list[str]:
	paths = sorted((path for path in dta_dir.glob("*.dta") if path.is_file()), key=lambda path: (path.stat().st_size, path.stem.upper()))
	return [path.stem.upper() for path in paths[:count]]

@functools.lru_cache(maxsize=1)
def load_harvest_year_buckets(csv_path: Path = HARVEST_YEAR_BUCKETS_CSV) -> dict[int, int]:
	if not csv_path.exists(): raise FileNotFoundError(f"Run 0_choose_harvest_years.py first: {csv_path}")
	buckets: dict[int, int] = {}
	with open(csv_path, newline='') as f:
		for row in csv.DictReader(f):
			buckets[int(row["original_harvest_year"])] = int(row["chosen_harvest_year"])
	floor_year = min(buckets.values())
	for year in range(3, min(buckets)):
		buckets[year] = floor_year
	return buckets

def calculate_harvest_year(k: np.ndarray, time_horizon_years: int, discount_rate: float) -> np.ndarray:
	harvest_year = np.full(len(k), float(time_horizon_years + 1))
	for year in range(3, time_horizon_years + 1):
		term2 = (1 - np.exp(-k * (year - 2))) ** 2
		with np.errstate(invalid='ignore', divide='ignore'):
			rule_lhs = np.where(term2 == 0, np.nan, ((1 - np.exp(-k * (year - 1))) ** 2 - term2) / term2)
		rule_rhs = discount_rate / (1 - (1 + discount_rate) ** (-(year - 1)))
		harvest_year = np.where(rule_lhs > rule_rhs, float(year), harvest_year)
	buckets = load_harvest_year_buckets()
	max_yr = int(np.max(harvest_year)) if len(harvest_year) > 0 else time_horizon_years + 1
	lookup = np.arange(0, max_yr + 2, dtype=float)  # identity mapping by default
	for orig, chosen in buckets.items():
		if 0 <= orig <= max_yr + 1:
			lookup[orig] = float(chosen)
	return lookup[harvest_year.astype(int)]

def initialize_smdamage_database(output_dir: Path) -> None:
	db_path = output_dir / SMDAMAGE_DB_FILE
	output_dir.mkdir(parents=True, exist_ok=True)
	with sqlite3.connect(db_path) as conn:
		conn.execute("DROP TABLE IF EXISTS Busch2024_undiscounted_dta_output")
		conn.execute("DROP TABLE IF EXISTS Busch2024_undiscounted_tC_per_h_year")
		conn.execute(f"DROP TABLE IF EXISTS {SMDAMAGE_PIXEL_TABLE}")
		conn.execute(f"DROP TABLE IF EXISTS {SMDAMAGE_YEAR_TABLE}")
		conn.execute(f"CREATE TABLE {SMDAMAGE_PIXEL_TABLE} (country TEXT NOT NULL, pixel_id INTEGER NOT NULL, plantation_genus TEXT, selected_option TEXT, selected_A REAL, selected_k REAL, selected_rotation_year REAL, area_ha REAL, crop_va_USD_per_ha_per_year REAL, selected_establishment_cost_USD_per_ha REAL, p_USD_per_tC_harvested REAL)")
		conn.execute(f"CREATE TABLE {SMDAMAGE_YEAR_TABLE} (country TEXT NOT NULL, pixel_id INTEGER NOT NULL, year INTEGER NOT NULL, tC_per_ha_per_year REAL)")
		conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{SMDAMAGE_PIXEL_TABLE}_country_pixel ON {SMDAMAGE_PIXEL_TABLE} (country, pixel_id)")

def clear_smdamage_country(conn: sqlite3.Connection, country: str) -> None:
	conn.execute(f"DELETE FROM {SMDAMAGE_PIXEL_TABLE} WHERE country = ?", (country,))
	conn.execute(f"DELETE FROM {SMDAMAGE_YEAR_TABLE} WHERE country = ?", (country,))

# Execute one country pipeline from input .dta to SMDAMAGE exports.
def process_country(iso: str, dta_dir: Path, macc_dir: Path) -> dict[str, int]:
	# Find the input path.
	src = next((c for c in [dta_dir / iso, dta_dir / f"{iso}.dta", dta_dir / f"maps_{iso}.dta"] if c.exists() and c.is_file()), None)

	# Read .dta file via pyreadstat; output_format='dict' returns OrderedDict of Python lists.
	# np.array(lst, dtype=float) converts None (Stata missing) to NaN.
	raw_lists, _meta = pyreadstat.read_dta(str(src), output_format='dict', disable_datetime_conversion=True)
	d: dict[str, np.ndarray] = {k: np.array(v, dtype=float) for k, v in raw_lists.items()}
	N0 = len(next(iter(d.values())))
	d["id"] = np.arange(1, N0 + 1, dtype=float)

	# Data cleaning: ensure required columns exist with sensible defaults.
	for col in ["griscom", "brancalion", "bastin", "walker", "nr_buffer", "pl_buffer"]:
		if col not in d:
			d[col] = np.zeros(N0)
		else:
			d[col] = np.where(np.isnan(d[col]), 0.0, d[col])
	for col in ["biomes", "nr_A", "type"]:
		if col not in d: d[col] = np.full(N0, np.nan)
	for col in ["nr_cost", "ep_cost", "np_cost", "crop_va"]:
		if col not in d: d[col] = np.zeros(N0)
	if "fao_ecoz" not in d: d["fao_ecoz"] = np.full(N0, np.nan)

	# Drop deserts/xeric (biomes=13) and mangroves (biomes=14); drop missing nr_A and type.
	keep = (d["biomes"] != 13) & (d["biomes"] != 14) & ~np.isnan(d["nr_A"]) & ~np.isnan(d["type"])
	d = {k: v[keep] for k, v in d.items()}
	N = len(d["id"])

	# Keep all valid pixels, or one selected pixel when RUN_ALL_PIXELS is False.
	if not RUN_ALL_PIXELS:
		keep2 = d["id"] == float(TARGET_PIXEL_ID)
		d = {k: v[keep2] for k, v in d.items()}
		N = len(d["id"])

	# Convert 2011 establishment costs to 2020 USD.
	for c in ["nr_cost", "ep_cost", "np_cost"]: d[c] = d[c] * 1.1577

	# Truncate negative A and k values at zero before growth calculations.
	for col in ["nr_A", "nr_k"] + [f"{g}_{s}" for g in GENUS_ORDER for s in ["A", "k"]]:
		if col in d:
			d[col] = d[col].copy()
			d[col][d[col] < 0] = 0.0

	# Plantation genus mapping.
	type_int = d["type"].astype(int)
	genus_arr = np.array([GENUS_BY_TYPE.get(t, "") for t in type_int], dtype=object)
	d["genus"] = genus_arr
	d["genus_A"] = np.full(N, np.nan)
	d["genus_k"] = np.full(N, np.nan)
	for g in GENUS_ORDER:
		mg = genus_arr == g
		if np.any(mg):
			d["genus_A"][mg] = d[f"{g}_A"][mg]
			d["genus_k"][mg] = d[f"{g}_k"][mg]

	# Convert area m² → ha.
	d["area_ha"] = d["area_m2"] / 10000.0

	# Root-shoot ratios (IPCC 2019 V4 Ch4 Table 4.4).
	d["nr_rootshoot"] = np.full(N, 0.26)
	d["pl_rootshoot"] = np.full(N, 0.26)
	continent = get_continent(iso)
	is_brde_nede = np.isin(genus_arr, ["brde", "nede"])
	is_pinu_cunn_brev_neev = np.isin(genus_arr, ["pinu", "cunn", "brev", "neev"])
	fao = d["fao_ecoz"]

	for (eco, cont), value in NR_ROOTSHOOT_RULES.items():
		if cont == continent: d["nr_rootshoot"][fao == eco] = value
	for (eco, cont), value in PL_ROOTSHOOT_RULES.items():
		if cont == continent: d["pl_rootshoot"][fao == eco] = value

	if continent == "AMER":
		d["nr_rootshoot"][(fao == 31) & is_brde_nede] = 0.466
		d["nr_rootshoot"][(fao == 31) & is_pinu_cunn_brev_neev] = 0.337
		d["nr_rootshoot"][(fao == 32) & is_pinu_cunn_brev_neev] = 0.237
		d["pl_rootshoot"][(fao == 31) & is_pinu_cunn_brev_neev] = 0.203
		d["pl_rootshoot"][(fao == 32) & is_pinu_cunn_brev_neev] = 0.237
	elif continent == "ASIA":
		d["nr_rootshoot"][(fao == 31) & is_brde_nede] = 0.225
		d["nr_rootshoot"][(fao == 31) & is_pinu_cunn_brev_neev] = 0.243
		d["nr_rootshoot"][(fao == 32) & is_brde_nede] = 0.225
		d["nr_rootshoot"][(fao == 32) & is_pinu_cunn_brev_neev] = 0.243
		d["pl_rootshoot"][(fao == 31) & is_brde_nede] = 0.307
		d["pl_rootshoot"][(fao == 31) & is_pinu_cunn_brev_neev] = 0.224
		d["pl_rootshoot"][(fao == 32) & is_brde_nede] = 0.307
		d["pl_rootshoot"][(fao == 32) & is_pinu_cunn_brev_neev] = 0.224

	# Country-level scalars stored as full arrays.
	d["w"]      = np.full(N, mapped_value(WOOD_PRODUCT_STORAGE_MAP, iso))
	d["p"]      = np.full(N, mapped_value(WOOD_PRODUCT_REVENUE_MAP, iso))
	d["exotic"] = np.full(N, mapped_value(EXOTIC_SHARE_MAP, iso))
	d["native"] = 1.0 - d["exotic"]
	d["nr_npv_estcost_USD_per_ha"] = d["nr_cost"].copy()

	# NR total carbon (telescoped Chapman-Richards sum over full time horizon).
	Years_time_horizon = TIME_HORIZON_YEARS
	T = Years_time_horizon
	nr_A  = d["nr_A"]
	nr_k  = d.get("nr_k", np.zeros(N))
	nr_rs = d["nr_rootshoot"]
	d["nr_wohrv_total_tC_per_ha"] = (
		(AGB_C_POOL + BGB_C_POOL * nr_rs) * nr_A * (1 - np.exp(-nr_k * T)) ** 2
		+ SOIL_C_POOL * T * (NATURAL_SOIL_C + PLANTATION_SOIL_C)
	)

	# NR annual carbon schedules: (N, T) float32 array computed in row chunks.
	years_arr = np.arange(1, T + 1, dtype=np.float32)
	k_nr  = nr_k.astype(np.float32)
	A_nr  = nr_A.astype(np.float32)
	rs_nr = nr_rs.astype(np.float32)
	nr_annual = np.empty((N, T), dtype=np.float32)
	nr_chunk_rows = 120_000
	for i0 in range(0, N, nr_chunk_rows):
		i1 = min(i0 + nr_chunk_rows, N)
		k_ch = k_nr[i0:i1, None]; A_ch = A_nr[i0:i1, None]; rs_ch = rs_nr[i0:i1, None]
		nr_cum_ch = ((AGB_C_POOL + BGB_C_POOL * rs_ch) * A_ch * (1 - np.exp(-k_ch * years_arr)) ** 2
		             + SOIL_C_POOL * years_arr * (NATURAL_SOIL_C + PLANTATION_SOIL_C)).astype(np.float32, copy=False)
		nr_annual[i0:i1, 0] = nr_cum_ch[:, 0]
		nr_annual[i0:i1, 1:] = np.diff(nr_cum_ch, axis=1)
	np.nan_to_num(nr_annual, copy=False, nan=0.0)
	pl_annual = np.zeros((N, T), dtype=np.float32)

	# Initialise output columns.
	d["nr_harvestyear"] = np.full(N, float(T + 1))
	for col in ["pl_wohrv_total_tC_per_ha", "pl_whrv0_total_tC_per_ha", "pl_whrv_total_tC_per_ha",
	            "pl_npv_estcost_USD_per_ha", "pl_harvested_tC_per_ha"]:
		d[col] = np.full(N, np.nan)
	d["selected_plantation_rotation_year"] = np.full(N, np.nan)
	d["nr_wohrv_costeff_USD_per_tCO2"]  = np.full(N, np.nan)
	d["pl_wohrv_costeff_USD_per_tCO2"]  = np.full(N, np.nan)
	d["pl_whrv0_costeff_USD_per_tCO2"]  = np.full(N, np.nan)
	d["pl_whrv_costeff_USD_per_tCO2"]   = np.full(N, np.nan)
	d["refortype_whrv_tCO2_per_USD"]    = np.full(N, "", dtype=object)
	d["mi_whrv_total_tC_per_ha"]        = np.full(N, np.nan)
	d["mi_whrv_costeff_USD_per_tCO2"]   = np.full(N, np.nan)

	# NR harvest year (Faustmann-style; numpy array version).
	discount_rate_for_harvest = DISCOUNT_RATE
	d["nr_harvestyear"] = calculate_harvest_year(nr_k, T, discount_rate_for_harvest)

	# Discounted opportunity-cost sum.
	if "crop_va" not in d: d["crop_va"] = np.zeros(N)
	oppcost_discount_sum = float(np.sum((1.0 - DISCOUNT_RATE) ** (np.arange(1, T + 1, dtype=np.float64) - 1.0)))

	for g in GENUS_ORDER:
		mask_g = genus_arr == g
		if not np.any(mask_g): continue
		rows = np.flatnonzero(mask_g)
		A1d      = d[f"{g}_A"][mask_g].astype(np.float32)
		k1d_f64  = d[f"{g}_k"][mask_g]                     # float64 for harvest-year rule
		k1d      = k1d_f64.astype(np.float32)               # float32 for vectorised loops
		g_harvestyear = calculate_harvest_year(k1d_f64, T, discount_rate_for_harvest)
		if g == "pinu":   g_npv_init = d["exotic"][mask_g] * d["ep_cost"][mask_g] + d["native"][mask_g] * d["np_cost"][mask_g]
		elif g == "cunn": g_npv_init = d["np_cost"][mask_g] if iso == "CHN" else d["ep_cost"][mask_g]
		elif g == "euca": g_npv_init = d["ep_cost"][mask_g]
		else:             g_npv_init = d["exotic"][mask_g] * d["ep_cost"][mask_g] + d["native"][mask_g] * d["np_cost"][mask_g]
		hy1d     = g_harvestyear.astype(np.float32)
		hy1d_int = np.maximum(hy1d.astype(np.int32), 1)
		rs1d = d["pl_rootshoot"][mask_g].astype(np.float32)
		w1d  = d["w"][mask_g].astype(np.float32)
		g_sf = np.empty(len(rows), dtype=np.float32)
		g_hf = np.empty(len(rows), dtype=np.float32)
		harvest_counts = np.empty(len(rows), dtype=np.int16)
		yr = years_arr[None, :]
		chunk_rows = 100_000
		for c0 in range(0, len(rows), chunk_rows):
			c1 = min(c0 + chunk_rows, len(rows))
			A_ch  = A1d[c0:c1, None]; k_ch  = k1d[c0:c1, None]
			hy_ch = hy1d[c0:c1, None]; rs_ch = rs1d[c0:c1, None]; w_ch  = w1d[c0:c1, None]
			yr_mod   = np.mod(yr, hy_ch)
			harvest  = (yr_mod == 0)
			g_stock_2d = (A_ch * (1 - np.exp(-k_ch * yr_mod)) ** 2).astype(np.float32, copy=False)
			harv_c = (A_ch * (1 - np.exp(-k_ch * yr)) ** 2).astype(np.float32, copy=False)
			harv_c *= harvest
			harv_c *= ((1.0 - DISCOUNT_RATE) ** (yr - 1.0)).astype(np.float32, copy=False)
			np.cumsum(harv_c, axis=1, out=harv_c)
			g_soil_2d = (PLANTATION_SOIL_C * np.minimum(np.maximum(hy_ch - 1.0, 0.0), yr)).astype(np.float32, copy=False)
			# Use in-place ops on a copy to match original's float32 truncation sequence exactly.
			g_cum_2d = g_stock_2d.copy()
			g_cum_2d *= (1.0 + rs_ch)
			g_cum_2d += g_soil_2d
			g_cum_2d += w_ch * harv_c
			pl_g = np.empty_like(g_cum_2d, dtype=np.float32)
			pl_g[:, 0] = g_cum_2d[:, 0]
			pl_g[:, 1:] = np.diff(g_cum_2d, axis=1)
			np.nan_to_num(pl_g, copy=False, nan=0.0)
			pl_annual[rows[c0:c1], :] = pl_g
			g_sf[c0:c1] = g_cum_2d[:, -1]   # original aliases g_stock_2d=g_cum_2d, so [:,-1] is the full cumulative value
			g_hf[c0:c1] = harv_c[:, -1]
			harvest_counts[c0:c1] = harvest.sum(axis=1).astype(np.int16, copy=False)
		g_sf_soil  = PLANTATION_SOIL_C * np.minimum(np.maximum(hy1d - 1.0, 0.0), float(T))
		max_cycles = int(np.max(np.floor_divide(T, hy1d_int))) if len(hy1d_int) else 0
		if max_cycles > 0:
			m = np.arange(1, max_cycles + 1, dtype=np.int32)
			harvest_years = hy1d_int[:, None] * m[None, :]
			valid   = harvest_years <= T
			weights = (1.0 - DISCOUNT_RATE) ** (harvest_years.astype(np.float64) - 1.0)
			g_npv_f = g_npv_init * np.prod(np.where(valid, 1.0 + weights, 1.0), axis=1)
		else:
			g_npv_f = g_npv_init
		g_wohrv_total = (AGB_C_POOL * A1d * (1 - np.exp(-k1d * T)) ** 2
		                 + BGB_C_POOL * (A1d * (1 - np.exp(-k1d * T)) ** 2) * rs1d
		                 + SOIL_C_POOL * T * PLANTATION_SOIL_C)
		g_whrv_total  = AGB_C_POOL * g_sf + BGB_C_POOL * g_sf * rs1d + SOIL_C_POOL * g_sf_soil + HARVEST_C_POOL * w1d * g_hf
		d["pl_wohrv_total_tC_per_ha"][mask_g]          = g_wohrv_total
		d["pl_whrv0_total_tC_per_ha"][mask_g]          = g_whrv_total
		d["pl_whrv_total_tC_per_ha"][mask_g]           = g_whrv_total
		d["pl_npv_estcost_USD_per_ha"][mask_g]         = g_npv_f
		d["pl_harvested_tC_per_ha"][mask_g]            = g_hf
		d["selected_plantation_rotation_year"][mask_g] = g_harvestyear

	# Cost-effectiveness in $/tCO2.
	def _safe_denom(arr: np.ndarray) -> np.ndarray:
		out = arr.copy().astype(float); out[out == 0] = np.nan; return out

	d["nr_wohrv_costeff_USD_per_tCO2"] = (d["crop_va"] * oppcost_discount_sum + d["nr_cost"]) / _safe_denom(3.67 * d["nr_wohrv_total_tC_per_ha"])
	d["pl_wohrv_costeff_USD_per_tCO2"] = (d["crop_va"] * oppcost_discount_sum + d["pl_npv_estcost_USD_per_ha"]) / _safe_denom(3.67 * d["pl_wohrv_total_tC_per_ha"])
	d["pl_whrv0_costeff_USD_per_tCO2"] = (d["crop_va"] * oppcost_discount_sum + d["pl_npv_estcost_USD_per_ha"]) / _safe_denom(3.67 * d["pl_whrv0_total_tC_per_ha"])
	d["pl_whrv_costeff_USD_per_tCO2"]  = (d["crop_va"] * oppcost_discount_sum + d["pl_npv_estcost_USD_per_ha"] - d["p"] * d["pl_harvested_tC_per_ha"]) / _safe_denom(3.67 * d["pl_whrv_total_tC_per_ha"])

	# Compare plantation vs natural regeneration for whrv scenario.
	lhs_nr = 3.67 * d["nr_wohrv_total_tC_per_ha"] / _safe_denom(d["crop_va"] * oppcost_discount_sum + d["nr_cost"])
	lhs_pl = 3.67 * d["pl_whrv_total_tC_per_ha"] / _safe_denom(d["crop_va"] * oppcost_discount_sum + d["pl_npv_estcost_USD_per_ha"] - d["p"] * d["pl_harvested_tC_per_ha"])

	# Determine minimum cost-effectiveness.
	refortype = np.where(lhs_pl > lhs_nr, "P", np.where(lhs_pl < lhs_nr, "N", "."))
	refortype[lhs_pl < 0] = "P"
	d["refortype_whrv_tCO2_per_USD"] = refortype
	mi_counts = {"refortype_dot_total": int(np.sum(refortype == "."))}
	for col in ["nr_wohrv_total_tC_per_ha", "pl_whrv_total_tC_per_ha", "crop_va", "nr_cost",
	            "pl_npv_estcost_USD_per_ha", "p", "pl_harvested_tC_per_ha"]:
		mi_counts[f"{col}_dot_null_or_nan"] = int(np.sum(np.isnan(d[col].astype(float)[refortype == "."])))

	# Minimum-cost option columns.
	d["mi_whrv_total_tC_per_ha"]    = np.where(refortype == "P", d["pl_whrv_total_tC_per_ha"],
	                                             np.where(refortype == "N", d["nr_wohrv_total_tC_per_ha"], 0.0))
	d["mi_whrv_costeff_USD_per_tCO2"] = np.where(refortype == "P", d["pl_whrv_costeff_USD_per_tCO2"],
	                                               np.where(refortype == "N", d["nr_wohrv_costeff_USD_per_tCO2"], np.nan))
	is_P_1d = refortype == "P"
	is_N_1d = refortype == "N"

	# Build SMDAMAGE export arrays.
	selected_option = np.where(refortype == "P", "plantation",
	                            np.where(refortype == "N", "natural_regeneration", "undetermined"))
	selected_rotation_year = np.where(refortype == "P", d["selected_plantation_rotation_year"],
	                                   np.where(refortype == "N", float(T), np.nan))
	macc_dir.mkdir(parents=True, exist_ok=True)

	if SKIP_SQLITE_EXPORT:
		print(f"[SKIP_SQLITE_EXPORT] {iso}", flush=True)
	else:
		db_path = macc_dir / SMDAMAGE_DB_FILE
		with sqlite3.connect(db_path) as conn:
			conn.execute("PRAGMA synchronous = OFF")
			conn.execute("PRAGMA cache_size = -131072")
			clear_smdamage_country(conn, iso)
			# Helper: convert float array to list, replacing NaN with None for SQLite NULL.
			def _sql(arr: np.ndarray) -> list:
				lst = arr.astype(float).tolist()
				return [None if (v != v) else v for v in lst]
			sel_A    = np.where(refortype == "P", d["genus_A"], np.where(refortype == "N", d["nr_A"], np.nan))
			sel_k    = np.where(refortype == "P", d["genus_k"], np.where(refortype == "N", d["nr_k"], np.nan))
			sel_cost = np.where(refortype == "P", d["pl_npv_estcost_USD_per_ha"], np.where(refortype == "N", d["nr_cost"], np.nan))
			p_harv   = np.where(refortype == "P", d["p"], 0.0)
			pixel_rows = list(zip(
				[iso] * N, [int(x) for x in d["id"]], d["genus"].tolist(), selected_option.tolist(),
				_sql(sel_A), _sql(sel_k), _sql(selected_rotation_year),
				_sql(d["area_ha"]), _sql(d["crop_va"]), _sql(sel_cost), _sql(p_harv),
			))
			conn.executemany(
				f"INSERT INTO {SMDAMAGE_PIXEL_TABLE} (country, pixel_id, plantation_genus, selected_option, "
				"selected_A, selected_k, selected_rotation_year, area_ha, crop_va_USD_per_ha_per_year, "
				"selected_establishment_cost_USD_per_ha, p_USD_per_tC_harvested) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
				pixel_rows
			)
			# Annual carbon rows — identical write logic to the pandas version.
			sel_rot = np.where(np.isnan(selected_rotation_year), -1.0, selected_rotation_year)
			pixel_ids_arr = d["id"]
			insert_sql = f"INSERT INTO {SMDAMAGE_YEAR_TABLE} (country, pixel_id, year, tC_per_ha_per_year) VALUES (?,?,?,?)"
			for yr_idx in range(T):
				year = yr_idx + 1
				ym = sel_rot >= float(year)
				if not np.any(ym): continue
				p_mask = is_P_1d[ym]; n_mask = is_N_1d[ym]
				vals = nr_annual[ym, yr_idx].copy()
				vals[p_mask] = pl_annual[ym, yr_idx][p_mask]
				vals[~(p_mask | n_mask)] = 0.0
				pids = [int(x) for x in pixel_ids_arr[ym]]
				conn.executemany(insert_sql, zip([iso] * len(pids), pids, [year] * len(pids), vals.tolist()))

	mode_label = "all pixels" if RUN_ALL_PIXELS else f"pixel {TARGET_PIXEL_ID}"
	print(f"[OK] {iso} {mode_label}", flush=True)
	return mi_counts

# Iterate across requested ISO codes and run the country pipeline.
def main() -> None:
	parser = argparse.ArgumentParser(description="Pandas-free version of 1_import_Busch2024_to_SMDAMAGE.py")
	parser.add_argument("--dta-dir", type=Path, required=True, help="Directory containing per-country .dta input files")
	parser.add_argument("--macc-dir", type=Path, required=True, help="Directory to write output SQLite database")
	parser.add_argument("--iso", nargs="*", help="Optional ISO code filter.")
	parser.add_argument("--smallest-n", type=int, help="Optional count of smallest .dta files to process.")
	args = parser.parse_args()
	requested_isos = sorted(iso.upper() for iso in args.iso) if args.iso else iter_smallest_input_isos(args.dta_dir, args.smallest_n) if args.smallest_n else iter_input_isos(args.dta_dir)
	initialize_smdamage_database(args.macc_dir)
	all_blank_counts: dict[str, dict[str, int]] = {}
	for iso in requested_isos: all_blank_counts[iso] = process_country(iso=iso, dta_dir=args.dta_dir, macc_dir=args.macc_dir)

if __name__ == "__main__":
	base_dir = Path(__file__).resolve().parent.parent  # Busch2024/
	db_dir   = base_dir / "Output" / "Databases"
	db_path  = db_dir / SMDAMAGE_DB_FILE
	dta_dir  = base_dir / "Input"
	cutoff_src = next((c for c in [dta_dir / RESUME_AFTER_ISO, dta_dir / f"{RESUME_AFTER_ISO}.dta", dta_dir / f"maps_{RESUME_AFTER_ISO}.dta"] if c.exists() and c.is_file()), None)
	if cutoff_src is None: raise FileNotFoundError(f"{RESUME_AFTER_ISO} input file not found in {dta_dir}")
	cutoff_size = cutoff_src.stat().st_size
	requested_isos = [path.stem.upper() for path in sorted((path for path in dta_dir.glob("*.dta") if path.is_file() and path.stat().st_size > cutoff_size), key=lambda path: (path.stat().st_size, path.stem.upper()))]
	if db_path.exists(): print(f"Resume mode: assuming all countries up to {RESUME_AFTER_ISO} ({cutoff_size/1e6:.1f} MB) are complete; skipping DB completeness checks.", flush=True)
	else:
		print(f"No existing {db_path.name}; creating a fresh database.", flush=True)
		initialize_smdamage_database(db_dir)
	print(f"Processing {len(requested_isos)} countries with .dta size > {RESUME_AFTER_ISO}.", flush=True)
	all_blank_counts: dict[str, dict[str, int]] = {}
	for iso in requested_isos:
		src = next((c for c in [dta_dir / iso, dta_dir / f"{iso}.dta", dta_dir / f"maps_{iso}.dta"] if c.exists() and c.is_file()), None)
		size_mb = src.stat().st_size / 1e6 if src else 0
		t0 = time.perf_counter()
		all_blank_counts[iso] = process_country(iso=iso, dta_dir=dta_dir, macc_dir=db_dir)
		elapsed = time.perf_counter() - t0
		print(f"  {elapsed:.1f}s / {size_mb:.1f} MB = {elapsed/size_mb:.2f} s/MB", flush=True)
	print("Building year table index...", flush=True)
	with sqlite3.connect(db_path) as _c:
		_c.execute("PRAGMA synchronous = OFF")
		_c.execute(f"CREATE INDEX IF NOT EXISTS idx_{SMDAMAGE_YEAR_TABLE}_country_pixel_year ON {SMDAMAGE_YEAR_TABLE} (country, pixel_id, year)")
	print("Index built.", flush=True)
