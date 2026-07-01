# Current per-pixel export pipeline for the Busch 2024 reforestation model.
# Reads per-country .dta files via pd.read_stata, applies biome filtering, IPCC root-shoot ratios, continent/ecozonal rules, cost conversions, 
# and plantation genus selection, then computes annual undiscounted carbon sequestration for each pixel under natural regeneration and plantation options.
# Selects the cost-effective option per pixel and writes results to an sqlite file: 
# one row per pixel in Undiscounted_dta_output and one row per pixel-year in tC_per_h_per_year. 

from __future__ import annotations
import argparse
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

SMDAMAGE_DB_FILE = "Busch2024_to_SMDAMAGE.sqlite"
SMDAMAGE_PIXEL_TABLE = "Undiscounted_dta_output"
SMDAMAGE_YEAR_TABLE = "tC_per_h_per_year"
TARGET_PIXEL_ID = 5119
RUN_ALL_PIXELS = True
GENUS_ORDER = ["pinu", "cunn", "euca", "brde", "brev", "nede", "neev"]
GENUS_BY_TYPE = {1: "neev", 2: "brev", 3: "brde", 4: "neev", 5: "cunn", 6: "euca", 7: "nede", 8: "neev", 9: "pinu", 10: "brde", 11: "neev", 12: "brde", 13: "brev", 14: "brde", 15: "neev"}
TIME_HORIZON_YEARS = 35
DISCOUNT_RATE = 0.05
AUSTMANN_F = 1
NATURAL_SOIL_C = 0.415107735
PLANTATION_SOIL_C = 0.092749069
AGB_C_POOL = 1
BGB_C_POOL = 1
SOIL_C_POOL = 1
HARVEST_C_POOL = 1
SMALL_DTA_COUNT = 50
SKIP_SQLITE_EXPORT = False

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

def calculate_harvest_year(k: pd.Series, time_horizon_years: int, discount_rate: float) -> pd.Series:
	harvest_year = pd.Series(float(time_horizon_years + 1), index=k.index)
	for year in range(3, time_horizon_years + 1):
		rule_lhs = ((1 - np.exp(-k*(year - 1))) ** 2 - (1 - np.exp(-k*(year - 2))) ** 2) / ((1 - np.exp(-k*(year - 2))) ** 2).replace(0, np.nan)
		rule_rhs = discount_rate / (1 - (1 + discount_rate) ** (-(year - 1)))
		harvest_year = harvest_year.where(~(rule_lhs > rule_rhs), float(year))
	# I wanted to reduce the number of possible harvest years so SMDAMAGE is easier to solve. I used choose_harvest_years.py to calculate these.
	HARVEST_YEAR_BUCKETS = {3:3, 4:4, 5:6, 6:6, 7:6, 8:10, 9:10, 10:10, 11:10, 12:10, 13:17, 14:17, 15:17, 16:17, 17:17, 18:17, 19:17, 20:17, 21:17, 22:17, 23:35, 24:35, 25:35, 26:35, 27:35, 28:35, 29:35, 30:35, 31:35, 32:35, 33:35, 34:35, 35:35, 36:35, 37:35, 38:35, 39:35, 40:35, 41:35, 42:35, 43:35, 44:35, 45:35, 46:35, 47:35, 48:35, 49:35, 50:35, 51:35,}
	return harvest_year.map(HARVEST_YEAR_BUCKETS).fillna(harvest_year).astype(float)

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
		conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{SMDAMAGE_YEAR_TABLE}_country_pixel_year ON {SMDAMAGE_YEAR_TABLE} (country, pixel_id, year)")

def clear_smdamage_country(conn: sqlite3.Connection, country: str) -> None:
	conn.execute(f"DELETE FROM {SMDAMAGE_PIXEL_TABLE} WHERE country = ?", (country,))
	conn.execute(f"DELETE FROM {SMDAMAGE_YEAR_TABLE} WHERE country = ?", (country,))

def write_smdamage_database(output_dir: Path, pixel_export: pd.DataFrame, annual_export: pd.DataFrame) -> None:
	db_path = output_dir / SMDAMAGE_DB_FILE
	with sqlite3.connect(db_path) as conn:
		countries = pixel_export["country"].dropna().astype(str).unique().tolist()
		for country in countries:
			clear_smdamage_country(conn, country)
		pixel_export.to_sql(SMDAMAGE_PIXEL_TABLE, conn, if_exists="append", index=False)
		annual_export.to_sql(SMDAMAGE_YEAR_TABLE, conn, if_exists="append", index=False)

# Plain-English explanations for dataframe columns used or created in this pipeline.
df_explanation = {
	"id": "Sequential pixel identifier created in this script.",
	"genus": "Selected plantation genus code for the pixel.",
	"area_m2": "Pixel area in square meters from input.",
	"area_ha": "Pixel area in hectares. Original .do variable name after conversion: area.",
	"type": "Input plantation type code used to map to a genus.",
	"biomes": "Biome code used for filtering out excluded biomes.",
	"fao_ecoz": "FAO ecological zone code used in root-to-shoot rules.",
	"continent": "Continent label used for ecological lookups.",
	"griscom": "Baseline binary geographic screen for reforestation suitability (based on Griscom et al. map in the paper workflow); used to include or exclude pixels.",
	"brancalion": "Alternative binary geographic screen for sensitivity analysis (Brancalion tropical restoration feasibility map); used as a substitute for the baseline screen.",
	"bastin": "Alternative binary geographic screen for sensitivity analysis (Bastin global tree restoration potential map); used as a substitute for the baseline screen.",
	"walker": "Alternative binary geographic screen for sensitivity analysis (Walker high unrealized woody restoration potential map); used as a substitute for the baseline screen.",
	"nr_buffer": "Natural regeneration buffer variable from source dataset.",
	"pl_buffer": "Plantation buffer variable from source dataset.",
	"nr_A": "Natural regeneration Chapman-Richards asymptote parameter A.",
	"nr_k": "Natural regeneration Chapman-Richards growth rate parameter k.",
	"exotic": "Country-level share of plantations that are exotic species.",
	
	"nr_rootshoot": "Natural regeneration root-to-shoot ratio.",
	"pl_rootshoot": "Plantation root-to-shoot ratio.",
	"w": "Fraction of carbon at time of harvest that remains stored in long-lived wood products (discounted flow).",
	
	"nr_harvestyear": "Economically selected rotation length proxy for natural regeneration.",
	
	"crop_va": "Annual crop value per hectare used as opportunity cost.",
	"nr_cost": "Natural regeneration establishment or program cost per hectare in 2020 USD.",
	"ep_cost": "Exotic plantation establishment cost per hectare in 2020 USD.",
	"np_cost": "Native plantation establishment cost per hectare in 2020 USD.",
	"p": "Revenue from wood products (net of harvest and delivery costs) as a scalar of carbon at time of harvest.",
	
	"nr_wohrv_total_tC_per_ha": "Natural regeneration total carbon per hectare without harvest over the time horizon. Original .do variable name: nr_wohrv_totalC.",
	"pl_wohrv_total_tC_per_ha": "Plantation total carbon per hectare without harvest. Original .do variable name: pl_wohrv_totalC.",
	"pl_whrv0_total_tC_per_ha": "Plantation total carbon per hectare in the with-harvest scenario for whrv0; in this implementation the carbon equation matches whrv and the distinction is in the economic term. Original .do variable name: pl_whrv0_totalC.",
	"pl_whrv_total_tC_per_ha": "Plantation total carbon per hectare in the with-harvest scenario for whrv; in this implementation the carbon equation matches whrv0 and the distinction is in the economic term. Original .do variable name: pl_whrv_totalC.",
	"pl_npv_estcost_USD_per_ha": "Plantation net present value establishment cost per hectare. Original .do variable name: pl_npv_estcost.",
	"pl_harvested_tC_per_ha": "Plantation harvested carbon per hectare over the time horizon. Original .do variable name: pl_harvested.",
	"nr_wohrv_costeff_USD_per_tCO2": "Natural regeneration abatement cost-effectiveness in USD per ton CO2. Original .do variable name: nr_wohrv_costeff.",
	"pl_wohrv_costeff_USD_per_tCO2": "Plantation abatement cost-effectiveness without harvest in USD per ton CO2. Original .do variable name: pl_wohrv_costeff.",
	"pl_whrv0_costeff_USD_per_tCO2": "Plantation abatement cost-effectiveness with harvest, excluding wood product revenue, in USD per ton CO2. Original .do variable name: pl_whrv0_costeff.",
	"pl_whrv_costeff_USD_per_tCO2": "Plantation abatement cost-effectiveness with harvest, including wood product revenue, in USD per ton CO2. Original .do variable name: pl_whrv_costeff.",
	"refortype_whrv_tCO2_per_USD": "Preferred reforestation type under harvest scenario (P plantation, N natural regeneration). Original .do variable name: refortype_whrv.",
	"mi_whrv_total_tC_per_ha": "Minimum-cost option carbon per hectare under harvest scenario. Original .do variable name: mi_whrv_totalC.",
	"mi_whrv_costeff_USD_per_tCO2": "Minimum-cost option cost-effectiveness under harvest scenario in USD per ton CO2. Original .do variable name: mi_whrv_costeff.",
}

for g in GENUS_ORDER:
	df_explanation[f"{g}_A"] = f"Chapman-Richards asymptote parameter A for {g} plantation genus."
	df_explanation[f"{g}_k"] = f"Chapman-Richards growth rate parameter k for {g} plantation genus."
	df_explanation[f"{g}_harvestyear"] = f"Economically selected harvest rotation year for {g} plantation genus."
	df_explanation[f"{g}_stock_tC_per_ha"] = f"Standing aboveground carbon stock per hectare for {g} plantation genus. Original .do variable name: {g}_stock."
	df_explanation[f"{g}_harvested_tC_per_ha"] = f"Cumulative harvested carbon per hectare for {g} plantation genus. Original .do variable name: {g}_harvested."
	df_explanation[f"{g}_soilC_whrv_tC_per_ha"] = f"Soil carbon per hectare for {g} plantation genus under harvest scenario. Original .do variable name: {g}_soilC_whrv."
	df_explanation[f"{g}_wohrv_total_tC_per_ha"] = f"Total carbon per hectare for {g} plantation genus without harvest. Original .do variable name: {g}_wohrv_totalC."
	df_explanation[f"{g}_whrv0_total_tC_per_ha"] = f"Total carbon per hectare for {g} plantation genus in the with-harvest scenario for whrv0; in this implementation the carbon equation matches whrv and the distinction is in the economic term. Original .do variable name: {g}_whrv0_totalC."
	df_explanation[f"{g}_whrv_total_tC_per_ha"] = f"Total carbon per hectare for {g} plantation genus in the with-harvest scenario for whrv; in this implementation the carbon equation matches whrv0 and the distinction is in the economic term. Original .do variable name: {g}_whrv_totalC."
	df_explanation[f"{g}_npv_estcost_USD_per_ha"] = f"Net present value establishment cost per hectare for {g} plantation genus in USD. Original .do variable name: {g}_npv_estcost."

# Execute one country pipeline from input .dta to SMDAMAGE exports.
def process_country(iso: str, dta_dir: Path, macc_dir: Path) -> dict[str, int]:
	# Find the input paths.
	src = next((c for c in [dta_dir / iso, dta_dir / f"{iso}.dta", dta_dir / f"maps_{iso}.dta"] if c.exists() and c.is_file()), None)
	df = pd.read_stata(src)
	df["id"] = np.arange(1, len(df) + 1)

	# Data cleaning.
	for col in ["griscom", "brancalion", "bastin", "walker", "nr_buffer", "pl_buffer"]:
		if col in df.columns: df[col] = df[col].fillna(0)
		else: df[col] = 0

	for col in ["biomes", "nr_A", "type"]:  # for drop logic
		if col not in df.columns: df[col] = 0.0
	for col in ["nr_cost", "ep_cost", "np_cost", "crop_va"]:
		if col not in df.columns: df[col] = 0.0
	if "fao_ecoz" not in df.columns: df["fao_ecoz"] = np.nan
	# Drop deserts/xeric (biomes=13) and mangroves (biomes=14).
	df = df[df["biomes"] != 13]
	df = df[df["biomes"] != 14]
	df = df[df["nr_A"].notna()]
	df = df[df["type"].notna()]

	# Keep all valid pixels, or one selected pixel when RUN_ALL_PIXELS is False.
	if not RUN_ALL_PIXELS: df = df[df["id"] == TARGET_PIXEL_ID]
	for c in ["nr_cost", "ep_cost", "np_cost"]: df[c] = df[c]*1.1577 # Convert 2011 costs to 2020.

	# Truncate negative A and k values at zero before growth calculations.
	for col in ["nr_A", "nr_k"] + [f"{g}_{suffix}" for g in GENUS_ORDER for suffix in ["A", "k"]]:
		if col in df.columns: df.loc[df[col] < 0, col] = 0

	# Plantation genus
	df["genus"] = df["type"].map(GENUS_BY_TYPE).fillna("")
	df["genus_A"] = np.nan
	df["genus_k"] = np.nan
	for g in GENUS_ORDER:
		mask = df["genus"] == g
		df.loc[mask, "genus_A"] = df.loc[mask, f"{g}_A"]
		df.loc[mask, "genus_k"] = df.loc[mask, f"{g}_k"]
	df["area_ha"] = df["area_m2"] / 10000.0 # Convert area to hectares.
	df.drop(columns=["area_m2"], inplace=True)
	
	# Root-shoot ratio based on IPCC 2019 V4 Ch4 Table 4.4.
	df["nr_rootshoot"] = 0.26
	df["pl_rootshoot"] = 0.26
	df["continent"] = get_continent(iso)
	is_brde_nede = df["genus"].isin(["brde", "nede"])

	for (eco, cont), value in NR_ROOTSHOOT_RULES.items(): df.loc[(df["fao_ecoz"] == eco) & (df["continent"] == cont), "nr_rootshoot"] = value
	for (eco, cont), value in PL_ROOTSHOOT_RULES.items(): df.loc[(df["fao_ecoz"] == eco) & (df["continent"] == cont), "pl_rootshoot"] = value

	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "AMER") & is_brde_nede, "nr_rootshoot"] = 0.466
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "AMER") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "nr_rootshoot"] = 0.337
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "ASIA") & is_brde_nede, "nr_rootshoot"] = 0.225
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "ASIA") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "nr_rootshoot"] = 0.243
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "AMER") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "nr_rootshoot"] = 0.237
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "ASIA") & is_brde_nede, "nr_rootshoot"] = 0.225
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "ASIA") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "nr_rootshoot"] = 0.243

	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "AMER") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "pl_rootshoot"] = 0.203
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "ASIA") & is_brde_nede, "pl_rootshoot"] = 0.307
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "ASIA") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "pl_rootshoot"] = 0.224
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "AMER") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "pl_rootshoot"] = 0.237
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "ASIA") & is_brde_nede, "pl_rootshoot"] = 0.307
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "ASIA") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "pl_rootshoot"] = 0.224

	# Fraction of aboveground carbon at time of harvest stored in long-lived wood products, w (default=0)
	df["w"] = mapped_value(WOOD_PRODUCT_STORAGE_MAP, iso)
	
	# Revenue from wood products (net of harvest and delivery costs) as a scalar of aboveground carbon at time of harvest
	df["p"] = mapped_value(WOOD_PRODUCT_REVENUE_MAP, iso)
	df["exotic"] = mapped_value(EXOTIC_SHARE_MAP, iso)
	df["native"] = 1 - df["exotic"]
	df["nr_npv_estcost_USD_per_ha"] = df["nr_cost"]

	# Pre-loop: compute nr aggregate fields (accum telescoped, soilC, bgC, totalC) needed by all genus breakeven blocks.
	# (Chapman-Richards annual increments telescope: sum_y=1^Y A*[(1-e^{-ky})^2-(1-e^{-k(y-1)})^2] = A*(1-e^{-kY})^2.)
	Years_time_horizon = TIME_HORIZON_YEARS
	df["nr_wohrv_total_tC_per_ha"] = (AGB_C_POOL + BGB_C_POOL*df["nr_rootshoot"])*df["nr_A"]*(1 - np.exp(-df["nr_k"]*Years_time_horizon))**2 + SOIL_C_POOL*Years_time_horizon*(NATURAL_SOIL_C + PLANTATION_SOIL_C)
	# Track annual carbon removed for each pixel in tC_per_ha_per_year.
	nr_year_cols = [f"nr_wohrv_year{year}_tC_per_ha_per_year" for year in range(1, Years_time_horizon + 1)]
	pl_year_cols = [f"pl_whrv_year{year}_tC_per_ha_per_year" for year in range(1, Years_time_horizon + 1)]
	mi_year_cols = [f"mi_whrv_year{year}_tC_per_ha_per_year" for year in range(1, Years_time_horizon + 1)]
	new_columns: dict[str, np.ndarray] = {col: np.zeros(len(df), dtype=np.float32) for col in nr_year_cols + pl_year_cols + mi_year_cols}
	new_columns["nr_harvestyear"] = np.full(len(df), float(Years_time_horizon + 1))
	for col in [f"pl_{f}" for f in ["wohrv_total_tC_per_ha", "whrv0_total_tC_per_ha", "whrv_total_tC_per_ha", "npv_estcost_USD_per_ha", "harvested_tC_per_ha"]]:
		new_columns[col] = np.full(len(df), np.nan)
	new_columns["selected_plantation_rotation_year"] = np.full(len(df), np.nan)
	new_columns["nr_wohrv_costeff_USD_per_tCO2"] = np.full(len(df), np.nan)
	new_columns["pl_wohrv_costeff_USD_per_tCO2"] = np.full(len(df), np.nan)
	new_columns["pl_whrv0_costeff_USD_per_tCO2"] = np.full(len(df), np.nan)
	new_columns["pl_whrv_costeff_USD_per_tCO2"] = np.full(len(df), np.nan)
	new_columns["refortype_whrv_tCO2_per_USD"] = np.full(len(df), "", dtype=object)
	new_columns["mi_whrv_total_tC_per_ha"] = np.full(len(df), np.nan)
	new_columns["mi_whrv_costeff_USD_per_tCO2"] = np.full(len(df), np.nan)
	df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
	nr_cum_prev = pd.Series(0.0, index=df.index)
	
	# Convert cumulative natural-regeneration carbon at each horizon year into annual increments by differencing against the previous year.
	for year in range(1, Years_time_horizon + 1):
		nr_cum_year = (AGB_C_POOL + BGB_C_POOL*df["nr_rootshoot"])*df["nr_A"]*(1 - np.exp(-df["nr_k"]*year))**2 + SOIL_C_POOL*year*(NATURAL_SOIL_C + PLANTATION_SOIL_C)
		df[f"nr_wohrv_year{year}_tC_per_ha_per_year"] = (nr_cum_year - nr_cum_prev).fillna(0).astype(np.float32)
		nr_cum_prev = nr_cum_year

	# Calculate natural regeneration "nr" harvest year (Faustmann-style rule; used only for nr_harvestyear column, later dropped).
	# JFR: Plan is to use harvestyear as the length of a contract in SMDAMAGE.
	discount_rate_for_harvest = DISCOUNT_RATE # Should be used only for choosing the harvest year.
	df["nr_harvestyear"] = calculate_harvest_year(df["nr_k"], Years_time_horizon, discount_rate_for_harvest)

	# Discounted present value of establishment costs.
	if "crop_va" not in df.columns: df["crop_va"] = 0.0

	for g in GENUS_ORDER: # Calculate harvest year (Faustmann-style rule; year loop 1, merged).
		mask = df["genus"] == g
		if not mask.any(): continue
		g_A = df.loc[mask, f"{g}_A"]
		g_k = df.loc[mask, f"{g}_k"]
		g_harvestyear = calculate_harvest_year(g_k, Years_time_horizon, discount_rate_for_harvest)

		# Establishment cost initial value (genus-specific), restricted to the selected genus rows.
		if g == "pinu": g_npv_estcost = df.loc[mask, "exotic"]*df.loc[mask, "ep_cost"] + df.loc[mask, "native"]*df.loc[mask, "np_cost"]
		elif g == "cunn": g_npv_estcost = df.loc[mask, "np_cost"] if iso == "CHN" else df.loc[mask, "ep_cost"]
		elif g == "euca": g_npv_estcost = df.loc[mask, "ep_cost"]
		else: g_npv_estcost = df.loc[mask, "exotic"]*df.loc[mask, "ep_cost"] + df.loc[mask, "native"]*df.loc[mask, "np_cost"]

		# Year loop 2: stock, npv_estcost accumulation, and harvested biomass.
		g_stock = pd.Series(0.0, index=df.index[mask])
		g_harvested = pd.Series(0.0, index=df.index[mask])
		g_cum_prev = pd.Series(0.0, index=df.index[mask])
		g_harvestyear_nonzero = g_harvestyear.replace(0, np.nan)
		for year in range(1, Years_time_horizon + 1):
			g_stock = g_stock + (g_A*((1 - np.exp(-g_k*np.mod(year, g_harvestyear_nonzero))) ** 2 - (1 - np.exp(-g_k*np.mod(year - 1, g_harvestyear_nonzero))) ** 2)).fillna(0)
			g_npv_estcost += g_npv_estcost.where(np.mod(year, g_harvestyear) == 0, 0)
			g_harvested = g_harvested + (g_A*(1 - np.exp(-g_k*year)) ** 2).where(np.mod(year, g_harvestyear) == 0, 0)
			g_soil_cum_year = PLANTATION_SOIL_C*(g_harvestyear - 1).clip(lower=0, upper=year).fillna(0)
			g_cum_year = g_stock + g_stock*df.loc[mask, "pl_rootshoot"] + g_soil_cum_year + df.loc[mask, "w"]*g_harvested
			df.loc[mask, f"pl_whrv_year{year}_tC_per_ha_per_year"] = (g_cum_year - g_cum_prev).fillna(0).astype(np.float32)
			g_cum_prev = g_cum_year

		# Derived carbon fields. Carbon per hectare.
		g_soil_total = PLANTATION_SOIL_C*(g_harvestyear - 1).clip(lower=0, upper=Years_time_horizon).fillna(0)
		g_wohrv_total = AGB_C_POOL*g_A*(1 - np.exp(-g_k*Years_time_horizon)) ** 2 + BGB_C_POOL*(g_A*(1 - np.exp(-g_k*Years_time_horizon)) ** 2)*df.loc[mask, "pl_rootshoot"] + SOIL_C_POOL*Years_time_horizon*PLANTATION_SOIL_C
		g_whrv_total = AGB_C_POOL*g_stock + BGB_C_POOL*g_stock*df.loc[mask, "pl_rootshoot"] + SOIL_C_POOL*g_soil_total + HARVEST_C_POOL*df.loc[mask, "w"]*g_harvested
		df.loc[mask, "pl_wohrv_total_tC_per_ha"] = g_wohrv_total
		df.loc[mask, "pl_whrv0_total_tC_per_ha"] = g_whrv_total
		df.loc[mask, "pl_whrv_total_tC_per_ha"] = g_whrv_total
		df.loc[mask, "pl_npv_estcost_USD_per_ha"] = g_npv_estcost
		df.loc[mask, "pl_harvested_tC_per_ha"] = g_harvested
		df.loc[mask, "selected_plantation_rotation_year"] = g_harvestyear
	
	# Compute cost-effectiveness in $/tCO2 from per-hectare costs and per-hectare carbon.
	df["nr_wohrv_costeff_USD_per_tCO2"] = (df["crop_va"]*Years_time_horizon + df["nr_cost"]) / (3.67*df["nr_wohrv_total_tC_per_ha"]).replace(0, np.nan)
	df["pl_wohrv_costeff_USD_per_tCO2"] = (df["crop_va"]*Years_time_horizon + df["pl_npv_estcost_USD_per_ha"]) / (3.67*df["pl_wohrv_total_tC_per_ha"]).replace(0, np.nan)
	df["pl_whrv0_costeff_USD_per_tCO2"] =(df["crop_va"]*Years_time_horizon + df["pl_npv_estcost_USD_per_ha"]) / (3.67*df["pl_whrv0_total_tC_per_ha"]).replace(0, np.nan)
	df["pl_whrv_costeff_USD_per_tCO2"] = (df["crop_va"]*Years_time_horizon + df["pl_npv_estcost_USD_per_ha"] - df["p"]*df["pl_harvested_tC_per_ha"]) / (3.67*df["pl_whrv_total_tC_per_ha"]).replace(0, np.nan)
	
	# Compare plantation vs natural regeneration for whrv scenario.
	lhs_nr_tCO2_per_USD = 3.67*df["nr_wohrv_total_tC_per_ha"] / (df["crop_va"]*Years_time_horizon + df["nr_cost"]).replace(0, np.nan)
	lhs_pl_wh_tCO2_per_USD = 3.67*df["pl_whrv_total_tC_per_ha"] / (df["crop_va"]*Years_time_horizon + df["pl_npv_estcost_USD_per_ha"] - df["p"]*df["pl_harvested_tC_per_ha"]).replace(0, np.nan)
	
	# Determine minimum cost-effectiveness (plantation vs natural regeneration).
	df["refortype_whrv_tCO2_per_USD"] = np.where(lhs_pl_wh_tCO2_per_USD > lhs_nr_tCO2_per_USD, "P", np.where(lhs_pl_wh_tCO2_per_USD < lhs_nr_tCO2_per_USD, "N", "."))
	df.loc[lhs_pl_wh_tCO2_per_USD < 0, "refortype_whrv_tCO2_per_USD"] = "P"
	mi_counts = {"refortype_dot_total": int((df["refortype_whrv_tCO2_per_USD"] == ".").sum()),}
	for col in ["nr_wohrv_total_tC_per_ha", "pl_whrv_total_tC_per_ha", "crop_va", "nr_cost", "pl_npv_estcost_USD_per_ha", "p", "pl_harvested_tC_per_ha"]:
		mi_counts[f"{col}_dot_null_or_nan"] = int(df.loc[df["refortype_whrv_tCO2_per_USD"] == ".", col].isna().sum())

	# Generate minimum output columns (keep only whrv scenario for MACC).
	df.loc[:, "mi_whrv_total_tC_per_ha"] = np.where(df["refortype_whrv_tCO2_per_USD"] == "P", df["pl_whrv_total_tC_per_ha"], np.where(df["refortype_whrv_tCO2_per_USD"] == "N", df["nr_wohrv_total_tC_per_ha"], 0))
	df.loc[:, "mi_whrv_costeff_USD_per_tCO2"] = np.where(df["refortype_whrv_tCO2_per_USD"] == "P", df["pl_whrv_costeff_USD_per_tCO2"], np.where(df["refortype_whrv_tCO2_per_USD"] == "N", df["nr_wohrv_costeff_USD_per_tCO2"], np.nan))
	
	for year in range(1, Years_time_horizon + 1):
		pl_col = f"pl_whrv_year{year}_tC_per_ha_per_year"
		nr_col = f"nr_wohrv_year{year}_tC_per_ha_per_year"
		mi_col = f"mi_whrv_year{year}_tC_per_ha_per_year"
		df.loc[:, mi_col] = np.where(df["refortype_whrv_tCO2_per_USD"] == "P", df[pl_col], np.where(df["refortype_whrv_tCO2_per_USD"] == "N", df[nr_col], 0))

	# Print all columns as one item/value per row.
	# item_value_df = df.rename_axis("row").reset_index().melt(id_vars="row", var_name="item", value_name="value")
	# print(item_value_df.to_string(index=False))

	# Diagnose cost-effectiveness values before binning.
	# print(df[["mi_whrv_costeff_USD_per_tCO2", "mi_whrv_total_tC_per_ha", "refortype_whrv_tCO2_per_USD", "area_ha"]].to_string())

	# MARGINAL ABATEMENT COST CURVE bins in $/tCO2: sum area in each 5-unit bin.
	# price_bins = list(range(0, 201, 5))
	# macc_curve = {p: 0.0 for p in price_bins}
	# Price_per_tCO2 = df["mi_whrv_costeff_USD_per_tCO2"]
	# area_ha = df["area_ha"].fillna(0)
	# Negative cost-effectiveness means wood revenue exceeds costs; treat as $0/tCO2.
	# macc_curve[0] += area_ha.where(Price_per_tCO2 <= 0, 0).sum(skipna=True)
	# # Example: area with 15 < price <= 20 is added to macc_curve[20].
	# for p in price_bins[1:]:
	# 	mask = (Price_per_tCO2 > p - 5) & (Price_per_tCO2 <= p)
	# 	macc_curve[p] += area_ha.where(mask, 0).sum(skipna=True)

	# macc = pd.DataFrame({"price_per_tCO2": list(macc_curve.keys()), "area_ha": list(macc_curve.values())})
	# # Save MAC curves and map-ready outputs.
	# macc_dir.mkdir(parents=True, exist_ok=True)
	# macc.to_csv(macc_dir / f"MACC_{iso}.csv", index=False)

	# Map model outputs into a wide SMDAMAGE-style export.
	# - plantation_genus: fixed plantation genus available on the pixel.
	# - selected_option: best model choice between plantation and natural regeneration.
	# - selected_A / selected_k / selected_rotation_year: parameters for the chosen option.
	#   Natural regeneration uses Years_time_horizon as the contract/rotation length in this export.
	# - selected_establishment_cost_USD_per_ha: model-used establishment cost for the chosen option.
	# - p_USD_per_tC_harvested: wood-product revenue scalar, zeroed when the selected option has no harvest revenue.
	selected_option = df["refortype_whrv_tCO2_per_USD"].map({"P": "plantation", "N": "natural_regeneration", ".": "undetermined"})
	selected_rotation_year = np.where(df["refortype_whrv_tCO2_per_USD"] == "P", df["selected_plantation_rotation_year"], np.where(df["refortype_whrv_tCO2_per_USD"] == "N", float(Years_time_horizon), np.nan))
	macc_dir.mkdir(parents=True, exist_ok=True)

	smdamage_export = pd.DataFrame({
		"country": iso,
		"pixel_id": df["id"],
		"plantation_genus": df["genus"],
		"selected_option": selected_option,
		"selected_A": np.where(df["refortype_whrv_tCO2_per_USD"] == "P", df["genus_A"], np.where(df["refortype_whrv_tCO2_per_USD"] == "N", df["nr_A"], np.nan)),
		"selected_k": np.where(df["refortype_whrv_tCO2_per_USD"] == "P", df["genus_k"], np.where(df["refortype_whrv_tCO2_per_USD"] == "N", df["nr_k"], np.nan)),
		"selected_rotation_year": selected_rotation_year,
		"area_ha": df["area_ha"],
		"crop_va_USD_per_ha_per_year": df["crop_va"],
		"selected_establishment_cost_USD_per_ha": np.where(df["refortype_whrv_tCO2_per_USD"] == "P", df["pl_npv_estcost_USD_per_ha"], np.where(df["refortype_whrv_tCO2_per_USD"] == "N", df["nr_cost"], np.nan)),
		"p_USD_per_tC_harvested": np.where(df["refortype_whrv_tCO2_per_USD"] == "P", df["p"], 0.0),
	})
	if SKIP_SQLITE_EXPORT: print(f"[SKIP_SQLITE_EXPORT] {iso}", flush=True)
	else:
		db_path = macc_dir / SMDAMAGE_DB_FILE
		with sqlite3.connect(db_path) as conn:
			clear_smdamage_country(conn, iso)
			smdamage_export.to_sql(SMDAMAGE_PIXEL_TABLE, conn, if_exists="append", index=False)
			for year in range(1, Years_time_horizon + 1):
				year_mask = selected_rotation_year >= year
				if not year_mask.any(): continue
				annual_export = pd.DataFrame({
					"country": iso,
					"pixel_id": df.loc[year_mask, "id"].to_numpy(),
					"year": year,
					"tC_per_ha_per_year": df.loc[year_mask, f"mi_whrv_year{year}_tC_per_ha_per_year"].to_numpy(),})
				annual_export.to_sql(SMDAMAGE_YEAR_TABLE, conn, if_exists="append", index=False)

	# Show carbon buckets.
	# total_C_categories = {}
	# for total_carbon in df[mi_year_cols].sum(axis=1):
	# 	bucket = round(total_carbon, 0)
	# 	total_C_categories[bucket] = total_C_categories.get(bucket, 0) + 1
	# print(dict(sorted(total_C_categories.items())))

	mode_label = "all pixels" if RUN_ALL_PIXELS else f"pixel {TARGET_PIXEL_ID}"
	print(f"[OK] {iso} {mode_label}", flush=True)
	# print(f"[MI_COUNTS] {iso} {mi_counts}", flush=True)
	return mi_counts

# Iterate across requested ISO codes and run the country pipeline.
def main() -> None:
	parser = argparse.ArgumentParser(description="Python conversion of 1. Model loop all data.do")
	parser.add_argument("--dta-dir", type=Path, required=True, help="Directory containing per-country .dta input files",)
	parser.add_argument("--macc-dir", type=Path, required=True, help="Directory to write MACC_*.csv files")
	parser.add_argument("--iso", nargs="*", help="Optional ISO code filter. If omitted, run all countries in the input directory.")
	parser.add_argument("--smallest-n", type=int, help="Optional count of smallest .dta files to process by file size.")
	args = parser.parse_args()
	requested_isos = sorted(iso.upper() for iso in args.iso) if args.iso else iter_smallest_input_isos(args.dta_dir, args.smallest_n) if args.smallest_n else iter_input_isos(args.dta_dir)
	initialize_smdamage_database(args.macc_dir)
	all_blank_counts: dict[str, dict[str, int]] = {}
	for iso in requested_isos: all_blank_counts[iso] = process_country(iso=iso, dta_dir=args.dta_dir, macc_dir=args.macc_dir)

if __name__ == "__main__":
	# Set this to True to run all countries from in-script settings (no CLI args).
	RUN_WITH_INLINE_SETTINGS = True
	if RUN_WITH_INLINE_SETTINGS:
		base_dir = Path(__file__).resolve().parent.parent.parent
		initialize_smdamage_database(base_dir / "Output")
		requested_isos = iter_input_isos(base_dir / "Input")  # all 138 countries; use iter_smallest_input_isos(base_dir / "Input", SMALL_DTA_COUNT) for a partial run
		all_blank_counts: dict[str, dict[str, int]] = {}
		for iso in requested_isos:
			all_blank_counts[iso] = process_country(iso=iso, dta_dir=base_dir / "Input", macc_dir=base_dir / "Output")
		# print(f"[MI_COUNTS_SUMMARY] {all_blank_counts}", flush=True)
	else:
		main()