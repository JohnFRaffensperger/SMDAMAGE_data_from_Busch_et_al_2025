# 1_model_loop_all_data.py | Created 2026-03-26
# Recreates the original Stata model loop in Python, computing restoration economics, harvest timing, annual carbon flows, and country outputs files.
"""
Python translation of: 1. Model loop all data.do. This script mirrors the original Stata workflow as closely as practical:
- loops through ISO-level input .dta files,
- computes carbon, cost-effectiveness, and break-even metrics,
- builds MAC curve tables,
- writes per-country MACC and maps outputs as .dta files.
Dependencies: pip install pandas numpy pyreadstat
Example: python 1_model_loop_all_data.py \
		--dta-dir "c:/Users/johnr/Downloads/09_output_dtas/Input" \
		--macc-dir "c:/Users/johnr/Downloads/09_output_dtas/Output" \
		--maps-dir "c:/Users/johnr/Downloads/09_output_dtas/Output"

Stata source note:
Busch, J., Bukoski, J.J., Cook-Patton, S.C., Griscom, B., Kaczan, D., Potts, M.D., Yi, Y., Vincent, J. (2024).
"Tree planting vs. natural forest regeneration: relative cost-effectiveness at mitigating climate change." Nature Climate Change.

Abbreviations used in variable names (from the .do file):
pl=plantation
nr=natural regeneration
np=plantation (native species)
ep=plantation (exotic species)
mi=minimum (cost option between plantation and natural regeneration)
wohrv=without harvest
whrv0=with harvest, excluding wood product revenue
whrv=with harvest, including wood product revenue
w=fraction of carbon at harvest remaining in wood products (discounted flow)
p=revenue from wood products as a scalar of carbon at harvest
s=screen
Gr=Griscom
Br=Brancalion
Ba=Bastin
Wa=Walker
bu=buffer
ad=additional
rel=relative (cost effectiveness)
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd

ISO_LIST = ["AFG", "AGO", "ARG", "ARM", "ATG", "AZE", "BDI", "BEN", "BFA", "BGD", "BLZ", "BOL", "BRA", "BRB", "BTN", "BWA", "CAF",
 "CHL", "CHN", "CIV", "CMR", "COD", "COG", "COK", "COL", "COM", "CPV", "CRI", "CUB", "DJI", "DMA", "DOM", "DZA", "ECU", "EGY",
 "ERI", "ESH", "ETH", "FJI", "FSM", "GAB", "GEO", "GHA", "GIN", "GMB", "GNB", "GNQ", "GRD", "GTM", "GUF", "GUY", "HND", "HTI",
 "IDN", "IND", "IRN", "IRQ", "JAM", "JOR", "KAZ", "KEN", "KGZ", "KHM", "KNA", "LAO", "LBN", "LBR", "LBY", "LCA", "LKA", "LSO",
 "MAR", "MDG", "MDV", "MEX", "MHL", "MLI", "MMR", "MNG", "MOZ", "MRT", "MUS", "MWI", "MYS", "NAM", "NER", "NGA", "NIC", "NPL",
 "NRU", "OMN", "PAK", "PAN", "PER", "PHL", "PLW", "PNG", "PRK", "PRY", "PSE", "RWA", "SDN", "SEN", "SLB", "SLE", "SLV", "SOM",
 "SSD", "STP", "SUR", "SWZ", "SYC", "SYR", "TCD", "TGO", "THA", "TJK", "TKL", "TKM", "TLS", "TON", "TTO", "TUN", "TUR", "TUV",
 "TZA", "UGA", "URY", "UZB", "VCT", "VEN", "VNM", "VUT", "WSM", "YEM", "ZAF", "ZMB", "ZWE",]

GENUS_ORDER = ["pinu", "cunn", "euca", "brde", "brev", "nede", "neev"]

AFRI = {"DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COM", "COG", "CIV", "COD", "DJI", "EGY", "GNQ",
 "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY", "MDG", "MWI", "MDV", "MLI", "MRT", "MUS",
 "MAR", "MOZ", "NAM", "NER", "NGA", "RWA", "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "TUN", "UGA",
 "ESH", "ZMB", "ZWE",}

AMER = {"ATG", "ARG", "BRB", "BLZ", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DMA", "DOM", "ECU", "SLV", "GUF", "GRD", "GTM",
	"GUY", "HTI", "HND", "JAM", "MEX", "NIC", "PAN", "PRY", "PER", "KNA", "LCA", "VCT", "SUR", "TTO", "URY", "VEN",}

ASIA = {"AFG", "BGD", "BTN", "KHM", "CHN", "COK", "PRK", "FJI", "IND", "IDN", "IRN", "IRQ", "JOR", "KAZ", "KIR", "KGZ", "LAO",
 "LBN", "MYS", "MHL", "FSM", "MNG", "MMR", "NRU", "NPL", "OMN", "PAK", "PLW", "PSE", "PNG", "PHL", "WSM", "SLB", "LKA", "SYR",
 "TJK", "THA", "TLS", "TKL", "TON", "TKM", "TUV", "UZB", "VUT", "VNM", "YEM", "ARM", "AZE", "GEO", "TUR",}

# Ensure required columns exist before vectorized model calculations.
def ensure_columns(df: pd.DataFrame, cols: Iterable[str], default: float = 0.0) -> None:
	for col in cols:
		if col not in df.columns: df[col] = default

# Return the sum of annual discount weights used in NPV-style terms.
def discounted_factor(y: int, d: float) -> float: return float(sum((1 - d) ** (i - 1) for i in range(1, y + 1)))

# Compute discounted aboveground carbon accumulation from Chapman-Richards growth.
def compute_accum (df: pd.DataFrame, asymptotic_carbon_col: str, growth_rate_col: str,
	years: int, discount_rate: float,) -> pd.Series:
	"""Return discounted aboveground carbon accumulated over the modeling horizon.
	The original Stata code computes the present value of annual carbon increments
	implied by a Chapman-Richards curve:  A * (1 - exp(-k * t))^2
	where:
		``A`` is the asymptotic aboveground carbon stock for each pixel/genus
		``k`` is the growth-rate parameter
		``t`` is the year within the time horizon

	For each year, this function calculates the incremental increase in standing
	aboveground carbon from year ``t-1`` to year ``t``, discounts that increment
	back to the present, and sums the discounted increments across the full horizon.

	Args:
		df: DataFrame containing the growth parameters. Each row is one pixel.
		asymptotic_carbon_col: Column holding the Chapman-Richards ``A`` term.
		growth_rate_col: Column holding the Chapman-Richards ``k`` term.
		years: Number of years in the model horizon.
		discount_rate: Annual discount rate applied to each year's increment.

	Returns: A Series aligned to ``df.index`` with discounted aboveground carbon accumulation for each row.
	"""
	discounted_accumulation = pd.Series(0.0, index=df.index)
	for year in range(1, years + 1):
		discounted_increment = ((1 - discount_rate)**(year - 1)*df[asymptotic_carbon_col]
			*((1 - np.exp(-df[growth_rate_col]*year))**2 - (1 - np.exp(-df[growth_rate_col]*(year - 1)))**2))
		discounted_accumulation = discounted_accumulation + discounted_increment
	print(discounted_accumulation)
	return discounted_accumulation

# Compute yearly undiscounted aboveground carbon increments and write them to text.
def compute_aboveground_carbon(df: pd.DataFrame, pixel_id_col: str, asymptotic_carbon_col: str,
	growth_rate_col: str, years: int, output_txt_path: Path,) -> None: 
	
	"""Write per-pixel annual undiscounted aboveground-carbon increments to text.
	Each output row corresponds to one pixel. The first column is the pixel
	identifier, followed by one column per year (year_1 ... year_N) containing
	the undiscounted increase in standing aboveground carbon from year t-1 to t
	under the Chapman-Richards curve.
	"""
	if pixel_id_col not in df.columns: raise KeyError(f"Missing pixel identifier column: {pixel_id_col}")
	if years < 1: raise ValueError("years must be >= 1")
	output_txt_path.parent.mkdir(parents=True, exist_ok=True)
	year_columns = [f"year_{year}" for year in range(1, years + 1)]
	with output_txt_path.open("w", encoding="utf-8", newline="") as out_file:
		writer = csv.writer(out_file, delimiter="\t")
		writer.writerow([pixel_id_col, *year_columns])

		for _, row in df.iterrows():
			asymptotic_carbon = row[asymptotic_carbon_col]
			growth_rate = row[growth_rate_col]
			yearly_increments: List[float] = []

			for year in range(1, years + 1):
				standing_t = asymptotic_carbon * (1 - np.exp(-growth_rate * year)) ** 2
				standing_t_minus_1 = asymptotic_carbon * (1 - np.exp(-growth_rate * (year - 1))) ** 2
				yearly_increments.append(float(standing_t - standing_t_minus_1))
			writer.writerow([row[pixel_id_col], *yearly_increments])

# Estimate harvest timing from marginal growth heuristics (Faustmann-style modes).
def compute_harvest_year(df: pd.DataFrame, A_col: str, k_col: str, years: int, discount_rate: float, harvest_rule: int) -> pd.Series:
	harvest_year = pd.Series(float(years + 1), index=df.index)
	growth_rate_by_year = {}
	for year in range(2, years + 1):
		incremental_stock = (1 - np.exp(-df[k_col]*year)) ** 2 - (1 - np.exp(-df[k_col]*(year - 1))) ** 2
		prior_stock = (1 - np.exp(-df[k_col]*(year - 1))) ** 2
		growth_rate_by_year[year] = incremental_stock / prior_stock.replace(0, np.nan)

	for year in range(3, years + 1):
		prior_growth_rate = growth_rate_by_year[year - 1]
		if harvest_rule == 0: cond = prior_growth_rate > discount_rate
		elif harvest_rule == 1:
			decision_threshold = discount_rate / (1 - (1 + discount_rate) ** (-(year - 1)))
			cond = prior_growth_rate > decision_threshold
		else: cond = prior_growth_rate > (1 / year)
		harvest_year = harvest_year.where(~cond, float(year))
	return harvest_year

# Compute discounted harvested carbon pulses at harvest years.
# Build one MAC curve as cumulative abatement below each carbon price step.
def mac_curve(df: pd.DataFrame, total_col: str, cost_col: str, prices: np.ndarray) -> pd.Series:
	cumulative_abatement = []
	for price_threshold in prices: cumulative_abatement.append(df.loc[df[cost_col] <= price_threshold, total_col].sum(skipna=True))
	return pd.Series(cumulative_abatement)

# Choose plantation or natural-regeneration value by cost-effective method flag.
def choose_by_refortype(df: pd.DataFrame, out_col: str, ref_col: str, p_col: str, n_col: str, default: float, ) -> None: df[out_col] = np.where(df[ref_col] == "P", df[p_col], np.where(df[ref_col] == "N", df[n_col], default))

# Execute the full country-level pipeline from input .dta to MACC/maps outputs.
def process_country(iso: str, dta_dir: Path, macc_dir: Path, maps_dir: Path) -> None:
	# Find the input paths.
	src = next((c for c in [dta_dir / iso, dta_dir / f"{iso}.dta", dta_dir / f"maps_{iso}.dta"] if c.exists() and c.is_file()), None)
	if src is None:
		print(f"[WARN] {iso}: input not found, skipping")
		return

	df = pd.read_stata(src)
	if df.empty:
		print(f"[WARN] {iso}: empty dataset, skipping")
		return

	df = df.copy()
	# Add sequential ID to sort back to original order, if needed.
	df["id"] = np.arange(1, len(df) + 1)

	# Data cleaning.
	for col in ["griscom", "brancalion", "bastin", "walker", "nr_buffer", "pl_buffer"]:
		if col in df.columns: df[col] = df[col].fillna(0)
		else: df[col] = 0

	ensure_columns(df, ["biomes", "nr_A", "type"])  # for drop logic
	# Drop deserts/xeric (biomes=13) and mangroves (biomes=14).
	df = df[df["biomes"] != 13]
	df = df[df["biomes"] != 14]
	df = df[df["nr_A"].notna()]
	df = df[df["type"].notna()]

	if df.empty:
		print(f"[WARN] {iso}: no rows after cleaning, skipping")
		return

	# Implementation costs were given in 2011 USD; inflate to 2020 USD by multiplying by 1.1577
	for c in ["nr_cost", "ep_cost", "np_cost"]:
		if c not in df.columns: df[c] = 0.0
		df[c] = df[c]*1.1577 # Convert 2011 costs to 2020.

	# Plantation genus
	GENUS_BY_TYPE = {1: "neev", 2: "brev", 3: "brde", 4: "neev", 5: "cunn", 6: "euca", 7: "nede", 8: "neev", 9: "pinu", 10: "brde", 11: "neev", 12: "brde", 13: "brev", 14: "brde", 15: "neev",}
	df["genus"] = df["type"].map(GENUS_BY_TYPE).fillna("")
	if "area_m2" in df.columns:
		df["area"] = df["area_m2"] / 10000.0 # Rescale area from m^2 to ha
		df.drop(columns=["area_m2"], inplace=True)
	else: df["area"] = 0.0

	continent = "AFRI" if iso in AFRI else "AMER" if iso in AMER else "ASIA" if iso in ASIA else ""
	df["continent"] = continent

	# Root-shoot ratio based on IPCC 2019 V4 Ch4 Table 4.4.
	df["nr_rootshoot"] = 0.26
	df["pl_rootshoot"] = 0.26
	ensure_columns(df, ["fao_ecoz"], np.nan)

	nr_rootshoot_rules = {(11, "AFRI"): 0.825, (11, "AMER"): 0.221, (11, "ASIA"): 0.207, (12, "AFRI"): 0.232, (12, "AMER"): 0.2845, (12, "ASIA"): 0.323,
	(13, "AFRI"): 0.332, (13, "AMER"): 0.334, (13, "ASIA"): 0.440, (16, "AMER"): 0.348, (16, "ASIA"): 0.322, (21, "AFRI"): 0.232, (21, "AMER"): 0.175,
	(21, "ASIA"): 0.230, (22, "AMER"): 0.336, (22, "ASIA"): 0.440, (23, "AMER"): 1.338, (23, "ASIA"): 1.338,}

	for (eco, cont), value in nr_rootshoot_rules.items(): df.loc[(df["fao_ecoz"] == eco) & (df["continent"] == cont), "nr_rootshoot"] = value

	# replace nr_rootshoot=??? if fao_ecoz==16 & continent=="AFRI"
	# replace nr_rootshoot=0.??? if fao_ecoz==22 & continent=="AFRI"
	# replace nr_rootshoot=0.??? if fao_ecoz==23 & continent=="AFRI"

	is_brde_nede = df["genus"].isin(["brde", "nede"])

	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "AMER") & is_brde_nede, "nr_rootshoot"] = (0.466)
	# Compute discounted harvested carbon pulses at harvest years.
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "AMER") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "nr_rootshoot"] = 0.337
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "ASIA") & is_brde_nede, "nr_rootshoot"] = (0.225)
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "ASIA") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "nr_rootshoot"] = 0.243
	# replace nr_rootshoot=0.??? if fao_ecoz==32 & =="AMER" & genus=="brde" | genus=="nede"
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "AMER") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "nr_rootshoot"] = 0.237
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "ASIA") & is_brde_nede, "nr_rootshoot"] = (0.225)
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "ASIA") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "nr_rootshoot"] = 0.243
	# replace nr_rootshoot=0.??? if fao_ecoz==24, 25, 33, 34, 35, 41, 42, 43, 50, 90, 99

	# replace pl_rootshoot=??? if fao_ecoz==11 & continent=="AFRI"
	# replace pl_rootshoot=??? if fao_ecoz==12 & continent=="AFRI"
	# replace pl_rootshoot=??? if fao_ecoz==12 & continent=="AMER"
	# replace pl_rootshoot=??? if fao_ecoz==12 & continent=="ASIA"
	# replace pl_rootshoot=??? if fao_ecoz==13 & continent=="AFRI"
	# replace pl_rootshoot=??? if fao_ecoz==13 & continent=="AMER"
	# replace pl_rootshoot=??? if fao_ecoz==13 & continent=="ASIA"
	# replace pl_rootshoot=??? if fao_ecoz==16 & continent=="AFRI"
	# replace pl_rootshoot=??? if fao_ecoz==16 & continent=="ASIA"
	# replace pl_rootshoot=??? if fao_ecoz==21 & continent=="AFRI"
	# replace pl_rootshoot=??? if fao_ecoz==21 & continent=="AMER"
	# replace pl_rootshoot=??? if fao_ecoz==21 & continent=="ASIA"
	# replace pl_rootshoot=??? if fao_ecoz==22 & continent=="AFRI"
	# replace pl_rootshoot=??? if fao_ecoz==22 & continent=="AMER"
	# replace pl_rootshoot=??? if fao_ecoz==22 & continent=="ASIA"
	# replace pl_rootshoot=??? if fao_ecoz==23 & continent=="AFRI"
	# replace pl_rootshoot=??? if fao_ecoz==23 & continent=="AMER"
	pl_rootshoot_rules = {(11, "AMER"): 0.170, (11, "ASIA"): 0.325, (16, "AMER"): 2.158, (23, "ASIA"): 2.158,}
	for (eco, cont), value in pl_rootshoot_rules.items(): df.loc[(df["fao_ecoz"] == eco) & (df["continent"] == cont), "pl_rootshoot"] = value
	# replace pl_rootshoot=0.??? if fao_ecoz==31 & continent=="AMER" & genus=="brde" | genus=="nede"
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "AMER") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "pl_rootshoot"] = 0.203
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "ASIA") & is_brde_nede, "pl_rootshoot"] = (0.307)
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "ASIA") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "pl_rootshoot"] = 0.224
	# replace pl_rootshoot=0.??? if fao_ecoz==32 & continent=="AMER" & genus=="brde" | genus=="nede"
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "AMER") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "pl_rootshoot"] = 0.237
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "ASIA") & is_brde_nede, "pl_rootshoot"] = (0.307)
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "ASIA") & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "pl_rootshoot"] = 0.224

	base_cols = ["nr_A", "nr_k"]
	for g in GENUS_ORDER: base_cols.extend([f"{g}_A", f"{g}_k"])
	ensure_columns(df, base_cols, 0.0)
	# Truncate negative values of A and k at zero
	for c in base_cols: df[c] = df[c].clip(lower=0)

	# Set parameters (scalars).
	# time horizon (default=30)
	Y = 30
	# discount rate (default=0.05)
	d = 0.05
	# set F(austmann)=0 for rotation length decision based on single harvest cycle;
	# set F(austmann)=1 for rotation length decision based on infinite harvest cycle; (default)
	# set F(austmann)=2 for rotation length decision to maximize mean annual interval
	F = 1

	# Export per-pixel undiscounted annual aboveground increments (natural regeneration).
	# compute_aboveground_carbon(df=df, pixel_id_col="id", asymptotic_carbon_col="nr_A",
	#     growth_rate_col="nr_k", years=50, output_txt_path=maps_dir / f"aboveground_carbon_{iso}.txt",)

	# annual carbon accumulation in soil
	nrsoil = 0.415107735
	plantsoil = 0.092749069
	# carbon pools to include in total carbon (totalC). set=1 to include, =0 to exclude. (default=1 for agbC, bgbC, soilC; =0 for harvestC)
	agbC = 1
	bgbC = 1
	soilC = 1
	harvestC = 1

	# Fraction of aboveground carbon at time of harvest stored in long-lived wood products, w (default=0)
	w_map = {0.188: {"ATG", "BLZ", "BRB", "CRI", "CUB", "DMA", "DOM", "GRD", "GTM", "HND", "HTI", "JAM", "KNA", "LCA", "MEX", "NIC", "PAN", "SLV", "TTO", "VCT", },
	0.268: {"ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUF", "GUY", "PER", "PRY", "SUR", "URY", "VEN", },
	0.355: {"AFG", "ARM", "AZE", "BGD", "BTN", "CHN", "GEO", "IDN", "IND", "IRN", "IRQ", "JOR", "KAZ", "KGZ", "KHM", "LAO", "LBN", "LKA", "MMR", "MNG", "MYS", "NPL", "OMN", "PAK", "PHL", "PNG", "PRK", "PSE", "SYR", "THA", "TJK", "TKM", "TLS", "TUR", "UZB", "VNM", "YEM", },
	0.392: {"AGO", "BDI", "BEN", "BFA", "BWA", "CAF", "CIV", "CMR", "COD", "COG", "COM", "CPV", "DJI", "DZA", "EGY", "ERI", "ESH", "ETH", "GAB", "GHA", "GIN", "GMB", "GNB", "GNQ", "KEN", "LBR", "LBY", "LSO", "MAR", "MDG", "MDV", "MLI", "MOZ", "MRT", "MUS", "MWI", "NAM", "NER", "NGA", "RWA", "SDN", "SEN", "SLE", "SOM", "SSD", "STP", "SWZ", "SYC", "TCD", "TGO", "TUN", "TZA", "UGA", "ZAF", "ZMB", "ZWE", },
	0.415: {"COK", "FJI", "FSM", "KIR", "MHL", "NRU", "PLW", "SLB", "TKL", "TON", "TUV", "VUT", "WSM", },}
	# Revenue from wood products (net of harvest and delivery costs) as a scalar of aboveground carbon at time of harvest
	p_map = {52.65: {"ATG", "BLZ", "BRB", "CRI", "CUB", "DMA", "DOM", "GRD", "GTM", "HND", "HTI", "JAM", "KNA", "LCA", "NIC", "PAN", "SLV", "TTO", "VCT", },
	59.73: {"MEX"},
	61.04: {"BOL", "BRA", "COL", "ECU", "GUF", "GUY", "PER", "SUR", "VEN"},
	67.76: {"BGD", "IDN", "KHM", "LAO", "LKA", "MMR", "MYS", "OMN", "PHL", "PNG", "THA", "TLS", "VNM", "YEM", },
	69.26: {"PRY"},
	71.24: {"AGO", "BDI", "BEN", "BFA", "CAF", "CIV", "CMR", "COD", "COG", "COM", "CPV", "DJI", "ERI", "ETH", "GAB", "GHA", "GIN", "GMB", "GNB", "GNQ", "KEN", "LBR", "MDG", "MDV", "MLI", "MOZ", "MRT", "MUS", "MWI", "NER", "NGA", "RWA", "SDN", "SEN", "SLE", "SOM", "SSD", "STP", "SYC", "TCD", "TGO", "TZA", "UGA", "ZMB", "ZWE", },
	76.08: {"COK", "FJI", "FSM", "KIR", "MHL", "NRU", "PLW", "SLB", "TKL", "TON", "TUV", "VUT", "WSM", },
	76.88: {"IND"},
	80.003: {"ARG", "CHL", "URY"},
	80.83: {"BWA", "NAM"},
	88.84: {"AFG", "ARM", "AZE", "BTN", "CHN", "GEO", "IRN", "IRQ", "JOR", "KAZ", "KGZ", "LBN", "MNG", "NPL", "PAK", "PRK", "PSE", "SYR", "TJK", "TKM", "TUR", "UZB", },
	93.40: {"DZA", "EGY", "ESH", "LBY", "LSO", "MAR", "SWZ", "TUN", "ZAF"}, }

	df["w"] = next((v for v, s in w_map.items() if iso in s), 0.0)
	df["p"] = next((v for v, s in p_map.items() if iso in s), 0.0)

	# generate local to use in many functions involving {pinu cunn euca brde brev nede neev}
	df["nr_Y30_d0"] = df["nr_A"]*(1 - np.exp(-df["nr_k"]*Y)) ** 2

	# Chapman-Richards carbon accumulation in year 30 (no harvest)
	df["nr_accum"] = compute_accum(df, "nr_A", "nr_k", Y, d)

	# Discounted present value of aboveground carbon accumulation at discount rate=d (no harvest)
	for g in GENUS_ORDER:
		df[f"{g}_Y30_d0"] = df[f"{g}_A"]*(1 - np.exp(-df[f"{g}_k"]*Y)) ** 2
		df[f"{g}_accum"] = compute_accum(df, f"{g}_A", f"{g}_k", Y, d)

	# sum nr* pinu* cunn* euca* brde* brev* nede* neev*

	# Calculate harvest year.
	df["nr_harvestyear"] = compute_harvest_year(df, "nr_A", "nr_k", Y, d, F)
	for g in GENUS_ORDER:
		df[f"{g}_harvestyear"] = compute_harvest_year(df, f"{g}_A", f"{g}_k", Y, d, F)

	# Compute discounted standing-stock carbon under repeated harvest cycles.
	for g in GENUS_ORDER:
		_stock = pd.Series(0.0, index=df.index)
		_harvest_interval = df[f"{g}_harvestyear"].replace(0, np.nan)
		for year in range(1, Y + 1):
			_ysh = np.mod(year, _harvest_interval)
			_pysh = np.mod(year - 1, _harvest_interval)
			_disc_inc = ((1 - d) ** (year - 1)*df[f"{g}_A"]*((1 - np.exp(-df[f"{g}_k"]*_ysh)) ** 2 - (1 - np.exp(-df[f"{g}_k"]*_pysh)) ** 2))
			_stock = _stock + _disc_inc.fillna(0)
		df[f"{g}_stock"] = _stock

	# heatplot euca_stock y x, xbins(200) ybins(200) cut(0(10)100)

	# Consolidate blocks before the next wave of column creation.
	df = df.copy()

	# % of plantations that are exotic (FAO FRA 2020, Table 24), for all except cunn (native to China only), euca (exotic everywhere)
	exotic_map = {0.74: {"AGO", "BWA", "COM", "DJI", "ERI", "SWZ", "ETH", "KEN", "LSO", "MDG", "MWI", "MDV", "MUS", "MOZ", "NAM", "SYC", "SOM", "ZAF", "SSD", "SDN", "TZA", "UGA", "ZMB", "ZWE", "BEN", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COG", "CIV", "COD", "GNQ", "GAB", "GMB", "GHA", "GIN", "GNB", "LBR", "MLI", "MRT", "NER", "NGA", "RWA", "STP", "SEN", "SLE", "TGO", }, 0.50: {"DZA", "EGY", "LBY", "MAR", "TUN", "ESH"}, 0.31: {"CHN", "PRK", "MNG"}, 0.40: {"BGD", "BTN", "KHM", "IND", "IDN", "LAO", "MYS", "MMR", "NPL", "PAK", "PHL", "LKA", "THA", "TLS", "VNM", }, 0.05: {"AFG", "IRN", "IRQ", "JOR", "KAZ", "KGZ", "LBN", "OMN", "PSE", "SYR", "TJK", "TKM", "UZB", "YEM", "ARM", "AZE", "GEO", "TUR", }, 0.78: {"COK", "FJI", "KIR", "MHL", "FSM", "NRU", "PLW", "PNG", "WSM", "SLB", "TKL", "TON", "TUV", "VUT", }, 0.32: {"ATG", "BRB", "CUB", "DMA", "DOM", "GRD", "HTI", "KNA", "LCA", "VCT", "TTO"}, 0.18: {"BLZ", "CRI", "SLV", "GTM", "HND", "JAM", "MEX", "NIC", "PAN"}, 0.97: {"ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUF", "GUY", "PRY", "PER", "SUR", "URY", "VEN", }, }

	# Discounted present value of establishment costs.
	df["exotic"] = next((v for v, s in exotic_map.items() if iso in s), 0.0)
	df["native"] = 1 - df["exotic"]
	df["nr_npv_estcost"] = df["nr_cost"]
	df["pinu_npv_estcost"] = df["exotic"]*df["ep_cost"] + df["native"]*df["np_cost"]
	df["cunn_npv_estcost"] = df["ep_cost"]
	if iso == "CHN": df["cunn_npv_estcost"] = df["np_cost"]
	df["euca_npv_estcost"] = df["ep_cost"]
	for g in ["brde", "brev", "nede", "neev"]: df[f"{g}_npv_estcost"] = df["exotic"]*df["ep_cost"] + df["native"]*df["np_cost"]

	for i in range(1, Y + 1):
		disc = (1 - d) ** (i - 1)
		for g in GENUS_ORDER:
			col = f"{g}_npv_estcost"
			mask = np.mod(i, df[f"{g}_harvestyear"]) == 0
			df[col] = df[col] + (disc*df[col]).where(mask, 0)

	# Discounted present value of opportunity costs.
	ensure_columns(df, ["crop_va"], 0.0)
	df["crop_npv_oppcost"] = df["crop_va"]*discounted_factor(Y, d)

	# Compute discounted harvested carbon pulses at harvest years.
	for g in GENUS_ORDER:
		_harvested = pd.Series(0.0, index=df.index)
		_harvest_interval = df[f"{g}_harvestyear"]
		for year in range(1, Y + 1):
			_is_harvest_year = np.mod(year, _harvest_interval) == 0
			_disc_harvest = (1 - d) ** (year - 1)*df[f"{g}_A"]*(1 - np.exp(-df[f"{g}_k"]*year)) ** 2
			_harvested = _harvested + _disc_harvest.where(_is_harvest_year, 0)
		df[f"{g}_harvested"] = _harvested

	# Discounted present value of belowground biomass carbon, w/o harvest
	df["nr_bgC"] = df["nr_accum"]*df["nr_rootshoot"]
	for g in GENUS_ORDER: df[f"{g}_bgC_wohrv"] = df[f"{g}_accum"]*df["pl_rootshoot"]

	# Discounted present value of belowground biomass carbon, w/ harvest
	for g in GENUS_ORDER: df[f"{g}_bgC_whrv"] = df[f"{g}_stock"]*df["pl_rootshoot"]

	# Discounted present value of soil carbon, w/o harvest
	df["nr_soilC"] = 0.0
	for g in GENUS_ORDER:
		df[f"{g}_soilC_wohrv"] = 0.0
		df[f"{g}_soilC_whrv"] = 0.0

	for i in range(1, Y + 1):
		disc = (1 - d) ** (i - 1)
		df["nr_soilC"] = df["nr_soilC"] + disc*nrsoil
		for g in GENUS_ORDER: df[f"{g}_soilC_wohrv"] = df[f"{g}_soilC_wohrv"] + disc*plantsoil

	# Discounted present value of soil carbon, w/harvest
	# This intentionally mirrors the second nr_soilC accumulation in the original do-file.
	for i in range(1, Y + 1):
		disc = (1 - d) ** (i - 1)
		df["nr_soilC"] = df["nr_soilC"] + disc*nrsoil
		for g in GENUS_ORDER:
			mask = i < df[f"{g}_harvestyear"]
			df[f"{g}_soilC_whrv"] = df[f"{g}_soilC_whrv"] + (disc*plantsoil)*mask.astype(float)

	# Consolidate blocks before creating totalC and break-even columns.
	df = df.copy()

	# total carbon without harvest
	df["nr_wohrv_totalC"] = agbC*df["nr_accum"] + bgbC*df["nr_bgC"] + soilC*df["nr_soilC"]
	for g in GENUS_ORDER: df[f"{g}_wohrv_totalC"] = (agbC*df[f"{g}_accum"] + bgbC*df[f"{g}_bgC_wohrv"] + soilC*df[f"{g}_soilC_wohrv"]        )

	# total carbon with harvest
	for g in GENUS_ORDER:
		df[f"{g}_whrv0_totalC"] = (agbC*df[f"{g}_stock"] + bgbC*df[f"{g}_bgC_whrv"] + soilC*df[f"{g}_soilC_whrv"] + harvestC*df["w"]*df[f"{g}_harvested"]        )
		df[f"{g}_whrv_totalC"] = df[f"{g}_whrv0_totalC"]

	# Calculate breakeven carbon storage % (w1) (includes carbon accumulation only, but not cost)
	for g in GENUS_ORDER:
		den = df[f"{g}_harvested"]
		df[f"{g}_breakw1_nr"] = (df["nr_wohrv_totalC"] - df[f"{g}_whrv_totalC"]) / den.replace(0, np.nan)
		df[f"{g}_breakw1_{g}"] = (df[f"{g}_wohrv_totalC"] - df[f"{g}_whrv_totalC"]) / den.replace(0, np.nan)

		# Calculate breakeven carbon storage % (w2) (includes both carbon accumulation and cost)
		_d_nr = (df["crop_npv_oppcost"] + df["nr_npv_estcost"]).replace(0, np.nan)
		_d_g = (df["crop_npv_oppcost"] + df[f"{g}_npv_estcost"]).replace(0, np.nan)
		ratio_nr = (df["crop_npv_oppcost"] + df[f"{g}_npv_estcost"]) / _d_nr
		ratio_g = (df["crop_npv_oppcost"] + df[f"{g}_npv_estcost"]) / _d_g
		df[f"{g}_breakw2_nr"] = (df["nr_wohrv_totalC"]*ratio_nr - df[f"{g}_whrv_totalC"]) / den.replace(0, np.nan)
		df[f"{g}_breakw2_{g}"] = (df[f"{g}_wohrv_totalC"]*ratio_g - df[f"{g}_whrv_totalC"]) / den.replace(0, np.nan)

		# Calculate breakeven carbon storage % (w3) (includes both carbon accumulation and cost, less revenue from wood products)
		ratio_nr3 = (df["crop_npv_oppcost"] + df[f"{g}_npv_estcost"] - df["p"]*den) / _d_nr
		ratio_g3 = (df["crop_npv_oppcost"] + df[f"{g}_npv_estcost"] - df["p"]*den) / _d_g
		df[f"{g}_breakw3_nr"] = (df["nr_wohrv_totalC"]*ratio_nr3 - df[f"{g}_whrv_totalC"]) / den.replace(0, np.nan)
		df[f"{g}_breakw3_{g}"] = (df[f"{g}_wohrv_totalC"]*ratio_g3 - df[f"{g}_whrv_totalC"]) / den.replace(0, np.nan)

		# sum *breakw*
		# sum *breakw*, d
		# Calculate breakeven timber sale revenue (p) ($/tC harvested biomass) (includes both carbon accumulation and cost. Is dependent on w=carbon storage %)
		_den0 = den.replace(0, np.nan); _nr_wc0 = (den * df["nr_wohrv_totalC"]).replace(0, np.nan); _g_wc0 = (den * df[f"{g}_wohrv_totalC"]).replace(0, np.nan)
		df[f"{g}_breakp_nr"] = (df["crop_npv_oppcost"] + df[f"{g}_npv_estcost"] / _den0
			- (df["crop_npv_oppcost"] + df["nr_npv_estcost"])*df[f"{g}_whrv_totalC"] / _nr_wc0
			- df["w"] * (df["crop_npv_oppcost"] + df["nr_npv_estcost"]) / df["nr_wohrv_totalC"].replace(0, np.nan))
		df[f"{g}_breakp_{g}"] = (df["crop_npv_oppcost"] + df[f"{g}_npv_estcost"] / _den0
			- (df["crop_npv_oppcost"] + df[f"{g}_npv_estcost"])*df[f"{g}_whrv_totalC"] / _g_wc0
			- df["w"] * (df["crop_npv_oppcost"] + df[f"{g}_npv_estcost"]) / df[f"{g}_wohrv_totalC"].replace(0, np.nan))

	# sum *breakp*
	# sum *breakp*, d

	# Consolidate blocks before selecting plantation columns and cost metrics.
	df = df.copy()

	# Choose the most likely genus for plantation-side fields.
	pl_fields = [ "wohrv_totalC", "stock", "bgC_whrv", "soilC_whrv", "whrv0_totalC", "whrv_totalC", "npv_estcost", "harvested", "breakw1_nr", "breakw2_nr", "breakw3_nr", "breakp_nr", ]
	for col in [f"pl_{f}" for f in pl_fields]: df[col] = np.nan

	for g in GENUS_ORDER:
		m = df["genus"] == g
		for field in pl_fields: df.loc[m, f"pl_{field}"] = df.loc[m, f"{g}_{field}"]

	# Cost effectiveness.
	df["nr_wohrv_costeff"] = (df["crop_npv_oppcost"] + df["nr_npv_estcost"]) / (3.67*df["nr_wohrv_totalC"]).replace(0, np.nan)
	df["pl_wohrv_costeff"] = (df["crop_npv_oppcost"] + df["pl_npv_estcost"]) / (3.67*df["pl_wohrv_totalC"]).replace(0, np.nan)
	df["pl_whrv0_costeff"] = (df["crop_npv_oppcost"] + df["pl_npv_estcost"]) / (3.67*df["pl_whrv0_totalC"]).replace(0, np.nan)
	df["pl_whrv_costeff"] = (df["crop_npv_oppcost"] + df["pl_npv_estcost"] - df["p"]*df["pl_harvested"]) / (3.67*df["pl_whrv_totalC"]).replace(0, np.nan)

	lhs_nr = 3.67*df["nr_wohrv_totalC"] / (df["crop_npv_oppcost"] + df["nr_npv_estcost"]).replace(0, np.nan)
	lhs_pl_wo = 3.67*df["pl_wohrv_totalC"] / (df["crop_npv_oppcost"] + df["pl_npv_estcost"]).replace(0, np.nan)
	lhs_pl_wh0 = 3.67*df["pl_whrv0_totalC"] / (df["crop_npv_oppcost"] + df["pl_npv_estcost"]).replace(0, np.nan)
	lhs_pl_wh = 3.67*df["pl_whrv_totalC"] / (df["crop_npv_oppcost"] + df["pl_npv_estcost"] - df["p"]*df["pl_harvested"]).replace(0, np.nan)

	# which refor type is more costeffective
	df["refortype_wohrv"] = np.where(lhs_pl_wo > lhs_nr, "P", np.where(lhs_pl_wo < lhs_nr, "N", "."))
	df["refortype_whrv0"] = np.where(lhs_pl_wh0 > lhs_nr, "P", np.where(lhs_pl_wh0 < lhs_nr, "N", "."))
	df["refortype_whrv"] = np.where(lhs_pl_wh > lhs_nr, "P", np.where(lhs_pl_wh < lhs_nr, "N", "."))
	df.loc[lhs_pl_wh < 0, "refortype_whrv"] = "P"

	lhs_nr_bu = 3.67*df["nr_wohrv_totalC"]*df["nr_buffer"] / (df["crop_npv_oppcost"] + df["nr_npv_estcost"]).replace(0, np.nan)
	lhs_pl_wo_bu = 3.67*df["pl_wohrv_totalC"]*df["pl_buffer"] / (df["crop_npv_oppcost"] + df["pl_npv_estcost"]).replace(0, np.nan)
	lhs_pl_wh0_bu = 3.67*df["pl_whrv0_totalC"]*df["pl_buffer"] / (df["crop_npv_oppcost"] + df["pl_npv_estcost"]).replace(0, np.nan)
	lhs_pl_wh_bu = 3.67*df["pl_whrv_totalC"]*df["pl_buffer"] / (df["crop_npv_oppcost"] + df["pl_npv_estcost"] - df["p"]*df["pl_harvested"]).replace(0, np.nan)

	df["refortype_wohrv_bu"] = np.where(lhs_pl_wo_bu > lhs_nr_bu, "P", np.where(lhs_pl_wo_bu < lhs_nr_bu, "N", "."))
	df["refortype_whrv0_bu"] = np.where(lhs_pl_wh0_bu > lhs_nr_bu, "P", np.where(lhs_pl_wh0_bu < lhs_nr_bu, "N", "."))
	df["refortype_whrv_bu"] = np.where(lhs_pl_wh_bu > lhs_nr_bu, "P", np.where(lhs_pl_wh_bu < lhs_nr_bu, "N", "."))
	df.loc[lhs_pl_wh_bu < 0, "refortype_whrv_bu"] = "P"

	# Map of relative cost-effectiveness (unbuffered only).
	_r = df["nr_wohrv_costeff"] / df["pl_wohrv_costeff"].replace(0, np.nan); df["rel_wohrv_costeff"] = np.log(_r.where(_r > 0))
	_r = df["nr_wohrv_costeff"] / df["pl_whrv0_costeff"].replace(0, np.nan); df["rel_whrv0_costeff"] = np.log(_r.where(_r > 0))
	_r = df["nr_wohrv_costeff"] / df["pl_whrv_costeff"].replace(0, np.nan); df["rel_whrv_costeff"] = np.log(_r.where(_r > 0))

	# generate minimums
	for scenario, ref_col, p_total, p_cost in [("wohrv", "refortype_wohrv", "pl_wohrv_totalC", "pl_wohrv_costeff"), ("whrv0", "refortype_whrv0", "pl_whrv0_totalC", "pl_whrv0_costeff"), ("whrv", "refortype_whrv", "pl_whrv_totalC", "pl_whrv_costeff"), ]:
		choose_by_refortype(df, f"mi_{scenario}_totalC", ref_col, p_total, "nr_wohrv_totalC", 0)
		choose_by_refortype(df, f"mi_{scenario}_costeff", ref_col, p_cost, "nr_wohrv_costeff", np.nan)
		choose_by_refortype(df, f"mi_{scenario}_bu_totalC", f"{ref_col}_bu", p_total, "nr_wohrv_totalC", 0)
		choose_by_refortype(df, f"mi_{scenario}_bu_costeff", f"{ref_col}_bu", p_cost, "nr_wohrv_costeff", np.nan)

	# Consolidate blocks before MAC/screening feature columns.
	df = df.copy()

	# MARGINAL ABATEMENT COST CURVES
	# Odd but necessary hack to ensure that the MAC curves code still runs for countries that have fewer than 100 observations remaining: add 100 blank rows
	# In the pandas translation this padding step is not needed.
	# Convert C values from per-hectare to per-cell for MAC curves.
	totals_to_scale = [ "nr_wohrv_totalC", "pl_wohrv_totalC", "pl_whrv0_totalC", "pl_whrv_totalC", "mi_wohrv_totalC", "mi_whrv0_totalC", "mi_whrv_totalC", "mi_wohrv_bu_totalC", "mi_whrv0_bu_totalC", "mi_whrv_bu_totalC", ]
	for c in totals_to_scale: df[c] = df[c]*df["area"]

	prices = np.array([(i - 5)*5 for i in range(1, 26)], dtype=float)
	# change the name of the frame so that I can use MAC curves in a separate frame later
	# Set up a new frame to put the MAC curves in
	# set obs 100
	# gen price=_n
	macc = pd.DataFrame({"price": prices})

	# generate local to use in many functions involving {pinu cunn euca brde brev nede neev}
	# MAC curves - unscreened.
	MAClocal = ["nr_wohrv", "pl_wohrv", "pl_whrv0", "pl_whrv", "mi_wohrv", "mi_whrv0", "mi_whrv"]
	# gen price=_n
	# forvalues i=1/100 {# egen temp=total(`item'_totalC) if `item'_costeff<=`i'
	for item in MAClocal: macc[f"MAC_{item}"] = mac_curve(df, f"{item}_totalC", f"{item}_costeff", prices)

	# MAC curves - screened by Griscom / Brancalion / Bastin / Walker.
	# screen MAC curve based on reforestation potential map (Griscom et al 2017)
	# **gen nr_wohrv_sGr=nr_wohrv_totalC*griscom
	# gen pl_wohrv_sGr=pl_wohrv_totalC*griscom
	# gen pl_whrv0_sGr=pl_whrv0_totalC*griscom
	# **gen pl_whrv_sGr=pl_whrv_totalC*griscom
	# gen mi_wohrv_sGr=mi_wohrv_totalC*griscom
	# gen mi_whrv0_sGr=mi_wohrv0_totalC*griscom
	df["mi_whrv_sGr"] = df["mi_whrv_totalC"]*df["griscom"]
	df["mi_whrv_sBr"] = df["mi_whrv_totalC"]*df["brancalion"]
	df["mi_whrv_sBa"] = df["mi_whrv_totalC"]*df["bastin"]
	df["mi_whrv_sWa"] = df["mi_whrv_totalC"]*df["walker"]
	# gen price=_n
	# forvalues i=1/100 {# egen temp=total(mi_whrv_s`item') if mi_whrv_costeff<=`i'
	for item in ["Gr", "Br", "Ba", "Wa"]: macc[f"MAC_mi_whrv_s{item}"] = mac_curve(df, f"mi_whrv_s{item}", "mi_whrv_costeff", prices)

	# MAC curves - screened, buffered.
	df["mi_whrv_sGr_bu"] = df["mi_whrv_bu_totalC"]*df["griscom"]
	# gen price=_n
	# forvalues i=1/100 {# egen temp=total(mi_whrv_sGr_bu) if mi_whrv_bu_costeff<=`i'
	macc["MAC_mi_whrv_sGr_bu"] = mac_curve(df, "mi_whrv_sGr_bu", "mi_whrv_bu_costeff", prices)

	# MAC curves - screened, buffered, additional.
	# reduce MAC based on additionality, i.e. subtracts reforestation (as a % of a cell) that "would have happened anyway" under Busch et al 2019 business-as-usual projection
	# the coefficients of 0.510928, 0.305911, 0.183161 are to upweight near-term reforestation consistent with a discount rate of 5%
	# ideally, we'd multiply carbon accumulation in each year by BAU in that year, but this is a reasonable approximation
	ensure_columns(df, ["bau_2020_2030", "bau_2030_2040", "bau_2040_2050"], np.nan)
	df["BAUrefor"] = (df["bau_2020_2030"]*0.510928 + df["bau_2030_2040"]*0.305911 + df["bau_2040_2050"]*0.183161)
	# Outside of tropics, replace BAU missing values with default value
	df["BAUrefor"] = df["BAUrefor"].fillna(0.0571795)

	# MAC - more costeffective (w/ harvest) (screened) (buffered) (additional)
	# gen price=_n
	# forvalues i=1/100 {# egen temp=total(mi_whrv_sGr_bu_ad) if mi_whrv_bu_costeff<=`i'
	df["mi_whrv_sGr_bu_ad"] = (1 - df["BAUrefor"])*df["mi_whrv_sGr_bu"]
	macc["MAC_mi_whrv_sGr_bu_ad"] = mac_curve(df, "mi_whrv_sGr_bu_ad", "mi_whrv_bu_costeff", prices)

	# MAC - all seven curves (screened) (buffered) (additional)
	for item in ["nr_wohrv", "pl_wohrv", "pl_whrv0", "pl_whrv"]: df[f"{item}_bu_costeff"] = df[f"{item}_costeff"]

	df["nr_wohrv_sGr_bu_ad"] = (df["nr_wohrv_totalC"]*df["griscom"]*df["nr_buffer"]*(1 - df["BAUrefor"]))
	df["pl_wohrv_sGr_bu_ad"] = (df["pl_wohrv_totalC"]*df["griscom"]*df["pl_buffer"]*(1 - df["BAUrefor"]))
	df["pl_whrv0_sGr_bu_ad"] = (df["pl_whrv0_totalC"]*df["griscom"]*df["pl_buffer"]*(1 - df["BAUrefor"]))
	df["pl_whrv_sGr_bu_ad"] = (df["pl_whrv_totalC"]*df["griscom"]*df["pl_buffer"]*(1 - df["BAUrefor"]))
	df["mi_wohrv_sGr_bu_ad"] = df["mi_wohrv_bu_totalC"]*df["griscom"]*(1 - df["BAUrefor"])
	df["mi_whrv0_sGr_bu_ad"] = df["mi_whrv0_bu_totalC"]*df["griscom"]*(1 - df["BAUrefor"])

	for item in MAClocal: macc[f"MAC_{item}_sGr_bu_ad"] = mac_curve(df, f"{item}_sGr_bu_ad", f"{item}_bu_costeff", prices)

	# MAC curves - disaggregated by biome (shown here for more costeffective w/harvest, screened, buffered, additional)
	# gen price=_n
	# forvalues i=1/100{# egen temp=total(mi_whrv_sGr_bu_ad_b`j') if mi_whrv_bu_costeff<=`i'
	for j in range(1, 13):
		col = f"mi_whrv_sGr_bu_ad_b{j}"
		df[col] = np.where(df["biomes"] == j, df["mi_whrv_sGr_bu_ad"], 0)
		macc[f"MAC_mi_whrv_sGr_bu_ad_b{j}"] = mac_curve(df, col, "mi_whrv_bu_costeff", prices)

	# Disaggregate MAC curve by reforestation type/genus (shown here for more costeffective w/harvest screened)
	# gen price=_n
	# forvalues i=1/100{# egen temp=total(mi_whrv_sGr_bu_ad_t`j') if mi_whrv_bu_costeff<=`i'
	df["Genus"] = np.nan
	df.loc[df["refortype_whrv_bu"] == "N", "Genus"] = 0
	for t in range(1, 16): df.loc[(df["refortype_whrv_bu"] == "P") & (df["type"] == t), "Genus"] = t

	for j in range(0, 16):
		col = f"mi_whrv_sGr_bu_ad_t{j}"
		df[col] = np.where(df["Genus"] == j, df["mi_whrv_sGr_bu_ad"], 0)
		macc[f"MAC_mi_whrv_sGr_bu_ad_t{j}"] = mac_curve(df, col, "mi_whrv_bu_costeff", prices)

	# Final consolidation before export steps.
	df = df.copy()

	# Save MAC curves
	# Save MAC curves and map-ready outputs.
	macc_dir.mkdir(parents=True, exist_ok=True)
	maps_dir.mkdir(parents=True, exist_ok=True)
	macc_path = macc_dir / f"MACC_{iso}.dta"
	macc.to_stata(macc_path, write_index=False)
	drop_exact = {"biomes", "fao_ecoz", "id", "native", "exotic", "continent", "nr_Y30_d0", "nr_harvestyear", "mi_whrv_bu_costeff", "Genus", "np_cost", "ep_cost", }

	# Keep only map-facing columns by removing intermediate model internals.
	def should_drop(c: str) -> bool:
		return (c in drop_exact or c.startswith("bau") or c.endswith(("_A", "_k", "_rootshoot", "totalC"))
			or c.startswith(("pinu", "cunn", "euca", "brde", "brev", "nede", "neev", "mi_wohrv", "refortype", "rel_", "mi_whrv_s", )))

	drop_cols = [c for c in df.columns if should_drop(c)]
	maps_df = df.drop(columns=sorted(set(drop_cols)), errors="ignore").copy()
	maps_df["country"] = iso
	maps_path = maps_dir / f"maps_{iso}.dta"
	maps_df.to_stata(maps_path, write_index=False)
	print(f"[OK] {iso}: {src.name} -> {macc_path.name}, {maps_path.name}")

# Parse command-line paths and optional ISO subset selection.
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Python conversion of 1. Model loop all data.do")
	parser.add_argument("--dta-dir", type=Path, required=True, help="Directory containing per-country .dta input files", )
	parser.add_argument("--macc-dir", type=Path, required=True, help="Directory to write MACC_*.dta files")
	parser.add_argument("--maps-dir", type=Path, required=True, help="Directory to write maps_*.dta files")
	parser.add_argument("--iso", nargs="*", default=ISO_LIST, help="Optional subset of ISO codes to run")
	return parser.parse_args()

# Iterate across requested ISO codes and run the country pipeline.
def main() -> None:
	args = parse_args()
	for iso in args.iso: process_country(iso, args.dta_dir, args.macc_dir, args.maps_dir)

if __name__ == "__main__":
	# Set this to True to run one country from in-script settings (no CLI args).
	RUN_WITH_INLINE_SETTINGS = True

	if RUN_WITH_INLINE_SETTINGS:
		base_dir = Path(__file__).resolve().parent.parent
		process_country(iso="AFG",
			dta_dir=base_dir / "Input",
			macc_dir=base_dir / "Output",
			maps_dir=base_dir / "Output",)
	else:
		main()