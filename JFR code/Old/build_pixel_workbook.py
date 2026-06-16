# build_pixel_workbook.py | Created 2026-03-27
# Builds an Excel workbook from selected pixel data, organizing inputs, outputs, and diagnostics for manual inspection and comparison across cases.
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
MODEL_PATH = CODE_DIR / "1_model_loop_all_data.py"
GENUS_ORDER = ["pinu", "cunn", "euca", "brde", "brev", "nede", "neev"]

def load_model_module() -> Any:
	spec = importlib.util.spec_from_file_location("model_loop_all_data", MODEL_PATH)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Unable to load model module from {MODEL_PATH}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module

def resolve_iso(iso_code: str, model: Any) -> str:
	iso_code = iso_code.strip().upper()
	if iso_code in model.ISO_LIST: return iso_code
	prefix_matches = [iso for iso in model.ISO_LIST if iso.startswith(iso_code)]
	if len(prefix_matches) == 1: return prefix_matches[0]
	raise ValueError(f"Could not resolve ISO code {iso_code!r} to a single dataset.")

def safe_div_scalar(num: float, den: float) -> float:
	if pd.isna(den) or den == 0: return np.nan
	return num / den

def safe_log_scalar(value: float) -> float:
	if pd.isna(value) or value <= 0: return np.nan
	return float(np.log(value))

def compute_accum_silent(df: pd.DataFrame, asymptotic_carbon_col: str, growth_rate_col: str, years: int) -> pd.Series:
	accumulation = pd.Series(0.0, index=df.index)
	for year in range(1, years + 1):
		increment = (df[asymptotic_carbon_col]
			* ((1 - np.exp(-df[growth_rate_col] * year)) ** 2
				- (1 - np.exp(-df[growth_rate_col] * (year - 1))) ** 2))
		accumulation = accumulation + increment
	return accumulation

def to_vertical_table(items: Iterable[tuple[str, Any]]) -> pd.DataFrame:
	return pd.DataFrame(items, columns=["variable", "value"])

def build_country_dataframe(model: Any, iso: str, dta_dir: Path) -> pd.DataFrame:
	src = next((candidate
		for candidate in [dta_dir / iso, dta_dir / f"{iso}.dta", dta_dir / f"maps_{iso}.dta"]
		if candidate.exists() and candidate.is_file()), None,)
	if src is None: raise FileNotFoundError(f"No input found for {iso} in {dta_dir}")

	df = pd.read_stata(src)
	if df.empty: raise ValueError(f"Input dataset for {iso} is empty.")

	df = df.copy()
	df["id"] = np.arange(1, len(df) + 1)

	for col in ["griscom", "brancalion", "bastin", "walker", "nr_buffer", "pl_buffer"]:
		if col in df.columns: df[col] = df[col].fillna(0)
		else: df[col] = 0

	model.ensure_columns(df, ["biomes", "nr_A", "type"])
	df = df[df["biomes"] != 13]
	df = df[df["biomes"] != 14]
	df = df[df["nr_A"].notna()]
	df = df[df["type"].notna()]
	if df.empty: raise ValueError(f"No rows remain for {iso} after cleaning.")

	for col in ["nr_cost", "ep_cost", "np_cost"]:
		if col not in df.columns: df[col] = 0.0
		df[col] = df[col] * 1.1577

	df["genus"] = df["type"].map(model.GENUS_BY_TYPE).fillna("")
	if "area_m2" in df.columns:
		df["area"] = df["area_m2"] / 10000.0
		df.drop(columns=["area_m2"], inplace=True)
	else: df["area"] = 0.0

	continent = "AFRI" if iso in model.AFRI else "AMER" if iso in model.AMER else "ASIA" if iso in model.ASIA else ""
	df["continent"] = continent

	df["nr_rootshoot"] = 0.26
	df["pl_rootshoot"] = 0.26
	model.ensure_columns(df, ["fao_ecoz"], np.nan)

	nr_rootshoot_rules = {(11, "AFRI"): 0.825,
		(11, "AMER"): 0.221,
		(11, "ASIA"): 0.207,
		(12, "AFRI"): 0.232,
		(12, "AMER"): 0.2845,
		(12, "ASIA"): 0.323,
		(13, "AFRI"): 0.332,
		(13, "AMER"): 0.334,
		(13, "ASIA"): 0.440,
		(16, "AMER"): 0.348,
		(16, "ASIA"): 0.322,
		(21, "AFRI"): 0.232,
		(21, "AMER"): 0.175,
		(21, "ASIA"): 0.230,
		(22, "AMER"): 0.336,
		(22, "ASIA"): 0.440,
		(23, "AMER"): 1.338,
		(23, "ASIA"): 1.338,}
	for (eco, cont), value in nr_rootshoot_rules.items(): df.loc[(df["fao_ecoz"] == eco) & (df["continent"] == cont), "nr_rootshoot"] = value

	is_brde_nede = df["genus"].isin(["brde", "nede"])
	is_other = df["genus"].isin(["pinu", "cunn", "brev", "neev"])
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "AMER") & is_brde_nede, "nr_rootshoot"] = 0.466
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "AMER") & is_other, "nr_rootshoot"] = 0.337
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "ASIA") & is_brde_nede, "nr_rootshoot"] = 0.225
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "ASIA") & is_other, "nr_rootshoot"] = 0.243
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "AMER") & is_other, "nr_rootshoot"] = 0.237
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "ASIA") & is_brde_nede, "nr_rootshoot"] = 0.225
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "ASIA") & is_other, "nr_rootshoot"] = 0.243

	pl_rootshoot_rules = {(11, "AMER"): 0.170, (11, "ASIA"): 0.325, (16, "AMER"): 2.158, (23, "ASIA"): 2.158}
	for (eco, cont), value in pl_rootshoot_rules.items():
		df.loc[(df["fao_ecoz"] == eco) & (df["continent"] == cont), "pl_rootshoot"] = value
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "AMER") & is_other, "pl_rootshoot"] = 0.203
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "ASIA") & is_brde_nede, "pl_rootshoot"] = 0.307
	df.loc[(df["fao_ecoz"] == 31) & (df["continent"] == "ASIA") & is_other, "pl_rootshoot"] = 0.224
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "AMER") & is_other, "pl_rootshoot"] = 0.237
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "ASIA") & is_brde_nede, "pl_rootshoot"] = 0.307
	df.loc[(df["fao_ecoz"] == 32) & (df["continent"] == "ASIA") & is_other, "pl_rootshoot"] = 0.224

	base_cols = ["nr_A", "nr_k"]
	for genus in model.GENUS_ORDER:
		base_cols.extend([f"{genus}_A", f"{genus}_k"])
	model.ensure_columns(df, base_cols, 0.0)
	for col in base_cols:
		df[col] = df[col].clip(lower=0)

	years = 30
	harvest_rule = 1
	nrsoil = 0.415107735
	plantsoil = 0.092749069
	agb_c = 1
	bgb_c = 1
	soil_c = 1
	harvest_c = 1

	w_map = {0.188: {"ATG", "BLZ", "BRB", "CRI", "CUB", "DMA", "DOM", "GRD", "GTM", "HND", "HTI", "JAM", "KNA", "LCA", "MEX", "NIC", "PAN", "SLV", "TTO", "VCT"},
		0.268: {"ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUF", "GUY", "PER", "PRY", "SUR", "URY", "VEN"},
		0.355: {"AFG", "ARM", "AZE", "BGD", "BTN", "CHN", "GEO", "IDN", "IND", "IRN", "IRQ", "JOR", "KAZ", "KGZ", "KHM", "LAO", "LBN", "LKA", "MMR", "MNG", "MYS", "NPL", "OMN", "PAK", "PHL", "PNG", "PRK", "PSE", "SYR", "THA", "TJK", "TKM", "TLS", "TUR", "UZB", "VNM", "YEM"},
		0.392: {"AGO", "BDI", "BEN", "BFA", "BWA", "CAF", "CIV", "CMR", "COD", "COG", "COM", "CPV", "DJI", "DZA", "EGY", "ERI", "ESH", "ETH", "GAB", "GHA", "GIN", "GMB", "GNB", "GNQ", "KEN", "LBR", "LBY", "LSO", "MAR", "MDG", "MDV", "MLI", "MOZ", "MRT", "MUS", "MWI", "NAM", "NER", "NGA", "RWA", "SDN", "SEN", "SLE", "SOM", "SSD", "STP", "SWZ", "SYC", "TCD", "TGO", "TUN", "TZA", "UGA", "ZAF", "ZMB", "ZWE"},
		0.415: {"COK", "FJI", "FSM", "KIR", "MHL", "NRU", "PLW", "SLB", "TKL", "TON", "TUV", "VUT", "WSM"},}
	p_map = {52.65: {"ATG", "BLZ", "BRB", "CRI", "CUB", "DMA", "DOM", "GRD", "GTM", "HND", "HTI", "JAM", "KNA", "LCA", "NIC", "PAN", "SLV", "TTO", "VCT"},
		59.73: {"MEX"},
		61.04: {"BOL", "BRA", "COL", "ECU", "GUF", "GUY", "PER", "SUR", "VEN"},
		67.76: {"BGD", "IDN", "KHM", "LAO", "LKA", "MMR", "MYS", "OMN", "PHL", "PNG", "THA", "TLS", "VNM", "YEM"},
		69.26: {"PRY"},
		71.24: {"AGO", "BDI", "BEN", "BFA", "CAF", "CIV", "CMR", "COD", "COG", "COM", "CPV", "DJI", "ERI", "ETH", "GAB", "GHA", "GIN", "GMB", "GNB", "GNQ", "KEN", "LBR", "MDG", "MDV", "MLI", "MOZ", "MRT", "MUS", "MWI", "NER", "NGA", "RWA", "SDN", "SEN", "SLE", "SOM", "SSD", "STP", "SYC", "TCD", "TGO", "TZA", "UGA", "ZMB", "ZWE"},
		76.08: {"COK", "FJI", "FSM", "KIR", "MHL", "NRU", "PLW", "SLB", "TKL", "TON", "TUV", "VUT", "WSM"},
		76.88: {"IND"},
		80.003: {"ARG", "CHL", "URY"},
		80.83: {"BWA", "NAM"},
		88.84: {"AFG", "ARM", "AZE", "BTN", "CHN", "GEO", "IRN", "IRQ", "JOR", "KAZ", "KGZ", "LBN", "MNG", "NPL", "PAK", "PRK", "PSE", "SYR", "TJK", "TKM", "TUR", "UZB"},
		93.40: {"DZA", "EGY", "ESH", "LBY", "LSO", "MAR", "SWZ", "TUN", "ZAF"},}
	exotic_map = {0.74: {"AGO", "BWA", "COM", "DJI", "ERI", "SWZ", "ETH", "KEN", "LSO", "MDG", "MWI", "MDV", "MUS", "MOZ", "NAM", "SYC", "SOM", "ZAF", "SSD", "SDN", "TZA", "UGA", "ZMB", "ZWE", "BEN", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COG", "CIV", "COD", "GNQ", "GAB", "GMB", "GHA", "GIN", "GNB", "LBR", "MLI", "MRT", "NER", "NGA", "RWA", "STP", "SEN", "SLE", "TGO"},
		0.50: {"DZA", "EGY", "LBY", "MAR", "TUN", "ESH"},
		0.31: {"CHN", "PRK", "MNG"},
		0.40: {"BGD", "BTN", "KHM", "IND", "IDN", "LAO", "MYS", "MMR", "NPL", "PAK", "PHL", "LKA", "THA", "TLS", "VNM"},
		0.05: {"AFG", "IRN", "IRQ", "JOR", "KAZ", "KGZ", "LBN", "OMN", "PSE", "SYR", "TJK", "TKM", "UZB", "YEM", "ARM", "AZE", "GEO", "TUR"},
		0.78: {"COK", "FJI", "KIR", "MHL", "FSM", "NRU", "PLW", "PNG", "WSM", "SLB", "TKL", "TON", "TUV", "VUT"},
		0.32: {"ATG", "BRB", "CUB", "DMA", "DOM", "GRD", "HTI", "KNA", "LCA", "VCT", "TTO"},
		0.18: {"BLZ", "CRI", "SLV", "GTM", "HND", "JAM", "MEX", "NIC", "PAN"},
		0.97: {"ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUF", "GUY", "PRY", "PER", "SUR", "URY", "VEN"},}

	df["w"] = model.iso_value(iso, w_map, 0.0)
	df["p"] = model.iso_value(iso, p_map, 0.0)
	df["nr_Y30_d0"] = df["nr_A"] * (1 - np.exp(-df["nr_k"] * years)) ** 2
	df["nr_accum"] = compute_accum_silent(df, "nr_A", "nr_k", years)

	for genus in model.GENUS_ORDER:
		df[f"{genus}_Y30_d0"] = df[f"{genus}_A"] * (1 - np.exp(-df[f"{genus}_k"] * years)) ** 2
		df[f"{genus}_accum"] = compute_accum_silent(df, f"{genus}_A", f"{genus}_k", years)

	df["nr_harvestyear"] = model.compute_harvest_year(df, "nr_A", "nr_k", years, 0.0, harvest_rule)
	for genus in model.GENUS_ORDER:
		df[f"{genus}_harvestyear"] = model.compute_harvest_year(df, f"{genus}_A", f"{genus}_k", years, 0.0, harvest_rule)
		df[f"{genus}_stock"] = model.compute_stock(df, f"{genus}_A", f"{genus}_k", f"{genus}_harvestyear", years, 0.0)

	df = df.copy()
	df["exotic"] = model.iso_value(iso, exotic_map, 0.0)
	df["native"] = 1 - df["exotic"]
	df["nr_npv_estcost"] = df["nr_cost"]
	df["pinu_npv_estcost"] = df["exotic"] * df["ep_cost"] + df["native"] * df["np_cost"]
	df["cunn_npv_estcost"] = df["ep_cost"]
	if iso == "CHN":
		df["cunn_npv_estcost"] = df["np_cost"]
	df["euca_npv_estcost"] = df["ep_cost"]
	for genus in ["brde", "brev", "nede", "neev"]:
		df[f"{genus}_npv_estcost"] = df["exotic"] * df["ep_cost"] + df["native"] * df["np_cost"]

	for year in range(1, years + 1):
		for genus in model.GENUS_ORDER:
			column = f"{genus}_npv_estcost"
			mask = np.mod(year, df[f"{genus}_harvestyear"]) == 0
			df[column] = df[column] + df[column].where(mask, 0)

	model.ensure_columns(df, ["crop_va"], 0.0)
	df["crop_npv_oppcost"] = df["crop_va"] * model.discounted_factor(years, 0.0)

	for genus in model.GENUS_ORDER:
		df[f"{genus}_harvested"] = model.compute_harvested(df, f"{genus}_A", f"{genus}_k", f"{genus}_harvestyear", years, 0.0)

	df["nr_bgC"] = df["nr_accum"] * df["nr_rootshoot"]
	for genus in model.GENUS_ORDER:
		df[f"{genus}_bgC_wohrv"] = df[f"{genus}_accum"] * df["pl_rootshoot"]
		df[f"{genus}_bgC_whrv"] = df[f"{genus}_stock"] * df["pl_rootshoot"]

	df["nr_soilC"] = 0.0
	for genus in model.GENUS_ORDER:
		df[f"{genus}_soilC_wohrv"] = 0.0
		df[f"{genus}_soilC_whrv"] = 0.0

	for year in range(1, years + 1):
		df["nr_soilC"] = df["nr_soilC"] + nrsoil
		for genus in model.GENUS_ORDER:
			df[f"{genus}_soilC_wohrv"] = df[f"{genus}_soilC_wohrv"] + plantsoil

	for year in range(1, years + 1):
		df["nr_soilC"] = df["nr_soilC"] + nrsoil
		for genus in model.GENUS_ORDER:
			mask = year < df[f"{genus}_harvestyear"]
			df[f"{genus}_soilC_whrv"] = df[f"{genus}_soilC_whrv"] + plantsoil * mask.astype(float)

	df = df.copy()
	df["nr_wohrv_totalC"] = agb_c * df["nr_accum"] + bgb_c * df["nr_bgC"] + soil_c * df["nr_soilC"]
	for genus in model.GENUS_ORDER:
		df[f"{genus}_wohrv_totalC"] = agb_c * df[f"{genus}_accum"] + bgb_c * df[f"{genus}_bgC_wohrv"] + soil_c * df[f"{genus}_soilC_wohrv"]
		df[f"{genus}_whrv0_totalC"] = agb_c * df[f"{genus}_stock"] + bgb_c * df[f"{genus}_bgC_whrv"] + soil_c * df[f"{genus}_soilC_whrv"] + harvest_c * df["w"] * df[f"{genus}_harvested"]
		df[f"{genus}_whrv_totalC"] = df[f"{genus}_whrv0_totalC"]

	for genus in model.GENUS_ORDER:
		denominator = df[f"{genus}_harvested"]
		df[f"{genus}_breakw1_nr"] = model.safe_div(df["nr_wohrv_totalC"] - df[f"{genus}_whrv_totalC"], denominator)
		df[f"{genus}_breakw1_{genus}"] = model.safe_div(df[f"{genus}_wohrv_totalC"] - df[f"{genus}_whrv_totalC"], denominator)

		ratio_nr = model.safe_div(df["crop_npv_oppcost"] + df[f"{genus}_npv_estcost"], df["crop_npv_oppcost"] + df["nr_npv_estcost"])
		ratio_genus = model.safe_div(df["crop_npv_oppcost"] + df[f"{genus}_npv_estcost"], df["crop_npv_oppcost"] + df[f"{genus}_npv_estcost"])
		df[f"{genus}_breakw2_nr"] = model.safe_div(df["nr_wohrv_totalC"] * ratio_nr - df[f"{genus}_whrv_totalC"], denominator)
		df[f"{genus}_breakw2_{genus}"] = model.safe_div(df[f"{genus}_wohrv_totalC"] * ratio_genus - df[f"{genus}_whrv_totalC"], denominator)

		ratio_nr3 = model.safe_div(df["crop_npv_oppcost"] + df[f"{genus}_npv_estcost"] - df["p"] * denominator, df["crop_npv_oppcost"] + df["nr_npv_estcost"])
		ratio_genus3 = model.safe_div(df["crop_npv_oppcost"] + df[f"{genus}_npv_estcost"] - df["p"] * denominator, df["crop_npv_oppcost"] + df[f"{genus}_npv_estcost"])
		df[f"{genus}_breakw3_nr"] = model.safe_div(df["nr_wohrv_totalC"] * ratio_nr3 - df[f"{genus}_whrv_totalC"], denominator)
		df[f"{genus}_breakw3_{genus}"] = model.safe_div(df[f"{genus}_wohrv_totalC"] * ratio_genus3 - df[f"{genus}_whrv_totalC"], denominator)

		df[f"{genus}_breakp_nr"] = (df["crop_npv_oppcost"]
			+ model.safe_div(df[f"{genus}_npv_estcost"], denominator)
			- model.safe_div((df["crop_npv_oppcost"] + df["nr_npv_estcost"]) * df[f"{genus}_whrv_totalC"], denominator * df["nr_wohrv_totalC"])
			- df["w"] * model.safe_div(df["crop_npv_oppcost"] + df["nr_npv_estcost"], df["nr_wohrv_totalC"]))
		df[f"{genus}_breakp_{genus}"] = (df["crop_npv_oppcost"]
			+ model.safe_div(df[f"{genus}_npv_estcost"], denominator)
			- model.safe_div((df["crop_npv_oppcost"] + df[f"{genus}_npv_estcost"]) * df[f"{genus}_whrv_totalC"], denominator * df[f"{genus}_wohrv_totalC"])
			- df["w"] * model.safe_div(df["crop_npv_oppcost"] + df[f"{genus}_npv_estcost"], df[f"{genus}_wohrv_totalC"]))

	df = df.copy()
	pl_fields = ["wohrv_totalC", "stock", "bgC_whrv", "soilC_whrv", "whrv0_totalC", "whrv_totalC", "npv_estcost", "harvested", "breakw1_nr", "breakw2_nr", "breakw3_nr", "breakp_nr"]
	for col in [f"pl_{field}" for field in pl_fields]:
		df[col] = np.nan
	for genus in model.GENUS_ORDER:
		mask = df["genus"] == genus
		for field in pl_fields:
			df.loc[mask, f"pl_{field}"] = df.loc[mask, f"{genus}_{field}"]

	df["nr_wohrv_costeff"] = model.safe_div(df["crop_npv_oppcost"] + df["nr_npv_estcost"], 3.67 * df["nr_wohrv_totalC"])
	df["pl_wohrv_costeff"] = model.safe_div(df["crop_npv_oppcost"] + df["pl_npv_estcost"], 3.67 * df["pl_wohrv_totalC"])
	df["pl_whrv0_costeff"] = model.safe_div(df["crop_npv_oppcost"] + df["pl_npv_estcost"], 3.67 * df["pl_whrv0_totalC"])
	df["pl_whrv_costeff"] = model.safe_div(df["crop_npv_oppcost"] + df["pl_npv_estcost"] - df["p"] * df["pl_harvested"], 3.67 * df["pl_whrv_totalC"])

	lhs_nr = model.safe_div(3.67 * df["nr_wohrv_totalC"], df["crop_npv_oppcost"] + df["nr_npv_estcost"])
	lhs_pl_wo = model.safe_div(3.67 * df["pl_wohrv_totalC"], df["crop_npv_oppcost"] + df["pl_npv_estcost"])
	lhs_pl_wh0 = model.safe_div(3.67 * df["pl_whrv0_totalC"], df["crop_npv_oppcost"] + df["pl_npv_estcost"])
	lhs_pl_wh = model.safe_div(3.67 * df["pl_whrv_totalC"], df["crop_npv_oppcost"] + df["pl_npv_estcost"] - df["p"] * df["pl_harvested"])

	df["refortype_wohrv"] = np.where(lhs_pl_wo > lhs_nr, "P", np.where(lhs_pl_wo < lhs_nr, "N", "."))
	df["refortype_whrv0"] = np.where(lhs_pl_wh0 > lhs_nr, "P", np.where(lhs_pl_wh0 < lhs_nr, "N", "."))
	df["refortype_whrv"] = np.where(lhs_pl_wh > lhs_nr, "P", np.where(lhs_pl_wh < lhs_nr, "N", "."))
	df.loc[lhs_pl_wh < 0, "refortype_whrv"] = "P"

	lhs_nr_bu = model.safe_div(3.67 * df["nr_wohrv_totalC"] * df["nr_buffer"], df["crop_npv_oppcost"] + df["nr_npv_estcost"])
	lhs_pl_wo_bu = model.safe_div(3.67 * df["pl_wohrv_totalC"] * df["pl_buffer"], df["crop_npv_oppcost"] + df["pl_npv_estcost"])
	lhs_pl_wh0_bu = model.safe_div(3.67 * df["pl_whrv0_totalC"] * df["pl_buffer"], df["crop_npv_oppcost"] + df["pl_npv_estcost"])
	lhs_pl_wh_bu = model.safe_div(3.67 * df["pl_whrv_totalC"] * df["pl_buffer"], df["crop_npv_oppcost"] + df["pl_npv_estcost"] - df["p"] * df["pl_harvested"])

	df["refortype_wohrv_bu"] = np.where(lhs_pl_wo_bu > lhs_nr_bu, "P", np.where(lhs_pl_wo_bu < lhs_nr_bu, "N", "."))
	df["refortype_whrv0_bu"] = np.where(lhs_pl_wh0_bu > lhs_nr_bu, "P", np.where(lhs_pl_wh0_bu < lhs_nr_bu, "N", "."))
	df["refortype_whrv_bu"] = np.where(lhs_pl_wh_bu > lhs_nr_bu, "P", np.where(lhs_pl_wh_bu < lhs_nr_bu, "N", "."))
	df.loc[lhs_pl_wh_bu < 0, "refortype_whrv_bu"] = "P"

	df["rel_wohrv_costeff"] = model.safe_log(model.safe_div(df["nr_wohrv_costeff"], df["pl_wohrv_costeff"]))
	df["rel_whrv0_costeff"] = model.safe_log(model.safe_div(df["nr_wohrv_costeff"], df["pl_whrv0_costeff"]))
	df["rel_whrv_costeff"] = model.safe_log(model.safe_div(df["nr_wohrv_costeff"], df["pl_whrv_costeff"]))

	for scenario, ref_col, p_total, p_cost in [("wohrv", "refortype_wohrv", "pl_wohrv_totalC", "pl_wohrv_costeff"),
		("whrv0", "refortype_whrv0", "pl_whrv0_totalC", "pl_whrv0_costeff"), ("whrv", "refortype_whrv", "pl_whrv_totalC", "pl_whrv_costeff"),]:
		model.choose_by_refortype(df, f"mi_{scenario}_totalC", ref_col, p_total, "nr_wohrv_totalC", 0)
		model.choose_by_refortype(df, f"mi_{scenario}_costeff", ref_col, p_cost, "nr_wohrv_costeff", np.nan)
		model.choose_by_refortype(df, f"mi_{scenario}_bu_totalC", f"{ref_col}_bu", p_total, "nr_wohrv_totalC", 0)
		model.choose_by_refortype(df, f"mi_{scenario}_bu_costeff", f"{ref_col}_bu", p_cost, "nr_wohrv_costeff", np.nan)

	return df

def build_traces(row_inputs: pd.Series, row_pre_mac: pd.Series) -> dict[str, pd.DataFrame]:
	years = int(row_pre_mac["Y"])
	nrsoil = float(row_pre_mac["nrsoil"])
	plantsoil = float(row_pre_mac["plantsoil"])
	traces: dict[str, pd.DataFrame] = {}

	nr_rows: list[dict[str, Any]] = []
	prev_standing = 0.0
	nr_soil_wo = 0.0
	nr_soil_second_pass = 0.0
	for year in range(1, years + 1):
		standing = float(row_inputs["nr_A"] * (1 - np.exp(-row_inputs["nr_k"] * year)) ** 2)
		increment = standing - prev_standing
		nr_soil_wo += nrsoil
		nr_soil_second_pass += nrsoil
		growth_rate = np.nan
		decision_threshold = np.nan
		harvest_trigger = False
		if year >= 2:
			prior_stock = float(row_inputs["nr_A"] * (1 - np.exp(-row_inputs["nr_k"] * (year - 1))) ** 2)
			if prior_stock != 0:
				growth_rate = increment / prior_stock
		if year >= 3:
			decision_threshold = 0.0
			harvest_trigger = bool(growth_rate > decision_threshold) if not pd.isna(growth_rate) else False
		nr_rows.append({"year": year,
				"standing_no_harvest": standing,
				"increment_no_harvest": increment,
				"accum_increment": increment,
				"growth_rate": growth_rate,
				"harvest_decision_threshold_F1": decision_threshold,
				"harvest_trigger_F1": harvest_trigger,
				"soil_increment_pass_1": nrsoil,
				"nr_soilC_after_pass_1": nr_soil_wo,
				"soil_increment_pass_2": nrsoil,
				"nr_soilC_after_pass_2": nr_soil_wo + nr_soil_second_pass,})
		prev_standing = standing
	traces["nr_trace"] = pd.DataFrame(nr_rows)

	for genus in GENUS_ORDER:
		harvest_year = float(row_pre_mac[f"{genus}_harvestyear"])
		npv_estcost = float(row_pre_mac[f"{genus}_npv_estcost"])
		running_estcost = float(row_inputs["ep_cost"])
		if genus == "pinu":
			running_estcost = float(row_pre_mac["exotic"] * row_inputs["ep_cost"] + row_pre_mac["native"] * row_inputs["np_cost"])
		elif genus == "cunn":
			running_estcost = float(row_inputs["ep_cost"])
		elif genus == "euca":
			running_estcost = float(row_inputs["ep_cost"])
		else:
			running_estcost = float(row_pre_mac["exotic"] * row_inputs["ep_cost"] + row_pre_mac["native"] * row_inputs["np_cost"])

		rows: list[dict[str, Any]] = []
		prev_standing = 0.0
		soil_wo = 0.0
		soil_w = 0.0
		for year in range(1, years + 1):
			standing = float(row_inputs[f"{genus}_A"] * (1 - np.exp(-row_inputs[f"{genus}_k"] * year)) ** 2)
			increment = standing - prev_standing
			prior_stock = prev_standing
			growth_rate = np.nan
			if year >= 2 and prior_stock != 0:
				growth_rate = increment / prior_stock
			decision_threshold = np.nan
			harvest_trigger = False
			if year >= 3:
				decision_threshold = 0.0
				harvest_trigger = bool(growth_rate > decision_threshold) if not pd.isna(growth_rate) else False
			years_since_harvest = np.nan if pd.isna(harvest_year) or harvest_year == 0 else float(np.mod(year, harvest_year))
			prior_years_since_harvest = np.nan if pd.isna(harvest_year) or harvest_year == 0 else float(np.mod(year - 1, harvest_year))
			stock_current = np.nan if pd.isna(years_since_harvest) else float(row_inputs[f"{genus}_A"] * (1 - np.exp(-row_inputs[f"{genus}_k"] * years_since_harvest)) ** 2)
			stock_previous = np.nan if pd.isna(prior_years_since_harvest) else float(row_inputs[f"{genus}_A"] * (1 - np.exp(-row_inputs[f"{genus}_k"] * prior_years_since_harvest)) ** 2)
			stock_increment = np.nan if pd.isna(stock_current) or pd.isna(stock_previous) else (stock_current - stock_previous)
			is_harvest_year = False if pd.isna(harvest_year) or harvest_year == 0 else bool(np.mod(year, harvest_year) == 0)
			harvest_pulse = standing if is_harvest_year else 0.0
			soil_wo += plantsoil
			soil_w_increment = plantsoil if year < harvest_year else 0.0
			soil_w += soil_w_increment
			estcost_added = running_estcost if is_harvest_year else 0.0
			if is_harvest_year:
				running_estcost = running_estcost + estcost_added
			rows.append({"year": year,
					"standing_no_harvest": standing,
					"increment_no_harvest": increment,
					"accum_increment": increment,
					"growth_rate": growth_rate,
					"harvest_decision_threshold_F1": decision_threshold,
					"harvest_trigger_F1": harvest_trigger,
					"years_since_harvest": years_since_harvest,
					"prior_years_since_harvest": prior_years_since_harvest,
					"stock_increment": stock_increment,
					"harvest_pulse": harvest_pulse,
					"soil_increment_wohrv": plantsoil,
					"soilC_wohrv_running": soil_wo,
					"soil_increment_whrv": soil_w_increment,
					"soilC_whrv_running": soil_w,
					"npv_estcost_addition": estcost_added,
					"npv_estcost_running": running_estcost,})
			prev_standing = standing

		traces[f"{genus}_trace"] = pd.DataFrame(rows)
		traces[f"{genus}_summary"] = to_vertical_table([("harvestyear", row_pre_mac[f"{genus}_harvestyear"]),
				("Y30_d0", row_pre_mac[f"{genus}_Y30_d0"]),
				("accum", row_pre_mac[f"{genus}_accum"]),
				("stock", row_pre_mac[f"{genus}_stock"]),
				("harvested", row_pre_mac[f"{genus}_harvested"]),
				("npv_estcost_final", npv_estcost),
				("bgC_wohrv", row_pre_mac[f"{genus}_bgC_wohrv"]),
				("bgC_whrv", row_pre_mac[f"{genus}_bgC_whrv"]),
				("soilC_wohrv", row_pre_mac[f"{genus}_soilC_wohrv"]),
				("soilC_whrv", row_pre_mac[f"{genus}_soilC_whrv"]),
				("wohrv_totalC", row_pre_mac[f"{genus}_wohrv_totalC"]),
				("whrv_totalC", row_pre_mac[f"{genus}_whrv_totalC"]),])

	return traces

def build_formula_sheet() -> pd.DataFrame:
	return pd.DataFrame([("standing stock", "A * (1 - exp(-k * t))^2"),
			("annual increment", "standing_t - standing_(t-1)"),
			("accumulated increment", "sum annual increment over t=1..Y"),
			("harvest rule F=1", "growth_(t-1) > 0"),
			("stock with harvest", "increment evaluated on mod(t, harvestyear)"),
			("harvested pulse", "standing_t when mod(t, harvestyear) = 0"),
			("nr wohrv totalC", "agbC*nr_accum + bgbC*nr_bgC + soilC*nr_soilC"),
			("genus whrv totalC", "agbC*stock + bgbC*bgC_whrv + soilC*soilC_whrv + harvestC*w*harvested"),
			("cost effectiveness", "(opportunity cost + establishment cost - revenue term) / (3.67 * totalC)"),
			("refortype", "Plantation if plantation carbon-per-dollar > natural regeneration carbon-per-dollar"),
			("mi_*", "Select plantation or natural-regeneration value according to refortype"),
			("BAUrefor", "0.510928*bau_2020_2030 + 0.305911*bau_2030_2040 + 0.183161*bau_2040_2050, else 0.0571795 if missing"),
			("screened buffered additional", "totalC * griscom * buffer * (1 - BAUrefor)"),],
		columns=["item", "formula"],)

def build_comparison_sheet(row_pre_mac: pd.Series, row_post_mac: pd.Series) -> pd.DataFrame:
	lhs_nr = safe_div_scalar(3.67 * row_pre_mac["nr_wohrv_totalC"], row_pre_mac["crop_npv_oppcost"] + row_pre_mac["nr_npv_estcost"])
	lhs_pl_wo = safe_div_scalar(3.67 * row_pre_mac["pl_wohrv_totalC"], row_pre_mac["crop_npv_oppcost"] + row_pre_mac["pl_npv_estcost"])
	lhs_pl_wh0 = safe_div_scalar(3.67 * row_pre_mac["pl_whrv0_totalC"], row_pre_mac["crop_npv_oppcost"] + row_pre_mac["pl_npv_estcost"])
	lhs_pl_wh = safe_div_scalar(3.67 * row_pre_mac["pl_whrv_totalC"], row_pre_mac["crop_npv_oppcost"] + row_pre_mac["pl_npv_estcost"] - row_pre_mac["p"] * row_pre_mac["pl_harvested"])
	lhs_nr_bu = safe_div_scalar(3.67 * row_pre_mac["nr_wohrv_totalC"] * row_pre_mac["nr_buffer"], row_pre_mac["crop_npv_oppcost"] + row_pre_mac["nr_npv_estcost"])
	lhs_pl_wo_bu = safe_div_scalar(3.67 * row_pre_mac["pl_wohrv_totalC"] * row_pre_mac["pl_buffer"], row_pre_mac["crop_npv_oppcost"] + row_pre_mac["pl_npv_estcost"])
	lhs_pl_wh0_bu = safe_div_scalar(3.67 * row_pre_mac["pl_whrv0_totalC"] * row_pre_mac["pl_buffer"], row_pre_mac["crop_npv_oppcost"] + row_pre_mac["pl_npv_estcost"])
	lhs_pl_wh_bu = safe_div_scalar(3.67 * row_pre_mac["pl_whrv_totalC"] * row_pre_mac["pl_buffer"], row_pre_mac["crop_npv_oppcost"] + row_pre_mac["pl_npv_estcost"] - row_pre_mac["p"] * row_pre_mac["pl_harvested"])

	items = [("lhs_nr", lhs_nr),
		("lhs_pl_wo", lhs_pl_wo),
		("lhs_pl_wh0", lhs_pl_wh0),
		("lhs_pl_wh", lhs_pl_wh),
		("lhs_nr_bu", lhs_nr_bu),
		("lhs_pl_wo_bu", lhs_pl_wo_bu),
		("lhs_pl_wh0_bu", lhs_pl_wh0_bu),
		("lhs_pl_wh_bu", lhs_pl_wh_bu),
		("refortype_wohrv", row_pre_mac["refortype_wohrv"]),
		("refortype_whrv0", row_pre_mac["refortype_whrv0"]),
		("refortype_whrv", row_pre_mac["refortype_whrv"]),
		("refortype_wohrv_bu", row_pre_mac["refortype_wohrv_bu"]),
		("refortype_whrv0_bu", row_pre_mac["refortype_whrv0_bu"]),
		("refortype_whrv_bu", row_pre_mac["refortype_whrv_bu"]),
		("rel_wohrv_costeff", safe_log_scalar(safe_div_scalar(row_pre_mac["nr_wohrv_costeff"], row_pre_mac["pl_wohrv_costeff"]))),
		("rel_whrv0_costeff", safe_log_scalar(safe_div_scalar(row_pre_mac["nr_wohrv_costeff"], row_pre_mac["pl_whrv0_costeff"]))),
		("rel_whrv_costeff", safe_log_scalar(safe_div_scalar(row_pre_mac["nr_wohrv_costeff"], row_pre_mac["pl_whrv_costeff"]))),
		("mi_whrv_sGr_bu_ad", row_post_mac["mi_whrv_sGr_bu_ad"]),
		("Genus", row_post_mac["Genus"]),]
	return to_vertical_table(items)

def build_mac_contribution_sheet(row_post_mac: pd.Series) -> pd.DataFrame:
	prices = [(i - 5) * 5 for i in range(1, 26)]
	rows: list[dict[str, Any]] = []
	mac_items = ["nr_wohrv", "pl_wohrv", "pl_whrv0", "pl_whrv", "mi_wohrv", "mi_whrv0", "mi_whrv"]
	screen_items = ["Gr", "Br", "Ba", "Wa"]
	for price in prices:
		row = {"price": price}
		for item in mac_items:
			row[f"MAC_{item}_pixel_contribution"] = row_post_mac[f"{item}_totalC"] if row_pre_threshold(row_post_mac, f"{item}_costeff", price) else 0.0
			row[f"MAC_{item}_sGr_bu_ad_pixel_contribution"] = row_post_mac.get(f"{item}_sGr_bu_ad", np.nan) if row_pre_threshold(row_post_mac, f"{item}_bu_costeff", price) else 0.0
		for item in screen_items:
			row[f"MAC_mi_whrv_s{item}_pixel_contribution"] = row_post_mac[f"mi_whrv_s{item}"] if row_pre_threshold(row_post_mac, "mi_whrv_costeff", price) else 0.0
		row["MAC_mi_whrv_sGr_bu_pixel_contribution"] = row_post_mac["mi_whrv_sGr_bu"] if row_pre_threshold(row_post_mac, "mi_whrv_bu_costeff", price) else 0.0
		row["MAC_mi_whrv_sGr_bu_ad_pixel_contribution"] = row_post_mac["mi_whrv_sGr_bu_ad"] if row_pre_threshold(row_post_mac, "mi_whrv_bu_costeff", price) else 0.0
		rows.append(row)
	return pd.DataFrame(rows)

def row_pre_threshold(row: pd.Series, cost_col: str, price: float) -> bool:
	value = row.get(cost_col, np.nan)
	return False if pd.isna(value) else bool(value <= price)

def write_workbook(output_path: Path, sheets: dict[str, pd.DataFrame]) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
		for sheet_name, dataframe in sheets.items():
			dataframe.to_excel(writer, sheet_name=sheet_name[:31], index=False)

	from openpyxl import load_workbook

	workbook = load_workbook(output_path)
	for worksheet in workbook.worksheets:
		worksheet.freeze_panes = "A2"
		for column_cells in worksheet.columns:
			values = ["" if cell.value is None else str(cell.value) for cell in column_cells]
			width = min(max(len(value) for value in values) + 2, 60)
			worksheet.column_dimensions[column_cells[0].column_letter].width = width
	workbook.save(output_path)

def main() -> None:
	parser = argparse.ArgumentParser(description="Build an Excel workbook showing all calculations for one country pixel.")
	parser.add_argument("--iso", required=True, help="Country code, exact or unique prefix (for example AF or AFG)")
	parser.add_argument("--pixel-id", required=True, type=int, help="Pixel id after the model assigns sequential ids")
	parser.add_argument("--dta-dir", type=Path, default=PROJECT_DIR / "Input")
	parser.add_argument("--output", type=Path, default=PROJECT_DIR / "Output" / "AF_5119.xlsx")
	args = parser.parse_args()

	model = load_model_module()
	iso = resolve_iso(args.iso, model)
	processed = build_country_dataframe(model, iso, args.dta_dir)
	if args.pixel_id not in processed["id"].values:
		raise ValueError(f"Pixel id {args.pixel_id} is not present after cleaning for {iso}.")

	pre_mac = processed.copy()
	row_pre_mac = pre_mac.loc[pre_mac["id"] == args.pixel_id].iloc[0].copy()
	raw_path = args.dta_dir / f"{iso}.dta"
	raw_df = pd.read_stata(raw_path).copy()
	raw_df["id"] = np.arange(1, len(raw_df) + 1)
	row_raw = raw_df.loc[raw_df["id"] == args.pixel_id].iloc[0].copy()

	pre_mac["Y"] = 30
	pre_mac["F"] = 1
	pre_mac["nrsoil"] = 0.415107735
	pre_mac["plantsoil"] = 0.092749069
	row_pre_mac = pre_mac.loc[pre_mac["id"] == args.pixel_id].iloc[0].copy()

	for item in ["nr_wohrv", "pl_wohrv", "pl_whrv0", "pl_whrv", "mi_wohrv", "mi_whrv0", "mi_whrv"]:
		pre_mac[f"{item}_bu_costeff"] = pre_mac[f"{item}_costeff"]
	totals_to_scale = ["nr_wohrv_totalC", "pl_wohrv_totalC", "pl_whrv0_totalC", "pl_whrv_totalC", "mi_wohrv_totalC", "mi_whrv0_totalC", "mi_whrv_totalC", "mi_wohrv_bu_totalC", "mi_whrv0_bu_totalC", "mi_whrv_bu_totalC"]
	for col in totals_to_scale:
		pre_mac[col] = pre_mac[col] * pre_mac["area"]
	model.ensure_columns(pre_mac, ["bau_2020_2030", "bau_2030_2040", "bau_2040_2050"], np.nan)
	pre_mac["BAUrefor"] = pre_mac["bau_2020_2030"] * 0.510928 + pre_mac["bau_2030_2040"] * 0.305911 + pre_mac["bau_2040_2050"] * 0.183161
	pre_mac["BAUrefor"] = pre_mac["BAUrefor"].fillna(0.0571795)
	pre_mac["mi_whrv_sGr"] = pre_mac["mi_whrv_totalC"] * pre_mac["griscom"]
	pre_mac["mi_whrv_sBr"] = pre_mac["mi_whrv_totalC"] * pre_mac["brancalion"]
	pre_mac["mi_whrv_sBa"] = pre_mac["mi_whrv_totalC"] * pre_mac["bastin"]
	pre_mac["mi_whrv_sWa"] = pre_mac["mi_whrv_totalC"] * pre_mac["walker"]
	pre_mac["mi_whrv_sGr_bu"] = pre_mac["mi_whrv_bu_totalC"] * pre_mac["griscom"]
	pre_mac["mi_whrv_sGr_bu_ad"] = (1 - pre_mac["BAUrefor"]) * pre_mac["mi_whrv_sGr_bu"]
	pre_mac["nr_wohrv_sGr_bu_ad"] = pre_mac["nr_wohrv_totalC"] * pre_mac["griscom"] * pre_mac["nr_buffer"] * (1 - pre_mac["BAUrefor"])
	pre_mac["pl_wohrv_sGr_bu_ad"] = pre_mac["pl_wohrv_totalC"] * pre_mac["griscom"] * pre_mac["pl_buffer"] * (1 - pre_mac["BAUrefor"])
	pre_mac["pl_whrv0_sGr_bu_ad"] = pre_mac["pl_whrv0_totalC"] * pre_mac["griscom"] * pre_mac["pl_buffer"] * (1 - pre_mac["BAUrefor"])
	pre_mac["pl_whrv_sGr_bu_ad"] = pre_mac["pl_whrv_totalC"] * pre_mac["griscom"] * pre_mac["pl_buffer"] * (1 - pre_mac["BAUrefor"])
	pre_mac["mi_wohrv_sGr_bu_ad"] = pre_mac["mi_wohrv_bu_totalC"] * pre_mac["griscom"] * (1 - pre_mac["BAUrefor"])
	pre_mac["mi_whrv0_sGr_bu_ad"] = pre_mac["mi_whrv0_bu_totalC"] * pre_mac["griscom"] * (1 - pre_mac["BAUrefor"])
	pre_mac["Genus"] = np.nan
	pre_mac.loc[pre_mac["refortype_whrv_bu"] == "N", "Genus"] = 0
	for genus_type in range(1, 16):
		pre_mac.loc[(pre_mac["refortype_whrv_bu"] == "P") & (pre_mac["type"] == genus_type), "Genus"] = genus_type
	row_post_mac = pre_mac.loc[pre_mac["id"] == args.pixel_id].iloc[0].copy()

	metadata = to_vertical_table([("requested_iso", args.iso.upper()),
			("resolved_iso", iso),
			("pixel_id", args.pixel_id),
			("output_file", str(args.output)),
			("selected_genus", row_pre_mac["genus"]),
			("type", row_pre_mac["type"]),
			("biomes", row_pre_mac["biomes"]),
			("fao_ecoz", row_pre_mac.get("fao_ecoz", np.nan)),
			("area_ha", row_pre_mac["area"]),])
	raw_inputs = to_vertical_table([(name, value) for name, value in row_raw.items()])
	cleaned_inputs = to_vertical_table([(name, row_pre_mac[name])
			for name in ["id", "type", "genus", "continent", "biomes", "fao_ecoz", "area", "griscom", "brancalion", "bastin", "walker",
				"nr_buffer", "pl_buffer", "nr_cost", "ep_cost", "np_cost", "crop_va", "w", "p", "exotic", "native",
				"nr_A", "nr_k", "pinu_A", "pinu_k", "cunn_A", "cunn_k", "euca_A", "euca_k", "brde_A", "brde_k",
				"brev_A", "brev_k", "nede_A", "nede_k", "neev_A", "neev_k", "nr_rootshoot", "pl_rootshoot", "Y", "F", "nrsoil", "plantsoil"]
			if name in row_pre_mac.index])
	pre_mac_outputs = to_vertical_table([(name, value) for name, value in row_pre_mac.items()])
	post_mac_outputs = to_vertical_table([(name, value) for name, value in row_post_mac.items()])
	traces = build_traces(row_pre_mac, row_pre_mac)
	comparison = build_comparison_sheet(row_pre_mac, row_post_mac)
	mac_contrib = build_mac_contribution_sheet(row_post_mac)

	sheets: dict[str, pd.DataFrame] = {"metadata": metadata,
		"raw_inputs": raw_inputs,
		"cleaned_inputs": cleaned_inputs,
		"pre_mac_outputs": pre_mac_outputs,
		"post_mac_outputs": post_mac_outputs,
		"comparison": comparison,
		"mac_contributions": mac_contrib,
		"formulas": build_formula_sheet(),}
	sheets.update(traces)
	write_workbook(args.output, sheets)
	print(f"Created {args.output}")

if __name__ == "__main__":
	main()
