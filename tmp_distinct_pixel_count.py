import pandas as pd
import numpy as np
from pathlib import Path

base = Path(r"C:\Users\johnr\Documents\Work documents\2 Research\Global warming\Numerical Simulation\Busch2024")
input_dir = base / "Input"

GENUS_ORDER = ["pinu", "cunn", "euca", "brde", "brev", "nede", "neev"]
GENUS_BY_TYPE = {1: "neev", 2: "brev", 3: "brde", 4: "neev", 5: "cunn", 6: "euca", 7: "nede", 8: "neev", 9: "pinu", 10: "brde", 11: "neev", 12: "brde", 13: "brev", 14: "brde", 15: "neev"}
AFRI = {"DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COM", "COG", "CIV", "COD", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY", "MDG", "MWI", "MDV", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA", "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "TUN", "UGA", "ESH", "ZMB", "ZWE"}
AMER = {"ATG", "ARG", "BRB", "BLZ", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DMA", "DOM", "ECU", "SLV", "GUF", "GRD", "GTM", "GUY", "HTI", "HND", "JAM", "MEX", "NIC", "PAN", "PRY", "PER", "KNA", "LCA", "VCT", "SUR", "TTO", "URY", "VEN"}
ASIA = {"AFG", "BGD", "BTN", "KHM", "CHN", "COK", "PRK", "FJI", "IND", "IDN", "IRN", "IRQ", "JOR", "KAZ", "KIR", "KGZ", "LAO", "LBN", "MYS", "MHL", "FSM", "MNG", "MMR", "NRU", "NPL", "OMN", "PAK", "PLW", "PSE", "PNG", "PHL", "WSM", "SLB", "LKA", "SYR", "TJK", "THA", "TLS", "TKL", "TON", "TKM", "TUV", "UZB", "VUT", "VNM", "YEM", "ARM", "AZE", "GEO", "TUR"}
NR_ROOTSHOOT_RULES = {(11, "AFRI"): 0.825, (11, "AMER"): 0.221, (11, "ASIA"): 0.207, (12, "AFRI"): 0.232, (12, "AMER"): 0.2845, (12, "ASIA"): 0.323, (13, "AFRI"): 0.332, (13, "AMER"): 0.334, (13, "ASIA"): 0.440, (16, "AMER"): 0.348, (16, "ASIA"): 0.322, (21, "AFRI"): 0.232, (21, "AMER"): 0.175, (21, "ASIA"): 0.230, (22, "AMER"): 0.336, (22, "ASIA"): 0.440, (23, "AMER"): 1.338, (23, "ASIA"): 1.338}
PL_ROOTSHOOT_RULES = {(11, "AMER"): 0.170, (11, "ASIA"): 0.325, (16, "AMER"): 2.158, (23, "ASIA"): 2.158}
WOOD_PRODUCT_STORAGE_MAP = {0.188: {"ATG", "BLZ", "BRB", "CRI", "CUB", "DMA", "DOM", "GRD", "GTM", "HND", "HTI", "JAM", "KNA", "LCA", "MEX", "NIC", "PAN", "SLV", "TTO", "VCT"}, 0.268: {"ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUF", "GUY", "PER", "PRY", "SUR", "URY", "VEN"}, 0.355: {"AFG", "ARM", "AZE", "BGD", "BTN", "CHN", "GEO", "IDN", "IND", "IRN", "IRQ", "JOR", "KAZ", "KGZ", "KHM", "LAO", "LBN", "LKA", "MMR", "MNG", "MYS", "NPL", "OMN", "PAK", "PHL", "PNG", "PRK", "PSE", "SYR", "THA", "TJK", "TKM", "TLS", "TUR", "UZB", "VNM", "YEM"}, 0.392: {"AGO", "BDI", "BEN", "BFA", "BWA", "CAF", "CIV", "CMR", "COD", "COG", "COM", "CPV", "DJI", "DZA", "EGY", "ERI", "ESH", "ETH", "GAB", "GHA", "GIN", "GMB", "GNB", "GNQ", "KEN", "LBR", "LBY", "LSO", "MAR", "MDG", "MDV", "MLI", "MOZ", "MRT", "MUS", "MWI", "NAM", "NER", "NGA", "RWA", "SDN", "SEN", "SLE", "SOM", "SSD", "STP", "SWZ", "SYC", "TCD", "TGO", "TUN", "TZA", "UGA", "ZAF", "ZMB", "ZWE"}, 0.415: {"COK", "FJI", "FSM", "KIR", "MHL", "NRU", "PLW", "SLB", "TKL", "TON", "TUV", "VUT", "WSM"}}
WOOD_PRODUCT_REVENUE_MAP = {52.65: {"ATG", "BLZ", "BRB", "CRI", "CUB", "DMA", "DOM", "GRD", "GTM", "HND", "HTI", "JAM", "KNA", "LCA", "NIC", "PAN", "SLV", "TTO", "VCT"}, 59.73: {"MEX"}, 61.04: {"BOL", "BRA", "COL", "ECU", "GUF", "GUY", "PER", "SUR", "VEN"}, 67.76: {"BGD", "IDN", "KHM", "LAO", "LKA", "MMR", "MYS", "OMN", "PHL", "PNG", "THA", "TLS", "VNM", "YEM"}, 69.26: {"PRY"}, 71.24: {"AGO", "BDI", "BEN", "BFA", "CAF", "CIV", "CMR", "COD", "COG", "COM", "CPV", "DJI", "ERI", "ETH", "GAB", "GHA", "GIN", "GMB", "GNB", "GNQ", "KEN", "LBR", "MDG", "MDV", "MLI", "MOZ", "MRT", "MUS", "MWI", "NER", "NGA", "RWA", "SDN", "SEN", "SLE", "SOM", "SSD", "STP", "SYC", "TCD", "TGO", "TZA", "UGA", "ZMB", "ZWE"}, 76.08: {"COK", "FJI", "FSM", "KIR", "MHL", "NRU", "PLW", "SLB", "TKL", "TON", "TUV", "VUT", "WSM"}, 76.88: {"IND"}, 80.003: {"ARG", "CHL", "URY"}, 80.83: {"BWA", "NAM"}, 88.84: {"AFG", "ARM", "AZE", "BTN", "CHN", "GEO", "IRN", "IRQ", "JOR", "KAZ", "KGZ", "LBN", "MNG", "NPL", "PAK", "PRK", "PSE", "SYR", "TJK", "TKM", "TUR", "UZB"}, 93.40: {"DZA", "EGY", "ESH", "LBY", "LSO", "MAR", "SWZ", "TUN", "ZAF"}}
EXOTIC_SHARE_MAP = {0.74: {"AGO", "BWA", "COM", "DJI", "ERI", "SWZ", "ETH", "KEN", "LSO", "MDG", "MWI", "MDV", "MUS", "MOZ", "NAM", "SYC", "SOM", "ZAF", "SSD", "SDN", "TZA", "UGA", "ZMB", "ZWE", "BEN", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COG", "CIV", "COD", "GNQ", "GAB", "GMB", "GHA", "GIN", "GNB", "LBR", "MLI", "MRT", "NER", "NGA", "RWA", "STP", "SEN", "SLE", "TGO"}, 0.50: {"DZA", "EGY", "LBY", "MAR", "TUN", "ESH"}, 0.31: {"CHN", "PRK", "MNG"}, 0.40: {"BGD", "BTN", "KHM", "IND", "IDN", "LAO", "MYS", "MMR", "NPL", "PAK", "PHL", "LKA", "THA", "TLS", "VNM"}, 0.05: {"AFG", "IRN", "IRQ", "JOR", "KAZ", "KGZ", "LBN", "OMN", "PSE", "SYR", "TJK", "TKM", "UZB", "YEM", "ARM", "AZE", "GEO", "TUR"}, 0.78: {"COK", "FJI", "KIR", "MHL", "FSM", "NRU", "PLW", "PNG", "WSM", "SLB", "TKL", "TON", "TUV", "VUT"}, 0.32: {"ATG", "BRB", "CUB", "DMA", "DOM", "GRD", "HTI", "KNA", "LCA", "VCT", "TTO"}, 0.18: {"BLZ", "CRI", "SLV", "GTM", "HND", "JAM", "MEX", "NIC", "PAN"}, 0.97: {"ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUF", "GUY", "PRY", "PER", "SUR", "URY", "VEN"}}

def get_continent(iso: str) -> str:
    return "AFRI" if iso in AFRI else "AMER" if iso in AMER else "ASIA" if iso in ASIA else ""

def mapped_value(mapping, iso, default=0.0):
    for value, isos in mapping.items():
        if iso in isos:
            return float(value)
    return float(default)

needed = [
    "biomes", "nr_A", "nr_k", "type", "fao_ecoz", "crop_va", "nr_cost", "ep_cost", "np_cost",
    "pinu_A", "pinu_k", "cunn_A", "cunn_k", "euca_A", "euca_k", "brde_A", "brde_k", "brev_A", "brev_k", "nede_A", "nede_k", "neev_A", "neev_k",
    "x", "y", "area", "area_m2", "area_ha"
]

files = sorted([p for p in input_dir.glob("*.dta") if p.is_file()])
rows_raw = 0
rows_post_filter = 0
excluded_found = set()
full_hashes = set()
lean_hashes = set()

for fp in files:
    iso = fp.stem.upper()
    try:
        df = pd.read_stata(fp, columns=needed)
    except Exception:
        tmp = pd.read_stata(fp)
        keep = [c for c in needed if c in tmp.columns]
        df = tmp[keep]

    for col in ["x", "y", "area", "area_m2", "area_ha"]:
        if col in df.columns:
            excluded_found.add(col)
    rows_raw += len(df)

    for col in ["biomes", "nr_A", "type"]:
        if col not in df.columns:
            df[col] = 0.0
    for col in ["nr_cost", "ep_cost", "np_cost", "crop_va"]:
        if col not in df.columns:
            df[col] = 0.0
    if "fao_ecoz" not in df.columns:
        df["fao_ecoz"] = np.nan

    df = df[(df["biomes"] != 13) & (df["biomes"] != 14) & df["nr_A"].notna() & df["type"].notna()].copy()
    rows_post_filter += len(df)

    for c in ["nr_cost", "ep_cost", "np_cost"]:
        df[c] = df[c] * 1.1577
    for c in ["nr_A", "nr_k"] + [f"{g}_{s}" for g in GENUS_ORDER for s in ["A", "k"]]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = 0

    df["genus"] = df["type"].map(GENUS_BY_TYPE).fillna("")
    df["genus_A"] = np.nan
    df["genus_k"] = np.nan
    for g in GENUS_ORDER:
        ac = f"{g}_A"
        kc = f"{g}_k"
        if ac in df.columns and kc in df.columns:
            m = df["genus"] == g
            df.loc[m, "genus_A"] = df.loc[m, ac]
            df.loc[m, "genus_k"] = df.loc[m, kc]

    continent = get_continent(iso)
    df["nr_rootshoot"] = 0.26
    df["pl_rootshoot"] = 0.26
    is_brde_nede = df["genus"].isin(["brde", "nede"])

    for (eco, cont), val in NR_ROOTSHOOT_RULES.items():
        if cont == continent:
            df.loc[df["fao_ecoz"] == eco, "nr_rootshoot"] = val
    for (eco, cont), val in PL_ROOTSHOOT_RULES.items():
        if cont == continent:
            df.loc[df["fao_ecoz"] == eco, "pl_rootshoot"] = val

    if continent == "AMER":
        df.loc[(df["fao_ecoz"] == 31) & is_brde_nede, "nr_rootshoot"] = 0.466
        df.loc[(df["fao_ecoz"] == 31) & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "nr_rootshoot"] = 0.337
        df.loc[(df["fao_ecoz"] == 32) & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "nr_rootshoot"] = 0.237
        df.loc[(df["fao_ecoz"] == 31) & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "pl_rootshoot"] = 0.203
        df.loc[(df["fao_ecoz"] == 32) & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "pl_rootshoot"] = 0.237
    elif continent == "ASIA":
        df.loc[(df["fao_ecoz"] == 31) & is_brde_nede, "nr_rootshoot"] = 0.225
        df.loc[(df["fao_ecoz"] == 31) & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "nr_rootshoot"] = 0.243
        df.loc[(df["fao_ecoz"] == 32) & is_brde_nede, "nr_rootshoot"] = 0.225
        df.loc[(df["fao_ecoz"] == 32) & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "nr_rootshoot"] = 0.243
        df.loc[(df["fao_ecoz"] == 31) & is_brde_nede, "pl_rootshoot"] = 0.307
        df.loc[(df["fao_ecoz"] == 31) & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "pl_rootshoot"] = 0.224
        df.loc[(df["fao_ecoz"] == 32) & is_brde_nede, "pl_rootshoot"] = 0.307
        df.loc[(df["fao_ecoz"] == 32) & df["genus"].isin(["pinu", "cunn", "brev", "neev"]), "pl_rootshoot"] = 0.224

    w = mapped_value(WOOD_PRODUCT_STORAGE_MAP, iso)
    p = mapped_value(WOOD_PRODUCT_REVENUE_MAP, iso)
    exotic = mapped_value(EXOTIC_SHARE_MAP, iso)
    native = 1.0 - exotic

    full_sig = pd.DataFrame({
        "genus": df["genus"], "genus_A": df["genus_A"], "genus_k": df["genus_k"], "nr_A": df["nr_A"], "nr_k": df["nr_k"],
        "nr_rootshoot": df["nr_rootshoot"], "pl_rootshoot": df["pl_rootshoot"], "crop_va": df["crop_va"],
        "nr_cost_2020": df["nr_cost"], "ep_cost_2020": df["ep_cost"], "np_cost_2020": df["np_cost"],
        "w": w, "p": p, "exotic": exotic, "native": native,
    })
    full_hashes.update(pd.util.hash_pandas_object(full_sig, index=False).to_numpy().tolist())

    lean_sig = pd.DataFrame({
        "genus": df["genus"], "genus_A": df["genus_A"], "genus_k": df["genus_k"], "nr_A": df["nr_A"], "nr_k": df["nr_k"],
        "nr_rootshoot": df["nr_rootshoot"], "pl_rootshoot": df["pl_rootshoot"], "crop_va": df["crop_va"],
        "nr_cost_2020": df["nr_cost"], "pl_init_cost_2020": exotic*df["ep_cost"] + native*df["np_cost"],
        "w": w, "p": p,
    })
    lean_hashes.update(pd.util.hash_pandas_object(lean_sig, index=False).to_numpy().tolist())

print("FILES", len(files))
print("ROWS_TOTAL_RAW", rows_raw)
print("ROWS_POST_FILTER", rows_post_filter)
print("EXCLUDED_COLUMNS_FOUND", sorted(excluded_found))
print("DISTINCT_FULL_SIGNATURE_EXCL_X_Y_AREA", len(full_hashes))
print("DISTINCT_LEAN_SIGNATURE_EXCL_X_Y_AREA", len(lean_hashes))
