# validate_1a.py — Compare output of 1_import (pandas) vs 1a_import (pandas-free) for 40k pixels.
# Imports both modules, runs process_country on the same ISO codes, writes to separate DBs,
# then compares Undiscounted_dta_output and tC_per_h_per_year row by row.
from __future__ import annotations
import importlib.util, sqlite3, math, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DTA_DIR = BASE / "Input"
ORIG_DIR = BASE / "Output" / "Databases_orig"
NEW_DIR  = BASE / "Output" / "Databases_new"

# --- load modules without triggering their __main__ blocks ---
def _load(name: str) -> object:
    path = BASE / "JFR code" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __main__ guard: __name__ != "__main__" here
    return mod

m1  = _load("1_import_Busch2024_to_SMDAMAGE")
m1a = _load("1a_import_Busch2024_to_SMDAMAGE")

# Target: 40 000 pixels.  Pick smallest countries by .dta file size until total >= 40 000.
iso_list_full = m1.iter_smallest_input_isos(DTA_DIR, 999)
chosen_isos: list[str] = []
total_pixels = 0
for iso in iso_list_full:
    if total_pixels >= 40_000: break
    chosen_isos.append(iso)
    # Rough pixel estimate from file size (skip exact count for speed)
    src = next((c for c in [DTA_DIR / iso, DTA_DIR / f"{iso}.dta"] if c.is_file()), None)
    if src: total_pixels += src.stat().st_size // 300  # ~300 bytes / row (rough estimate)

print(f"Selected {len(chosen_isos)} ISOs: {chosen_isos}, estimated ~{total_pixels} pixels")

# --- run original ---
print("\n=== Running ORIGINAL (pandas) ===")
ORIG_DIR.mkdir(parents=True, exist_ok=True)
m1.initialize_smdamage_database(ORIG_DIR)
for iso in chosen_isos:
    m1.process_country(iso=iso, dta_dir=DTA_DIR, macc_dir=ORIG_DIR)

# --- run new ---
print("\n=== Running NEW (pandas-free) ===")
NEW_DIR.mkdir(parents=True, exist_ok=True)
m1a.initialize_smdamage_database(NEW_DIR)
for iso in chosen_isos:
    m1a.process_country(iso=iso, dta_dir=DTA_DIR, macc_dir=NEW_DIR)

# --- compare ---
print("\n=== Comparing outputs ===")
PIXEL_TABLE = m1.SMDAMAGE_PIXEL_TABLE
YEAR_TABLE  = m1.SMDAMAGE_YEAR_TABLE
DB_FILE     = m1.SMDAMAGE_DB_FILE
FLOAT_TOL   = 1e-4  # relative tolerance for float comparison

def _nan_eq(a, b) -> bool:
    if a is None and b is None: return True
    if a is None or b is None: return False
    try:
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb): return True
        if math.isnan(fa) or math.isnan(fb): return False
        if fa == 0.0 and fb == 0.0: return True
        return abs(fa - fb) / (max(abs(fa), abs(fb), 1e-300)) <= FLOAT_TOL
    except (TypeError, ValueError):
        return a == b

orig_conn = sqlite3.connect(ORIG_DIR / DB_FILE)
new_conn  = sqlite3.connect(NEW_DIR / DB_FILE)
orig_conn.row_factory = sqlite3.Row
new_conn.row_factory  = sqlite3.Row

errors: list[str] = []

# --- pixel table ---
orig_pixels = {(r["country"], r["pixel_id"]): r for r in orig_conn.execute(f"SELECT * FROM {PIXEL_TABLE} ORDER BY country, pixel_id")}
new_pixels  = {(r["country"], r["pixel_id"]): r for r in new_conn.execute(f"SELECT * FROM {PIXEL_TABLE} ORDER BY country, pixel_id")}
print(f"Pixel rows: orig={len(orig_pixels)}, new={len(new_pixels)}")
if len(orig_pixels) != len(new_pixels):
    errors.append(f"PIXEL row count mismatch: orig={len(orig_pixels)} new={len(new_pixels)}")

cols_px = [c[1] for c in orig_conn.execute(f"PRAGMA table_info({PIXEL_TABLE})")]
mismatch_px = 0
for key in orig_pixels:
    if key not in new_pixels:
        errors.append(f"Missing pixel in new: {key}")
        continue
    ro, rn = orig_pixels[key], new_pixels[key]
    for col in cols_px:
        if col in ("country", "pixel_id"): continue
        if not _nan_eq(ro[col], rn[col]):
            mismatch_px += 1
            if mismatch_px <= 20:
                errors.append(f"Pixel {key} col={col}: orig={ro[col]} new={rn[col]}")
if mismatch_px: errors.append(f"... total pixel column mismatches: {mismatch_px}")

# --- year table (sample first 40 000 rows for speed) ---
orig_yr = {(r["country"], r["pixel_id"], r["year"]): r["tC_per_ha_per_year"]
           for r in orig_conn.execute(f"SELECT * FROM {YEAR_TABLE} ORDER BY country, pixel_id, year LIMIT 40000")}
new_yr  = {(r["country"], r["pixel_id"], r["year"]): r["tC_per_ha_per_year"]
           for r in new_conn.execute(f"SELECT * FROM {YEAR_TABLE} ORDER BY country, pixel_id, year LIMIT 40000")}
print(f"Year rows (sampled 40k): orig={len(orig_yr)}, new={len(new_yr)}")
mismatch_yr = 0
for key in orig_yr:
    if key not in new_yr:
        mismatch_yr += 1
        if mismatch_yr <= 5: errors.append(f"Missing year row in new: {key}")
        continue
    if not _nan_eq(orig_yr[key], new_yr[key]):
        mismatch_yr += 1
        if mismatch_yr <= 20:
            errors.append(f"Year {key}: orig={orig_yr[key]} new={new_yr[key]}")
if mismatch_yr: errors.append(f"... total year row mismatches: {mismatch_yr}")

orig_conn.close(); new_conn.close()

if errors:
    print("\nMISMATCHES FOUND:")
    for e in errors: print(" ", e)
    sys.exit(1)
else:
    print("\nAll values match within tolerance. PASS.")
