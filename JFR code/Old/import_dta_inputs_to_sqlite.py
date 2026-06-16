# import_dta_inputs_to_sqlite.py | Created 2026-04-08
# Imports Busch input .dta files into SQLite tables, standardizing schema and enabling faster query-based validation and preprocessing work across countries.
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_DIR / "Input"
OUTPUT_DIR = PROJECT_DIR / "Output"
DATABASE_FILE = OUTPUT_DIR / "Busch2024_inputs.sqlite"
PRIMARY_TABLE_NAME = "Busch2024_inputs"
DESCRIPTION_TABLE_NAME = "Description"

LONG_DESCRIPTIONS = {
	"source_iso": "ISO3 country code inferred from the input filename.",
	"source_file": "Original .dta filename imported into this row.",
	"x": "Pixel centroid x coordinate from the source dataset.",
	"y": "Pixel centroid y coordinate from the source dataset.",
	"nr_A": "Natural regeneration Chapman-Richards asymptote parameter A.",
	"nr_k": "Natural regeneration Chapman-Richards growth rate parameter k.",
	"pinu_A": "Chapman-Richards asymptote parameter A for Pinus plantations.",
	"pinu_k": "Chapman-Richards growth rate parameter k for Pinus plantations.",
	"cunn_A": "Chapman-Richards asymptote parameter A for Cunninghamia plantations.",
	"cunn_k": "Chapman-Richards growth rate parameter k for Cunninghamia plantations.",
	"euca_A": "Chapman-Richards asymptote parameter A for Eucalyptus plantations.",
	"euca_k": "Chapman-Richards growth rate parameter k for Eucalyptus plantations.",
	"brde_A": "Chapman-Richards asymptote parameter A for broadleaf deciduous plantations.",
	"brde_k": "Chapman-Richards growth rate parameter k for broadleaf deciduous plantations.",
	"brev_A": "Chapman-Richards asymptote parameter A for broadleaf evergreen plantations.",
	"brev_k": "Chapman-Richards growth rate parameter k for broadleaf evergreen plantations.",
	"nede_A": "Chapman-Richards asymptote parameter A for needleleaf deciduous plantations.",
	"nede_k": "Chapman-Richards growth rate parameter k for needleleaf deciduous plantations.",
	"neev_A": "Chapman-Richards asymptote parameter A for needleleaf evergreen plantations.",
	"neev_k": "Chapman-Richards growth rate parameter k for needleleaf evergreen plantations.",
	"biomes": "Biome code used to identify ecological zone exclusions and filters.",
	"type": "Plantation type code used to map each pixel to a plantation genus.",
	"griscom": "Binary reforestation suitability screen based on the Griscom et al. layer.",
	"brancalion": "Binary reforestation suitability screen based on the Brancalion layer.",
	"bastin": "Binary reforestation suitability screen based on the Bastin layer.",
	"walker": "Binary reforestation suitability screen based on the Walker layer.",
	"nr_buffer": "Natural-regeneration buffer variable from the source dataset.",
	"pl_buffer": "Plantation buffer variable from the source dataset.",
	"bau_2020_2030": "Business-as-usual reforestation measure for 2020-2030 from the source dataset.",
	"bau_2030_2040": "Business-as-usual reforestation measure for 2030-2040 from the source dataset.",
	"bau_2040_2050": "Business-as-usual reforestation measure for 2040-2050 from the source dataset.",
	"crop_va": "Annual crop value per hectare used as the opportunity-cost input.",
	"nr_cost": "Natural regeneration establishment or program cost per hectare.",
	"np_cost": "Native plantation establishment cost per hectare.",
	"ep_cost": "Exotic plantation establishment cost per hectare.",
	"fao_ecoz": "FAO ecological zone code used in root-to-shoot rules.",
	"area_m2": "Pixel area in square meters.",
	"pxl_id": "Pixel identifier from the source dataset.",
}

def read_variable_labels(data_path: Path) -> dict[str, str]:
	reader = pd.read_stata(data_path, iterator=True)
	return reader.variable_labels()

def list_input_files(input_dir: Path) -> list[Path]:
	data_files = sorted(path for path in input_dir.glob("*.dta") if path.is_file())
	if not data_files: raise FileNotFoundError(f"No .dta files found in {input_dir}")
	return data_files

def ensure_single_schema(data_files: list[Path]) -> tuple[str, ...]:
	reference_columns = tuple(read_variable_labels(data_files[0]).keys())
	mismatches = [path.name for path in data_files[1:] if tuple(read_variable_labels(path).keys()) != reference_columns]
	if mismatches: raise ValueError(f"Input files do not share one schema. First mismatches: {mismatches[:10]}")
	return reference_columns

def apply_processing_filters(dataframe: pd.DataFrame) -> pd.DataFrame:
	return dataframe[(dataframe["biomes"] != 13) & (dataframe["biomes"] != 14) & dataframe["nr_A"].notna() & dataframe["type"].notna()].copy()

def import_inputs(connection: sqlite3.Connection, table_name: str, data_files: list[Path]) -> int:
	total_rows = 0
	for file_index, data_file in enumerate(data_files, start=1):
		dataframe = pd.read_stata(data_file, convert_categoricals=False)
		input_rows = len(dataframe)
		dataframe = apply_processing_filters(dataframe)
		dataframe.insert(0, "source_file", data_file.name)
		dataframe.insert(0, "source_iso", data_file.stem.upper())
		dataframe.to_sql(table_name, connection, if_exists="replace" if file_index == 1 else "append", index=False)
		connection.commit()
		total_rows += len(dataframe)
		print(f"[OK] Imported {data_file.name}: kept {len(dataframe)} of {input_rows} row(s)", flush=True)
	return total_rows

def build_description_rows(columns: tuple[str, ...], variable_labels: dict[str, str]) -> list[tuple[str, str]]:
	ordered_columns = ("source_iso", "source_file", *columns)
	rows: list[tuple[str, str]] = []
	for column in ordered_columns:
		label = (variable_labels.get(column) or "").strip()
		long_description = LONG_DESCRIPTIONS.get(column) or label or ""
		rows.append((column, long_description))
	return rows

def write_description_table(connection: sqlite3.Connection, rows: list[tuple[str, str]]) -> None:
	connection.execute(f"DROP TABLE IF EXISTS {DESCRIPTION_TABLE_NAME}")
	connection.execute(f"CREATE TABLE {DESCRIPTION_TABLE_NAME} (variable_name TEXT PRIMARY KEY, long_description TEXT NOT NULL)")
	connection.executemany(f"INSERT INTO {DESCRIPTION_TABLE_NAME} (variable_name, long_description) VALUES (?, ?)", rows)

def create_indexes(connection: sqlite3.Connection, table_name: str) -> None:
	index_sql = [
		f"CREATE INDEX IF NOT EXISTS idx_{table_name}_source_file ON {table_name} (source_file)",
		f"CREATE INDEX IF NOT EXISTS idx_{table_name}_source_iso ON {table_name} (source_iso)",
		f"CREATE INDEX IF NOT EXISTS idx_{table_name}_source_iso_pxl_id ON {table_name} (source_iso, pxl_id)",
		f"CREATE INDEX IF NOT EXISTS idx_{table_name}_type ON {table_name} (type)",
		f"CREATE INDEX IF NOT EXISTS idx_{table_name}_biomes ON {table_name} (biomes)",
		f"CREATE INDEX IF NOT EXISTS idx_{table_name}_fao_ecoz ON {table_name} (fao_ecoz)",
	]
	for statement in index_sql: connection.execute(statement)

def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	data_files = list_input_files(INPUT_DIR)
	columns = ensure_single_schema(data_files)
	variable_labels = read_variable_labels(data_files[0])
	description_rows = build_description_rows(columns, variable_labels)
	with sqlite3.connect(DATABASE_FILE) as connection:
		connection.execute("PRAGMA journal_mode=WAL")
		connection.execute("PRAGMA synchronous=OFF")
		connection.execute("PRAGMA temp_store=MEMORY")
		connection.execute(f"DROP TABLE IF EXISTS {PRIMARY_TABLE_NAME}")
		total_rows = import_inputs(connection, PRIMARY_TABLE_NAME, data_files)
		write_description_table(connection, description_rows)
		create_indexes(connection, PRIMARY_TABLE_NAME)
		connection.commit()

	print(f"Created database: {DATABASE_FILE}")
	print(f"Imported files into {PRIMARY_TABLE_NAME}: {len(data_files)}")
	print(f"- {PRIMARY_TABLE_NAME}: {total_rows} row(s)")
	print(f"- {DESCRIPTION_TABLE_NAME}: {len(description_rows)} row(s)")

if __name__ == "__main__":
	main()