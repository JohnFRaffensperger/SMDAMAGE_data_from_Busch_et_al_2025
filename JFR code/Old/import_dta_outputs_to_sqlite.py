# import_dta_outputs_to_sqlite.py | Created 2026-03-23
# Imports Busch output .dta files into SQLite, preserving mapped result fields for ad hoc analysis, validation, and reporting tasks locally.
# This script reads the maps_*.dta files and outputs sqlite database Busch2024_dta_outputs.sqlite.
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "Output"
REFERENCE_FILE = OUTPUT_DIR / "maps_ATG.dta"
DATABASE_FILE = OUTPUT_DIR / "Busch2024_dta_outputs.sqlite"
PRIMARY_TABLE_NAME = "Busch2024_dta_outputs"

def read_columns(data_path: Path) -> tuple[str, ...]:
	reader = pd.read_stata(data_path, iterator=True)
	return tuple(reader.variable_labels().keys())

def build_schema_groups(data_files: list[Path]) -> dict[tuple[str, ...], list[Path]]:
	schema_groups: dict[tuple[str, ...], list[Path]] = {}
	for data_file in data_files:
		columns = read_columns(data_file)
		schema_groups.setdefault(columns, []).append(data_file)
	return schema_groups

def import_group(connection: sqlite3.Connection, table_name: str, data_files: list[Path], ) -> int:
	total_rows = 0

	for file_index, data_file in enumerate(sorted(data_files), start=1):
		dataframe = pd.read_stata(data_file, convert_categoricals=False)
		dataframe.insert(0, "source_file", data_file.name)
		dataframe.to_sql(table_name, connection, if_exists="replace" if file_index == 1 else "append", index=False,)
		total_rows += len(dataframe)

	return total_rows

def main() -> None:
	if not REFERENCE_FILE.exists(): raise FileNotFoundError(f"Reference file not found: {REFERENCE_FILE}")

	data_files = sorted(OUTPUT_DIR.glob("maps_*.dta"))
	if not data_files: raise FileNotFoundError(f"No .dta files found in {OUTPUT_DIR}")

	reference_columns = read_columns(REFERENCE_FILE)
	matching_files = [data_file for data_file in data_files if read_columns(data_file) == reference_columns]
	skipped_files = [data_file for data_file in data_files if data_file not in matching_files]
	if not matching_files: raise FileNotFoundError("No .dta files matched the reference schema.")

	with sqlite3.connect(DATABASE_FILE) as connection:
		connection.execute(f"DROP TABLE IF EXISTS {PRIMARY_TABLE_NAME}")
		total_rows = import_group(connection, PRIMARY_TABLE_NAME, matching_files)

	print(f"Created database: {DATABASE_FILE.name}")
	print(f"Imported files into {PRIMARY_TABLE_NAME}: {len(matching_files)}")
	print(f"Skipped files with non-matching schema: {len(skipped_files)}")
	print(f"- {PRIMARY_TABLE_NAME}: {total_rows} row(s)")
	if skipped_files:
		print("Skipped:")
		for data_file in skipped_files:
			print(f"- {data_file.name}")

if __name__ == "__main__": main()
