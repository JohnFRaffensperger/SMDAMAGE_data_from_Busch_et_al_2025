# inspect_maps_brb.py | Created 2026-03-23
# Inspects Barbados map outputs, summarizing fields and values to understand dataset contents, anomalies, and structure before downstream processing or export.
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Output"

def main() -> None:
    data_path = OUTPUT_DIR / "maps_ATG.dta"

    if not data_path.exists():
        raise FileNotFoundError(f"Stata file not found: {data_path}")

    dataframe = pd.read_stata(data_path)

    info_buffer = StringIO()
    dataframe.info(buf=info_buffer)

    print(f"File: {data_path.name}")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")
    print()
    print("Dataset structure")
    print("=" * 80)
    print(info_buffer.getvalue())

    print("Column dtypes")
    print("=" * 80)
    print(dataframe.dtypes)
    print()

    print("Sample data (first 5 rows)")
    print("=" * 80)
    print(dataframe.head(5).to_string(index=False))


if __name__ == "__main__":
    main()