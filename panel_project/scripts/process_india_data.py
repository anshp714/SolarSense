"""
scripts/process_india_data.py
==============================
Aggregate the raw 64MB India weather/rainfall Excel file into a lightweight
district-level summary CSV used by the Streamlit frontend.

Usage:
    python scripts/process_india_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.config import RAW_EXCEL_PATH, DISTRICT_DATA_PATH
except ImportError as e:
    print(f"ERROR: Could not import backend config: {e}")
    print("Run from project root: python scripts/process_india_data.py")
    sys.exit(1)


def main():
    # ── Check source file ────────────────────────────────────────────────────
    if not RAW_EXCEL_PATH.exists():
        print(f"ERROR: Source Excel not found: {RAW_EXCEL_PATH}")
        print("Please place 'india_weather_rainfall_data.xlsx' in the data/ directory.")
        sys.exit(1)

    # ── Check openpyxl available ─────────────────────────────────────────────
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("ERROR: openpyxl is required to read Excel files.")
        print("Install it with: pip install openpyxl")
        sys.exit(1)

    print(f"Loading Excel dataset from {RAW_EXCEL_PATH} …")
    print("(This may take 30–90 seconds for a 64 MB file)")

    try:
        df = pd.read_excel(RAW_EXCEL_PATH, engine="openpyxl")
    except Exception as exc:
        print(f"ERROR reading Excel file: {exc}")
        sys.exit(1)

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")
    print(f"Columns found: {list(df.columns)}")

    # ── Validate expected columns ────────────────────────────────────────────
    expected = {"state", "district"}
    optional = {"elevation", "rainfall", "avg_temp", "latitude", "longitude"}
    col_set  = set(df.columns)

    missing_required = expected - col_set
    if missing_required:
        print(f"ERROR: Required columns missing: {missing_required}")
        sys.exit(1)

    present_optional = optional & col_set
    missing_optional = optional - col_set
    if missing_optional:
        print(f"WARNING: Optional columns not found (will be estimated): {missing_optional}")

    # ── Aggregation ──────────────────────────────────────────────────────────
    agg_dict = {}
    if "elevation" in col_set:
        agg_dict["elevation"] = "mean"
    if "rainfall" in col_set:
        agg_dict["rainfall"] = "mean"
    if "avg_temp" in col_set:
        agg_dict["avg_temp"] = "mean"
    if "latitude" in col_set:
        agg_dict["latitude"] = "mean"
    if "longitude" in col_set:
        agg_dict["longitude"] = "mean"

    print("Aggregating by state + district…")
    try:
        summary = df.groupby(["state", "district"]).agg(agg_dict).reset_index()
    except Exception as exc:
        print(f"ERROR during aggregation: {exc}")
        sys.exit(1)

    # ── Convert daily → annual rainfall ──────────────────────────────────────
    if "rainfall" in summary.columns:
        summary["annual_rainfall"] = (summary["rainfall"] * 365).round(0)
        summary.drop(columns=["rainfall"], inplace=True)
    elif "annual_rainfall" not in summary.columns:
        print("WARNING: No rainfall column found — setting annual_rainfall to 0.")
        summary["annual_rainfall"] = 0.0

    # ── Fill missing opt cols with estimates ──────────────────────────────────
    if "elevation" not in summary.columns:
        summary["elevation"] = 200.0  # sea-level default
    if "avg_temp" not in summary.columns:
        summary["avg_temp"] = (30.0 - 0.006 * summary["elevation"]).round(1)
    if "latitude" not in summary.columns:
        summary["latitude"] = np.nan
    if "longitude" not in summary.columns:
        summary["longitude"] = np.nan

    # ── Round & sort ──────────────────────────────────────────────────────────
    summary["elevation"]       = summary["elevation"].round(0)
    summary["annual_rainfall"] = summary["annual_rainfall"].round(0)
    summary["avg_temp"]        = summary["avg_temp"].round(1)
    summary["latitude"]        = summary["latitude"].round(4)
    summary["longitude"]       = summary["longitude"].round(4)

    summary = summary.sort_values(["state", "district"]).reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    DISTRICT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DISTRICT_DATA_PATH, index=False)

    print(f"\n✅ Summary saved → {DISTRICT_DATA_PATH}")
    print(f"   Total districts: {len(summary)}")
    print(f"   Columns: {list(summary.columns)}")
    print(summary.head())


if __name__ == "__main__":
    main()
