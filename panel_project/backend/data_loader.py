"""
backend/data_loader.py
========================
Safe data loading with schema validation, type coercion, and graceful fallbacks.
All functions return None (not raise) on failure so the UI can degrade gracefully.
"""

from __future__ import annotations
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import (
    REQUIRED_TRAINING_COLS,
    REQUIRED_DISTRICT_COLS,
    SOIL_MAPPING,
)
from backend.feature_engineering import encode_soil_column, derive_features

logger = logging.getLogger(__name__)


# ── Training Dataset ────────────────────────────────────────────────────────

def load_training_data(path: Path) -> pd.DataFrame | None:
    """Load and validate the training dataset.

    Returns:
        Cleaned DataFrame with all required + derived columns, or None on failure.
    """
    path = Path(path)
    if not path.exists():
        logger.error("Training dataset not found: %s", path)
        return None

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        logger.error("Failed to read training CSV (%s): %s", path, exc)
        return None

    # ── Schema validation ───────────────────────────────────────────────────
    missing = [c for c in REQUIRED_TRAINING_COLS if c not in df.columns]
    if missing:
        logger.error(
            "Training CSV is missing required columns: %s  (found: %s)",
            missing, list(df.columns)
        )
        return None

    if df.empty:
        logger.error("Training dataset loaded but is empty.")
        return None

    # ── Type coercion ───────────────────────────────────────────────────────
    numeric_cols = ["Elevation", "Slope", "Rainfall"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows_before = len(df)
    df.dropna(subset=numeric_cols + ["Suitability"], inplace=True)
    if len(df) < rows_before:
        logger.warning(
            "Dropped %d rows with NaN values in core columns.",
            rows_before - len(df)
        )

    # ── Suitability must be binary ───────────────────────────────────────────
    df["Suitability"] = df["Suitability"].astype(int)
    unique_vals = df["Suitability"].unique()
    if not set(unique_vals).issubset({0, 1}):
        logger.error(
            "Suitability column contains unexpected values: %s  (expected 0 or 1)",
            unique_vals
        )
        return None

    # Check for class imbalance extremes (< 2 samples in a class → stratify breaks)
    vc = df["Suitability"].value_counts()
    for cls, cnt in vc.items():
        if cnt < 10:
            logger.warning("Class %d has only %d samples. Model may be unreliable.", cls, cnt)

    # ── Encode soil & derive features ───────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = encode_soil_column(df)
        df = derive_features(df)

    logger.info(
        "Training data loaded: %d rows, %d cols, %d suitable / %d not suitable",
        len(df), len(df.columns),
        int(vc.get(1, 0)), int(vc.get(0, 0))
    )
    return df


# ── Indian Districts Summary ────────────────────────────────────────────────

def load_district_data(path: Path) -> pd.DataFrame | None:
    """Load and validate the Indian districts summary CSV.

    Returns:
        Cleaned DataFrame, or None on failure.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("District data not found: %s", path)
        return None

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        logger.error("Failed to read district CSV (%s): %s", path, exc)
        return None

    # ── Schema ───────────────────────────────────────────────────────────────
    missing = [c for c in REQUIRED_DISTRICT_COLS if c not in df.columns]
    if missing:
        logger.error(
            "District CSV missing required columns: %s  (found: %s)",
            missing, list(df.columns)
        )
        return None

    if df.empty:
        logger.warning("District CSV loaded but is empty.")
        return None

    # ── Type coercion ─────────────────────────────────────────────────────────
    df["elevation"]       = pd.to_numeric(df["elevation"], errors="coerce").fillna(0.0)
    df["annual_rainfall"] = pd.to_numeric(df["annual_rainfall"], errors="coerce").fillna(0.0)

    if "avg_temp" in df.columns:
        df["avg_temp"] = pd.to_numeric(df["avg_temp"], errors="coerce")
    else:
        df["avg_temp"] = derive_features(
            pd.DataFrame({"Elevation": df["elevation"], "Slope": 0, "Rainfall": df["annual_rainfall"]})
        )["Temperature"].values

    if "latitude" not in df.columns:
        df["latitude"]  = np.nan
    if "longitude" not in df.columns:
        df["longitude"] = np.nan

    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Normalise string columns
    df["state"]    = df["state"].astype(str).str.strip().str.title()
    df["district"] = df["district"].astype(str).str.strip().str.title()

    logger.info("District data loaded: %d districts across states.", len(df))
    return df


# ── Data Quality Report ─────────────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a data quality summary dict for display in the UI.

    Checks:
      - Total rows / cols
      - Missing value counts per column
      - Outlier counts (IQR method) for numeric columns
      - Class distribution
      - Duplicate row count
    """
    report: dict = {}

    report["n_rows"]   = len(df)
    report["n_cols"]   = len(df.columns)
    report["n_dupes"]  = int(df.duplicated().sum())

    # Missing values
    missing = df.isnull().sum()
    report["missing"] = {
        col: int(cnt) for col, cnt in missing.items() if cnt > 0
    }
    report["total_missing"] = int(missing.sum())

    # Outliers via IQR on numeric columns
    num_df = df.select_dtypes(include=[np.number])
    outlier_counts: dict[str, int] = {}
    for col in num_df.columns:
        q1  = num_df[col].quantile(0.25)
        q3  = num_df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((num_df[col] < lo) | (num_df[col] > hi)).sum())
        if n_out > 0:
            outlier_counts[col] = n_out
    report["outliers"] = outlier_counts

    # Class balance
    if "Suitability" in df.columns:
        vc = df["Suitability"].value_counts()
        report["class_balance"] = {int(k): int(v) for k, v in vc.items()}
        report["class_ratio"]   = round(
            float(vc.get(1, 0)) / max(float(vc.get(0, 1)), 1.0), 3
        )

    return report
