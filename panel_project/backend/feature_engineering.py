"""
backend/feature_engineering.py
================================
Single canonical implementation of ALL feature derivation formulas.
Used by: model training, dataset generation script, and live prediction.

Having ONE source of truth here means:
 - Adding a new feature only requires changing this file.
 - The training dataset and live inference ALWAYS use identical formulas.
 - Unit-testable without importing Streamlit.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from backend.config import (
    SOIL_MAPPING, SOIL_LABELS,
    ELEVATION_BINS, ELEVATION_LABELS,
    REQUIRED_TRAINING_COLS,
)


# ── Individual derivation formulas (stateless, vectorized) ─────────────────

def derive_temperature(elevation: np.ndarray | float) -> np.ndarray | float:
    """Environmental lapse rate: −0.006 °C per metre of elevation.
    Base temperature at sea level: 30 °C.
    """
    return 30.0 - (0.006 * elevation)


def derive_ndvi(rainfall: np.ndarray | float) -> np.ndarray | float:
    """Simplified NDVI proxy from annual rainfall.
    NDVI ranges 0.2 (sparse) → ~1.0 (dense vegetation).
    """
    return np.clip(0.2 + (rainfall / 2500.0), 0.0, 1.0)


def derive_soil_moisture(
    rainfall: np.ndarray | float,
    slope: np.ndarray | float,
) -> np.ndarray | float:
    """Soil moisture proxy: rainfall drives saturation, slope drives runoff."""
    return (rainfall / 2000.0) - (slope / 100.0)


def derive_elevation_zone(elevation: np.ndarray | float) -> np.ndarray | float:
    """Discrete elevation band:
      0 = Lowland   (< 500 m)
      1 = Mid-hill  (500–1500 m)
      2 = Highland  (1500–2500 m)
      3 = Alpine    (> 2500 m)
    """
    bins   = np.array(ELEVATION_BINS)
    labels = np.array(ELEVATION_LABELS)
    # np.searchsorted works on scalars and arrays
    idx = np.searchsorted(bins[1:], elevation, side="left")
    return labels[np.clip(idx, 0, len(labels) - 1)]


# ── DataFrame-level feature derivation ─────────────────────────────────────

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived columns to a copy of *df* if they are missing.

    Safe to call on the training dataset (all 100k rows) or on a
    single-row inference DataFrame.

    Returns a new DataFrame; does not mutate the input.
    """
    df = df.copy()

    if "Temperature" not in df.columns or df["Temperature"].isna().any():
        df["Temperature"] = derive_temperature(df["Elevation"].values)

    if "NDVI" not in df.columns or df["NDVI"].isna().any():
        df["NDVI"] = derive_ndvi(df["Rainfall"].values)

    if "Soil_Moisture" not in df.columns or df["Soil_Moisture"].isna().any():
        df["Soil_Moisture"] = derive_soil_moisture(
            df["Rainfall"].values, df["Slope"].values
        )

    if "Elevation_Zone" not in df.columns or df["Elevation_Zone"].isna().any():
        df["Elevation_Zone"] = derive_elevation_zone(df["Elevation"].values).astype(int)

    return df


def encode_soil(soil_str: str) -> int:
    """Encode soil type string to integer.

    Raises:
        ValueError: if *soil_str* is not a recognised soil type.
    """
    if soil_str not in SOIL_MAPPING:
        valid = list(SOIL_MAPPING.keys())
        raise ValueError(
            f"Unknown soil type '{soil_str}'. Valid options: {valid}"
        )
    return SOIL_MAPPING[soil_str]


def decode_soil(code: int) -> str:
    """Decode integer back to soil type string, or 'Unknown' on failure."""
    return SOIL_LABELS.get(int(code), "Unknown")


def encode_soil_column(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the Soil_Type column in-place (string → int).
    Also writes a Soil_Type_Label column preserving the original string.

    Safely handles:
     - Already-encoded (int) columns.
     - Unknown soil types (replaced with most common class).
    """
    df = df.copy()
    if df["Soil_Type"].dtype == object:
        df["Soil_Type_Label"] = df["Soil_Type"]
        df["Soil_Type"] = (
            df["Soil_Type"]
            .map(SOIL_MAPPING)
            .fillna(SOIL_MAPPING.get("Loamy", 1))   # fallback to Loamy (most common)
            .astype(int)
        )
    else:
        df["Soil_Type"] = df["Soil_Type"].astype(int)
        df["Soil_Type_Label"] = df["Soil_Type"].map(SOIL_LABELS).fillna("Unknown")
    return df


# ── Single-row convenience wrapper (for live prediction UI) ────────────────

def derive_single(elevation: float, slope: float, rainfall: float) -> dict:
    """Compute all derived features for a single set of user inputs.

    Returns a dict with keys: temperature, ndvi, soil_moisture, elevation_zone.
    """
    return {
        "temperature":    round(float(derive_temperature(elevation)), 2),
        "ndvi":           round(float(derive_ndvi(rainfall)), 4),
        "soil_moisture":  round(float(derive_soil_moisture(rainfall, slope)), 4),
        "elevation_zone": int(derive_elevation_zone(elevation)),
    }
