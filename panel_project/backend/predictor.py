"""
backend/predictor.py
=====================
High-level prediction interfaces consumed by the Streamlit frontend.
All functions return structured dicts / DataFrames with explicit error keys
so the UI can render graceful error states instead of crashing.
"""

from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from backend.config import FEATURE_COLS, CONDITION_THRESHOLDS, confidence_tier
from backend.feature_engineering import (
    derive_single, encode_soil, derive_features, encode_soil_column
)

logger = logging.getLogger(__name__)


# ── Single-location prediction ──────────────────────────────────────────────

def predict(
    model,
    elevation: float,
    slope: float,
    rainfall: float,
    soil_str: str,
) -> dict[str, Any]:
    """Predict land suitability for a single location.

    Returns:
        dict with keys:
          prediction (int), confidence (float), label (str), conf_tier (str),
          conf_color (str), temperature, ndvi, soil_moisture, elevation_zone,
          conditions (list of condition dicts)
          OR  error (str) if something went wrong.
    """
    try:
        soil_enc = encode_soil(soil_str)
    except ValueError as exc:
        return {"error": str(exc)}

    try:
        derived = derive_single(elevation, slope, rainfall)
    except Exception as exc:
        logger.error("Feature derivation failed: %s", exc)
        return {"error": f"Feature derivation error: {exc}"}

    try:
        fv = pd.DataFrame(
            [[
                elevation, slope, rainfall, soil_enc,
                derived["temperature"], derived["ndvi"],
                derived["soil_moisture"], derived["elevation_zone"],
            ]],
            columns=FEATURE_COLS,
        )
        pred   = int(model.predict(fv)[0])
        probas = model.predict_proba(fv)[0]
        conf   = float(max(probas))
    except Exception as exc:
        logger.error("Model inference error: %s", exc)
        return {"error": f"Model inference failed: {exc}"}

    tier, color = confidence_tier(conf)

    # ── Condition analysis ──────────────────────────────────────────────────
    live_vals = {
        "Rainfall":      rainfall,
        "Slope":         slope,
        "Elevation":     elevation,
        "NDVI":          derived["ndvi"],
        "Soil Moisture": derived["soil_moisture"],
        "Temperature":   derived["temperature"],
    }
    conditions = []
    for name, cfg in CONDITION_THRESHOLDS.items():
        val    = live_vals[name]
        ideal  = cfg["ideal"]
        op     = cfg["op"]
        passed = (val >= ideal) if op == ">=" else (val <= ideal)
        conditions.append({
            "name":   name,
            "val":    round(val, 3),
            "ideal":  ideal,
            "op":     op,
            "unit":   cfg["unit"],
            "passed": passed,
        })

    return {
        "prediction":    pred,
        "confidence":    conf,
        "label":         "Suitable" if pred == 1 else "Not Suitable",
        "conf_tier":     tier,
        "conf_color":    color,
        "temperature":   derived["temperature"],
        "ndvi":          derived["ndvi"],
        "soil_moisture": derived["soil_moisture"],
        "elevation_zone":derived["elevation_zone"],
        "conditions":    conditions,
    }


# ── Batch prediction ─────────────────────────────────────────────────────────

def predict_batch(model, df_input: pd.DataFrame) -> pd.DataFrame:
    """Predict suitability for a DataFrame of locations.

    Expected columns (at minimum): Elevation, Slope, Rainfall, Soil_Type.
    Returns the input DataFrame augmented with:
      Prediction, Confidence, Label, Conf_Tier columns.

    Missing optional columns (Temperature, NDVI, etc.) are derived automatically.
    Rows with unrecognisable Soil_Type are set to Loamy as fallback.
    """
    if df_input is None or df_input.empty:
        return pd.DataFrame()

    required = ["Elevation", "Slope", "Rainfall", "Soil_Type"]
    missing  = [c for c in required if c not in df_input.columns]
    if missing:
        raise ValueError(f"Batch input missing columns: {missing}")

    df = df_input.copy()

    # Coerce numerics
    for col in ["Elevation", "Slope", "Rainfall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Elevation", "Slope", "Rainfall"], inplace=True)

    # Encode soil — unknown types → Loamy
    df = encode_soil_column(df)
    df = derive_features(df)

    X = df[FEATURE_COLS].copy()

    try:
        preds  = model.predict(X)
        probas = model.predict_proba(X)
        confs  = probas.max(axis=1)
    except Exception as exc:
        logger.error("Batch inference failed: %s", exc)
        raise

    df["Prediction"]  = preds.astype(int)
    df["Confidence"]  = np.round(confs, 4)
    df["Label"]       = df["Prediction"].map({1: "Suitable", 0: "Not Suitable"})
    df["Conf_Tier"]   = df["Confidence"].apply(
        lambda c: confidence_tier(c)[0]
    )

    return df


# ── Find similar real-world locations ────────────────────────────────────────

def find_similar_districts(
    df_districts: pd.DataFrame,
    target_elevation: float,
    target_rainfall: float,
    model=None,
    n: int = 10,
) -> pd.DataFrame | None:
    """Find the N most environmentally similar Indian districts.

    Matching is done via MinMaxScaler + Euclidean distance on
    (elevation, annual_rainfall).

    If *model* is provided, each matched district gets a Suitability prediction.

    Returns:
        Ranked DataFrame of top-N districts, or None on failure.
    """
    if df_districts is None or df_districts.empty:
        return None

    needed = ["elevation", "annual_rainfall"]
    if any(c not in df_districts.columns for c in needed):
        logger.error("District data missing elevation/annual_rainfall columns.")
        return None

    try:
        X_all = df_districts[needed].fillna(0).values
        if len(X_all) < 2:
            logger.warning("Not enough districts to compute similarity.")
            return df_districts.copy()

        query = np.array([[target_elevation, target_rainfall]])

        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X_all)
        q_norm = scaler.transform(query)

        dists   = np.linalg.norm(X_norm - q_norm, axis=1)
        actual_n = min(n, len(dists))
        top_idx = np.argsort(dists)[:actual_n]

        result = df_districts.iloc[top_idx].copy()
        result["Match_Distance"] = np.round(dists[top_idx], 4)
        result = result.reset_index(drop=True)
        result.index += 1

        # ── Optionally predict suitability for each matched district ────────
        if model is not None:
            predictions, confidences = [], []
            for _, row in result.iterrows():
                try:
                    res = predict(
                        model,
                        elevation=float(row["elevation"]),
                        slope=5.0,          # typical moderate slope as default
                        rainfall=float(row["annual_rainfall"]),
                        soil_str="Loamy",   # neutral default
                    )
                    predictions.append(res.get("label", "N/A"))
                    confidences.append(res.get("confidence", 0.0))
                except Exception:
                    predictions.append("N/A")
                    confidences.append(0.0)
            result["Predicted_Suitability"] = predictions
            result["Prediction_Confidence"] = np.round(confidences, 3)

        return result

    except Exception as exc:
        logger.error("find_similar_districts failed: %s", exc)
        return None
