"""
scripts/generate_dataset.py
============================
Synthesise the 100,000-sample training dataset for the Land Suitability model.

Uses the canonical feature derivation formulas from backend/feature_engineering.py
so training data and live inference are ALWAYS consistent.

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --samples 200000 --seed 99 --output data/custom.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Resolve project root and add to path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Use canonical feature formulas from backend ──────────────────────────────
try:
    from backend.feature_engineering import (
        derive_temperature, derive_ndvi,
        derive_soil_moisture, derive_elevation_zone,
    )
    from backend.config import DATASET_PATH, SOIL_MAPPING
except ImportError as e:
    print(f"ERROR: Could not import backend package: {e}")
    print("Run this script from the project root: python scripts/generate_dataset.py")
    sys.exit(1)


def generate(samples: int = 100_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    elevation = rng.uniform(0,    3000, samples)
    slope     = rng.uniform(0,    45,   samples)
    rainfall  = rng.uniform(200,  2000, samples)
    soil      = rng.choice(list(SOIL_MAPPING.keys()), samples)

    # Derived features — identical formulas to backend/feature_engineering.py
    temperature   = derive_temperature(elevation)
    ndvi          = derive_ndvi(rainfall)
    soil_moisture = derive_soil_moisture(rainfall, slope)

    data = pd.DataFrame({
        "Elevation":     elevation,
        "Slope":         slope,
        "Rainfall":      rainfall,
        "Soil_Type":     soil,
        "Temperature":   temperature,
        "NDVI":          ndvi,
        "Soil_Moisture": soil_moisture,
    })

    # ── Suitability score (probabilistic, not hard threshold) ─────────────────
    # Each component scored 0–1 then summed → max 4.0
    elevation_score = np.clip(1.0 - (elevation / 2000), 0, 1)
    slope_score     = np.clip(1.0 - (slope     / 25),   0, 1)
    rainfall_score  = np.clip((rainfall - 400)  / 1000, 0, 1)
    ndvi_score      = np.clip((ndvi - 0.2)      / 0.6,  0, 1)

    base_score = elevation_score + slope_score + rainfall_score + ndvi_score

    # Gaussian noise — models real-world unmeasured variability, prevents overfitting
    noise      = rng.normal(0, 0.4, samples)
    final_score = base_score + noise

    data["Suitability"] = (final_score > 2.5).astype(int)

    return data


def main():
    parser = argparse.ArgumentParser(description="Generate Land Suitability training dataset")
    parser.add_argument("--samples", type=int,   default=100_000, help="Number of samples (default: 100000)")
    parser.add_argument("--seed",    type=int,   default=42,      help="Random seed (default: 42)")
    parser.add_argument("--output",  type=str,   default=None,    help="Output CSV path (default: data/dataset_100k.csv)")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else DATASET_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.samples:,} samples (seed={args.seed})…")
    data = generate(args.samples, args.seed)
    data.to_csv(out_path, index=False)

    suitable     = int(data["Suitability"].sum())
    not_suitable = len(data) - suitable
    print(f"✅ Dataset saved → {out_path}")
    print(f"   Suitable     : {suitable:,}  ({suitable/len(data):.1%})")
    print(f"   Not Suitable : {not_suitable:,}  ({not_suitable/len(data):.1%})")


if __name__ == "__main__":
    main()