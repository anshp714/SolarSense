"""
backend/config.py
=================
Single source of truth for all paths, constants, and hyperparameters.
ALL paths are relative to the project root — no hardcoded absolute paths.
"""

from pathlib import Path

# ── Project Root (two levels up from this file) ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data Paths ─────────────────────────────────────────────────────────────
DATA_DIR              = PROJECT_ROOT / "data"
DATASET_PATH          = DATA_DIR / "dataset_100k.csv"
DISTRICT_DATA_PATH    = DATA_DIR / "indian_districts_summary.csv"
RAW_EXCEL_PATH        = DATA_DIR / "india_weather_rainfall_data.xlsx"

# ── Output Paths ───────────────────────────────────────────────────────────
OUTPUTS_DIR           = PROJECT_ROOT / "outputs"
MODEL_PATH            = OUTPUTS_DIR / "model.joblib"
MODEL_METADATA_PATH   = OUTPUTS_DIR / "model_metadata.json"

# ── Feature Definitions ────────────────────────────────────────────────────
SOIL_MAPPING: dict[str, int] = {
    "Clay":  0,
    "Loamy": 1,
    "Sandy": 2,
    "Silty": 3,
}
SOIL_LABELS: dict[int, str] = {v: k for k, v in SOIL_MAPPING.items()}

FEATURE_COLS: list[str] = [
    "Elevation",
    "Slope",
    "Rainfall",
    "Soil_Type",
    "Temperature",
    "NDVI",
    "Soil_Moisture",
    "Elevation_Zone",
]

# Columns that MUST exist in dataset_100k.csv for the model to train
REQUIRED_TRAINING_COLS: list[str] = [
    "Elevation", "Slope", "Rainfall", "Soil_Type", "Suitability"
]

# Columns that MUST exist in indian_districts_summary.csv
REQUIRED_DISTRICT_COLS: list[str] = [
    "state", "district", "elevation", "annual_rainfall"
]

# ── sklearn Hyperparameters (CPU fallback, anti-overfitting tuned) ──────────
MODEL_HYPERPARAMS: dict = {
    "n_estimators":      200,
    "max_depth":         12,
    "min_samples_leaf":  30,
    "min_samples_split": 50,
    "max_features":      "sqrt",
    "class_weight":      "balanced",
    "random_state":      42,
    "n_jobs":            -1,
}

# ── XGBoost Hyperparameters (GPU-primary path) ──────────────────────────────
# XGBoost is the GPU backend for Windows (RTX 3050 / any CUDA device).
# scikit-learn's RandomForest does NOT support CUDA — XGBoost does natively.
#
# Fallback chain (auto-detected at runtime in model.py):
#   cuML RF  →  XGBoost + CUDA  →  XGBoost + CPU  →  sklearn RF
XGBOOST_HYPERPARAMS: dict = {
    "n_estimators":     200,
    "max_depth":        8,        # boosted trees are additive — shallower needed
    "learning_rate":    0.05,     # slow learning = better generalisation
    "subsample":        0.8,      # row subsampling per tree (like RF's bootstrap)
    "colsample_bytree": 0.8,      # feature subsampling per tree
    "min_child_weight": 30,       # equivalent to min_samples_leaf
    "reg_alpha":        0.1,      # L1 regularisation
    "reg_lambda":       1.0,      # L2 regularisation
    "random_state":     42,
    "eval_metric":      "logloss",
    "use_label_encoder": False,
    # 'device' key is injected dynamically in model.py after GPU detection
}

CROSS_VAL_FOLDS: int = 5   # stratified k-fold for overfitting detection

# ── Prediction Condition Thresholds (UI display) ───────────────────────────
CONDITION_THRESHOLDS: dict = {
    "Rainfall":      {"ideal": 600,   "op": ">=", "unit": "mm"},
    "Slope":         {"ideal": 20,    "op": "<=", "unit": "°"},
    "Elevation":     {"ideal": 2000,  "op": "<=", "unit": "m"},
    "NDVI":          {"ideal": 0.40,  "op": ">=", "unit": ""},
    "Soil Moisture": {"ideal": 0.10,  "op": ">=", "unit": ""},
    "Temperature":   {"ideal": 15,    "op": ">=", "unit": "°C"},
}

# ── Confidence Tier Labels ─────────────────────────────────────────────────
def confidence_tier(conf: float) -> tuple[str, str]:
    """Return (label, color) for a confidence value in [0, 1]."""
    if conf >= 0.85:
        return "High Confidence", "#4caf50"
    elif conf >= 0.65:
        return "Moderate Confidence", "#ff9800"
    else:
        return "Low Confidence — Borderline Case", "#f44336"

# ── Elevation Zone Bins ────────────────────────────────────────────────────
ELEVATION_BINS   = [0, 500, 1500, 2500, float("inf")]
ELEVATION_LABELS = [0, 1, 2, 3]
