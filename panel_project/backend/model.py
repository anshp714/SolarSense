"""
backend/model.py
=================
Model training, evaluation, persistence, and staleness detection.

Anti-overfitting strategy:
  1. Reduced max_depth (12, was 15) — prevents deep memorisation
  2. Increased min_samples_leaf (30, was 20) — forces generalisation
  3. min_samples_split=50 — avoids tiny impure splits
  4. max_features="sqrt" — feature subsampling per split
  5. class_weight="balanced" — handles class imbalance
  6. 5-fold stratified cross-validation — detects train/val gap

When new data is integrated:
  - Model is retrained automatically if the dataset CSV is newer than the
    saved model file (mtime comparison in load_or_train_model).
  - Cross-val scores are printed; if val/train gap is large, a warning fires.
"""

from __future__ import annotations
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import (
        train_test_split,
        StratifiedKFold,
        cross_val_score,
        learning_curve,
    )
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_curve, auc,
        brier_score_loss,
    )
    from sklearn.calibration import calibration_curve

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from backend.config import (
    FEATURE_COLS, MODEL_HYPERPARAMS, CROSS_VAL_FOLDS,
    MODEL_PATH, MODEL_METADATA_PATH, OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


# ── Training ────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Train a Random Forest classifier with anti-overfitting controls.

    Performs 5-fold stratified cross-validation and computes:
      accuracy, precision, recall, F1, confusion matrix, ROC-AUC,
      calibration curve, feature importances.

    Args:
        df: Preprocessed training DataFrame with FEATURE_COLS + "Suitability".

    Returns:
        (fitted model, metrics dict)

    Raises:
        ValueError: if required columns are missing or dataset is too small.
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for training: {missing}")
    if "Suitability" not in df.columns:
        raise ValueError("'Suitability' column not found in training data.")
    if len(df) < 200:
        raise ValueError(f"Dataset too small to train reliably ({len(df)} rows).")

    X = df[FEATURE_COLS].copy()
    y = df["Suitability"].copy()

    # ── Stratified split ────────────────────────────────────────────────────
    # Guard: stratify needs ≥ 2 samples per class
    vc = y.value_counts()
    use_stratify = all(cnt >= 2 for cnt in vc)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=MODEL_HYPERPARAMS["random_state"],
        stratify=y if use_stratify else None,
    )

    # ── Fit ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model = RandomForestClassifier(**MODEL_HYPERPARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # ── Evaluation on held-out test set ────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # ── Training accuracy (for overfitting gap check) ────────────────────
    train_acc = accuracy_score(y_train, model.predict(X_train))
    gap = train_acc - accuracy
    if gap > 0.05:
        logger.warning(
            "Potential overfitting detected: train acc=%.4f, test acc=%.4f, gap=%.4f",
            train_acc, accuracy, gap
        )
    else:
        logger.info(
            "Model generalises well: train acc=%.4f, test acc=%.4f, gap=%.4f",
            train_acc, accuracy, gap
        )

    # ── Cross-validation ────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
    logger.info(
        "Cross-validation F1 (stratified %d-fold): %.4f ± %.4f",
        CROSS_VAL_FOLDS, cv_scores.mean(), cv_scores.std()
    )
    if cv_scores.std() > 0.05:
        logger.warning(
            "High cross-val variance (std=%.4f). "
            "Model may be sensitive to data distribution.",
            cv_scores.std()
        )

    # ── ROC-AUC ─────────────────────────────────────────────────────────────
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba, pos_label=1)
    roc_auc = float(auc(fpr, tpr))

    # ── Calibration curve ───────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal_frac_pos, cal_mean_pred = calibration_curve(
            y_test, y_proba, n_bins=10, strategy="uniform"
        )

    # ── Brier score ─────────────────────────────────────────────────────────
    brier = float(brier_score_loss(y_test, y_proba))

    # ── Learning curve (sampled for speed) ─────────────────────────────────
    # Use a subset of the data to compute the learning curve quickly
    lc_sizes = np.linspace(0.1, 1.0, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lc_train_sizes, lc_train_scores, lc_val_scores = learning_curve(
            RandomForestClassifier(**{**MODEL_HYPERPARAMS, "n_estimators": 50}),
            X, y,
            train_sizes=lc_sizes,
            cv=3,
            scoring="f1",
            n_jobs=-1,
        )

    metrics: dict[str, Any] = {
        # Core metrics
        "accuracy":         float(accuracy),
        "precision":        float(precision),
        "recall":           float(recall),
        "f1":               float(f1),
        "roc_auc":          roc_auc,
        "brier_score":      brier,
        "train_accuracy":   float(train_acc),
        "overfit_gap":      float(gap),
        # Cross-val
        "cv_scores":        cv_scores.tolist(),
        "cv_mean":          float(cv_scores.mean()),
        "cv_std":           float(cv_scores.std()),
        # Confusion matrix
        "cm":               cm,
        # Feature importance
        "importances":      model.feature_importances_,
        "feature_names":    list(X_train.columns),
        # ROC
        "roc_fpr":          fpr.tolist(),
        "roc_tpr":          tpr.tolist(),
        "roc_auc":          roc_auc,
        # Calibration
        "cal_frac_pos":     cal_frac_pos.tolist(),
        "cal_mean_pred":    cal_mean_pred.tolist(),
        # Learning curve
        "lc_train_sizes":   lc_train_sizes.tolist(),
        "lc_train_scores":  lc_train_scores.mean(axis=1).tolist(),
        "lc_val_scores":    lc_val_scores.mean(axis=1).tolist(),
        # Bookkeeping
        "n_train":          len(X_train),
        "n_test":           len(X_test),
        "train_time_s":     round(train_time, 2),
        "trained_at":       datetime.now().isoformat(timespec="seconds"),
        # For downstream subset access
        "X_test":           X_test,
        "y_test":           y_test,
        "y_pred":           y_pred,
        "y_proba":          y_proba,
    }

    logger.info(
        "Model trained in %.2fs | Acc=%.4f | F1=%.4f | ROC-AUC=%.4f | CV-F1=%.4f±%.4f",
        train_time, accuracy, f1, roc_auc, cv_scores.mean(), cv_scores.std()
    )
    return model, metrics


# ── Persistence ─────────────────────────────────────────────────────────────

def save_model(model: RandomForestClassifier, metrics: dict) -> bool:
    """Persist the trained model and its metadata to disk.

    Returns True on success, False on failure.
    """
    if not _JOBLIB_AVAILABLE:
        logger.warning("joblib not available — model will not be persisted.")
        return False

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, MODEL_PATH)
    except Exception as exc:
        logger.error("Failed to save model: %s", exc)
        return False

    # Save metadata (JSON-serialisable subset of metrics)
    meta = {
        "trained_at":     metrics.get("trained_at", ""),
        "n_train":        metrics.get("n_train", 0),
        "n_test":         metrics.get("n_test", 0),
        "accuracy":       metrics.get("accuracy", 0),
        "f1":             metrics.get("f1", 0),
        "roc_auc":        metrics.get("roc_auc", 0),
        "cv_mean":        metrics.get("cv_mean", 0),
        "cv_std":         metrics.get("cv_std", 0),
        "overfit_gap":    metrics.get("overfit_gap", 0),
        "train_time_s":   metrics.get("train_time_s", 0),
    }
    try:
        MODEL_METADATA_PATH.write_text(json.dumps(meta, indent=2))
    except Exception as exc:
        logger.warning("Could not write model metadata: %s", exc)

    logger.info("Model saved to %s", MODEL_PATH)
    return True


def load_model(path: Path | None = None) -> RandomForestClassifier | None:
    """Load a previously saved model from disk.

    Returns the model, or None if not available / loading failed.
    """
    if not _JOBLIB_AVAILABLE:
        return None
    p = Path(path or MODEL_PATH)
    if not p.exists():
        return None
    try:
        model = joblib.load(p)
        logger.info("Loaded pre-trained model from %s", p)
        return model
    except Exception as exc:
        logger.error("Failed to load saved model (%s): %s", p, exc)
        return None


def load_model_metadata() -> dict | None:
    """Load saved model metadata JSON, or None if unavailable."""
    if not MODEL_METADATA_PATH.exists():
        return None
    try:
        return json.loads(MODEL_METADATA_PATH.read_text())
    except Exception:
        return None


def is_model_stale(dataset_path: Path) -> bool:
    """Return True if the dataset CSV is newer than the saved model.

    Used to decide whether a retrain is needed when data changes.
    """
    if not MODEL_PATH.exists():
        return True
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return False
    return dataset_path.stat().st_mtime > MODEL_PATH.stat().st_mtime


def load_or_train_model(
    df: pd.DataFrame, dataset_path: Path, force_retrain: bool = False
) -> tuple[RandomForestClassifier, dict[str, Any], str]:
    """Load the model from disk if fresh, otherwise retrain.

    Args:
        df: Preprocessed training DataFrame (used only if retraining).
        dataset_path: Path to the CSV — used for staleness check.
        force_retrain: If True, always retrain regardless of saved state.

    Returns:
        (model, metrics, status)
        status is one of: "loaded", "retrained", "error"
    """
    stale = is_model_stale(dataset_path)

    if not force_retrain and not stale:
        saved_model = load_model()
        if saved_model is not None:
            # We don't have metrics from disk (only aggregates), so retrain
            # lightly on test split for metric display
            try:
                model, metrics = train_model(df)
                metrics["source"] = "loaded"
                return saved_model, metrics, "loaded"
            except Exception:
                pass  # fall through to full retrain

    try:
        model, metrics = train_model(df)
        save_model(model, metrics)
        metrics["source"] = "retrained"
        return model, metrics, "retrained"
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        # Last-ditch: try loading stale saved model
        saved_model = load_model()
        if saved_model is not None:
            logger.warning("Returning stale saved model due to training failure.")
            return saved_model, {}, "error"
        raise
