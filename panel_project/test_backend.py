"""Quick smoke-test for backend modules."""
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from backend.data_loader import load_training_data, load_district_data
from backend.model import train_model
from backend.predictor import predict, find_similar_districts
from backend.config import DATASET_PATH, DISTRICT_DATA_PATH

print("=== Loading data ===")
df = load_training_data(DATASET_PATH)
print(f"Training data: {len(df)} rows")

dd = load_district_data(DISTRICT_DATA_PATH)
print(f"District data: {len(dd)} districts")

print("\n=== Training model ===")
model, metrics = train_model(df)
print(f"Accuracy  : {metrics['accuracy']:.4f}")
print(f"F1 Score  : {metrics['f1']:.4f}")
print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
print(f"Brier     : {metrics['brier_score']:.4f}")
print(f"CV F1     : {metrics['cv_mean']:.4f} +/- {metrics['cv_std']:.4f}")
print(f"Overfit gap: {metrics['overfit_gap']:+.4f}")

print("\n=== Predictor tests ===")
# Normal
res = predict(model, 600, 8, 800, "Loamy")
assert "error" not in res, f"Unexpected error: {res.get('error')}"
print(f"Normal: {res['label']} | conf={res['confidence']:.3f} | tier={res['conf_tier']}")

# Extreme values
res2 = predict(model, 3000, 45, 200, "Sandy")
assert "error" not in res2
print(f"Extreme: {res2['label']} | conf={res2['confidence']:.3f}")

# Flat high-rain
res3 = predict(model, 0, 0, 2000, "Loamy")
assert "error" not in res3
print(f"Flat high-rain: {res3['label']} | conf={res3['confidence']:.3f}")

# Unknown soil type
res4 = predict(model, 600, 8, 800, "Gravel")
assert "error" in res4, "Expected error dict for unknown soil type"
print(f"Bad soil correctly caught: {res4['error'][:60]}")

print("\n=== Similar districts ===")
found = find_similar_districts(dd, 600, 800, model=model)
assert found is not None and len(found) > 0
print(f"Top match: {found.iloc[0]['district']}, dist={found.iloc[0]['Match_Distance']}")
if "Predicted_Suitability" in found.columns:
    print(f"Suitability of top match: {found.iloc[0]['Predicted_Suitability']}")

print("\n=== ALL TESTS PASSED ===")
