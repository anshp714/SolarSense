import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from backend.model import load_model
from backend.config import FEATURE_COLS
from backend.feature_engineering import encode_soil_column
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve

def main():
    print("Loading recent model and dataset...")
    DATA_PATH = PROJECT_ROOT / "data" / "dataset_100k.csv"
    OUT_DIR = PROJECT_ROOT / "outputs"
    
    model = load_model()
    if model is None:
        print("Error: Could not load the model from outputs/")
        return
        
    df = pd.read_csv(DATA_PATH)
    df = encode_soil_column(df)
    from backend.feature_engineering import derive_features
    df = derive_features(df)
    
    df_test = df.sample(min(20000, len(df)), random_state=42)
    X = df_test[FEATURE_COLS]
    y = df_test["Suitability"]
    
    print("Running inference...")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # 1. Confusion Matrix
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False,
                xticklabels=["Not Suitable", "Suitable"],
                yticklabels=["Not Suitable", "Suitable"])
    plt.title("Confusion Matrix (Latest Model)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved confusion_matrix.png")

    # 2. Evaluation Metrics Display
    plt.figure(figsize=(6,4))
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_proba)
    metrics = {"Accuracy": acc, "F1-Score": f1, "ROC-AUC": roc}
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
    plt.ylim(0, 1.1)
    plt.title("Model Evaluation Metrics")
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
    plt.savefig(OUT_DIR / "evaluation_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved evaluation_metrics.png")

    # 3. Feature Importance
    plt.figure(figsize=(8,6))
    imp = model.feature_importances_
    f_imp = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": imp}).sort_values('Importance', ascending=False)
    sns.barplot(data=f_imp, x="Importance", y="Feature", palette="magma")
    plt.title("Feature Importance (Gini)")
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved feature_importance.png")

    # 4. Prediction Explanation
    plt.figure(figsize=(6,5))
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc:.3f})")
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Prediction Integrity (ROC Curve)")
    plt.legend()
    plt.savefig(OUT_DIR / "prediction_explanation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved prediction_explanation.png")

    print("\n✅ Successfully updated standard PNGs in outputs/ to exactly match the latest underlying model readings.")

if __name__ == "__main__":
    main()
