import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Initializing environment...")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "dataset_100k.csv"
    OUT_DIR = PROJECT_ROOT / "outputs"
    OUT_DIR.mkdir(exist_ok=True)

    if not DATA_PATH.exists():
        print(f"Error: Dataset not found at {DATA_PATH}")
        print("Please run scripts/generate_dataset.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Take a sample for responsive rendering (15k points)
    df_sample = df.sample(min(15000, len(df)), random_state=42)
    df_sample["Class"] = df_sample["Suitability"].map({1: "Suitable", 0: "Not Suitable"})
    
    color_map = {"Suitable": "#4caf50", "Not Suitable": "#f44336"}

    # 1. Train Random Forest for Feature Importance
    print("Training Decision Tree to compute explicit Feature Importances...")
    # Clean the data for fitting
    X = df_sample.drop(columns=["Suitability", "Class"])
    if "Soil_Type" in X.columns:
        X = pd.get_dummies(X, columns=["Soil_Type"])
    y = df_sample["Suitability"]
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42)
    rf.fit(X, y)

    imp_df = pd.DataFrame({"Parameter": X.columns, "Importance": rf.feature_importances_})
    imp_df = imp_df.sort_values(by="Importance", ascending=True)

    fig_imp = px.bar(
        imp_df, x="Importance", y="Parameter", orientation='h', 
        title="Overall Feature Importances (Gini Index)",
        color="Importance", color_continuous_scale="Viridis"
    )
    fig_imp.write_html(OUT_DIR / "1_Overall_Feature_Importances.html")
    print(" -> Saved 1_Overall_Feature_Importances.html")

    # 2. Elevation Impact Matrix
    fig_elev = px.histogram(
        df_sample, x="Elevation", color="Class", barmode="overlay", 
        color_discrete_map=color_map, opacity=0.75,
        title="Elevation Influence on Solar Suitability (m)"
    )
    fig_elev.write_html(OUT_DIR / "2_Elevation_Importance.html")
    print(" -> Saved 2_Elevation_Importance.html")

    # 3. Slope Impact Matrix
    fig_slope = px.histogram(
        df_sample, x="Slope", color="Class", barmode="overlay", 
        color_discrete_map=color_map, opacity=0.75,
        title="Terrain Slope Influence on Solar Suitability (Degrees)"
    )
    fig_slope.write_html(OUT_DIR / "3_Slope_Importance.html")
    print(" -> Saved 3_Slope_Importance.html")

    # 4. Rainfall Impact Matrix
    fig_rain = px.histogram(
        df_sample, x="Rainfall", color="Class", barmode="overlay", 
        color_discrete_map=color_map, opacity=0.75,
        title="Annual Rainfall Influence on Solar Suitability (mm/yr)"
    )
    fig_rain.write_html(OUT_DIR / "4_Rainfall_Importance.html")
    print(" -> Saved 4_Rainfall_Importance.html")

    # 5. Combined Elevation vs Slope Scatter map
    fig_scatter = px.scatter(
        df_sample, x="Elevation", y="Slope", color="Class", 
        color_discrete_map=color_map, opacity=0.6,
        title="Multivariate Importance: Elevation vs Target Slope Limits"
    )
    fig_scatter.write_html(OUT_DIR / "5_Elevation_vs_Slope_Scatter.html")
    print(" -> Saved 5_Elevation_vs_Slope_Scatter.html")

    print(f"\n✅ All 5 Importance Graphs successfully generated in: {OUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
