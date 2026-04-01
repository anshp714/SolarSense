# Preliminary Design Review (PDR)
## Project: Land Suitability Web Application

---

## 1. Introduction
The Land Suitability Web Application is an interactive, fully offline machine-learning-powered tool that predicts whether a given piece of land is suitable for use. It utilizes a Random Forest Classifier trained on a synthesized dataset of 100,000 samples. In addition to real-time interactive predictions, the system is deeply integrated with real-world geographical and meteorogical data, allowing users to query environmental metrics (like annual rainfall and elevation) based on Indian states and districts, and to discover real-world locations that closely match their targeted environmental criteria.

### 1.1 Objectives
* Provide a dynamic, user-driven interface for land suitability analysis.
* Integrate real-world Indian environmental data to simplify user inputs (auto-filling environmental parameters).
* Map and find real-world Indian districts with environments similar to arbitrary constraints.
* Deliver real-time ML inference, complete with confidence scoring and probability assessments.
* Provide interactive dataset exploration and model insight dashboards for transparency.

## 2. System Architecture
The overall architecture relies on a local Streamlit web application that serves both the frontend user interface and the backend ML inference engine in a single, unified runtime.

* **Frontend & Backend Frame:** Streamlit (`app.py`), orchestrating the UI components, user session states, routing (Tabs), and caching.
* **Data Processing & ML:** `pandas`, `numpy`, and `scikit-learn` handling data ingestion, preprocessing, training, and inference.
* **Visualization:** `plotly.express` and `plotly.graph_objects` utilized for rich, interactive data visualizations and gauges.

### 2.1 Component Breakdown
1. **Frontend UI (`frontend/app.py`):**
   * Heavily styled with custom CSS (`Inter` font, dark gradients, animated result banners, tailored metric cards, visually enhanced tabs and buttons).
   * Structured into four primary functional tabs.
2. **Data Pipeline (`scripts/`):**
   * **Dataset Generation:** `generate_dataset.py` programmatically produces `dataset_100k.csv` incorporating derived features (Temperature, NDVI, Soil Moisture) and adding probabilistic noise to base scores to train a robust model.
   * **Real-world Data Processing:** `process_india_data.py` processes a 64MB raw Excel dataset into a lightweight `indian_districts_summary.csv` containing grouped means of rainfall, elevation, temperature, and coordinates per district.
3. **Machine Learning Model:**
   * **Type:** Random Forest Classifier (`n_estimators=100`, `max_depth=15`).
   * **Features (8):** Elevation, Slope, Rainfall, Soil_Type (Label Encoded), Temperature, NDVI, Soil_Moisture, Elevation_Zone.
   * Cached to memory at application load (`@st.cache_resource`).

## 3. Data Flow and Storage
The application operates entirely offline, utilizing pre-processed and locally stored CSV files.

1. **Model Training Data (`data/dataset_100k.csv`):** Loaded at startup and split (80/20) to train the Random Forest model on the fly. Results and model artifacts are cached in memory for the duration of the server run.
2. **Indian Districts Summary (`data/indian_districts_summary.csv`):** Loaded to populate state/district dropdowns.
3. **Inference Flow:** 
   * User inputs parameters via sliders (or auto-fills from a selected district).
   * Derived features are computationally determined on the fly (e.g., Temperature drops by 0.006°C per meter of Elevation).
   * A single-row DataFrame is passed to the trained Random Forest model.
   * Classification label and prediction probabilities are returned and mapped to the UI.

## 4. User Interface and Experience (UI/UX)
The interface is segmented into Four distinct tabs:

### 4.1 Tab 1: Predict Suitability (🗺️)
* Contains an expandable component for users to pick a State and District, automatically substituting the associated average Real-World Elevation and Rainfall via Session State.
* Real-time sliders for Elevation, Slope, and Rainfall.
* Live-updating derived features (Temperature, NDVI, Soil Moisture).
* **Results Display:** Conditional animated banners (Suitable / Not Suitable), a vibrant Plotly Gauge Chart showing probability confidence, and pass/fail cards detailing whether specific parameters met ideal thresholds.

### 4.2 Tab 2: Dataset Explorer (📊)
* Scatter charts mapping combinations of features against each other with target mapping.
* Pie charts displaying the generated dataset balance.
* Histograms highlighting distribution geometries for each feature class.
* A plotted Correlation Heatmap.

### 4.3 Tab 3: Model Insights (🤖)
* Horizontal Bar charts ranking **Feature Importance** across the trained model.
* A heat-mapped **Confusion Matrix** showing True Positive / False Positive rates.
* Visual bar charts highlighting the training distributions.

### 4.4 Tab 4: Find Similar Locations (🔍)
* Users input target constraints (Elevation and Rainfall).
* K-Nearest Neighbors mathematical matching (via `MinMaxScaler` and Euclidean distance vectors) locates the top 10 most similar districts.
* Outputs the closest districts securely on a tabular format alongside distance rankings.
* Geographically plots the resulting matches across an integrated map visualization utilizing latitude and longitude pairs.

## 5. Security and Deployment Context
* **Offline First:** No external APIs or active cloud databases are queried during execution, eliminating active network vulnerabilities. Data acts locally.
* **Performance:** Uses `@st.cache_data` and `@st.cache_resource` extensively to prevent latency during UI reflows or redundant model training processes.
* **Deployment:** Best deployed on an environment equipped with minimally 4GB+ RAM to handle the `dataset_100k.csv` memory requirements and real-time inference gracefully. Required Python packages strictly documented in `requirements.txt`.

## 6. Directory Structure Overview
```
panel_project/
│
├── frontend/
│   └── app.py (Main Streamlit Entrypoint)
│
├── scripts/
│   ├── generate_dataset.py (Synthesizes 100k samples)
│   └── process_india_data.py (Aggregates Excel state/district statistics)
│
├── data/
│   ├── dataset_100k.csv (12MB ML dataset)
│   ├── indian_districts_summary.csv (15KB fast geographic mapping records)
│   └── india_weather_rainfall_data.xlsx (Raw unaggregated weather dataset - 64MB)
│
├── outputs/
│   └── (Static Image Artifacts for Matrix/Evaluation reference)
│
└── requirements.txt
```

---
*Document produced systematically observing codebase architectures, pipelines, and data transformations.*
