# ☀️ SolarSense Flowcharts & Architecture Diagrams

This document outlines the systematic execution flows of the SolarSense application using Mermaid diagrams. You can render these diagrams in any markdown viewer that supports Mermaid (like GitHub or modern IDEs).

## 1. High-Level System Architecture

This diagram shows how the user interacts with the UI, and how the UI communicates with the isolated backend.

```mermaid
graph TD
    User([User / Planner]) --> |Interacts with| UI[Frontend: Streamlit App]
    
    subgraph Streamlit Frontend
        UI --> Home[Home / Landing Page]
        UI --> Analyze[Analyze Location UI]
        UI --> Dash[Dashboard UI]
    end
    
    subgraph Python Backend Core
        Analyze --> |Sends Params| BP[backend/predictor.py]
        Dash --> |Requests Details| BM[backend/model.py]
        Dash --> |Requests Details| BD[backend/data_loader.py]
        
        BP --> |Uses| BE[backend/feature_engineering.py]
        BP --> |Loads Saved| ModelData[(Trained Model PKL)]
    end
    
    BP --> |Returns Suitability| Analyze
```

---

## 2. Model Training Pipeline

This flowchart explains what happens behind the scenes when the dataset is generated or when the model is forced to retrain.

```mermaid
graph TD
    Start([Start Training]) --> LoadData[backend/data_loader.py loads dataset.csv]
    
    LoadData --> Check{Is Data Valid?}
    Check -->|No| Fail([Throw Error])
    Check -->|Yes| Eng[backend/feature_engineering.py]
    
    Eng --> DeriveTemp[Derive Temperature]
    Eng --> DeriveNDVI[Derive NDVI]
    Eng --> DeriveMoist[Derive Soil Moisture]
    
    DeriveTemp --> Merge
    DeriveNDVI --> Merge
    DeriveMoist --> Merge
    
    Merge[Combine Base + Derived Features] --> Split[Split Data 80% Train / 20% Test]
    
    Split --> Train[Train Random Forest Classifier]
    Train --> Evaluate[Calculate Accuracy, F1, ROC AUC]
    
    Evaluate --> Save[Save model.pkl and metrics.json]
    Save --> End([End Training / Ready])
```

---

## 3. Real-Time Prediction Execution Flow

This flowchart details exactly what happens when a user clicks "Predict Land Suitability" in the app.

```mermaid
sequenceDiagram
    participant User
    participant Streamlit (app.py)
    participant Predictor (backend)
    participant FeatEng (feature_engineering)
    participant Model (RandomForest)

    User->>Streamlit: Adjusts Sliders & Clicks "Predict"
    Streamlit->>Predictor: predict(elevation, slope, rainfall, soil)
    
    Predictor->>FeatEng: derive_single(elevation, rainfall, soil)
    FeatEng-->>Predictor: Returns (Temp, NDVI, Moisture)
    
    Predictor->>Predictor: Formats into DataFrame
    Predictor->>Model: model.predict_proba(DataFrame)
    Model-->>Predictor: Returns Probabilities (e.g., 0.85 Suitable)
    
    Predictor->>Predictor: Generate Explanations (What passed/failed)
    Predictor-->>Streamlit: Returns Result Dict (Status, Confidence, Explanations)
    
    Streamlit->>User: Renders Green/Red Banner & Cards
```

---

## 4. Batch Processing Flow

How the system handles predicting massive amounts of data at once via CSV upload.

```mermaid
graph LR
    A([User Uploads CSV]) --> B[Streamlit parses to Pandas DF]
    B --> C{Contains Required Columns?}
    C -->|No| D([Return Column Error])
    C -->|Yes| E[backend/predictor.batch_predict()]
    E --> F[Apply Feature Engineering to all rows]
    F --> G[Run Model Prediction over entire batch]
    G --> H[Append 'Prediction' and 'Probability' columns]
    H --> I([User Downloads Result CSV])
```
