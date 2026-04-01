# ☀️ SolarSense: The Complete Project Guide

*A guide for teammates, friends, and anyone who wants to understand how SolarSense works under the hood.*

---

## 🧐 What is SolarSense?
SolarSense is a web application that acts like an intelligent assistant for renewable energy planners. Its primary job is to answer one question: **"Is this specific plot of land good for building a solar farm?"**

Instead of guessing, SolarSense uses **Machine Learning** (specifically, an algorithm called *Random Forest*) to predict the suitability of land based on its environmental conditions.

## 🧩 How Does it Work? (The Big Picture)

Imagine you are trying to bake a cake. You need a recipe (the Model), ingredients (the Data), and a kitchen to bake and serve it in (the App/Frontend).

### 1. The Ingredients (Data & Feature Engineering)
We feed the system data about different plots of land. We look at:
- **Elevation:** How high above sea level is it? (Too high = harsh weather, too low = flood risk).
- **Slope:** Is the land flat or steep? (Steep land is terrible for installing flat solar panels).
- **Rainfall:** How much does it rain? (Heavy rain means heavy clouds; clouds block the sun).
- **Soil Type:** Is the ground clay, sand, or loam? (Sand might shift; clay holds water).

To make our AI even smarter, our `backend/feature_engineering.py` script calculates "Derived Features" mathematically:
- **Temperature:** We estimate this based on how high the land is (higher = colder).
- **NDVI (Vegetation Index):** We estimate how much plant life is naturally there based on rainfall.
- **Soil Moisture:** We calculate this combining rainfall and how well the specific soil type drains water.

### 2. The Recipe (The Machine Learning Model)
We use a **Random Forest Classifier** (`backend/model.py`).
Think of a Random Forest as a "council" of hundreds of different decision trees. 
- One tree might say: *"The slope is flat, so it's Suitable."*
- Another tree might say: *"Yes, but the rainfall is extremely high, so it's NOT Suitable."*
The forest takes a vote from all these trees, and the majority wins. This creates a highly accurate, robust prediction.

### 3. The Kitchen (The Streamlit Frontend)
The frontend (`frontend/app.py`) is what you actually see and interact with. It is built using **Streamlit**, a tool that turns Python scripts into interactive web apps.
- **Home:** A beautiful landing page explaining the "Why".
- **Analyze Location:** Sliders and buttons where you input your land's details and get instant predictions.
- **Dashboard:** Deep dives into the dataset, model accuracy (F1 score, AUC), and batch predictions for power-users.

## 🏗️ The Architecture (Frontend vs Backend)
One of the best design choices in SolarSense is the **decoupled architecture**.
- The `frontend/` folder ONLY handles drawing buttons, sliders, and colors on the screen. It knows nothing about math.
- The `backend/` folder handles all the heavy lifting, calculating, and model training.
If the frontend requests a prediction, it just asks the backend: *"Hey, slope is 5 and rainfall is 500. What's the result?"* The backend computes and sends back the answer.

## 🚀 How to Demo the Project to Others
If you are showing this to a professor or friend, follow this flow:
1. **Start on the Home Tab:** Talk about climate change and why solar energy is growing. Show the gorgeous UI.
2. **Go to Analyze Location:** Move the sliders. 
   - *Trick:* Set Slope to `45` (very steep) and watch the AI confidently predict "Not Suitable" because you can't build panels on a cliff.
3. **Show the Dashboard:** Click the "Model Insights" tab to prove that this isn't just a toy; it is backed by a real trained AI model with high accuracy metrics (usually ~85%+ Accuracy).
4. **Use Auto-fill:** Show the Indian District Auto-fill to demonstrate how the app utilizes real-world geospatial data averages instead of just random numbers.
