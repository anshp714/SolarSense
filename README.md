☀️ SolarSense: Land Suitability AI
SolarSense is an advanced, fully-offline machine learning application designed to predict the suitability of land for solar panel installation based on environmental and topographical factors.

📖 About the Project
The transition to renewable energy requires identifying optimal locations for solar infrastructure. SolarSense solves this by utilizing a Random Forest Classification model to analyze geographical parameters such as:

Elevation (meters)
Slope (degrees)
Annual Rainfall (mm/yr)
Soil Type (Clay, Sandy, Loamy)
Derived features: Temperature, NDVI (Normalized Difference Vegetation Index), Soil Moisture
The application is split into a robust Python backend handling all ML modeling, and an interactive frontend built with Streamlit featuring a modern, bright-theme UI.

🚀 How to Run
Prerequisites
Python 3.9+ installed
Git installed
1. Clone the Repository
git clone <your-repository-url>
cd panel_project
2. Install Dependencies
Install the required packages using pip:

pip install -r requirements.txt
3. Generate the Dataset
Before running the application, you must generate the synthetic training data:

python scripts/generate_dataset.py
(Optional) Process the real Indian district dataset for the auto-fill feature:

python scripts/process_india_data.py
4. Run the Application
Launch the Streamlit frontend:

streamlit run frontend/app.py
The application will open in your default web browser at http://localhost:8501.

🌳 Git Workflow (For Contributors)
If you are contributing to this project, follow this standard Git workflow:

Pull the latest changes:
git pull origin main
Create a new branch for your feature:
git checkout -b feature/your-feature-name
Stage your changes:
git add .
Commit your changes:
git commit -m "Add descriptive commit message here"
Push to the repository:
git push origin feature/your-feature-name
Open a Pull Request on GitHub/GitLab.
📁 Project Structure
frontend/: Contains the Streamlit web application (app.py).
backend/: Core ML logic, data loading, and feature engineering.
scripts/: Utility scripts for generating and processing datasets.
data/: Directory where generated datasets and models are saved.
