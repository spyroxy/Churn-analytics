# SYSTEM SPECIFICATION & INSTRUCTIONS FOR ANTIGRAVITY

> **INSTRUCTION FOR AI AGENT / ANTIGRAVITY:**  
> Read this document as a complete product requirement specification. Generate the entire project folder structure, modular code files (`src/`), main Streamlit application (`app.py`), data processing pipeline for local Excel files (`Data Absen.xlsx` and `Data Visit.xlsx`), and configuration files as specified below. Ensure all Python code is production-ready, clean, well-commented, and fully functional.

---

## 🚀 Project Overview
- **Project Name:** `fmcg-fieldforce-churn-analytics`
- **Domain:** FMCG Human Resources (HRD) & Field Operations Analytics.
- **Goal:** A Streamlit-based Machine Learning dashboard that ingests real-world field force operational data (`Data Absen.xlsx` & `Data Visit.xlsx`) to measure daily workload, visit compliance, effective working time, and overtime to predict employee attrition risk (*churn*) for Merchandisers, Sales Canvassers, and Field Force staff.

---

## 📁 Required File Structure
Generate the following file tree and populate each file with the corresponding implementation:

```text
fmcg-fieldforce-churn-analytics/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── Data Absen.xlsx
│   └── Data Visit.xlsx
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   └── predictor.py
├── app.py
├── requirements.txt
└── README.md


📊 Real Dataset Schema MappingThe application processes two primary local data sources mapped via Personil Code:1. Data Absen.xlsxKey Columns Utilized:Personil Code, Personil Name, Group Personil (User Info)Total Jam Absen (Total attendance time in minutes)Total Jam Efektif (JEM) (Effective work time in minutes)Visited, Not Visited, Effective Call (EC) (Visit achievements)Start Trip, End Trip, First Check In, Last Check Out (Timestamps)2. Data Visit.xlsxKey Columns Utilized:Personil Code, Personil Name, User Group (User Info)Interval (Menit) (Visit duration per outlet in minutes)Jumlah Activity, Jumlah Activity Items (Workload volume per visit)Latitude In, Longitude In, Latitude Out, Longitude Out (Geolocation coordinates)📐 Mathematical Feature Engineering Specs1. Feature Derivation Rules (src/feature_engineering.py)From the merged dataset (Personil Code), calculate the following key metrics:$$\text{Overtime Hours} = \max\left(0, \frac{\text{Total Jam Absen}}{60} - 8\right)$$$$\text{Effective Work Ratio (\%)} = \left(\frac{\text{Total Jam Efektif (JEM)}}{\text{Total Jam Absen}}\right) \times 100$$$$\text{Route Compliance Rate (\%)} = \left(\frac{\text{Visited}}{\text{Visited} + \text{Not Visited}}\right) \times 100$$$$\text{Workload Volume} = \sum (\text{Jumlah Activity Items})$$$$\text{Avg Visit Duration} = \text{Mean}(\text{Interval (Menit)})$$2. Risk Scoring LogicAssign synthetic/historical churn targets ($y \in \{0, 1\}$) during initial training based on realistic stress triggers:High Overtime ($> 2.5$ Hours/day)Low Effective Work Ratio ($< 60\%$)Low Route Compliance Rate ($< 70\%$)Excessive Workload ($> 15$ Activity Items/day)🧱 Module Implementation Specs1. src/data_loader.pyImplement functions:load_raw_data(absen_path: str, visit_path: str) -> tuple[pd.DataFrame, pd.DataFrame]: Loads both Excel sheets safely.merge_datasets(df_absen: pd.DataFrame, df_visit: pd.DataFrame) -> pd.DataFrame: Aggregate df_visit per Personil Code (summing Jumlah Activity Items, averaging Interval (Menit)) and perform a LEFT JOIN with df_absen on Personil Code.2. src/feature_engineering.pyImplement function:process_features(df_merged: pd.DataFrame) -> pd.DataFrame: Calculates Overtime_Hours, Effective_Work_Ratio, Route_Compliance, and handles missing/null values gracefully.3. src/model_trainer.pyImplement function:train_turnover_model(df_features: pd.DataFrame) -> dict:Trains a RandomForestClassifier to predict high churn risk.Extracts Feature Importance metrics.Returns model, scaler, and evaluation metrics (Accuracy, F1-Score, ROC-AUC).4. src/predictor.pyImplement functions:predict_personnel_risk(model_dict, input_features: dict) -> dict: Returns probability score (%) and risk level (Low, Medium, High).get_actionable_recommendations(risk_level: str, metrics: dict) -> list[str]: Generates specific HR action items (e.g., workload re-balancing, route optimization).5. app.py (Streamlit Dashboard)Build a multi-tab dashboard with an executive HR design:Sidebar:File path selector for Data Absen.xlsx and Data Visit.xlsx.Filter by Group Personil / Area ASSM / DSO Name.Tab 1: 📊 Operational Overview & Field Workforce KPIsKPI Cards (st.metric): Total Personnel, Total Visits Completed, Avg Work Duration (Hours), Avg Overtime Hours.Charts: Distribution of Effective Work Ratio vs Overtime, Visit Compliance by Area.Tab 2: 🔮 Individual Personnel Churn Risk EvaluatorDropdown to select specific personil by Personil Name / Personil Code.Gauge / Risk Score Meter indicating predicted churn probability.Breakdown of individual metrics vs average team benchmarks.HR Actionable Recommendations Box (st.warning / st.success).Tab 3: 🧠 ML Model Performance & Feature DriversDisplays Model Accuracy, ROC-AUC, and Feature Importance Chart.Download Button: Export High-Risk Personnel List to Excel (openpyxl).