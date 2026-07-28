# FMCG Fieldforce Churn Analytics

A Streamlit-based Machine Learning dashboard that ingests real-world field force operational data to measure daily workload, visit compliance, effective working time, and overtime to predict employee attrition risk (churn).

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate   # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `Data Absen.xlsx` and `Data Visit.xlsx` are located in the `data/` directory or at the root (you can select them from the sidebar).

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
