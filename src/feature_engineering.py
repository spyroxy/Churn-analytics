import pandas as pd
import numpy as np

def process_features(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Overtime_Hours, Effective_Work_Ratio, Route_Compliance, 
    and handles missing/null values gracefully.
    """
    if df_merged.empty:
        return df_merged

    df = df_merged.copy()
    
    # Fill NA for calculation columns
    df['Total Jam Absen'] = df['Total Jam Absen'].fillna(0)
    df['Total Jam Efektif (JEM)'] = df['Total Jam Efektif (JEM)'].fillna(0)
    df['Visited'] = df['Visited'].fillna(0)
    df['Not Visited'] = df['Not Visited'].fillna(0)
    df['Jumlah Activity Items'] = df['Jumlah Activity Items'].fillna(0)
    df['Interval (Menit)'] = df['Interval (Menit)'].fillna(0)

    # Calculate Overtime Hours: max(0, (Total Jam Absen / 60) - 8)
    df['Overtime_Hours'] = np.maximum(0, (df['Total Jam Absen'] / 60) - 8)

    # Calculate Effective Work Ratio (%)
    # Handle division by zero
    df['Effective_Work_Ratio'] = np.where(
        df['Total Jam Absen'] > 0,
        (df['Total Jam Efektif (JEM)'] / df['Total Jam Absen']) * 100,
        0
    )

    # Calculate Route Compliance Rate (%)
    total_route = df['Visited'] + df['Not Visited']
    df['Route_Compliance'] = np.where(
        total_route > 0,
        (df['Visited'] / total_route) * 100,
        0
    )
    
    df['Workload_Volume'] = df['Jumlah Activity Items']
    df['Avg_Visit_Duration'] = df['Interval (Menit)']

    # Risk Scoring Logic (Synthetic Churn Target generation for training)
    # High Overtime (> 2.5 Hours/day), Low Effective Work Ratio (< 60%), 
    # Low Route Compliance Rate (< 70%), Excessive Workload (> 15 Activity Items/day)
    def calculate_risk(row):
        stress_triggers = 0
        if row['Overtime_Hours'] > 2.5: stress_triggers += 1
        if row['Effective_Work_Ratio'] < 60: stress_triggers += 1
        if row['Route_Compliance'] < 70: stress_triggers += 1
        if row['Workload_Volume'] > 15: stress_triggers += 1
        
        # If at least 2 stress triggers are met, mark as high risk (churn = 1)
        return 1 if stress_triggers >= 2 else 0

    df['Churn_Target'] = df.apply(calculate_risk, axis=1)

    return df
