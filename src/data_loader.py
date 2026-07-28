import pandas as pd
import streamlit as st

@st.cache_data
def load_raw_data(absen_path: str, visit_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads both Excel sheets safely."""
    try:
        df_absen = pd.read_excel(absen_path)
        df_visit = pd.read_excel(visit_path)
        return df_absen, df_visit
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def merge_datasets(df_absen: pd.DataFrame, df_visit: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate df_visit per Personil Code (summing Jumlah Activity Items, averaging Interval (Menit)) 
    and perform a LEFT JOIN with df_absen on Personil Code.
    """
    if df_absen.empty or df_visit.empty:
        return pd.DataFrame()

    # Ensure numeric columns
    df_visit['Jumlah Activity Items'] = pd.to_numeric(df_visit['Jumlah Activity Items'], errors='coerce').fillna(0)
    df_visit['Interval (Menit)'] = pd.to_numeric(df_visit['Interval (Menit)'], errors='coerce').fillna(0)

    # Aggregate visit data
    visit_agg = df_visit.groupby('Personil Code').agg({
        'Jumlah Activity Items': 'sum',
        'Interval (Menit)': 'mean'
    }).reset_index()
    
    # Left join absen with aggregated visit
    df_merged = pd.merge(df_absen, visit_agg, on='Personil Code', how='left')
    return df_merged
