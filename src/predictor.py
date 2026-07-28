import pandas as pd

def predict_personnel_risk(model_dict: dict, input_features: dict) -> dict:
    """Returns probability score (%) and risk level (Low, Medium, High)."""
    if not model_dict:
        return {'probability': 0.0, 'risk_level': 'Tidak Diketahui'}

    model = model_dict['model']
    scaler = model_dict['scaler']
    features_list = model_dict['features_list']

    # Convert input to dataframe for scaler
    df_input = pd.DataFrame([input_features], columns=features_list)
    scaled_input = scaler.transform(df_input)

    # Predict
    prob = model.predict_proba(scaled_input)[0][1] * 100
    
    if prob < 30:
        risk_level = 'Rendah'
    elif prob < 70:
        risk_level = 'Sedang'
    else:
        risk_level = 'Tinggi'

    return {
        'probability': round(prob, 2),
        'risk_level': risk_level
    }

def get_actionable_recommendations(risk_level: str, metrics: dict) -> list[str]:
    """Generates specific HR action items based on risk level and metrics."""
    recommendations = []
    
    if risk_level == 'Rendah':
        recommendations.append("Karyawan bekerja dengan baik dan risiko kelelahan (burnout) rendah. Lanjutkan dukungan yang ada saat ini.")
        return recommendations
        
    if metrics.get('Overtime_Hours', 0) > 2.5:
        recommendations.append("Kurangi jam lembur dengan mendistribusikan ulang tugas harian atau menyesuaikan ukuran area/teritori tugas.")
    
    if metrics.get('Effective_Work_Ratio', 100) < 60:
        recommendations.append("Selidiki penyebab rendahnya rasio kerja efektif. Berikan pelatihan tambahan atau optimalkan rute operasional.")
        
    if metrics.get('Route_Compliance', 100) < 70:
        recommendations.append("Kepatuhan rute kunjungan rendah. Diskusikan kendala penyelesaian kunjungan (misalnya: kemacetan, toko tutup).")
        
    if metrics.get('Workload_Volume', 0) > 15:
        recommendations.append("Beban kerja terdeteksi tinggi. Pertimbangkan untuk menyeimbangkan kembali jumlah akun atau menambah staf pendukung.")
        
    if not recommendations and risk_level in ['Sedang', 'Tinggi']:
        recommendations.append("Lakukan sesi diskusi 1-on-1 (tatap muka) untuk membahas kesejahteraan secara umum dan kendala operasional harian.")
        
    return recommendations
