import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train_turnover_model(df_features: pd.DataFrame) -> dict:
    """
    Trains a RandomForestClassifier to predict high churn risk.
    Extracts Feature Importance metrics.
    Returns model, scaler, and evaluation metrics.
    """
    if df_features.empty:
        return {}

    # Define features and target
    features = ['Overtime_Hours', 'Effective_Work_Ratio', 'Route_Compliance', 'Workload_Volume', 'Avg_Visit_Duration']
    X = df_features[features]
    y = df_features['Churn_Target']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    # Handle case where test set might only have 1 class for AUC
    try:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    except:
        roc_auc = 0.0

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc
    }

    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'feature_importance': importance_df,
        'features_list': features
    }
