import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib

# Ensure UTF-8 output across Windows, Docker, and Linux consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

def load_data(path='data/Telco-Customer-Churn.csv'):
    """Load the Telco Customer Churn dataset."""
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    """Preprocess the dataset for modeling."""
    # Drop customerID as it's not useful for prediction
    df = df.drop('customerID', axis=1)
    
    # Handle TotalCharges: convert whitespace to NaN, fill with 0, cast to float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce').fillna(0.0)

    # Convert target variable to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Identify categorical columns (TotalCharges is now float so it won't be included)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler, label_encoders, X.columns.tolist()

def train_model(X_train, y_train, X_test, y_test, feature_names=None):
    """Train XGBoost model with optimal parameters."""
    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # Balanced by SMOTE
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    names = feature_names if feature_names is not None else [f"Feature {i}" for i in range(X_train.shape[1])]
    feature_importance = pd.DataFrame({
        'feature': names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return model

def save_artifacts(model, scaler, label_encoders, feature_names, path='.'):
    """Save model and preprocessing artifacts."""
    joblib.dump(model, f'{path}/xgboost_churn_model.pkl')
    joblib.dump(scaler, f'{path}/scaler.pkl')
    joblib.dump(label_encoders, f'{path}/label_encoders.pkl')
    joblib.dump(feature_names, f'{path}/feature_names.pkl')
    
    print("\n[SUCCESS] Model artifacts saved successfully!")
    print(f" - {path}/xgboost_churn_model.pkl")
    print(f" - {path}/scaler.pkl")
    print(f" - {path}/label_encoders.pkl")
    print(f" - {path}/feature_names.pkl")

def main():
    print("[INFO] Loading data...")
    df = load_data('data/Telco-Customer-Churn.csv')
    print(f"   Dataset shape: {df.shape}")
    
    print("\n[INFO] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = preprocess_data(df)
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    print(f"   Features ({len(feature_names)}): {feature_names}")
    print(f"   Churn distribution (train): {y_train.sum()} churn / {len(y_train)-y_train.sum()} no churn")
    
    print("\n[INFO] Training XGBoost model...")
    model = train_model(X_train, y_train, X_test, y_test, feature_names)
    
    print("\n[INFO] Saving artifacts...")
    save_artifacts(model, scaler, label_encoders, feature_names, path='.')
    
    print("\n" + "=" * 50)
    print("[SUCCESS] TRAINING COMPLETE")
    print("=" * 50)

if __name__ == '__main__':
    main()
    main()