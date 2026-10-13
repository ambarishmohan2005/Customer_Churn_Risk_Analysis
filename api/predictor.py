import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from api.schemas import CustomerInput, PredictionResponse, BatchPredictionResponse


class ChurnPredictor:
    """Enterprise Model Serving Engine for Customer Churn Prediction."""

    def __init__(self, artifacts_dir: str = "."):
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Loads serialized machine learning artifacts."""
        model_path = os.path.join(self.artifacts_dir, 'xgboost_churn_model.pkl')
        scaler_path = os.path.join(self.artifacts_dir, 'scaler.pkl')
        encoders_path = os.path.join(self.artifacts_dir, 'label_encoders.pkl')
        features_path = os.path.join(self.artifacts_dir, 'feature_names.pkl')

        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoders_path)):
            raise FileNotFoundError(
                f"Missing one or more required model artifacts in directory '{self.artifacts_dir}'."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)

        if os.path.exists(features_path):
            self.feature_names = joblib.load(features_path)
        else:
            self.feature_names = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                'MonthlyCharges', 'TotalCharges'
            ]

    def _preprocess_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Encodes and scales raw dataframe into numeric feature matrix."""
        df_encoded = df.copy()

        categorical_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]

        for col in categorical_cols:
            if col in self.label_encoders and col in df_encoded.columns:
                le = self.label_encoders[col]
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Align columns with exact training schema order
        df_encoded = df_encoded[self.feature_names].astype(float)

        # Apply standard scaling
        return self.scaler.transform(df_encoded)

    def _generate_recommendation(self, input_data: CustomerInput, proba: float) -> str:
        """Generates contextual business action recommendation."""
        if proba >= 0.5:
            recs = []
            if input_data.Contract == 'Month-to-month':
                recs.append("Incentivize upgrading to an annual contract with a discount.")
            if input_data.InternetService == 'Fiber optic' and input_data.TechSupport == 'No':
                recs.append("Bundle complimentary tech support and device security.")
            if input_data.PaymentMethod == 'Electronic check':
                recs.append("Encourage automated bank or credit card billing with a bill credit.")
            if not recs:
                recs.append("Trigger proactive VIP outreach and satisfaction survey.")
            return " ".join(recs)
        else:
            return "Customer health is solid. Maintain regular engagement and evaluate premium service upsell."

    def predict_single(self, customer: CustomerInput) -> PredictionResponse:
        """Executes real-time inference for a single customer record."""
        df = pd.DataFrame([customer.model_dump()])
        X_scaled = self._preprocess_dataframe(df)

        proba = float(self.model.predict_proba(X_scaled)[0, 1])
        prediction = int(self.model.predict(X_scaled)[0])

        risk_level = "High" if proba >= 0.6 else "Medium" if proba >= 0.35 else "Low"
        confidence = "High" if (proba > 0.7 or proba < 0.3) else "Medium"

        return PredictionResponse(
            churn_prediction=prediction,
            churn_label="Yes" if prediction == 1 else "No",
            churn_probability=round(proba, 4),
            risk_level=risk_level,
            confidence=confidence,
            key_recommendation=self._generate_recommendation(customer, proba)
        )

    def predict_batch(self, customers: List[CustomerInput]) -> BatchPredictionResponse:
        """Executes vector-accelerated batch inference for multiple customers."""
        df = pd.DataFrame([c.model_dump() for c in customers])
        X_scaled = self._preprocess_dataframe(df)

        probas = self.model.predict_proba(X_scaled)[:, 1]
        predictions = self.model.predict(X_scaled)

        results = []
        churn_count = 0

        for i, customer in enumerate(customers):
            proba = float(probas[i])
            pred = int(predictions[i])
            if pred == 1:
                churn_count += 1

            risk = "High" if proba >= 0.6 else "Medium" if proba >= 0.35 else "Low"
            conf = "High" if (proba > 0.7 or proba < 0.3) else "Medium"

            results.append(
                PredictionResponse(
                    churn_prediction=pred,
                    churn_label="Yes" if pred == 1 else "No",
                    churn_probability=round(proba, 4),
                    risk_level=risk,
                    confidence=conf,
                    key_recommendation=self._generate_recommendation(customer, proba)
                )
            )

        total = len(customers)
        return BatchPredictionResponse(
            total_records=total,
            churn_count=churn_count,
            non_churn_count=total - churn_count,
            overall_churn_rate=round(churn_count / total, 4) if total > 0 else 0.0,
            predictions=results
        )


# Global singleton instance
predictor = ChurnPredictor(artifacts_dir=".")
