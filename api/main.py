import io
from contextlib import asynccontextmanager
from typing import List
import pandas as pd
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from api.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchCustomerInput,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadataResponse,
)
from api.predictor import predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for warm startup and graceful shutdown."""
    print("[INFO] Starting up Churn Prediction Microservice...")
    try:
        predictor.load_artifacts()
        print(f"[INFO] Artifacts loaded. Features: {len(predictor.feature_names)}")
    except Exception as e:
        print(f"[ERROR] Failed to load model artifacts on startup: {e}")
    yield
    print("[INFO] Shutting down Churn Prediction Microservice...")


app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    description="Enterprise-grade Machine Learning microservice for real-time customer churn inference and risk analytics.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for React frontends (Vite, Next.js, CRA) and Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint welcoming clients and linking documentation."""
    return {
        "service": "Telecom Customer Churn Prediction API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Healthcheck endpoint for orchestrators (Kubernetes, AWS ECS, GCP Cloud Run)."""
    is_loaded = predictor.model is not None and predictor.scaler is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        version="1.0.0",
        num_features=len(predictor.feature_names) if predictor.feature_names else 0,
    )


@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Monitoring"])
async def model_metadata():
    """Returns technical metadata about the underlying ML model."""
    return ModelMetadataResponse(
        model_type="XGBoost Classifier (binary:logistic)",
        features=predictor.feature_names,
        target="Churn (1 = Yes, 0 = No)",
        imbalance_strategy="SMOTE (Synthetic Minority Over-sampling Technique)",
        training_roc_auc=0.8252,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(customer: CustomerInput):
    """Generates real-time churn prediction and business recommendations for a single customer."""
    try:
        return predictor.predict_single(customer)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failure: {str(exc)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(payload: BatchCustomerInput):
    """Executes high-throughput batch churn prediction for up to 5,000 customers."""
    try:
        return predictor.predict_batch(payload.customers)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failure: {str(exc)}",
        )


@app.post("/predict/csv", tags=["Inference"])
async def predict_csv(file: UploadFile = File(...)):
    """Accepts a CSV upload of customer records, runs batch predictions, and returns an enriched CSV with churn probabilities."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid .csv format.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Check required columns
        for col in predictor.feature_names:
            if col not in df.columns and col != 'TotalCharges':
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"CSV is missing required feature column: '{col}'",
                )

        if 'TotalCharges' not in df.columns:
            df['TotalCharges'] = pd.to_numeric(
                df['tenure'] * df['MonthlyCharges'], errors='coerce'
            ).fillna(0.0)
        else:
            df['TotalCharges'] = pd.to_numeric(
                df['TotalCharges'].replace(' ', None), errors='coerce'
            ).fillna(0.0)

        # Preprocess and predict
        X_scaled = predictor._preprocess_dataframe(df)
        probas = predictor.model.predict_proba(X_scaled)[:, 1]
        preds = predictor.model.predict(X_scaled)

        df['Churn_Prediction'] = preds
        df['Churn_Probability'] = probas.round(4)
        df['Risk_Level'] = [
            "High" if p >= 0.6 else "Medium" if p >= 0.35 else "Low" for p in probas
        ]

        # Output to CSV string
        stream = io.StringIO()
        df.to_csv(stream, index=False)

        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=scored_customers.csv"},
        )
        return response

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV processing failed: {str(exc)}",
        )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
