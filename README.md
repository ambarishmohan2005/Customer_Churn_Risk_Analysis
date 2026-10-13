# Enterprise Customer Churn Intelligence Platform

An end-to-end, enterprise-grade Machine Learning system designed for real-time customer churn risk assessment and proactive retention recommendations. Built with an asynchronous **FastAPI** model-serving backend, a **Streamlit** analytics dashboard, and a modern **React** single-page application.

---

## 🏛 Architecture Overview

```
                      ┌─────────────────────────────────────────┐
                      │             Clients / Users             │
                      └────────────────────┬────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │  Streamlit Dashboard  │                     │   React Web App       │
        │   (Port 8501)         │                     │    (Port 3000)        │
        └───────────┬───────────┘                     └───────────┬───────────┘
                    │                                             │
                    │               REST API (HTTP/JSON)          │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────┐
                    │          FastAPI Microservice           │
                    │               (Port 8000)               │
                    │   - Pydantic v2 Schema Validation       │
                    │   - CORS & Security Middleware          │
                    │   - High-Throughput Batch Processing    │
                    │   - OpenAPI / Swagger (/docs)           │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │         ML Inference Engine             │
                    │   - XGBoost Classifier (binary:logistic)│
                    │   - Scikit-Learn StandardScaler         │
                    │   - Categorical LabelEncoders           │
                    │   - Rule-based Retention Strategies     │
                    └─────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Customer_Churn_Project/
├── api/
│   ├── __init__.py
│   ├── schemas.py           # Pydantic v2 data models with field validations
│   ├── predictor.py         # ChurnPredictor singleton (loading, encoding, inference)
│   └── main.py              # FastAPI application (endpoints, CORS, error handling)
├── frontend-react/          # Modern React Single-Page Application
│   ├── package.json         # React & Vite dependencies
│   ├── vite.config.js       # Vite build & proxy config
│   ├── index.html           # HTML entry point
│   ├── Dockerfile.react     # Multi-stage production Nginx container
│   └── src/
│       ├── App.jsx          # React dashboard (form, probability meter, batch upload)
│       ├── main.jsx         # React DOM mount point
│       └── index.css        # Clean dark-mode enterprise UI styles
├── src/
│   └── train.py             # ML training pipeline (SMOTE oversampling + XGBoost)
├── app.py                   # Streamlit frontend client (connected to FastAPI API)
├── Dockerfile.api           # Container image for FastAPI Backend
├── Dockerfile.streamlit     # Container image for Streamlit Frontend
├── docker-compose.yml       # Production orchestration for all services
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
```

---

## ⚡ API Endpoint Specification

The FastAPI microservice provides interactive Swagger UI at `http://localhost:8000/docs` and ReDoc at `http://localhost:8000/redoc`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Healthcheck endpoint monitoring model readiness. |
| `GET` | `/model/metadata` | Metadata on underlying model type, features, and ROC-AUC. |
| `POST` | `/predict` | Real-time single customer inference + tailored retention advice. |
| `POST` | `/predict/batch` | High-throughput batch inference for up to 5,000 customers. |
| `POST` | `/predict/csv` | Multipart CSV upload: scores records and returns an enriched CSV. |

---

## 🚀 Step-by-Step Local Deployment

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start the FastAPI Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API is live at: `http://localhost:8000`
- Swagger UI docs: `http://localhost:8000/docs`

### Step 3: Run the Frontends (Choose Either or Both)

#### Option A: Streamlit Frontend
In a new terminal:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

#### Option B: React Frontend
In a new terminal:
```bash
cd frontend-react
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## 🐳 Production Deployment with Docker Compose (Recommended)

Orchestrate the FastAPI backend, Streamlit dashboard, and React application with a single command:

```bash
docker-compose up --build
```

### Service Map:
- **FastAPI Backend**: `http://localhost:8000` (`/docs` for API explorer)
- **Streamlit Frontend**: `http://localhost:8501`
- **React Frontend**: `http://localhost:3000`

To stop the services:
```bash
docker-compose down
```

---

## ☁️ Enterprise Cloud Deployment Guides

### 1. Google Cloud Run (Fully Managed Serverless Container)

Deploy the FastAPI backend microservice to Google Cloud Run:
```bash
# 1. Build and push the image to Google Artifact Registry
gcloud builds submit --tag gcr.io/<PROJECT_ID>/churn-fastapi -f Dockerfile.api .

# 2. Deploy to Cloud Run
gcloud run deploy churn-fastapi \
  --image gcr.io/<PROJECT_ID>/churn-fastapi \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 1Gi \
  --cpu 1
```

Deploy the Streamlit frontend with backend URL injected:
```bash
gcloud run deploy churn-streamlit \
  --image gcr.io/<PROJECT_ID>/churn-streamlit \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501 \
  --set-env-vars BACKEND_API_URL=https://churn-fastapi-<auto-hash>.run.app
```

---

### 2. AWS (ECS / Fargate with Application Load Balancer)

1. **Push images to Amazon ECR**:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t churn-fastapi -f Dockerfile.api .
   docker tag churn-fastapi:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/churn-fastapi:latest
   docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/churn-fastapi:latest
   ```
2. **Deploy on ECS Fargate**:
   - Create a Task Definition pointing to the container image.
   - Configure container port `8000` with health check path `/health`.
   - Attach an Application Load Balancer (ALB) routing traffic to the target group.

---

### 3. Render / Railway PaaS

1. Connect your GitHub repository.
2. **Backend Service**:
   - Environment: `Docker` (Dockerfile: `Dockerfile.api`).
   - Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
3. **Frontend Service**:
   - Environment: `Docker` (Dockerfile: `Dockerfile.streamlit`).
   - Environment Variable: `BACKEND_API_URL=https://<your-backend-service-url>.onrender.com`.

---

## 🔬 Model Retraining & Continuous Learning

To retrain the model with updated customer data:
```bash
python src/train.py
```
This updates:
- `xgboost_churn_model.pkl`
- `scaler.pkl`
- `label_encoders.pkl`
- `feature_names.pkl`

FastAPI dynamically reloads updated artifacts via its lifespan manager without requiring code rewrites.
- Jupyter Notebooks (analysis)