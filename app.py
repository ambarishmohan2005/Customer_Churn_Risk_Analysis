import os
import requests
import streamlit as st
import pandas as pd
import joblib

# API Gateway Configuration
API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# Page Configuration
st.set_page_config(
    page_title="Enterprise Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🏢 Telecom Customer Churn Intelligence Platform")
st.caption("Enterprise Machine Learning Service powered by XGBoost & FastAPI")

# Check Backend API Health
def check_backend_health():
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if res.status_code == 200:
            return True, res.json()
        return False, None
    except Exception:
        return False, None

backend_online, health_data = check_backend_health()

# Top KPI Status bar
col_stat1, col_stat2, col_stat3 = st.columns([1, 1, 2])
with col_stat1:
    if backend_online:
        st.success("🟢 FastAPI Backend: CONNECTED")
    else:
        st.warning("🟡 FastAPI Backend: OFFLINE (Local Mode Active)")

with col_stat2:
    st.info(f"Model: XGBoost (19 features)")

with col_stat3:
    st.markdown(f"**API Endpoint**: `{API_BASE_URL}` | **Docs**: [`/docs`]({API_BASE_URL}/docs)")

# Tabs for Single & Batch Predictions
tab_single, tab_batch = st.tabs(["👤 Single Customer Prediction", "📂 Batch CSV Prediction"])

# --- TAB 1: Single Prediction ---
with tab_single:
    st.sidebar.header("Customer Profile & Subscription")
    
    # Inputs
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.sidebar.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.sidebar.selectbox('Partner', ['No', 'Yes'])
    dependents = st.sidebar.selectbox('Dependents', ['No', 'Yes'])
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 12)
    
    phone_service = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.sidebar.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.sidebar.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    device_protection = st.sidebar.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.sidebar.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.sidebar.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    
    contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ['No', 'Yes'])
    payment_method = st.sidebar.selectbox('Payment Method', [
        'Electronic check', 
        'Mailed check', 
        'Bank transfer (automatic)', 
        'Credit card (automatic)'
    ])
    monthly_charges = st.sidebar.number_input('Monthly Charges ($)', min_value=0.0, max_value=500.0, value=70.0)
    total_charges = round(float(tenure * monthly_charges), 2)

    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == 'Yes' else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    with st.expander("🔍 Review Request Payload"):
        st.json(payload)

    if st.button("🚀 Run Churn Prediction", type="primary"):
        with st.spinner("Calling inference engine..."):
            if backend_online:
                try:
                    response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
                    if response.status_code == 200:
                        result = response.json()
                        pred = result["churn_prediction"]
                        proba = result["churn_probability"]
                        risk_level = result["risk_level"]
                        confidence = result["confidence"]
                        rec = result["key_recommendation"]
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                        st.stop()
                except Exception as e:
                    st.error(f"Failed to query backend: {e}")
                    st.stop()
            else:
                # Local Fallback
                try:
                    from api.predictor import predictor
                    result_obj = predictor.predict_single(pd.Series(payload))
                    pred = result_obj.churn_prediction
                    proba = result_obj.churn_probability
                    risk_level = result_obj.risk_level
                    confidence = result_obj.confidence
                    rec = result_obj.key_recommendation
                except Exception as e:
                    st.error(f"Local inference fallback failed: {e}")
                    st.stop()

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if pred == 1:
                    st.error(f"🚨 {risk_level.upper()} CHURN RISK")
                else:
                    st.success(f"✅ {risk_level.upper()} CHURN RISK")
                
                st.metric(label="Churn Probability", value=f"{proba:.2%}")
                st.metric(label="Model Confidence", value=confidence)

            with res_col2:
                st.subheader("💡 Strategic Retention Recommendation")
                st.info(f"**Recommended Action:** {rec}")
                
                st.markdown(f"""
                - **Churn Prediction**: `{'Will Churn (Yes)' if pred == 1 else 'Will Retain (No)'}`
                - **Risk Category**: `{risk_level}`
                - **Projected Monthly Loss**: `${monthly_charges if pred == 1 else 0.00:,.2f}`
                """)
                
            st.progress(float(proba))

# --- TAB 2: Batch CSV Upload ---
with tab_batch:
    st.subheader("📂 Batch Customer Scoring")
    st.markdown("Upload a CSV file containing customer records to score customer churn probabilities in bulk.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df_preview)}** customer records.")
            st.dataframe(df_preview.head(5))

            if st.button("⚡ Score Uploaded Dataset", type="primary"):
                uploaded_file.seek(0)
                if backend_online:
                    with st.spinner("Scoring via FastAPI `/predict/csv` microservice..."):
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                        res = requests.post(f"{API_BASE_URL}/predict/csv", files=files, timeout=60)
                        if res.status_code == 200:
                            import io
                            scored_df = pd.read_csv(io.StringIO(res.text))
                            st.success("Batch scoring complete!")
                            
                            c1, c2, c3 = st.columns(3)
                            total = len(scored_df)
                            churns = (scored_df['Churn_Prediction'] == 1).sum()
                            c1.metric("Total Customers", total)
                            c2.metric("Predicted Churn", churns)
                            c3.metric("Churn Rate", f"{churns/total:.1%}")

                            st.dataframe(scored_df)
                            st.download_button(
                                "📥 Download Scored CSV",
                                data=res.text,
                                file_name="churn_predictions_scored.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error(f"Error scoring batch: {res.text}")
                else:
                    st.warning("Please start the FastAPI backend to use batch CSV scoring.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# Footer
st.markdown("---")
st.caption("Enterprise ML Architecture: FastAPI Inference Service + Streamlit & React Frontends.")