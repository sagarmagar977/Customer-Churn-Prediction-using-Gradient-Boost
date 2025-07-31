import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load model and feature names ---
model = joblib.load("model.pkl")
with open("model_features.json", "r") as f:
    feature_names = json.load(f)

# --- Page Configuration ---
st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction App")
st.markdown("Provide customer details based on the top influential features to predict churn likelihood.")

# --- Two Equal Columns for Inputs ---
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=100.0)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    online_security = st.radio("Online Security", ["Yes", "No"], horizontal=True)

with col2:
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    paperless_billing = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

    # --- Predict button and result message side-by-side ---
    bcol1, bcol2 = st.columns([1, 2])

    with bcol1:
        predict = st.button("🎯 Predict", use_container_width=True)
    with bcol2:
        if 'prediction_made' not in st.session_state:
            st.session_state['prediction_made'] = False
        if 'last_prediction' not in st.session_state:
            st.session_state['last_prediction'] = ""

# --- Prepare Input for Prediction ---
input_data = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
input_data["tenure"] = tenure
input_data["TotalCharges"] = total_charges
input_data["MonthlyCharges"] = monthly_charges

# One-hot encodings
if internet_service == "Fiber optic":
    input_data["InternetService_Fiber optic"] = 1
elif internet_service == "No":
    input_data["InternetService_No"] = 1

if payment_method == "Electronic check":
    input_data["PaymentMethod_Electronic check"] = 1

if contract_type == "One year":
    input_data["Contract_One year"] = 1
elif contract_type == "Two year":
    input_data["Contract_Two year"] = 1

if online_security == "Yes":
    input_data["OnlineSecurity_Yes"] = 1
if paperless_billing == "Yes":
    input_data["PaperlessBilling_Yes"] = 1

# --- Make Prediction and Show Message Right of Button ---
if predict:
    prob = model.predict_proba(input_data)[0][1]
    st.session_state['prediction_made'] = True
    if prob > 0.5:
        st.session_state['last_prediction'] = f"⚠️ Likely to Churn (Probability: {prob:.2f})"
    else:
        st.session_state['last_prediction'] = f"✅ Likely to Stay (Probability: {1 - prob:.2f})"

if st.session_state['prediction_made']:
    with bcol2:
        if "Likely to Stay" in st.session_state['last_prediction']:
            st.success(st.session_state['last_prediction'])
        else:
            st.error(st.session_state['last_prediction'])

# --- Show Feature Importance Chart Below ---
with bcol1:
    show_chart = st.button("📊 Show Features", use_container_width=True)

if show_chart:
    st.markdown("### 🔍 Top 10 Important Features")
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = importances[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))  # ⬅️ Adjust chart size here
    ax.barh(top_features[::-1], top_scores[::-1], color='lightgreen')
    ax.set_xlabel("Feature Importance Score")
    ax.set_title("Top 10 Important Features - Gradient Boosting")
    st.pyplot(fig)
