import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Telecom Revenue Prediction",
    page_icon="📊",
    layout="centered"
)


# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = Path(__file__).parent / "models" / "ridge_regression_alpha100.joblib"

model = joblib.load(MODEL_PATH)


# -----------------------------
# App title
# -----------------------------
st.title("📊 Telecom Revenue Prediction")

st.write(
    "Predict a customer's aggregate telecom revenue "
    "using the trained Ridge Regression model."
)


# -----------------------------
# Input fields
# -----------------------------
st.subheader("Customer Information")

network_age = st.number_input(
    "Network Age",
    min_value=0.0,
    value=20.0
)

data_volume = st.number_input(
    "Aggregate Data Volume",
    min_value=0.0,
    value=1000.0
)

aggregate_calls = st.number_input(
    "Aggregate Calls",
    min_value=0.0,
    value=100.0
)

complaint_count = st.number_input(
    "Aggregate Complaint Count",
    min_value=0.0,
    value=0.0
)


# -----------------------------
# Categorical inputs
# -----------------------------

aug_user_type = st.selectbox(
    "August User Type",
    ["2G", "3G", "Other"]
)

sep_user_type = st.selectbox(
    "September User Type",
    ["2G", "3G", "Other"]
)

aug_fav_a = st.selectbox(
    "August Favorite Activity",
    ["ptcl", "ufone", "mobilink", "telenor", "zong", "warid", "0"]
)

sep_fav_a = st.selectbox(
    "September Favorite Activity",
    ["ufone", "ptcl", "mobilink", "telenor", "warid", "zong"]
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Revenue", type="primary"):

    input_data = pd.DataFrame({
        "network_age": [network_age],
        "Aggregate_Data_Vol": [data_volume],
        "Aggregate_Calls": [aggregate_calls],
        "Aggregate_complaint_count": [complaint_count],
        "aug_user_type": [aug_user_type],
        "sep_user_type": [sep_user_type],
        "aug_fav_a": [aug_fav_a],
        "sep_fav_a": [sep_fav_a]
    })

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")

    # Prediction result
    st.subheader("Prediction Result")

    st.metric(
        "Predicted Revenue",
        f"${prediction:,.2f}"
    )

    # Model information
    st.markdown("### Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model:** Ridge Regression")
        st.write("**R²:** 0.198")

    with col2:
        st.write("**MAE:** 487.13")

    # Prediction warning
    st.warning(
        "Predictions are estimates and may differ substantially "
        "from actual customer revenue."
    )