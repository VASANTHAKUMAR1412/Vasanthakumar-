import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved models and scaler
lr_model = joblib.load("linear_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="House Price Prediction", layout="centered")
st.title("Smart House Price Prediction App")

st.markdown("### Enter House Features to Predict Price")

# Get feature names from scaler
if hasattr(scaler, "feature_names_in_"):
    feature_names = scaler.feature_names_in_
else:
    st.error("Scaler does not have feature names. Please retrain using a DataFrame.")
    st.stop()

# Collect user input for each feature
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# When Predict is clicked
if st.button("Predict"):
    try:
        # Convert inputs to DataFrame
        input_df = pd.DataFrame([user_input])

        # Scale inputs
        input_scaled = scaler.transform(input_df)

        # Predict using all models
        lr_price = lr_model.predict(input_scaled)[0]
        rf_price = rf_model.predict(input_scaled)[0]
        xgb_price = xgb_model.predict(input_scaled)[0]

        # Display predictions
        st.markdown("### Predicted Prices:")
        st.success(f"Linear Regression: ${lr_price:,.2f}")
        st.success(f"Random Forest: ${rf_price:,.2f}")
        st.success(f"XGBoost: ${xgb_price:,.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
