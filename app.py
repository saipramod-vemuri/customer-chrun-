import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model and scaler
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Telecom Customer Churn Predictor")
st.write("Enter customer details to predict the likelihood of churn.")

# Sidebar inputs for the website
st.sidebar.header("Customer Information")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 150.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 9000.0, 500.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Processing the inputs
if st.button("Predict Churn"):
    # Convert inputs to the format the model expects
    # (Note: In a real app, you would include all features used during training)
    features = np.array([[tenure, monthly_charges, total_charges]]) # Simplified example
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"High Risk: This customer is likely to churn. (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk: This customer is likely to stay. (Probability: {1-probability:.2f})")
