import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load pre-trained model and scaler
gbr = joblib.load("gbr_model.pkl")
scaler = joblib.load("scaler.pkl")

# Town mappings (replace with actual names and encodings)
towns = {
    "Ang Mo Kio": 0,
    "Bedok": 1,
    "Bukit Batok": 2,
    "Bukit Merah": 3,
    "Bukit Panjang": 4,
    "Clementi": 5
}

# Flat type mappings (replace with actual names and encodings)
flat_types = {
    "1-Room": 0,
    "2-Room": 1,
    "3-Room": 2,
    "4-Room": 3,
    "5-Room": 4,
    "Executive": 5
}

# Flat model mappings (replace with actual names and encodings)
flat_models = {
    "Model A": 0,
    "Model B": 1,
    "Improved": 2,
    "Simplified": 3,
    "Standard": 4,
    "Premium Apartment": 5
}

# Streamlit app title
st.title("HDB Resale Price Prediction")

# Sidebar for user inputs
st.sidebar.header("Enter Property Details")

# User inputs
town = st.sidebar.selectbox("Town", list(towns.keys()))  # Dropdown with town names
flat_type = st.sidebar.selectbox("Flat Type", list(flat_types.keys()))  # Dropdown with flat type names
floor_area_sqm = st.sidebar.number_input("Floor Area (sqm)", min_value=10.0, max_value=300.0, step=1.0)
flat_model = st.sidebar.selectbox("Flat Model", list(flat_models.keys()))  # Dropdown with flat model names
flat_age = st.sidebar.number_input("Flat Age (years)", min_value=0, max_value=99, step=1)
remaining_lease_months = st.sidebar.number_input("Remaining Lease (months)", min_value=0, max_value=1200, step=1)
average_storey = st.sidebar.number_input("Average Storey", min_value=1, max_value=50, step=1)
price_per_sqm = st.sidebar.number_input("Price per sqm", min_value=1000.0, max_value=10000.0, step=10.0)
remaining_lease_ratio = st.sidebar.slider("Remaining Lease Ratio", min_value=0.0, max_value=1.0, step=0.01)

# Predict button
if st.sidebar.button("Predict"):
    # Prepare input data
    features = np.array([
        towns[town],  # Map town name to its encoding
        flat_types[flat_type],  # Map flat type name to its encoding
        floor_area_sqm,
        flat_models[flat_model],  # Map flat model name to its encoding
        flat_age,
        remaining_lease_months,
        average_storey,
        price_per_sqm,
        remaining_lease_ratio
    ]).reshape(1, -1)

    # Scale features
    scaled_features = scaler.transform(features)

    # Make prediction
    predicted_price = gbr.predict(scaled_features)[0]

    # Display prediction
    st.write(f"### Predicted Resale Price: ${predicted_price:,.2f}")
