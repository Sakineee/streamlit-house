import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the trained pipeline model
@st.cache(allow_output_mutation=True)
def load_model():
    pipeline = joblib.load("gbr_model_with_ohe_and_scaler.pkl")
    return pipeline

pipeline = load_model()

# Streamlit app title
st.title("HDB Resale Price Prediction")

# Sidebar for user inputs
st.sidebar.header("Enter Property Details")

# Town mappings (replace with actual names and encodings from dataset)
towns = {
    "ANG MO KIO": 0,
    "BEDOK": 1,
    "BISHAN": 2,
    "BUKIT BATOK": 3,
    "BUKIT MERAH": 4,
    "BUKIT PANJANG": 5,
    "BUKIT TIMAH": 6,
    "CENTRAL AREA": 7,
    "CHOA CHU KANG": 8,
    "CLEMENTI": 9,
    "GEYLANG": 10,
    "HOUGANG": 11,
    "JURONG EAST": 12,
    "JURONG WEST": 13,
    "KALLANG/WHAMPOA": 14,
    "MARINE PARADE": 15,
    "PASIR RIS": 16,
    "PUNGGOL": 17,
    "QUEENSTOWN": 18,
    "SEMBAWANG": 19,
    "SENGKANG": 20,
    "SERANGOON": 21,
    "TAMPINES": 22,
    "TOA PAYOH": 23,
    "WOODLANDS": 24,
    "YISHUN": 25
}

# Flat type mappings (replace with actual names and encodings)
flat_types = {
    "2 ROOM": 0,
    "3 ROOM": 1,
    "4 ROOM": 2,
    "5 ROOM": 3,
    "EXECUTIVE": 4,
    "1 ROOM": 5,
    "MULTI-GENERATION": 6
}

# Flat model mappings (replace with actual names and encodings)
flat_models = {
    "Improved": 0,
    "New Generation": 1,
    "DBSS": 2,
    "Standard": 3,
    "Apartment": 4,
    "Simplified": 5,
    "Model A": 6,
    "Premium Apartment": 7,
    "Adjoined flat": 8,
    "Model A-Maisonette": 9,
    "Maisonette": 10,
    "Type S1": 11,
    "Type S2": 12,
    "Model A2": 13,
    "Terrace": 14,
    "Improved-Maisonette": 15,
    "Premium Maisonette": 16,
    "Multi Generation": 17,
    "Premium Apartment Loft": 18,
    "2-room": 19,
    "3Gen": 20
}

# User inputs
town = st.sidebar.selectbox("Town", list(towns.keys()))  # Dropdown with town names
flat_type = st.sidebar.selectbox("Flat Type", list(flat_types.keys()))  # Dropdown with flat type names
floor_area_sqm = st.sidebar.number_input("Floor Area (sqm)", min_value=10.0, max_value=300.0, step=1.0)
flat_model = st.sidebar.selectbox("Flat Model", list(flat_models.keys()))  # Dropdown with flat model names
flat_age = st.sidebar.number_input("Flat Age (years)", min_value=0, max_value=99, step=1)
remaining_lease_months = st.sidebar.number_input("Remaining Lease (months)", min_value=0, max_value=1200, step=1)
average_storey = st.sidebar.number_input("Average Storey", min_value=1, max_value=50, step=1)
price_per_sqm = st.sidebar.number_input("Price per sqm", min_value=1000.0, max_value=10000.0, step=10.0)

# Prepare input data for prediction
input_data = {
    'town': towns[town],  # Map town name to its encoding
    'flat_type': flat_types[flat_type],  # Map flat type name to its encoding
    'flat_model': flat_models[flat_model],  # Map flat model name to its encoding
    'floor_area_sqm': floor_area_sqm,
    'flat_age': flat_age,
    'remaining_lease_months': remaining_lease_months,
    'average_storey': average_storey,
    'price_per_sqm': price_per_sqm,
    'remaining_lease_ratio': remaining_lease_months / (99 * 12),  # calculated ratio
}

# Convert the input_data into a pandas DataFrame
input_df = pd.DataFrame([input_data])

# Predict button
if st.sidebar.button("Predict"):
    # Make prediction using the preprocessed data
    predicted_price = pipeline.predict(input_df)[0]

    # Display prediction
    st.write(f"### Predicted Resale Price: ${predicted_price:,.2f}")

    # --- Impact of Floor Area on Resale Price ---
    floor_area_range = np.linspace(10, 300, 100)  # Floor area range from 10 to 300 sqm
    predicted_prices = []

    for area in floor_area_range:
        temp_input = input_data.copy()
        temp_input['floor_area_sqm'] = area  # Vary floor_area_sqm
        temp_df = pd.DataFrame([temp_input])
        predicted_prices.append(pipeline.predict(temp_df)[0])

    # Plot the impact of floor area on resale price
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(floor_area_range, predicted_prices, label="Predicted Resale Price")
    ax.set_title('Impact of Floor Area on Resale Price')
    ax.set_xlabel('Floor Area (sqm)')
    ax.set_ylabel('Predicted Resale Price')
    st.pyplot(fig)
