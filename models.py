import joblib

# Load the trained Gradient Boosting Regressor and scaler
gbr = joblib.load("gbr_model.pkl")
scaler = joblib.load("scaler.pkl")