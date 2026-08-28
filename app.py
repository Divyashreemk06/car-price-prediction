
import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoder
model = joblib.load("car_price_model.pkl")
encoder = joblib.load("car_price_encoder.pkl")

st.title("🚗 Car Price Prediction")
st.write("Enter the car details below to predict its selling price.")

# Numerical inputs
year = st.number_input(
    "Year of Manufacture",
    min_value=2000,
    max_value=2026,
    value=2018
)

present_price = st.number_input(
    "Present Price (₹ Lakhs)",
    min_value=0.1,
    max_value=100.0,
    value=8.5
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=1000000,
    value=30000
)

owner = st.number_input(
    "Number of Previous Owners",
    min_value=0,
    max_value=3,
    value=0,
    step=1
)

# Categorical inputs
fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

# Prediction button
if st.button("Predict Price"):

    new_car = pd.DataFrame({
        "Year": [year],
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Fuel_Type": [fuel_type],
        "Seller_Type": [seller_type],
        "Transmission": [transmission],
        "Owner": [owner]
    })

    # Numerical columns
    numeric_cols = [
        "Year",
        "Present_Price",
        "Kms_Driven",
        "Owner"
    ]

    # Categorical columns
    categorical_cols = [
        "Fuel_Type",
        "Seller_Type",
        "Transmission"
    ]

    new_numeric = new_car[numeric_cols]

    new_encoded = encoder.transform(
        new_car[categorical_cols]
    )

    new_encoded_df = pd.DataFrame(
        new_encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Combine numerical + encoded features
    new_final = pd.concat(
        [new_numeric.reset_index(drop=True),
         new_encoded_df],
        axis=1
    )

    # Predict
    prediction = model.predict(new_final)[0]

    # Prevent unrealistic negative display
    prediction = max(0, prediction)

    st.success(
        f"Estimated Selling Price: ₹{prediction:.2f} lakh"
    )
