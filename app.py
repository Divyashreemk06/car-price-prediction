import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# Load model and encoder
model = joblib.load("car_price_model.pkl")
encoder = joblib.load("car_price_encoder.pkl")

# Title
st.title("🚗 Car Price Prediction")
st.write("Predict the estimated resale value of a used car using Machine Learning.")

st.divider()

# Car details section
st.subheader("🚘 Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Year of Manufacture",
        min_value=2000,
        max_value=2026,
        value=2018,
        step=1
    )

    present_price = st.number_input(
        "Present Price (₹ Lakhs)",
        min_value=0.1,
        max_value=100.0,
        value=8.5,
        step=0.1
    )

    kms_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=1000000,
        value=30000,
        step=1000
    )

with col2:
    owner = st.number_input(
        "Previous Owners",
        min_value=0,
        max_value=3,
        value=0,
        step=1
    )

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

st.divider()

# Prediction button
if st.button("💰 Predict Selling Price", use_container_width=True):

    # Input validation
    if present_price <= 0:
        st.error("Present Price must be greater than 0.")

    elif kms_driven < 0:
        st.error("Kilometers Driven cannot be negative.")

    elif owner < 0 or owner > 3:
        st.error("Previous Owners must be between 0 and 3.")

    elif year > 2026:
        st.error("Please enter a valid year.")

    else:
        new_car = pd.DataFrame({
            "Year": [year],
            "Present_Price": [present_price],
            "Kms_Driven": [kms_driven],
            "Fuel_Type": [fuel_type],
            "Seller_Type": [seller_type],
            "Transmission": [transmission],
            "Owner": [owner]
        })

        numeric_cols = [
            "Year",
            "Present_Price",
            "Kms_Driven",
            "Owner"
        ]

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
            columns=encoder.get_feature_names_out(
                categorical_cols
            )
        )

        new_final = pd.concat(
            [
                new_numeric.reset_index(drop=True),
                new_encoded_df
            ],
            axis=1
        )

        prediction = model.predict(new_final)[0]

        prediction = max(0, prediction)

        st.success(
            f"### 💰 Estimated Selling Price: ₹{prediction:.2f} Lakh"
        )
st.divider()

st.caption(
    "Built using Python, Pandas, Scikit-learn and Streamlit."
)
