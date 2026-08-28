# 🚗 Car Price Prediction

A Machine Learning web application that predicts the estimated selling price of a used car based on its characteristics.

## 📌 Project Overview

The project uses the CarDekho used-car dataset to analyze factors affecting car selling prices and build a machine learning model for price prediction.

The complete workflow includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Categorical feature encoding
- Machine Learning model training
- Model evaluation
- Cross-validation
- Prediction
- Streamlit web application
- Cloud deployment

## 📊 Dataset

The dataset contains 299 car records and 9 features.

### Features

- Car_Name
- Year
- Selling_Price
- Present_Price
- Kms_Driven
- Fuel_Type
- Seller_Type
- Transmission
- Owner

### Target Variable

**Selling_Price**

The target represents the selling price of the used car in lakh rupees.

## 🔍 Exploratory Data Analysis

Some important observations from the analysis:

- Selling price ranges from approximately ₹0.1 lakh to ₹35 lakh.
- Average selling price is approximately ₹4.59 lakh.
- Average present price is approximately ₹7.54 lakh.
- Kilometers driven ranges from 500 to 500,000.
- Diesel cars had a higher average selling price than Petrol and CNG cars.
- Dealer-sold cars had a higher average selling price than individually sold cars.
- Automatic cars had a higher average selling price than manual cars.

## ⚙️ Preprocessing

The categorical features were converted into numerical form using One-Hot Encoding.

Categorical features:

- Fuel_Type
- Seller_Type
- Transmission

The final feature set contained **11 features**.

## 🤖 Machine Learning Models

Two regression models were evaluated:

### 1. Linear Regression

The Linear Regression model achieved:

- MAE: approximately 1.473
- RMSE: approximately 2.524
- Test R²: approximately 0.753
- Average 5-Fold Cross-Validation R²: approximately 0.852

### 2. Random Forest Regression

The Random Forest model achieved:

- MAE: approximately 1.540
- RMSE: approximately 3.718
- Test R²: approximately 0.464

Based on the evaluated test results, **Linear Regression performed better** for this project.

## 📈 Model Interpretation

The Linear Regression coefficients showed that features such as:

- Fuel type
- Transmission
- Seller type
- Present price
- Year

contributed to the model's predictions.

Kms_Driven had a relatively small linear coefficient in the trained model.

## 🌐 Web Application

The project includes a Streamlit web application where users can enter:

- Year of manufacture
- Present price
- Kilometers driven
- Previous owners
- Fuel type
- Seller type
- Transmission

The application then predicts the estimated selling price.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Google Colab
- GitHub

## 📁 Project Structure

```text
car-price-prediction/
│
├── app.py
├── requirements.txt
├── car_price_model.pkl
├── car_price_encoder.pkl
└── README.md
