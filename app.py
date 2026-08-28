import streamlit as st
import os

st.title("Car Price Prediction")

st.write("Files available in the app folder:")

for file in os.listdir("."):
    st.write(file)
