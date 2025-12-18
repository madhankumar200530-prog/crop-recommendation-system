import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("model/crop_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾")

st.title("🌾 Farmer Crop Recommendation System")
st.write("Enter soil and climate details to get the best crop recommendation.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("Soil pH", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# Predict button
if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    st.success(f"🌱 Recommended Crop: **{prediction[0]}**")
