import streamlit as st
import joblib
import numpy as np

model = joblib.load('model/forest_fire_model.pkl')

st.title("🌲 Forest Fire Early Detection System")
st.subheader("Predict fire risk using environmental conditions.")

temp = st.slider("Temperature (°C)", 0, 50, 25)
RH = st.slider("Relative Humidity (%)", 0, 100, 50)
wind = st.slider("Wind Speed (km/h)", 0.0, 20.0, 5.0)
rain = st.slider("Rainfall (mm)", 0.0, 10.0, 0.0)

if st.button("Predict Fire Risk"):
    input_data = np.array([[temp, RH, wind, rain]])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Forest Fire Detected!")
    else:
        st.success("✅ No Fire Risk Detected.")
