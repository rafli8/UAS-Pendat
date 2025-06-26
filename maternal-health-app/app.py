import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("maternal_model.pkl")
scaler = joblib.load("maternal_scaler.pkl")

st.set_page_config(page_title="Prediksi Risiko Kesehatan Ibu", layout="centered")
st.title("🩺 Prediksi Risiko Kesehatan Ibu (Maternal Health Risk)")
st.markdown("Masukkan data ibu hamil untuk memprediksi tingkat risikonya (Low, Mid, High).")

# Input dari user
age = st.number_input("Usia (tahun)", min_value=10.0, max_value=100.0, value=30.0)
sbp = st.number_input("Tekanan Darah Sistolik (SystolicBP)", value=120.0)
dbp = st.number_input("Tekanan Darah Diastolik (DiastolicBP)", value=80.0)
bs = st.number_input("Kadar Gula Darah (BS)", value=1.2)
temp = st.number_input("Suhu Tubuh (°C)", value=37.0)
hr = st.number_input("Detak Jantung (HeartRate)", value=80.0)

if st.button("Prediksi Risiko"):
    data = np.array([[age, sbp, dbp, bs, temp, hr]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    risk_labels = ["High Risk", "Low Risk", "Mid Risk"]
    st.success(f"Tingkat Risiko Kesehatan Ibu: **{risk_labels[pred]}**")

st.markdown("---")
st.caption(" ")