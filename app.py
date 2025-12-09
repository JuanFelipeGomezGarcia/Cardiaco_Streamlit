import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Load the model and the scaler
try:
  svc_model = joblib.load('svc_model.jb')
  scaler = joblib.load('scaler.jb')
except FileNotFoundError:
  st.error("Error: Archivos del modelo (svc_model.jb o scaler.jb) no encontrados.")
  st.stop()

# Function to make prediction
def predict_heart_problem(age, cholesterol, model, scaler):
  input_data = np.array([[age, cholesterol]])
  scaled_data = scaler.transform(input_data)
  prediction = model.predict(scaled_data)
  return prediction[0]

# Streamlit App
st.set_page_config(layout="wide", page_title="Predicción Cardíaca", page_icon="🫀")

st.title("🫀 Modelo IA para predicción de problemas cardiacos")

st.write("""
Este modelo de Inteligencia Artificial utiliza un algoritmo de Máquinas de Vectores de Soporte (SVC)
entrenado con datos de pacientes para predecir la probabilidad de sufrir problemas cardiacos
basado en la edad y el nivel de colesterol. El modelo aprende a identificar patrones en los datos
para separar a los pacientes en dos categorías: aquellos que probablemente no sufrirán problemas cardiacos
y aquellos que sí. Los datos de entrada (edad y colesterol) son escalados para mejorar el rendimiento del modelo.
""")

# Sidebar for user input
st.sidebar.header("Ingrese los datos del paciente")

age = st.sidebar.slider("Edad", min_value=20, max_value=80, value=20, step=1)
cholesterol = st.sidebar.slider("Colesterol", min_value=120, max_value=600, value=200, step=10)

# Make prediction
if st.sidebar.button("Predecir"):
  prediction = predict_heart_problem(age, cholesterol, svc_model, scaler)

  st.subheader("Resultado de la Predicción:")

  if prediction == 0:
    st.success("✅ **No sufrirá del corazón** 😊")
    st.balloons()
    st.markdown("""
    ### 🎉 ¡Buenas noticias!
    Según el modelo, basándose en la edad y nivel de colesterol proporcionados,
    **no se detecta riesgo significativo** de problemas cardíacos.
    
    **Recomendación:** Mantener hábitos saludables y chequeos regulares.
    """)
  else:
    st.error("⚠️ **Riesgo de problemas cardíacos detectado** 😟")
    st.markdown("""
    ### ⚠️ Atención necesaria
    Según el modelo, basándose en la edad y nivel de colesterol proporcionados,
    **se detecta un riesgo potencial** de problemas cardíacos.
    
    **Recomendación:** Consultar con un médico especialista lo antes posible.
    
    ⚕️ *Este es un modelo predictivo y no reemplaza el diagnóstico médico profesional.*
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    Elaborado por: Alfredo Diaz © UNAB 2025
</div>
""", unsafe_allow_html=True)
