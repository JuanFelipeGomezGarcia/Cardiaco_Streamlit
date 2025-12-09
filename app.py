# Deployment la aplicación para determinar si un paciente sufrirá o no del corazon. El modelo fué entrenado usando sklearn con SVC y los datos de entrada fueron escalados usando mixmax scaler. modelo: svc_model.jb escalador: scaler.jb Los modelo fueron guardadps usando joblib. Coloque un titulo asi: Modelo IA para predicción de problemas cardiacos. Haga un resumen de como funciona el modelo para los usarios. coloque en la parte abajo Elaborado por: Alfredo Diaz un emoji de copyright Unab 2025. En el ado izquierdo en sidebar donde con slider el usuario escoja lo siguiente: Edad: 20 años a 80 años, con incremento de 1 año. ColesterioL: Use los valores de parámetros de niveles de colesterol, desde 120 a 600 con incrementos de 10. Por defecto, los valores seleccionado seaN. Edad 20, Colesterol:200. Estos datos deben pasar por el scaler. Los resultados son: 0: No sufrira del corazón, ponerlo en fondo verde y letras negras y un emoji feliz y debajo aparece una imagen llamada Nosufre.jpg 1: Sufrirá del corazón con fondo rojo y letras negras y un emoji triste. y abajo una image Sisufre.jpg. Antes del titulo poner una inagen tipo banner llama cabezote.jpg. verificar si debe instalar y cargar la librearia de sklearn de svc

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model and the scaler
try:
  svc_model = joblib.load('svc_model.jb')
  scaler = joblib.load('scaler.jb')
except FileNotFoundError:
  st.error("Error: Archivos del modelo (svc_model.jb o scaler.jb) no encontrados.")
  st.stop()


# Function to make prediction
def predict_heart_problem(age, cholesterol, model, scaler):
  """
    Predicts the likelihood of a heart problem based on age and cholesterol.

    Args:
        age (int): The age of the patient.
        cholesterol (int): The cholesterol level of the patient.
        model: The trained machine learning model (SVC).
        scaler: The fitted scaler (MinMaxScaler).

    Returns:
        int: The predicted class (0 for no problem, 1 for problem).
  """
  input_data = np.array([[age, cholesterol]])
  scaled_data = scaler.transform(input_data)
  prediction = model.predict(scaled_data)
  return prediction[0]

# Streamlit App
st.set_page_config(layout="wide")

# Load and display the banner image
try:
  st.image('cabezote.jpg', use_column_width=True)
except FileNotFoundError:
  st.warning("Advertencia: Imagen 'cabezote.jpg' no encontrada.")

st.title("Modelo IA para predicción de problemas cardiacos")

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
    st.markdown(
        """
        <style>
        .green-box {
          background-color: #d4edda;
          color: #155724;
          padding: 10px;
          border-radius: 5px;
        }
        </style>
        <div class="green-box">
          <b>0: No sufrirá del corazón</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    try:
      st.image('Nosufre.jpg', caption='Resultado: No Sufrirá Problemas Cardiacos', use_column_width=True)
    except FileNotFoundError:
      st.warning("Advertencia: Imagen 'Nosufre.jpg' no encontrada.")
  else:
    st.markdown(
        """
        <style>
        .red-box {
          background-color: #f8d7da;
          color: #721c24;
          padding: 10px;
          border-radius: 5px;
        }
        </style>
        <div class="red-box">
          <b>1: Sufrirá del corazón</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    try:
      st.image('Sisufre.jpg', caption='Resultado: Sufrirá Problemas Cardiacos', use_column_width=True)
    except FileNotFoundError:
      st.warning("Advertencia: Imagen 'Sisufre.jpg' no encontrada.")
