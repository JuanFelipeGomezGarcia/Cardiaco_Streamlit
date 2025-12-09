import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    layout="wide", 
    page_title="Predicción Cardíaca IA", 
    page_icon="🫀",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
        margin: 20px 0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #4ECDC4;
        color: white;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        border: none;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #45b8b0;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        svc_model = joblib.load('svc_model.jb')
        scaler = joblib.load('scaler.jb')
        return svc_model, scaler
    except FileNotFoundError:
        st.error("❌ Error: Archivos del modelo no encontrados.")
        st.stop()

svc_model, scaler = load_models()

# Función de predicción
def predict_heart_problem(age, cholesterol, model, scaler):
    input_data = np.array([[age, cholesterol]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    return prediction[0]

# Header con imagen
st.markdown('<h1 class="main-header">🫀 Predicción de Problemas Cardíacos con IA</h1>', unsafe_allow_html=True)

# Banner image desde URL
st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=1200&h=300&fit=crop", 
         use_container_width=True)

# Información del modelo
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>🤖</h2>
        <h3>Algoritmo</h3>
        <p>Support Vector Machine (SVC)</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2>📊</h2>
        <h3>Variables</h3>
        <p>Edad y Colesterol</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2>⚡</h2>
        <h3>Precisión</h3>
        <p>Modelo Entrenado</p>
    </div>
    """, unsafe_allow_html=True)

# Descripción
st.markdown("""
<div class="info-box">
    <h3>📋 ¿Cómo funciona?</h3>
    <p>Este modelo de <b>Inteligencia Artificial</b> utiliza un algoritmo de <b>Máquinas de Vectores de Soporte (SVC)</b>
    entrenado con datos reales de pacientes para predecir la probabilidad de sufrir problemas cardíacos.</p>
    <p>El modelo analiza dos factores clave: <b>edad</b> y <b>nivel de colesterol</b>, identificando patrones
    que permiten clasificar a los pacientes en dos categorías de riesgo.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar mejorado
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2913/2913133.png", width=100)
st.sidebar.markdown("## 👤 Datos del Paciente")
st.sidebar.markdown("---")

age = st.sidebar.slider("🎂 Edad (años)", min_value=20, max_value=80, value=45, step=1, 
                        help="Seleccione la edad del paciente")
cholesterol = st.sidebar.slider("🩸 Colesterol (mg/dL)", min_value=120, max_value=600, value=200, step=10,
                                help="Nivel de colesterol en sangre")

# Mostrar valores actuales
st.sidebar.markdown("### 📌 Valores Seleccionados:")
st.sidebar.info(f"**Edad:** {age} años")
st.sidebar.info(f"**Colesterol:** {cholesterol} mg/dL")

# Interpretación de colesterol
if cholesterol < 200:
    st.sidebar.success("✅ Colesterol: Normal")
elif cholesterol < 240:
    st.sidebar.warning("⚠️ Colesterol: Límite alto")
else:
    st.sidebar.error("🔴 Colesterol: Alto")

st.sidebar.markdown("---")

# Botón de predicción
predict_button = st.sidebar.button("🔮 REALIZAR PREDICCIÓN", use_container_width=True)

# Resultados
if predict_button:
    with st.spinner('🔄 Analizando datos...'):
        prediction = predict_heart_problem(age, cholesterol, svc_model, scaler)
    
    st.markdown("---")
    st.markdown("## 📊 Resultado del Análisis")
    
    # Crear gráfico de gauge
    if prediction == 0:
        color = "green"
        risk_level = "BAJO"
        risk_value = 25
    else:
        color = "red"
        risk_level = "ALTO"
        risk_value = 75
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Nivel de Riesgo", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#d4edda'},
                {'range': [50, 100], 'color': '#f8d7da'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if prediction == 0:
            st.success("### ✅ RIESGO BAJO")
            st.image("https://cdn-icons-png.flaticon.com/512/5709/5709755.png", width=150)
            st.markdown("""
            ### 🎉 ¡Excelentes noticias!
            
            Según el análisis del modelo de IA:
            - ✅ **No se detecta riesgo significativo** de problemas cardíacos
            - 💚 Los valores analizados están dentro de parámetros favorables
            
            #### 📋 Recomendaciones:
            - Mantener hábitos de vida saludables
            - Realizar ejercicio regularmente
            - Chequeos médicos periódicos
            - Dieta balanceada
            """)
            st.balloons()
        else:
            st.error("### ⚠️ RIESGO DETECTADO")
            st.image("https://cdn-icons-png.flaticon.com/512/5709/5709654.png", width=150)
            st.markdown("""
            ### 🚨 Atención Requerida
            
            Según el análisis del modelo de IA:
            - ⚠️ **Se detecta un riesgo potencial** de problemas cardíacos
            - 🔴 Los valores analizados sugieren precaución
            
            #### 📋 Recomendaciones URGENTES:
            - 🏥 **Consultar con un cardiólogo** lo antes posible
            - 📊 Realizar exámenes médicos completos
            - 💊 Seguir indicaciones médicas estrictamente
            - 🥗 Modificar hábitos alimenticios
            - 🏃 Programa de ejercicio supervisado
            
            ---
            ⚕️ **IMPORTANTE:** Este es un modelo predictivo educativo y **NO reemplaza** 
            el diagnóstico médico profesional. Consulte siempre con un especialista.
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px;'>
    <p style='font-size: 18px; color: #555;'>
        <b>Desarrollado por:</b> Juan Felipe Gómez<br>
        <b>Institución:</b> Universidad Autónoma de Bucaramanga (UNAB)<br>
        <b>Año:</b> 2025<br>
        © Todos los derechos reservados
    </p>
    <p style='font-size: 14px; color: #888; margin-top: 10px;'>
        🤖 Powered by Machine Learning | 🐍 Python | ⚡ Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
