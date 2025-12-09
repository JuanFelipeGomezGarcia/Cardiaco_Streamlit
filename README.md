# 🫀 Predicción de Problemas Cardíacos con IA

<div align="center">

![Banner](https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=800&h=200&fit=crop)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cardiacoappjfgg.streamlit.app/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Aplicación web de Machine Learning para predecir riesgos cardíacos basada en edad y colesterol**

[Demo en Vivo](https://cardiacoappjfgg.streamlit.app/) • [Reportar Bug](https://github.com/JuanFelipeGomezGarcia/Cardiaco_Streamlit/issues) • [Solicitar Feature](https://github.com/JuanFelipeGomezGarcia/Cardiaco_Streamlit/issues)

</div>

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Demo](#-demo)
- [Tecnologías](#-tecnologías)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Modelo de Machine Learning](#-modelo-de-machine-learning)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Autor](#-autor)

---

## 🎯 Descripción

Esta aplicación utiliza **Inteligencia Artificial** para predecir la probabilidad de que un paciente sufra problemas cardíacos basándose en dos factores clave:

- 🎂 **Edad** (20-80 años)
- 🩸 **Nivel de Colesterol** (120-600 mg/dL)

El modelo fue entrenado utilizando **Support Vector Machine (SVC)** de scikit-learn con datos reales de pacientes, logrando identificar patrones que permiten clasificar el riesgo cardíaco.

> ⚠️ **Nota Importante:** Esta es una herramienta educativa y **NO reemplaza** el diagnóstico médico profesional.

---

## ✨ Características

### 🎨 Interfaz de Usuario
- ✅ Diseño moderno y responsivo
- 🌓 Soporte para modo claro y oscuro
- 📱 Compatible con dispositivos móviles
- 🎭 Animaciones y efectos visuales

### 📊 Visualizaciones
- 📈 Gráfico de medidor (gauge) de riesgo
- 🎨 Indicadores visuales de colesterol
- 📉 Interpretación automática de resultados

### 🤖 Machine Learning
- 🧠 Modelo SVC entrenado
- ⚡ Predicción en tiempo real
- 📊 Escalado de datos con MinMaxScaler
- 💾 Modelos pre-entrenados incluidos

### 🔧 Funcionalidades
- 🎚️ Sliders interactivos para entrada de datos
- 📋 Recomendaciones personalizadas según resultado
- 🎈 Animaciones para resultados positivos
- 📱 Interfaz intuitiva y fácil de usar

---

## 🎬 Demo

### Captura de Pantalla

![App Screenshot](Img/DemoSS.PNG)

### Prueba la App en Vivo

🔗 **[Abrir Aplicación](https://cardiacoappjfgg.streamlit.app/)**

---

## 🛠️ Tecnologías

### Backend
- ![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
- ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

### Frontend
- ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
- ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
- ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
- ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)

### Deployment
- ![Streamlit Cloud](https://img.shields.io/badge/Streamlit_Cloud-FF4B4B?style=flat&logo=streamlit&logoColor=white)
- ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)

---

## 📥 Instalación

### Prerrequisitos

- Python 3.10 (exactamente)
- pip (gestor de paquetes de Python)
- Git

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/JuanFelipeGomezGarcia/Cardiaco_Streamlit.git
cd Cardiaco_Streamlit
```

2. **Crear ambiente virtual (opcional pero recomendado)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación**
```bash
streamlit run app.py
```

5. **Abrir en el navegador**
```
La app se abrirá automáticamente en: http://localhost:8501
```

---

## 🚀 Uso

### Paso 1: Ingresar Datos del Paciente
En la barra lateral izquierda, ajusta los valores:
- 🎂 **Edad**: Usa el slider para seleccionar entre 20-80 años
- 🩸 **Colesterol**: Ajusta el nivel entre 120-600 mg/dL

### Paso 2: Realizar Predicción
Haz clic en el botón **"🔮 REALIZAR PREDICCIÓN"**

### Paso 3: Interpretar Resultados
La aplicación mostrará:
- 📊 Gráfico de medidor de riesgo
- ✅ o ⚠️ Clasificación de riesgo (Bajo/Alto)
- 📋 Recomendaciones personalizadas
- 🖼️ Visualización del resultado

---

## 🧠 Modelo de Machine Learning

### Algoritmo
**Support Vector Machine (SVC)** - Clasificador de vectores de soporte

### Características del Modelo
- **Entrada**: 2 variables (edad, colesterol)
- **Salida**: Clasificación binaria (0: Sin riesgo, 1: Con riesgo)
- **Preprocesamiento**: MinMaxScaler para normalización
- **Framework**: scikit-learn

### Pipeline de Entrenamiento
```python
1. Carga de datos de pacientes
2. Preprocesamiento con MinMaxScaler
3. Entrenamiento del modelo SVC
4. Validación y evaluación
5. Exportación con joblib
```

### Archivos del Modelo
- `svc_model.jb` - Modelo SVC entrenado
- `scaler.jb` - Escalador MinMaxScaler ajustado

---

## 📁 Estructura del Proyecto

```
Cardiaco_Streamlit/
│
├── app.py                  # Aplicación principal de Streamlit
├── svc_model.jb           # Modelo SVC entrenado
├── scaler.jb              # Escalador MinMaxScaler
├── requirements.txt       # Dependencias del proyecto
├── README.md             # Este archivo
├── .gitignore            # Archivos ignorados por Git
│
└── assets/               # (Opcional) Recursos adicionales
    └── images/
```

---


## 👨‍💻 Autor

**Juan Felipe Gómez García**

- 🎓 Universidad Autónoma de Bucaramanga (UNAB)
- 📧 Email: [juanfelipe.gomezgarcia@gmail.com](mailto:juanfelipe.gomezgarcia@gmail.com)
- 🐙 GitHub: [@JuanFelipeGomezGarcia](https://github.com/JuanFelipeGomezGarcia)

---


## 🙏 Agradecimientos

- 🏥 Datos de entrenamiento basados en registros médicos públicos
- 🎨 Iconos de [Flaticon](https://www.flaticon.com/)
- 📸 Imágenes de [Unsplash](https://unsplash.com/)
- 🚀 Deployment con [Streamlit Cloud](https://streamlit.io/cloud)
- 📚 Documentación de [Scikit-learn](https://scikit-learn.org/)

---

## 📊 Estadísticas del Proyecto

![GitHub stars](https://img.shields.io/github/stars/JuanFelipeGomezGarcia/Cardiaco_Streamlit?style=social)
![GitHub forks](https://img.shields.io/github/forks/JuanFelipeGomezGarcia/Cardiaco_Streamlit?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/JuanFelipeGomezGarcia/Cardiaco_Streamlit?style=social)

---

<div align="center">

**⭐ Si te gustó este proyecto, dale una estrella en GitHub ⭐**

Hecho con ❤️ y ☕ por Juan Felipe Gómez

© 2025 UNAB

</div>
