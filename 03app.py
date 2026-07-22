"""
Aplicación Streamlit — Clasificador de Enfermedades en Plantas
Pipeline: ResNet152 (extractor de características) + cabeza clasificadora Keras
Despliegue: Streamlit Cloud / GitHub
"""

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import load_model

# ───────────────────────────────────────────────
# Configuración de la página
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Enfermedades en Plantas",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 Clasificador de Enfermedades en Plantas")
st.markdown("Carga una imagen de hoja para obtener la predicción de enfermedad.")
st.markdown("<h2>Elaborado por: Orlando Advíncula Zeballos</h2>", unsafe_allow_html=True)
st.divider()

# ───────────────────────────────────────────────
# Rutas relativas al repositorio
# ───────────────────────────────────────────────
# Debes hacer el clases.txt

RUTA_MODELO  = "modelo.keras"   # archivo en la raíz del repo
RUTA_CLASES  = "clases.txt"      # una clase por línea, orden alfabético

# ───────────────────────────────────────────────
# Carga de modelos (cacheada — solo se ejecuta una vez)
# ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelos… puede tardar unos segundos.")
def cargar_modelos():
    base = ResNet152(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False
    head = load_model(RUTA_MODELO)
    return base, head

@st.cache_data(show_spinner=False)
def cargar_clases():
    with open(RUTA_CLASES, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# ───────────────────────────────────────────────
# Inicialización con validaciones
# ───────────────────────────────────────────────
try:
    base_model, head = cargar_modelos()
    CLASS_NAMES = cargar_clases()
except Exception as exc:
    st.error(f"Error al cargar el modelo o las clases: {exc}")
    st.stop()

# Validación defensiva: salidas del modelo vs clases
n_salidas = head.output_shape[-1]
if n_salidas != len(CLASS_NAMES):
    st.error(
        f"**Desajuste detectado:** el modelo tiene **{n_salidas} salidas** "
        f"pero `clases.txt` contiene **{len(CLASS_NAMES)} clases**. "
        "Verifica que ambos archivos correspondan al mismo entrenamiento."
    )
    st.stop()

# ───────────────────────────────────────────────
# Subida de imagen
# ───────────────────────────────────────────────
archivo = st.file_uploader(
    "Selecciona una imagen de hoja",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if archivo is None:
    st.info("Sube una imagen para comenzar.")
    st.stop()

# ───────────────────────────────────────────────
# Preprocesamiento — mismo pipeline que el notebook
# ───────────────────────────────────────────────
file_bytes = np.frombuffer(archivo.read(), np.uint8)
img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("No se pudo decodificar la imagen. Intenta con otro archivo.")
    st.stop()

img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))
img_pre     = preprocess_input(img_resized.astype("float32"))

# ───────────────────────────────────────────────
# Predicción
# ───────────────────────────────────────────────
with st.spinner("Analizando imagen…"):
    features = base_model.predict(np.expand_dims(img_pre, 0), verbose=0)
    probs    = head.predict(features, verbose=0)[0]

# Chequeo de sanidad sobre softmax
if abs(probs.sum() - 1.0) > 0.01:
    st.warning(
        f"Las probabilidades suman **{probs.sum():.4f}** (se esperaba ~1.0). "
        "Podría haber un desajuste en las características extraídas."
    )

idx_ganador    = int(np.argmax(probs))
clase_predicha = CLASS_NAMES[idx_ganador]
confianza      = float(probs[idx_ganador])
nombre_archivo = archivo.name

# ───────────────────────────────────────────────
# Resultados
# ───────────────────────────────────────────────
st.subheader("📋 Resultado")

col_img, col_info = st.columns([1, 1], gap="large")

with col_img:
    st.image(img_rgb, caption=nombre_archivo, use_container_width=True)

with col_info:
    st.metric("Archivo",        nombre_archivo)
    st.metric("Clase predicha", clase_predicha)
    st.metric("Confianza",      f"{confianza * 100:.2f}%")

st.divider()

# ───────────────────────────────────────────────
# Desglose de probabilidades (todas las clases)
# ───────────────────────────────────────────────
st.subheader("📊 Desglose de probabilidades")

sorted_idx    = np.argsort(probs)[::-1]
sorted_clases = [CLASS_NAMES[i] for i in sorted_idx]
sorted_probs  = [round(float(probs[i]) * 100, 2) for i in sorted_idx]

df_probs = pd.DataFrame({"Clase": sorted_clases, "Probabilidad (%)": sorted_probs})
st.dataframe(df_probs, use_container_width=True, hide_index=True)

st.bar_chart(
    data=df_probs.set_index("Clase"),
    use_container_width=True,
    color="#2ecc71",
)

#   python.exe -m streamlit run 03app.py
#   En la web de Streamlit, pasar a Python 3.11