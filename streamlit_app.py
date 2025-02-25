import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import difflib
import main 

# Cargar el modelo y los datos solo si no están en la sesión
if "model_loaded" not in st.session_state:
    with st.spinner("Cargando modelo..."):
        # Llama a la función cargar_modelo() de main.py
        main.cargar_modelo()
        
        # Almacena las variables globales en la sesión de Streamlit
        st.session_state["model_loaded"] = True
        st.session_state["X"] = main.X
        st.session_state["df_treatments"] = main.df_treatments
        st.session_state["mlb"] = main.mlb
        st.session_state["model"] = main.model

# Interfaz de usuario en Streamlit
st.title("Asistente Médico IA")
st.write("Ingresa tus síntomas para obtener un diagnóstico probable.")

# Input único para los síntomas
symptoms_input = st.text_input("Escribe los síntomas separados por comas", key="symptoms_input")

# Botón único para predecir enfermedades
if st.button("Predecir Enfermedades", key="predict_button"):
    symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
    corrected_symptoms = main.corregir_sintomas(symptoms, st.session_state["X"].columns)
    resultados = main.predict_all_diseases_with_treatments(corrected_symptoms)

    if not resultados:
        st.warning("No se encontraron enfermedades relacionadas con estos síntomas.")
    else:
        for enfermedad, probabilidad, tratamientos in resultados[:3]:
            st.subheader(f"{main.traducir_texto(enfermedad,"en","es")} - {probabilidad*100:.2f}% de probabilidad")
            st.write("### Posibles tratamientos:")
            for tratamiento in tratamientos:
                st.write(f"- {main.traducir_texto(tratamiento,"en","es")}")
