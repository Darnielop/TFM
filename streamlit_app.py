import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import difflib

# Cargar el modelo y los datos solo si no están en la sesión
if "model_loaded" not in st.session_state:
    with st.spinner("Cargando modelo..."):
        model = tf.keras.models.load_model("models/disease_nn_model.h5")
        mlb = joblib.load("datasets/label_binarizer.pkl")
        df_symptoms = pd.read_csv("datasets/Diseases_Symptoms_Processed.csv")
        df_treatments = pd.read_csv("datasets/Diseases_Treatments_Processed.csv")

        columnas_excluir = ["code", "name", "treatments"]
        columnas_presentes = [col for col in columnas_excluir if col in df_symptoms.columns]

        X = df_symptoms.drop(columns=columnas_presentes)
        X.columns = [col.lower() for col in X.columns]

        st.session_state["model_loaded"] = True
        st.session_state["X"] = X
        st.session_state["df_treatments"] = df_treatments
        st.session_state["mlb"] = mlb
        st.session_state["model"] = model

    st.success("Modelo cargado exitosamente.")

# Función para corregir síntomas
def corregir_sintomas(symptoms, available_symptoms):
    available_symptoms_lower = {s.lower(): s for s in available_symptoms}
    corrected = []

    for symptom in symptoms:
        symptom_lower = symptom.lower()
        closest_match = difflib.get_close_matches(symptom_lower, available_symptoms_lower.keys(), n=1, cutoff=0.5)
        if closest_match:
            corrected.append(available_symptoms_lower[closest_match[0]])

    return corrected

# Función para predecir enfermedades
def predict_all_diseases_with_treatments(symptom_input):
    symptom_input = [symptom.lower() for symptom in symptom_input]
    X = st.session_state["X"]
    df_treatments = st.session_state["df_treatments"]
    mlb = st.session_state["mlb"]
    model = st.session_state["model"]

    symptom_vector = np.array([[1 if symptom in symptom_input else 0 for symptom in X.columns]])
    symptom_vector = symptom_vector[:, :model.input_shape[1]]

    if symptom_vector.sum() == 0:
        return []

    probabilities = model.predict(symptom_vector)[0]
    disease_probabilities = {mlb.classes_[i]: prob for i, prob in enumerate(probabilities)}
    sorted_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)

    results = []
    for disease, probability in sorted_diseases:
        if probability >= 0.01:
            treatment_row = df_treatments[df_treatments["name"] == disease]
            if not treatment_row.empty:
                treatment_columns = [col for col in df_treatments.columns[3:] if "Unnamed" not in col]
                treatments = [col for col in treatment_columns if treatment_row.iloc[0][col] == 1]
                treatments = treatments if treatments else ["No hay tratamientos disponibles"]
            else:
                treatments = ["No hay tratamientos disponibles"]
            results.append((disease, probability, treatments))

    return results

# Interfaz de usuario en Streamlit
st.title("Asistente Médico IA")
st.write("Ingresa tus síntomas para obtener un diagnóstico probable.")

# Input único para los síntomas
symptoms_input = st.text_input("Escribe los síntomas separados por comas", key="symptoms_input")

# Botón único para predecir enfermedades
if st.button("Predecir Enfermedades", key="predict_button"):
    symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
    corrected_symptoms = corregir_sintomas(symptoms, st.session_state["X"].columns)
    resultados = predict_all_diseases_with_treatments(corrected_symptoms)

    if not resultados:
        st.warning("No se encontraron enfermedades relacionadas con estos síntomas.")
    else:
        for enfermedad, probabilidad, tratamientos in resultados[:3]:
            st.subheader(f"{enfermedad} - {probabilidad*100:.2f}% de probabilidad")
            st.write("### Posibles tratamientos:")
            for tratamiento in tratamientos:
                st.write(f"- {tratamiento}")
