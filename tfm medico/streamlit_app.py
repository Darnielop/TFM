import streamlit as st
import openai
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import difflib

# Configuración de la API de OpenAI
OPENAI_API_KEY = ""  # Reemplázala con tu clave real
openai.api_key = OPENAI_API_KEY

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
def predict_diseases(symptom_input):
    df_treatments = st.session_state["df_treatments"]
    symptom_input = [symptom.lower() for symptom in symptom_input]
    X = st.session_state["X"]
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
    for disease, prob in sorted_diseases:
        if prob >= 0.01:
            treatment_row = df_treatments[df_treatments["name"] == disease]
            if not treatment_row.empty:
                treatment_columns = [col for col in df_treatments.columns[3:] if "Unnamed" not in col]
                treatments = [col for col in treatment_columns if treatment_row.iloc[0][col] == 1]
                treatments = treatments if treatments else ["No hay tratamientos disponibles"]
            else:
                treatments = ["No hay tratamientos disponibles"]
            results.append((disease, prob, treatments))
    return results

# Función para interactuar con la API de ChatGPT
def chat_with_gpt(disease_predictions):
    if not disease_predictions:
        return "No se encontraron enfermedades relacionadas con estos síntomas."
    
    formatted_predictions = "\n".join([f"{disease} - {prob*100:.2f}%" for disease, prob, *_ in disease_predictions])
    prompt = f"""
    Eres un asistente médico experto. A continuación, te presento los resultados de un modelo de IA que analiza síntomas y predice enfermedades probables:
    
    {formatted_predictions}
    
    Para cada enfermedad detectada, también se incluyen tratamientos recomendados según el modelo. Explica los tratamientos detalladamente, incluyendo ejemplos, efectividad y posibles efectos secundarios.
    
    Tratamientos recomendados:
    """ + "\n".join([f"{disease}: {', '.join(treatments)}" for disease, _, treatments in disease_predictions])
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Eres un asistente médico experto."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error en la consulta: {str(e)}"
    
# Función mejorada para sugerir síntomas solo cuando no hay coincidencia exacta
def sugerir_sintomas(symptoms, available_symptoms):
    available_symptoms_lower = {s.lower(): s for s in available_symptoms}
    corrected = []

    for symptom in symptoms:
        symptom_lower = symptom.lower()

        # Si el síntoma ya está en el dataset, se usa directamente
        if symptom_lower in available_symptoms_lower:
            corrected.append(available_symptoms_lower[symptom_lower])
        else:
            # Buscar síntomas similares
            closest_matches = difflib.get_close_matches(symptom_lower, available_symptoms_lower.keys(), n=3, cutoff=0.4)

            if closest_matches:
                # Mostrar opciones al usuario
                selected_option = st.radio(
                    f"¿Te referías a '{symptom}'?", 
                    [available_symptoms_lower[m] for m in closest_matches] + ["Ninguna de las anteriores"], 
                    index=0
                )
                if selected_option != "Ninguna de las anteriores":
                    corrected.append(selected_option)
            else:
                st.warning(f"No se encontraron coincidencias para '{symptom}'.")
    
    return corrected

# Inicializar session_state para almacenar síntomas corregidos
if "symptoms_corrected" not in st.session_state:
    st.session_state["symptoms_corrected"] = {}

# Función mejorada para sugerir síntomas solo cuando no hay coincidencia exacta
def sugerir_sintomas(symptoms, available_symptoms):
    available_symptoms_lower = {s.lower(): s for s in available_symptoms}
    corrected = []

    for symptom in symptoms:
        symptom_lower = symptom.lower()

        # Si el síntoma ya está en el dataset, se usa directamente
        if symptom_lower in available_symptoms_lower:
            corrected.append(available_symptoms_lower[symptom_lower])
        else:
            # Si el usuario ya corrigió este síntoma, usar la opción guardada
            if symptom_lower in st.session_state["symptoms_corrected"]:
                corrected_symptom = st.session_state["symptoms_corrected"][symptom_lower]
                corrected.append(corrected_symptom)
            else:
                # Buscar síntomas similares
                closest_matches = difflib.get_close_matches(symptom_lower, available_symptoms_lower.keys(), n=3, cutoff=0.4)

                if closest_matches:
                    # Mostrar opciones al usuario
                    selected_option = st.radio(
                        f"¿'{symptom}' no es un síntoma registrado, te referias a ...?", 
                        [available_symptoms_lower[m] for m in closest_matches] + ["Ninguna de las anteriores"], 
                        index=0,
                        key=f"radio_{symptom_lower}"  # Clave única para evitar conflictos
                    )

                    if selected_option != "Ninguna de las anteriores":
                        corrected.append(selected_option)
                        st.session_state["symptoms_corrected"][symptom_lower] = selected_option  # Guardar selección del usuario
                    else:
                        corrected.append(symptom)  # Mantenerlo sin cambios si no hay corrección
                else:
                    st.warning(f"No se encontraron coincidencias para '{symptom}'.")
                    corrected.append(symptom)  # Mantenerlo sin cambios si no hay sugerencias

    return corrected

# Modificación en la interfaz
st.title("Asistente Médico IA con ChatGPT")
st.write("Ingresa tus síntomas para obtener un diagnóstico probable basado en un modelo de IA y una explicación de un chatbot médico.")

# Input de síntomas en minúsculas
symptoms_input = st.text_input("Escribe los síntomas separados por comas", key="symptoms_input").lower()

if st.button("Analizar síntomas", key="predict_button"):
    symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
    corrected_symptoms = sugerir_sintomas(symptoms, st.session_state["X"].columns)

    if corrected_symptoms:
        disease_predictions = predict_diseases(corrected_symptoms)
        chat_response = chat_with_gpt(disease_predictions)

        st.subheader("Resultados del Modelo de IA:")
        if disease_predictions:
            for enfermedad, probabilidad, *_ in disease_predictions:
                st.write(f"- {enfermedad}: {probabilidad*100:.2f}% de probabilidad")
        else:
            st.write("No se encontraron enfermedades relacionadas.")

        st.subheader("Explicación del Chatbot:")
        st.write(chat_response)
    else:
        st.warning("No se introdujeron síntomas válidos.")


