import streamlit as st
import openai
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import difflib
import os
import re
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración de la API de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

st.set_page_config(
    page_title="Diagnosis",
    page_icon="styles/logo-removebg.png",
    layout="wide"  # Hace que la página use todo el ancho disponible
)

# Cargar el archivo CSS
def load_css(file_name):
    with open(file_name, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("styles/styles.css")

# Cargar modelo y dataset si no están en session_state
if "model_loaded" not in st.session_state:
    with st.spinner("Cargando modelo..."):
        model = tf.keras.models.load_model("models/disease_nn_model.h5")
        mlb = joblib.load("datasets/label_binarizer.pkl")
        df_symptoms = pd.read_csv("datasets/df_Diseases_Symptoms_Processed.csv")
        df_treatments = pd.read_csv("datasets/df_Diseases_Treatments_Processed.csv")

        columnas_excluir = ["code", "name", "treatments"]
        columnas_presentes = [col for col in columnas_excluir if col in df_symptoms.columns]

        X = df_symptoms.drop(columns=columnas_presentes)
        X.columns = [col.lower() for col in X.columns]

        st.session_state["model_loaded"] = True
        st.session_state["X"] = X
        st.session_state["df_treatments"] = df_treatments
        st.session_state["mlb"] = mlb
        st.session_state["model"] = model

# Inicializar session_state para almacenar síntomas corregidos y pendientes
if "symptoms_corrected" not in st.session_state:
    st.session_state["symptoms_corrected"] = {}

if "pending_corrections" not in st.session_state:
    st.session_state["pending_corrections"] = {}

if "disease_predictions" not in st.session_state:
    st.session_state["disease_predictions"] = None


def traducir_texto(texto, src="español", dest="ingles"):
    """Traduce el texto de español a inglés utilizando la API de OpenAI y muestra el resultado en Streamlit."""
    try:
        prompt = f"Traduce el siguiente texto de {src} a {dest}:\n\n{texto}"
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un traductor experto. Responde unicamente con la palabra de la solución"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        translated_text = response.choices[0].message.content.strip()
        return translated_text

    except Exception as e:
        st.error(f"⚠️ Error al traducir: {e}")
        return texto  # Si hay error, retorna el texto original

# Función para sugerir síntomas y manejar términos desconocidos
def sugerir_sintomas(symptoms, available_symptoms):
    available_symptoms_lower = {s.lower(): s for s in available_symptoms}
    pending = {}
    all_simptoms = []

    for symptom in symptoms:
        print(f" sintoma {symptom}")
        symptom_lower = traducir_texto(symptom,"español","ingles")  # Pasar el síntoma como cadena, no como listast.write(f"Término traducido: {symptom_lower}")  # Depuración
        symptom_lower = symptom_lower.lower()
        print(f"minuscula:{symptom_lower}")

        if symptom_lower in available_symptoms_lower:        
            st.session_state["symptoms_corrected"][symptom_lower] = available_symptoms_lower[symptom_lower]
            all_simptoms.append(symptom_lower)

        elif symptom_lower in st.session_state["symptoms_corrected"]:

            continue  

        else:
            closest_matches = difflib.get_close_matches(symptom_lower, available_symptoms_lower.keys(), n=3, cutoff=0.4)
            if closest_matches:
                pending[symptom_lower] = closest_matches
            else:
                st.warning(f"El síntoma '{symptom}' no está registrado y no se encontraron coincidencias.")
                st.session_state["symptoms_corrected"][symptom_lower] = symptom  

    if pending:
        st.session_state["pending_corrections"] = pending
        st.rerun()  # 🔥 Recargar la interfaz inmediatamente para mostrar las sugerencias
    return all_simptoms

# Función para predecir enfermedades
def predict_diseases(symptom_input):
    df_treatments = st.session_state["df_treatments"]
    symptom_input = [symptom.lower() for symptom in symptom_input]
    X = st.session_state["X"]
    mlb = st.session_state["mlb"]
    model = st.session_state["model"]

    X.columns = [col.lower() for col in X.columns]

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

# Función para interactuar con ChatGPT
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
    
    with st.spinner("Generando respuesta del chatbot... ⏳"):
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
if "first_run" not in st.session_state:
    st.session_state.first_run = True
else:
    st.session_state.first_run = False 

if st.session_state.first_run:
    st.rerun()
st.image("styles/medico.png", use_container_width=True)

# Interfaz de usuario
titulo_placeholder = st.empty()  # Espacio reservado para el título
titulo_placeholder.title("Asistente Médico Virtual")
mensaje_placeholder = st.empty()  # Espacio reservado para evitar duplicación
mensaje_placeholder.write("Ingresa tus síntomas para obtener un diagnóstico basado en un modelo de IA y una explicación de un chatbot médico.")

# Input de síntomas
symptoms_input = st.text_input("Escribe los síntomas separados por comas", key="symptoms_input").lower()

# Si hay correcciones pendientes, mostrar opciones y ocultar botón de análisis
import re
import streamlit as st

if st.session_state["pending_corrections"]:
    st.subheader("Confirma los síntomas corregidos antes de continuar")

    with st.spinner("Procesando correcciones... ⏳"):  # 🔄 Spinner mientras carga
        for symptom, options in st.session_state["pending_corrections"].items():
            translated_options = [f"{traducir_texto(option, 'ingles', 'español')} ({option})" for option in options]
            selected_option = st.radio(
                f"¿'{traducir_texto(symptom, 'ingles', 'español')}' ({symptom}) no es un síntoma registrado, te referías a...?",
                translated_options + ["Ninguna de las anteriores"],
                index=0,
                key=f"radio_{symptom}"
            )
            selected_text = selected_option
            match = re.search(r"\((.*?)\)", selected_option)

            if match:
                selected_text = match.group(1)

            st.session_state["symptoms_corrected"][symptom] = selected_text if selected_option != "Ninguna de las anteriores" else symptom

    if st.button("Confirmar selección"):
            st.session_state["pending_corrections"] = {} 
            corrected_symptoms = list(st.session_state["symptoms_corrected"].values())
            print(f"sintomas corregidos {corrected_symptoms}")
            st.session_state["disease_predictions"] = predict_diseases(corrected_symptoms)
            st.rerun()

# Si no hay correcciones pendientes, analizar directamente
elif st.button("Analizar síntomas", key="predict_button"):
    # Reiniciar variables antes de ejecutar el análisis
    st.session_state["disease_predictions"] = None
    st.session_state["symptoms_corrected"] = {}
    st.session_state["pending_corrections"] = {}

    symptoms_sugeridos = []
    symptoms = [s.strip() for s in st.session_state["symptoms_input"].split(",") if s.strip()]
    print(f"antes de sugerir: {symptoms}")

    symptoms_sugeridos = sugerir_sintomas(symptoms, st.session_state["X"].columns)
    print(f"dsp de sugerir: {symptoms_sugeridos}")

    if not st.session_state["pending_corrections"]:
        st.session_state["disease_predictions"] = predict_diseases(symptoms_sugeridos)
        st.rerun()


# Mostrar resultados si ya se generaron
if st.session_state["disease_predictions"]:
    st.subheader("Resultados del Modelo de IA:")

    disease_names = [disease for disease, _, _ in st.session_state["disease_predictions"]]
    tabs = st.tabs(disease_names)  # Crear una pestaña para cada enfermedad

    for i, (disease, prob, treatments) in enumerate(st.session_state["disease_predictions"]):
        with tabs[i]:  # Mostrar cada enfermedad en su tab correspondiente
            st.markdown(f"### {disease}")
            st.write(f"**Probabilidad:** {prob*100:.2f}%")
            st.subheader("Explicación del Chatbot:")
            explanation = chat_with_gpt([(disease, prob, treatments)])  # Llamar a GPT solo con esta enfermedad
            st.write(explanation)
