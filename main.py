import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import difflib
from googletrans import Translator
from deep_translator import GoogleTranslator

# Traductor global
translator = Translator()

# Función para cargar el modelo y los datos
def cargar_modelo():
    
    global model, mlb, X, df_treatments
    try:
        # Rutas relativas a los archivos
        model = tf.keras.models.load_model("models/disease_nn_model.h5")  # Ruta relativa al modelo
        mlb = joblib.load("datasets/label_binarizer.pkl")  # Ruta relativa al binarizador
        df_symptoms = pd.read_csv("datasets/Diseases_Symptoms_Processed.csv")  # Ruta relativa al dataset de síntomas
        df_treatments = pd.read_csv("datasets/Diseases_Treatments_Processed.csv")  # Ruta relativa al dataset de tratamientos
        
        # Excluir columnas irrelevantes
        columnas_excluir = ["code", "name", "treatments"]
        columnas_presentes = [col for col in columnas_excluir if col in df_symptoms.columns]
        
        # Crear X (dataset de síntomas)
        X = df_symptoms.drop(columns=columnas_presentes)
        X.columns = [col.lower() for col in X.columns]
        
        print(f"✅ Dataset de síntomas cargado. Columnas disponibles: {X.columns.tolist()}")
    except Exception as e:
        print(f"⚠️ Error al cargar el modelo o los datos: {e}")
        raise e
    
    # Verificación de las columnas de X
    print(f"✅ Dataset de síntomas cargado. Columnas disponibles: {X.columns.tolist()}")
    
def traducir_texto(texto, src="es", dest="en"):
    """Traduce el texto siempre de español a inglés de manera síncrona."""
    try:
        # Traducción síncrona
        translated = translator.translate(texto, src=src, dest=dest)
        print(f"📝 Traducido '{texto}' -> '{translated.text}'")  # Muestra la traducción
        return translated.text  # Accede al texto traducido
    except Exception as e:
        print(f"⚠️ Error al traducir: {e}")
        return texto  # Si hay error, retorna el texto original

# Función para traducir los síntomas (sincrónica)
def traducir_sintomas(symptoms):
    """Traduce una lista de síntomas de español a inglés."""
    translated_symptoms = []
    for symptom in symptoms:
        # Llamamos a la función para traducir cada síntoma
        translated_symptom = traducir_texto(symptom)  # Llamada sincrónica
        if translated_symptom:  # Asegurarse de que no sea None
            translated_symptoms.append(translated_symptom)
    
    return translated_symptoms  # Devuelve una lista de síntomas traducidos

# Función para corregir los síntomas
def corregir_sintomas(symptoms, available_symptoms):
    """Traduce y corrige los síntomas según la lista disponible en inglés."""
    # Traducir los síntomas primero
    translated_symptoms = traducir_sintomas(symptoms)
    
    available_symptoms_lower = {s.lower(): s for s in available_symptoms}  # Diccionario en minúsculas
    corrected = []
    
    for symptom in translated_symptoms:
        print(f"🔍 Sintoma original: '{symptom}' -> Traducción: '{symptom}'")  # Imprime la traducción
        closest_match = difflib.get_close_matches(symptom.lower(), available_symptoms_lower.keys(), n=1, cutoff=0.5)
        
        print(f"🔍 Closest match: {closest_match}")
        
        if closest_match:
            corrected.append(available_symptoms_lower[closest_match[0]])  # Recupera el nombre original en inglés
        else:
            print(f"⚠️ No se encontró coincidencia exacta para '{symptom}' -> Traducción: '{symptom}'")
    
    return corrected

def predict_all_diseases_with_treatments(symptom_input):
    print(f"\n🔍 Síntomas ingresados: {symptom_input}")
    symptom_input = [symptom.lower() for symptom in symptom_input]
    
    # Asegurar que las columnas de X están en minúsculas
    print(f"🔍 Columnas en X antes de la predicción: {X.columns.tolist()}")
    X.columns = [col.lower() for col in X.columns]
    
    # Vector de síntomas
    symptom_vector = np.array([[1 if symptom in symptom_input else 0 for symptom in X.columns]])
    print(f"🔍 Vector de síntomas generado: {symptom_vector}")
    
    # Ajustar el tamaño del vector de acuerdo con la entrada del modelo
    symptom_vector = symptom_vector[:, :model.input_shape[1]]  # Ajustar al tamaño correcto
    
    num_sintomas_activos = symptom_vector.sum()
    print(f"✔️ Número de síntomas activos en el vector: {num_sintomas_activos}")

    if num_sintomas_activos == 0:
        print("⚠️ No se encontraron síntomas en el dataset. Revisa los síntomas ingresados.")
        return []
    
    # Hacer la predicción
    probabilities = model.predict(symptom_vector)[0]
    disease_probabilities = {mlb.classes_[i]: prob for i, prob in enumerate(probabilities)}
    sorted_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for disease, probability in sorted_diseases:
        if probability >= 0.01:  # Umbral del 1%
            treatment_row = df_treatments[df_treatments['name'] == disease]
            if not treatment_row.empty:
                # Filtrar tratamientos eliminando columnas no relevantes
                treatment_columns = [col for col in df_treatments.columns[3:] if "Unnamed" not in col]
                treatments = [col for col in treatment_columns if treatment_row.iloc[0][col] == 1]
                treatments = treatments if treatments else ["No hay tratamientos disponibles"]
            else:
                treatments = ["No hay tratamientos disponibles"]
            results.append((disease, probability, treatments))
    
    return results

def iniciar_chatbot():
    st.write("Hola, soy tu asistente médico. Voy a preguntarte sobre tus síntomas.")
    symptoms = []
    while True:
        symptom = st.text_input("Menciona un síntoma que tengas (o escribe 'listo' para terminar): ")
        if symptom.lower() == "listo":
            break
        symptoms.append(symptom)
    
    if not symptoms:
        st.write("No ingresaste ningún síntoma. Inténtalo de nuevo.")
        return
    
    st.write("\nAnalizando síntomas...")
    corrected_symptoms = corregir_sintomas(symptoms, X.columns)
    st.write(f"Síntomas corregidos: {corrected_symptoms}")
    
    resultados = predict_all_diseases_with_treatments(corrected_symptoms)
    if not resultados:
        st.write("No encontré enfermedades relacionadas con estos síntomas.")
        return
    
    enfermedad, probabilidad, tratamientos = resultados[0]
    st.write(f"\nSegún los síntomas proporcionados, podrías tener {enfermedad} con una probabilidad del {probabilidad*100:.2f}%.")
    if tratamientos:
        st.write("Posibles tratamientos:")
        for tratamiento in tratamientos:
            st.write(f"- {tratamiento}")
    else:
        st.write("No hay tratamientos disponibles en la base de datos.")

if __name__ == "__main__":
    cargar_modelo()
    iniciar_chatbot()
