import streamlit as st
import joblib
import numpy as np
from sklearn.svm import SVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Detector de Depresión (Híbrido)", 
    page_icon="🧠",
    layout="centered"
)

# --- INICIALIZAR VADER (Solo una vez) ---
analizador_sentimiento = SentimentIntensityAnalyzer()

# --- FUNCIÓN DE CARGA ---
@st.cache_resource
def cargar_modelo():
    # Busca el archivo en la misma carpeta
    # Asegúrate de que el nombre coincida con tu archivo .pkl real
    nombres_posibles = ['modelo_depresion_final_rbf.pkl', 'modelo_depresion.pkl', 'modelo_svm_optimizado.pkl']
    
    for nombre in nombres_posibles:
        try:
            data = joblib.load(nombre)
            # st.toast(f"Modelo cargado: {nombre}") # Descomentar para debug
            return data
        except FileNotFoundError:
            continue
    return None

# --- LÓGICA DE COHERENCIA (SVM + VADER) ---
def analizar_coherencia(texto, prediccion_svm):
    """
    Combina la predicción del modelo experto (SVM) con un análisis 
    de sentimiento general (VADER) para evitar falsos positivos/negativos obvios.
    """
    scores = analizador_sentimiento.polarity_scores(texto)
    compound_score = scores['compound'] # Va de -1 (Muy Negativo) a +1 (Muy Positivo)
    
    # CASO 1: El SVM dice "Depresión" (1) pero el texto es claramente Positivo
    # Ej: "I am very happy with my life" (SVM se confunde por la palabra 'life')
    if prediccion_svm == 1 and compound_score > 0.5:
        return 0, "Corregido por Tono Positivo (VADER)"
    
    # CASO 2: El SVM dice "Sano" (0) pero el texto es Extremadamente Negativo
    # Ej: "I feel empty and rot" (SVM quizás no conoce 'rot', VADER sí)
    if prediccion_svm == 0 and compound_score < -0.6:
        return 1, "Detectado por Tono Negativo Extremo (VADER)"
        
    # Si no hay contradicción fuerte, confiamos en el SVM
    return prediccion_svm, "Modelo SVM"

# --- INTERFAZ PRINCIPAL ---
def main():
    st.title("🧠 Detector de Patrones Depresivos")
    st.markdown("""
        Este sistema utiliza una **Arquitectura Híbrida** (SVM + Análisis de Sentimiento) para identificar indicadores lingüísticos de riesgo.
        
        *Nota: Esta herramienta es un prototipo académico y NO sustituye un diagnóstico profesional.*
    """)

    # Cargar modelo
    pack = cargar_modelo()
    
    if pack is None:
        st.error("❌ Error Crítico: No se encontró el archivo del modelo (.pkl).")
        st.warning("Asegúrate de tener 'modelo_depresion_final_rbf.pkl' en esta carpeta.")
        st.stop()

    modelo = pack['modelo']
    vectorizer = pack['vectorizer']
    
    # Intentamos recuperar el umbral óptimo si se guardó, sino usamos 0.5 por defecto
    umbral_optimo = pack.get('umbral_optimo', None)

    # Área de texto
    st.subheader("Ingresa el texto a analizar:")
    texto_usuario = st.text_area("Comentario:", height=150, placeholder="Escribe aquí en inglés (Ej: 'I feel empty inside')...")

    if st.button("Analizar Salud Mental"):
        if not texto_usuario.strip():
            st.warning("El texto está vacío.")
        else:
            with st.spinner("Procesando patrones lingüísticos..."):
                try:
                    # 1. Limpieza (Truncamiento a 25 palabras para evitar sesgo de longitud)
                    texto_truncado = " ".join(texto_usuario.split()[:25])
                    
                    # 2. Vectorización
                    texto_vec = vectorizer.transform([texto_truncado])
                    
                    # 3. Predicción Base (SVM)
                    # Usamos decision_function si existe para aplicar el umbral manual
                    if hasattr(modelo, "decision_function") and umbral_optimo is not None:
                        puntaje = modelo.decision_function(texto_vec)[0]
                        prediccion_base = 1 if puntaje > umbral_optimo else 0
                        confianza_visual = 1 / (1 + np.exp(-puntaje)) # Sigmoide simple para visualización
                    else:
                        # Fallback a predicción estándar
                        prediccion_base = modelo.predict(texto_vec)[0]
                        try:
                            probs = modelo.predict_proba(texto_vec)[0]
                            confianza_visual = probs[1]
                        except:
                            confianza_visual = 0.5

                    # 4. Capa de Corrección (VADER)
                    prediccion_final, fuente = analizar_coherencia(texto_usuario, prediccion_base)
                    
                    st.divider()

                    # 5. Mostrar Resultados
                    if prediccion_final == 1:
                        st.error("⚠️ Resultado: POSIBLE DEPRESIÓN")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Origen de la detección", value=fuente)
                        with col2:
                            # Mostramos la confianza del modelo base, aunque VADER haya corregido
                            st.metric("Intensidad SVM", value=f"{confianza_visual*100:.1f}%")
                        
                        st.info("El sistema ha detectado patrones semánticos o emocionales de alto riesgo.")
                        
                    else:
                        st.success("✅ Resultado: NO DEPRESIÓN")
                        st.metric("Fuente del análisis", value=fuente)
                        
                        if fuente != "Modelo SVM":
                            st.caption(f"Nota: El modelo SVM detectó riesgo, pero el análisis de sentimiento general (VADER) identificó un tono positivo, anulando la falsa alarma.")

                except Exception as e:
                    st.error(f"Error interno: {e}")

if __name__ == '__main__':
    main()