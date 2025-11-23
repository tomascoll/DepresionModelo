import streamlit as st
import joblib
import numpy as np
from sklearn.svm import SVC

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Detector de Depresión", 
    page_icon="🧠",
    layout="centered"
)

# --- FUNCIÓN DE CARGA ---
@st.cache_resource
def cargar_modelo():
    # Busca el archivo en la misma carpeta donde está app.py
    nombre_archivo = 'modelo_depresion.pkl'
    try:
        data = joblib.load(nombre_archivo)
        return data
    except FileNotFoundError:
        return None

# --- INTERFAZ PRINCIPAL ---
def main():
    st.title("🧠 Detector de Patrones Depresivos")
    st.markdown("""
        Este modelo analiza texto para identificar indicadores lingüísticos de depresión..
        
        *Nota: Esto es una herramienta de demostración y NO sustituye un diagnóstico profesional.*
        """)
    # Cargar modelo
    pack = cargar_modelo()
    
    if pack is None:
        st.error(f"❌ No se encuentra el archivo 'modelo_depresion_final_rbf.pkl'.")
        st.warning("Asegúrate de haber descargado el archivo .pkl de Colab y ponerlo en esta misma carpeta.")
        st.stop()

    modelo = pack['modelo']
    vectorizer = pack['vectorizer']

    # Área de texto
    st.subheader("Ingresa el texto a analizar:")
    texto_usuario = st.text_area("Comentario:", height=150, placeholder="Escribe aquí en inglés...")

    if st.button("Analizar Sentimiento"):
        if not texto_usuario.strip():
            st.warning("El texto está vacío.")
        else:
            with st.spinner("Procesando..."):
                try:
                    # 1. Limpieza
                    texto_truncado = " ".join(texto_usuario.split()[:25])
                    
                    # 2. Vectorización
                    texto_vec = vectorizer.transform([texto_truncado])
                    
                    # 3. Predicción
                    prediccion = modelo.predict(texto_vec)[0]
                    
                    # Intentamos sacar probabilidad si el modelo lo permite
                    try:
                        probs = modelo.predict_proba(texto_vec)[0]
                        confianza = probs[1] if prediccion == 1 else probs[0]
                    except:
                        confianza = 0.0 # Si no tiene probabilidad activada

                    st.divider()

                    if prediccion == 1:
                        st.error("⚠️ Resultado: POSIBLE DEPRESIÓN")
                        if confianza > 0:
                            st.write(f"Confianza del modelo: **{confianza*100:.1f}%**")
                        st.info("El modelo detectó palabras y estructuras asociadas a la clase 'Depresión'.")
                    else:
                        st.success("✅ Resultado: NO DEPRESIÓN")
                        if confianza > 0:
                            st.write(f"Confianza del modelo: **{confianza*100:.1f}%**")

                except Exception as e:
                    st.error(f"Error al procesar: {e}")

if __name__ == '__main__':
    main()