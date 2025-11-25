import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

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
    nombre_archivo = 'modelo_depresion.pkl' # Asegúrate que este sea el nombre correcto
    try:
        data = joblib.load(nombre_archivo)
        return data
    except FileNotFoundError:
        return None

# --- FUNCIÓN PARA MOSTRAR PESOS ---
def explicar_prediccion(modelo, vectorizer, texto_procesado):
    """
    Extrae los coeficientes de las palabras presentes en el texto.
    Solo funciona para modelos lineales (LogisticRegression, LinearSVC).
    """
    try:
        # 1. Verificar si el modelo tiene coeficientes (RBF no tiene)
        if not hasattr(modelo, 'coef_'):
            st.warning("⚠️ Este modelo no soporta la visualización de pesos directos (posible kernel RBF).")
            return

        # 2. Obtener el vocabulario y los pesos
        # get_feature_names_out() es para scikit-learn versiones nuevas
        # si te da error, cambia por get_feature_names()
        feature_names = vectorizer.get_feature_names_out()
        
        # 3. Transformar solo este texto para ver qué indices activa
        texto_vec = vectorizer.transform([texto_procesado])
        indices_activos = texto_vec.nonzero()[1] # Índices de las palabras encontradas
        
        datos_palabras = []
        
        # 4. Cruzar palabras encontradas con sus pesos
        for idx in indices_activos:
            palabra = feature_names[idx]
            peso = modelo.coef_[0][idx]
            impacto = "🔴 Depresivo" if peso > 0 else "🟢 Sano/Neutro"
            datos_palabras.append({
                "Palabra": palabra,
                "Peso (Coeficiente)": round(peso, 4),
                "Tendencia": impacto
            })
            
        # 5. Crear DataFrame y ordenar por impacto absoluto
        if datos_palabras:
            df = pd.DataFrame(datos_palabras)
            df = df.sort_values(by="Peso (Coeficiente)", ascending=False)
            
            st.markdown("##### ¿Por qué el modelo dijo esto?")
            st.dataframe(
                df.style.map(lambda x: 'color: red' if x > 0 else 'color: green', subset=['Peso (Coeficiente)']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No se encontraron palabras conocidas en el vocabulario del modelo.")

    except Exception as e:
        st.error(f"No se pudo explicar la predicción: {e}")

# --- INTERFAZ PRINCIPAL ---
def main():
    st.title("🧠 Detector de Patrones Depresivos")
    st.markdown("""
        Este modelo analiza texto para identificar indicadores lingüísticos de depresión.
        
        *Nota: Esto es una herramienta de demostración y NO sustituye un diagnóstico profesional.*
        """)
    
    # Cargar modelo
    pack = cargar_modelo()
    
    if pack is None:
        st.error(f"❌ No se encuentra el archivo 'modelo_depresion.pkl'.")
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
                    # 1. Limpieza (Truncamiento)
                    texto_truncado = " ".join(texto_usuario.split()[:25])
                    
                    # 2. Vectorización
                    texto_vec = vectorizer.transform([texto_truncado])
                    
                    # 3. Predicción
                    prediccion = modelo.predict(texto_vec)[0]
                    
                    # Intentamos sacar probabilidad
                    confianza = 0.0
                    try:
                        probs = modelo.predict_proba(texto_vec)[0]
                        confianza = probs[1] if prediccion == 1 else probs[0]
                    except:
                        pass # Si el modelo no tiene predict_proba

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

                    # --- NUEVA SECCIÓN: VISUALIZAR PESOS ---
                    with st.expander("🔍 Ver pesos de las palabras (Explicación)"):
                        explicar_prediccion(modelo, vectorizer, texto_truncado)
                    # ---------------------------------------

                except Exception as e:
                    st.error(f"Error al procesar: {e}")

    # Texto explicativo inferior (se mantiene igual)
    st.markdown("""
    ---
    ### Guía de Interpretación de Casos
    
    **Grupo 1: Los "Rescatados" (Sutiles)**
    * "I just want to stay in bed all day." -> Debería marcar Depresión (Anhedonia).
    
    **Grupo 2: Los "Falsos Positivos Esperados"**
    * "I hate rainy days." -> Posible Falsa Alarma por palabras negativas ("hate").
    
    **Grupo 3: La Prueba de Sanidad**
    * "I am very happy with my life." -> Debe marcar Sano.
    """)

if __name__ == '__main__':
    main()
