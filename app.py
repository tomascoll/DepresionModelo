import streamlit as st
import joblib
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
    st.markdown("""
    Grupo 1: Los "Rescatados" (Sutiles)

    Estas son las frases que un umbral de 0.50 hubiera ignorado, pero tu 0.35 debería atrapar.



    "I just want to stay in bed all day."

    Por qué: No dice "triste" ni "depresión". Solo implica falta de energía (anhedonia). El modelo debería marcarla como Depresión (1).

    "I don't know why I feel this way."

    Por qué: Es vago. Un modelo estricto lo ignoraría. Tu modelo sensible debería sospechar.

    "It's getting harder to pretend I'm okay."

    Por qué: La palabra "harder" y "pretend" suman puntos, pero quizás no llegaban a 0.5. Con 0.35, debería saltar la alerta.

    Grupo 2: Los "Falsos Positivos Esperados" (El costo a pagar)

    Estas frases NO son depresión clínica, son quejas normales. Pero como bajaste la vara, es muy probable que el modelo diga que SÍ son depresión. Esto es normal.



    "I am so stressed about the exam tomorrow."

    Predicción probable: Depresión (1) (Falsa alarma).

    Razón: "Stressed" tiene peso negativo. El modelo prefiere equivocarse y alertarte.

    "I hate rainy days, they make me lazy."

    Predicción probable: Depresión (1) (Falsa alarma).

    Razón: "Hate" y "lazy" son palabras negativas.

    "I am exhausted from the gym."

    Predicción probable: Depresión (1) (Falsa alarma).

    Razón: "Exhausted" es síntoma físico de depresión. El modelo SVM no entiende bien el contexto "gym".

    Grupo 3: Los "Falsos Negativos" (Si falla aquí, baja más el umbral)

    Incluso con 0.35, estas son difíciles. Si el modelo dice "Sano", es que el truncamiento o el vocabulario nos limitan.



    "I'm fine." (Dicho sarcásticamente, pero texto plano).

    Predicción: Sano (0). (El modelo no es adivino).

    "Whatever."

    Predicción: Sano (0). (Demasiado corto, poca señal).

    Grupo 4: La Prueba de Sanidad (Debe dar 0 sí o sí)

    Si el modelo marca esto como depresión, bajamos demasiado el umbral.



    "I am very happy with my life."

    Predicción: Sano (0).

    "The weather is beautiful today."

    Predicción: Sano (0).
        """)

if __name__ == '__main__':
    main()