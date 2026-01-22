import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACIÓN Y SEGURIDAD
# ==========================================
st.set_page_config(page_title="Actuarial Lab Pro", layout="wide")

# Gestión de Claves (Nube vs Local)
try:
    API_KEY = st.secrets["GEMINI_KEY"]
except:
    # 👇👇 PEGA TU CLAVE AQUÍ SI USAS EL PC 👇👇
    API_KEY = "PEGAR_TU_CLAVE_AQUI" 

# Configuración de Gemini
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')
except Exception as e:
    st.error(f"Error configurando API: {e}")

st.title("🛡️ Laboratorio Actuarial: Riesgo y Proyecciones")
st.markdown("---")

# ==========================================
# 2. INPUTS (BARRA LATERAL)
# ==========================================
st.sidebar.header("1. Configuración del Activo")
acciones = ["GOOGL", "AAPL", "NVDA", "TSLA", "MELI", "YPF", "GGAL", "BTC-USD", "SPY"]
ticker = st.sidebar.selectbox("Elige Activo:", acciones)
if st.sidebar.checkbox("Escribir otro manual"):
    ticker = st.sidebar.text_input("Ticker:", value="AMZN").upper()

st.sidebar.header("2. Hipótesis Actuariales")
dias_proyeccion = st.sidebar.slider("Horizonte (Días)", 30, 365, 30)
num_simulaciones = st.sidebar.slider("N° Escenarios", 100, 1000, 200)

st.sidebar.subheader("⚔️ Definición de Tendencia (Mu)")
metodo_mu = st.sidebar.radio("¿Qué retorno base usar?", 
                             ["Histórico (Lo que pasó)", "Hipótesis Manual (Lo que espero)"])

mu_manual = 0.0
if metodo_mu == "Hipótesis Manual (Lo que espero)":
    mu_manual_pct = st.sidebar.number_input("Rendimiento Anual Esperado (%)", value=15.0, step=1.0)
    mu_manual = mu_manual_pct / 100.0

# ==========================================
# 3. FUNCIONES (CORE)
# ==========================================
def obtener_datos(simbolo):
    """Descarga datos blindada contra errores de Yahoo"""
    try:
        data = yf.download(simbolo, period="5y", interval="1d", progress=False, auto_adjust=True)
        
        # Si no bajó nada, retornamos None
        if data is None or data.empty:
            return None

        # Fix para MultiIndex (Problema común de yfinance reciente)
        if isinstance(data.columns, pd.MultiIndex):
            # Intentar buscar 'Close' en nivel 0 o 1
            if 'Close' in data.columns.get_level_values(0):
                data = data.xs('Close', axis=1, level=0, drop_level=True)
            elif 'Close' in data.columns.get_level_values(1):
                data = data.xs('Close', axis=1, level=1, drop_level=True)
                
        # Limpieza final
        data = data.dropna()
        
        # Doble chequeo de vacío
        if data.empty:
            return None
            
        return data
    except Exception as e:
        return None

def monte_carlo_flexible(precios, dias, simulaciones, usar_manual=False, mu_anual_manual=0.10):
    # Validar que tengamos datos suficientes
    if len(precios) < 2:
        return None, 0, 0
        
    retornos_log = np.log(precios / precios.shift(1)).dropna()
    sigma_diaria = retornos_log.std()
    
    # Definir Drift (Tendencia)
    if usar_manual:
        # Convertir Tasa Anual a Diaria simple
        mu_diaria = mu_anual_manual / 252 
    else:
        mu_diaria = retornos_log.mean()

    ultimo_precio = precios.iloc[-1]
    dt = 1
    caminos = np.zeros((dias, simulaciones))
    caminos[0] = ultimo_precio
    
    for t in range(1, dias):
        Z = np.random.normal(0, 1, simulaciones)
        # Fórmula Geométrica Browniana: Drift real = mu - 0.5*sigma^2
        drift = (mu_diaria - 0.5 * sigma_diaria**2) * dt
        shock = sigma_diaria * np.sqrt(dt) * Z
        caminos[t] = caminos[t-1] * np.exp(drift + shock)
        
    return caminos, sigma_diaria * np.sqrt(252), mu_diaria * 252

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================
if st.button("🚀 Ejecutar Análisis"):
    with st.spinner(f"Descargando y simulando {ticker}..."):
        
        # 1. OBTENER DATOS (Con blindaje)
        df = obtener_datos(ticker)
        
        # Si falló la descarga, avisamos y paramos.
        if df is None or df.empty:
            st.error(f"❌ No se pudieron descargar datos para '{ticker}'.")
            st.warning("Posibles causas: Ticker incorrecto o Yahoo Finance está fallando temporalmente.")
            st.stop()
            
        # Validar columnas
        if ticker not in df.columns:
            # Si solo hay una columna, asumimos que es esa
            if len(df.columns) == 1:
                df.columns = [ticker]
            else:
                st.error("Error de formato en los datos descargados.")
                st.stop()

        # 2. SIMULACIÓN
        usar_manual = (metodo_mu == "Hipótesis Manual (Lo que espero)")
        caminos, vol_anual, mu_usado = monte_carlo_flexible(
            df[ticker], dias_proyeccion, num_simulaciones, 
            usar_manual=usar_manual, mu_anual_manual=mu_manual
        )
        
        if caminos is None:
            st.error("Datos insuficientes para simular.")
            st.stop()

        precio_actual = df[ticker].iloc[-1]
        precio_esperado = caminos[-1].mean()
        
        # 3. MOSTRAR RESULTADOS
        st.subheader(f"📊 Proyección a {dias_proyeccion} días")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio Actual", f"${precio_actual:.2f}")
        c2.metric("Tendencia Base (Mu)", f"{mu_usado:.1%}", help="Retorno anual antes de volatilidad")
        c3.metric("Volatilidad Anual", f"{vol_anual:.1%}", help="Riesgo medido como desviación estándar")

        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(caminos, color='gray', alpha=0.1)
        ax.plot(caminos.mean(axis=1), color='blue', linewidth=2, label='Promedio Esperado')
        ax.axhline(y=precio_actual, color='red', linestyle='--', label='Precio Hoy')
        ax.set_title(f"Monte Carlo: {ticker}")
        ax.legend()
        st.pyplot(fig)
        
        # Nota Teórica
        drift_real = mu_usado - 0.5 * (vol_anual**2)
        if drift_real < 0:
            st.warning(f"⚠️ Alerta Actuarial: Aunque esperas ganar {mu_usado:.1%}, la alta volatilidad ({vol_anual:.1%}) está destruyendo el valor compuesto. (Deriva real: {drift_real:.1%})")
        else:
            st.success(f"✅ Escenario favorable: La deriva compuesta es positiva ({drift_real:.1%}).")

        # 4. INTELIGENCIA ARTIFICIAL
        st.subheader("🤖 Análisis de Riesgos (IA)")
        prompt = f"""
        Actúa como consultor actuarial. Analiza:
        - Activo: {ticker}
        - Precio hoy: {precio_actual:.2f}
        - Volatilidad: {vol_anual:.1%}
        - Escenario proyectado (Media): {precio_esperado:.2f}
        
        Dame una recomendación de gestión de riesgos en 3 líneas.
        """
        try:
            response = model.generate_content(prompt)
            st.info(response.text)
        except Exception as e:
            st.warning(f"IA no disponible por cuota gratuita. (Cálculos matemáticos OK). Error: {e}")