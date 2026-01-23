import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ==========================================
# 1. CONFIGURACIÓN Y ESTILO
# ==========================================
st.set_page_config(page_title="Actuary Trader AI", layout="wide")

# CSS para estilo profesional
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div.stButton > button { background-color: #0068C9; color: white; border-radius: 8px; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# Gestión de Claves
try:
    API_KEY = st.secrets["GEMINI_KEY"]
except:
    API_KEY = "PEGAR_TU_CLAVE_AQUI" 

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')
except Exception as e:
    st.error(f"Error API: {e}")

st.title("📈 Actuary Trader: Simulación & Análisis Técnico")
st.markdown("---")


# 2. INPUTS

st.sidebar.header("Configuración")
usar_manual = st.sidebar.checkbox("✍️ Escribir Ticker manualmente")

if usar_manual:
    ticker = st.sidebar.text_input("Escribe el Ticker (ej: AMZN, KO, DIS):", value="MSFT").upper()
else:
    acciones = ["GOOGL", "AAPL", "NVDA", "TSLA", "MELI", "YPF", "GGAL", "BTC-USD", "SPY"]
    ticker = st.sidebar.selectbox("Elige un Activo:", acciones)

dias_proyeccion = st.sidebar.slider("Días a Predecir", 30, 90, 30)
num_simulaciones = st.sidebar.slider("Escenarios Monte Carlo", 50, 500, 200)

# 3. FUNCIONES

def obtener_datos(simbolo):
    try:
        data = yf.download(simbolo, period="2y", interval="1d", progress=False, auto_adjust=True)
        if data is None or data.empty: return None
        
        # Limpieza de MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                data = data.xs('Close', axis=1, level=0, drop_level=True)
            elif 'Close' in data.columns.get_level_values(1):
                data = data.xs('Close', axis=1, level=1, drop_level=True)
        
        data = data.dropna()
        if data.empty: return None
        # Asegurar nombre de columna
        if len(data.columns) == 1: data.columns = [simbolo]
        return data
    except: return None

def calcular_indicadores_tecnicos(df, ticker):
    precios = df[ticker]
    
    # 1. RSI (14 días)
    delta = precios.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 2. Medias Móviles (SMA)
    sma_50 = precios.rolling(window=50).mean()
    sma_200 = precios.rolling(window=200).mean()
    
    return rsi, sma_50, sma_200

def proyeccion_con_ajuste(precios, dias, simulaciones):
    # 1. AJUSTE HISTÓRICO (Regresión Log-Lineal)
    # Calculamos la tendencia matemática que traía la acción
    y_hist = np.log(precios.values)
    x_hist = np.arange(len(y_hist))
    slope, intercept, _, _, _ = linregress(x_hist, y_hist)
    
    # Línea de tendencia histórica (El "Ajuste del Modelo")
    tendencia_historica = np.exp(intercept + slope * x_hist)
    
    # 2. SIMULACIÓN FUTURA (Monte Carlo basado en esa tendencia)
    ultimo_precio = precios.iloc[-1]
    retornos_log = np.log(precios / precios.shift(1)).dropna()
    volatilidad_diaria = retornos_log.std()
    
    # Usamos la pendiente histórica (slope) como el "Drift" (Tendencia)
    # Esto hace que la proyección siga la inercia que traía la acción
    mu_diaria = slope 
    
    caminos = np.zeros((dias, simulaciones))
    caminos[0] = ultimo_precio
    dt = 1
    
    for t in range(1, dias):
        Z = np.random.normal(0, 1, simulaciones)
        drift = (mu_diaria - 0.5 * volatilidad_diaria**2) * dt
        shock = volatilidad_diaria * np.sqrt(dt) * Z
        caminos[t] = caminos[t-1] * np.exp(drift + shock)
        
    return caminos, tendencia_historica, slope * 252, volatilidad_diaria * np.sqrt(252)

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================
if st.button("🚀 Ejecutar Análisis Completo"):
    with st.spinner("Analizando mercado, ajustando modelos y consultando IA..."):
        
        # A. Datos
        df = obtener_datos(ticker)
        if df is None:
            st.error("Error descargando datos."); st.stop()
            
        # B. Indicadores Técnicos
        rsi_series, sma_50, sma_200 = calcular_indicadores_tecnicos(df, ticker)
        
        # Últimos valores para la IA
        precio_actual = df[ticker].iloc[-1]
        rsi_actual = rsi_series.iloc[-1]
        val_sma50 = sma_50.iloc[-1]
        val_sma200 = sma_200.iloc[-1]
        
        # Señales Técnicas
        senal_medias = "Neutro"
        if val_sma50 > val_sma200: senal_medias = "Alcista (Golden Cross)"
        if val_sma50 < val_sma200: senal_medias = "Bajista (Death Cross)"
        
        estado_rsi = "Neutro"
        if rsi_actual > 70: estado_rsi = "Sobrecompra (Posible corrección)"
        if rsi_actual < 30: estado_rsi = "Sobreventa (Oportunidad)"

        # C. Simulación y Ajuste
        caminos, tendencia_hist, drift_anual, vol_anual = proyeccion_con_ajuste(df[ticker], dias_proyeccion, num_simulaciones)
        precio_esperado = caminos[-1].mean()

        # --- D. VISUALIZACIÓN AVANZADA ---
        st.subheader("1. Ajuste del Modelo vs Realidad + Predicción")
        
        # Crear eje X continuo para unir pasado y futuro
        dias_hist = len(df)
        x_futuro = np.arange(dias_hist, dias_hist + dias_proyeccion)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 1. Pasado Real
        ax.plot(np.arange(dias_hist), df[ticker], color='white', linewidth=2, label='Precio Real Histórico')
        
        # 2. Ajuste del Modelo (Tendencia Matemática)
        ax.plot(np.arange(dias_hist), tendencia_hist, color='yellow', linestyle='--', alpha=0.7, label='Ajuste de Tendencia (Regresión)')
        
        # 3. Futuro (Simulación)
        ax.plot(x_futuro, caminos.mean(axis=1), color='#00CC96', linewidth=3, label='Predicción Media (Monte Carlo)')
        ax.fill_between(x_futuro, np.percentile(caminos, 5, axis=1), np.percentile(caminos, 95, axis=1), color='#00CC96', alpha=0.1, label='Rango de Probabilidad (95%)')
        
        ax.set_facecolor('#0E1117')
        fig.patch.set_facecolor('#0E1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values(): spine.set_edgecolor('white')
        
        ax.legend(facecolor='#262730', labelcolor='white')
        ax.set_title(f"Modelo Ajustado: {ticker}", color='white')
        st.pyplot(fig)

        # --- E. ANÁLISIS TÉCNICO IA ---
        st.subheader("2. Veredicto Técnico (IA)")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RSI (14)", f"{rsi_actual:.1f}")
        col2.metric("SMA 50 vs 200", senal_medias)
        col3.metric("Tendencia Anual", f"{drift_anual:.1%}")
        col4.metric("Predicción (Media)", f"${precio_esperado:.2f}")

        prompt = f"""
        Eres un Analista Técnico Senior de Wall Street. Analiza {ticker} con estos datos técnicos precisos:
        
        DATOS TÉCNICOS:
        1. Precio Actual: ${precio_actual:.2f}
        2. RSI (14): {rsi_actual:.1f} -> {estado_rsi}
        3. Medias Móviles: La SMA 50 está en ${val_sma50:.2f} y la SMA 200 en ${val_sma200:.2f}. Señal: {senal_medias}.
        4. Proyección Matemática: El modelo Monte Carlo ajustado proyecta un precio de ${precio_esperado:.2f} en {dias_proyeccion} días.
        
        TU MISIÓN:
        - Interpreta el cruce de medias (SMA) y el RSI. ¿Coinciden las señales?
        - Compara la proyección matemática con el análisis técnico. ¿Tiene sentido la subida/bajada proyectada?
        - Da una recomendación final de entrada/salida: (Compra Fuerte, Compra, Mantener, Venta).
        """
        
        try:
            response = model.generate_content(prompt)
            st.info(response.text)
        except Exception as e:
            st.warning("IA descansando (Cuota límite). Revisa los gráficos arriba.")

