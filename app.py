import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import os

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Industrial Gas Analytics", 
    layout="wide", 
    page_icon="🏭"
)
st.title("🏭 Panel de Control: Sensores de Gas")

# ==============================================================================
# BARRA LATERAL (Configuración)
# ==============================================================================
st.sidebar.header("Configuración de Datos")

# Selección del tipo de gas (Dataset)
gas_type = st.sidebar.selectbox("Seleccionar Mezcla", ["CO", "methane"])

# Rutas dinámicas según la selección
csv_path = f"datasets/ethylene_{gas_type}.csv"
model_path = f"models/dual_model_v2_{gas_type.lower()}.pkl"

# ==============================================================================
# FUNCIONES DE CARGA
# ==============================================================================
@st.cache_data
def get_data(path):
    """Carga los datos y toma una muestra aleatoria para asegurar variedad"""
    if os.path.exists(path):
        # 1. Leemos todo el archivo (para capturar todas las variaciones de gas)
        df_full = pd.read_csv(path)
        
        # 2. Si es muy grande, tomamos una muestra ALEATORIA de 5000 filas
        if len(df_full) > 5000:
            return df_full.sample(5000, random_state=42)
        
        return df_full
    return None

def load_model_metadata(path):
    """Carga el pickle del modelo para ver sus métricas internas"""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

# Cargar datos
df = get_data(csv_path)
model_data = load_model_metadata(model_path)

# ==============================================================================
# LÓGICA PRINCIPAL (PESTAÑAS)
# ==============================================================================

if df is not None and model_data is not None:
    
    tab1, tab2, tab3 = st.tabs(["📈 Exploración", "🎯 Métricas del Modelo", "🚀 Predicción API"])

    # --------------------------------------------------------------------------
    # TAB 1: EXPLORACIÓN
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader(f"Análisis de Sensores: Mezcla Ethylene + {gas_type}")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Resumen Estadístico (Muestra)**")
            st.dataframe(df.describe().T.head(16), height=500)
        
        with col2:
            st.write("**Distribución de Sensores Clave**")
            sensors_to_plot = ["Sensor_01", "Sensor_08", "Sensor_16"]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            df_melt = df[sensors_to_plot].melt(var_name="Sensor", value_name="Lectura")
            sns.boxplot(data=df_melt, x="Sensor", y="Lectura", ax=ax, palette="viridis")
            st.pyplot(fig)
            
            st.write("**Correlación con el Objetivo**")
            target_col_name = "CO_ppm" if gas_type == "CO" else "Methane_ppm"
            
            if target_col_name in df.columns:
                corr_cols = sensors_to_plot + [target_col_name]
                corr = df[corr_cols].corr()
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
                st.pyplot(fig2)

    # --------------------------------------------------------------------------
    # TAB 2: MÉTRICAS
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("Rendimiento del Modelo (Entrenamiento)")
        
        if 'metrics_val' in model_data:
            metrics = model_data['metrics_val']
            keys = list(metrics.keys())
            
            col_m1, col_m2 = st.columns(2)
            
            for i, gas_key in enumerate(keys):
                with (col_m1 if i == 0 else col_m2):
                    st.markdown(f"### Gas: {gas_key}")
                    c_a, c_b = st.columns(2)
                    c_a.metric("R² Score", f"{metrics[gas_key]['R2']:.4f}")
                    c_b.metric("MAE", f"{metrics[gas_key]['MAE']:.2f}")

        st.divider()
        st.subheader("🔥 Importancia de Variables (Top 10)")
        
        try:
            feature_names = model_data['feature_cols']
            model_keys = list(model_data['models'].keys())
            main_gas_key = next((k for k in model_keys if "Ethylene" not in k), model_keys[0])
            
            st.caption(f"Analizando importancia para el modelo de: **{main_gas_key}**")
            
            model_obj = model_data['models'][main_gas_key]['model_stable']
            
            if hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_
                
                imp_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values(by='importance', ascending=False).head(10)
                
                st.bar_chart(imp_df.set_index('feature'))
            else:
                st.warning("El modelo guardado no expone 'feature_importances_' directamente.")
                
        except Exception as e:
            st.error(f"No se pudo generar el gráfico de importancia: {e}")

    # --------------------------------------------------------------------------
    # TAB 3: PREDICCIÓN API (BENTOML)
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("Simulador de Inferencia")
        st.info(f"Conectando con servicio BentoML para: {gas_type}")
        
        with st.form("input_form"):
            st.write("Ajusta los sensores seleccionados (El resto tomará el valor promedio):")
            
            # Creamos 4 columnas para los sensores 1, 6, 11 y 16
            c1, c2, c3, c4 = st.columns(4)
            
            # Sensores específicos
            s1 = c1.slider("Sensor_01 (TGS2602)", 0.0, 100.0, 45.0)
            s6 = c2.slider("Sensor_06", 0.0, 100.0, 45.0)
            s11 = c3.slider("Sensor_11", 0.0, 100.0, 45.0)
            s16 = c4.slider("Sensor_16 (TGS2620)", 0.0, 100.0, 45.0)
            
            predict_btn = st.form_submit_button("Llamar a la API")

        if predict_btn:
            # 1. Calculamos promedio para rellenar
            avg_val = (s1 + s6 + s11 + s16) / 4
            sensor_values = {f"Sensor_{i:02d}": avg_val for i in range(1, 17)}
            
            # 2. Asignamos los valores específicos
            sensor_values["Sensor_01"] = s1
            sensor_values["Sensor_06"] = s6
            sensor_values["Sensor_11"] = s11
            sensor_values["Sensor_16"] = s16
            
            # 3. Empaquetamos en 'input_data'
            payload = {"input_data": sensor_values}
            
            endpoint = f"http://localhost:3000/predict_{gas_type.lower()}"
            
            try:
                with st.spinner("Procesando en el servidor..."):
                    response = requests.post(endpoint, json=payload)
                
                if response.status_code == 200:
                    data_res = response.json()
                    st.success("✅ Predicción recibida")
                    
                    r1, r2 = st.columns(2)
                    
                    ppm_main = data_res.get(f"{gas_type}_ppm", 0.0)
                    ppm_eth = data_res.get("Ethylene_ppm", 0.0)
                    
                    r1.metric(f"Concentración {gas_type}", f"{ppm_main:.2f} ppm")
                    r2.metric("Concentración Etileno", f"{ppm_eth:.2f} ppm")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ No se pudo conectar a BentoML (localhost:3000).")
                st.markdown("Revisa que tengas corriendo: `bentoml serve service.py:GasAnalyticsService --reload`")

else:
    st.error("⚠️ Faltan archivos críticos")
    st.markdown(f"No se encontró `{csv_path}` o `{model_path}`. Ejecuta el notebook primero.")