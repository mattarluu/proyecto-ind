import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import os
import json

#configuracion de la pagina
st.set_page_config(
    page_title="Industrial Gas Analytics", 
    layout="wide", 
    page_icon="🏭"
)

#estilos
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #1f77b4;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Panel de Control: Sensores de Gas Industriales</p>', unsafe_allow_html=True)

#barra lateral
st.sidebar.header("Configuración")

#seleccionar dataset
gas_type = st.sidebar.selectbox(
    "Seleccionar Mezcla de Gases", 
    ["CO", "methane"],
    help="Elige qué dataset analizar: Ethylene+CO o Ethylene+Methane"
)

#Rutas dinámicas según la selección
csv_path = f"datasets/ethylene_{gas_type}.csv"
model_path = f"models/dual_model_v2_{gas_type.lower()}.pkl"

#Info del dataset
st.sidebar.info(f"""
**Dataset Activo:**  
- Archivo: `ethylene_{gas_type}.csv`
- Modelo: `dual_model_v2_{gas_type}`
- Sensors: 16 (TGS-2600/2602/2610/2620)
""")

#funciones de carga
@st.cache_data
def get_data(path):
    """Carga los datos y toma una muestra aleatoria para asegurar variedad"""
    if os.path.exists(path):
        df_full = pd.read_csv(path)
        
        #Si es muy grande, tomamos muestra estratificada
        if len(df_full) > 10000:
            # Muestra que incluya variedad de concentraciones
            return df_full.sample(10000, random_state=42)
        
        return df_full
    return None

@st.cache_data
def load_model_metadata(path):
    """Carga el pickle del modelo para ver sus métricas internas"""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_analysis_results(gas_type):
    """Carga resultados de análisis (anomalías, robustez) si existen"""
    results = {}
    
    # Intentar cargar resumen completo
    summary_path = f'results/COMPLETE_SUMMARY_{gas_type.lower()}.json'
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)
    
    # Intentar cargar anomalías
    anomaly_path = f'results/anomalies_{gas_type.lower()}.json'
    if os.path.exists(anomaly_path):
        with open(anomaly_path, 'r') as f:
            results['anomalies'] = json.load(f)
    
    # Intentar cargar robustez
    robustness_path = f'results/robustness_{gas_type.lower()}.csv'
    if os.path.exists(robustness_path):
        results['robustness'] = pd.read_csv(robustness_path)
    
    return results

# Cargar datos
df = get_data(csv_path)
model_data = load_model_metadata(model_path)
analysis_results = load_analysis_results(gas_type)

#pestañas
if df is not None and model_data is not None:
    
    # Crear pestañas mejoradas
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Exploración",
        "Performance",
        "Anomalías",
        "Robustez",
        "Comparativas",
        "Predicción API"
    ])

    #exploracion
    with tab1:
        st.subheader(f"Análisis Exploratorio: Ethylene + {gas_type}")
        
        # Resumen de datos
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Muestras", f"{len(df):,}")
        with col2:
            st.metric("Sensores", "16")
        with col3:
            target_col = "CO_ppm" if gas_type == "CO" else "Methane_ppm"
            avg_concentration = df[target_col].mean()
            st.metric(f"Concentración Media {gas_type}", f"{avg_concentration:.2f} ppm")
        with col4:
            st.metric("Ethylene Medio", f"{df['Ethylene_ppm'].mean():.2f} ppm")
        
        st.divider()
        
        #Selector de sensores para visualizar
        st.write("**Distribución de Sensores**")
        sensor_cols = [f'Sensor_{i:02d}' for i in range(1, 17)]
        
        col_viz1, col_viz2 = st.columns([1, 2])
        
        with col_viz1:
            selected_sensors = st.multiselect(
                "Selecciona sensores para analizar:",
                sensor_cols,
                default=["Sensor_01", "Sensor_06", "Sensor_10", "Sensor_16"],
                help="Sensor_01-04: TGS2600, 05-08: TGS2602, 09-12: TGS2610, 13-16: TGS2620"
            )
            
            st.write("**Estadísticas de Sensores Seleccionados:**")
            if selected_sensors:
                st.dataframe(df[selected_sensors].describe().T, height=300)
        
        with col_viz2:
            if selected_sensors:
                #Boxplot
                fig1, ax1 = plt.subplots(figsize=(12, 5))
                df_melt = df[selected_sensors].melt(var_name="Sensor", value_name="Lectura")
                sns.boxplot(data=df_melt, x="Sensor", y="Lectura", ax=ax1, palette="viridis")
                ax1.set_title("Distribución de Lecturas por Sensor", fontsize=14, fontweight='bold')
                ax1.set_xlabel("Sensor")
                ax1.set_ylabel("Valor de Lectura")
                plt.xticks(rotation=45)
                st.pyplot(fig1)
                plt.close()
        
        st.divider()
        
        #Correlación con objetivos
        st.write("**Correlación con Gases Objetivo**")
        
        col_corr1, col_corr2 = st.columns(2)
        
        with col_corr1:
            #Correlación con gas principal
            target_col = "CO_ppm" if gas_type == "CO" else "Methane_ppm"
            correlations = df[sensor_cols + [target_col]].corr()[target_col].drop(target_col).sort_values(ascending=False)
            
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            correlations.head(10).plot(kind='barh', ax=ax2, color='steelblue')
            ax2.set_title(f"Top 10 Sensores - Correlación con {gas_type}", fontsize=12, fontweight='bold')
            ax2.set_xlabel("Correlación")
            st.pyplot(fig2)
            plt.close()
        
        with col_corr2:
            #Heatmap de sensores seleccionados
            if selected_sensors and len(selected_sensors) >= 2:
                corr_matrix = df[selected_sensors + [target_col, 'Ethylene_ppm']].corr()
                
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, ax=ax3, square=True)
                ax3.set_title("Matriz de Correlación", fontsize=12, fontweight='bold')
                st.pyplot(fig3)
                plt.close()
        
        st.divider()
        
        #Time series de concentraciones
        st.write("**Serie Temporal de Concentraciones**")
        
        #Tomar muestra para visualización
        sample_size = min(2000, len(df))
        df_sample = df.sample(sample_size, random_state=42).sort_index()
        
        fig4, ax4 = plt.subplots(figsize=(14, 4))
        ax4.plot(df_sample.index, df_sample[target_col], label=gas_type, linewidth=1, alpha=0.7)
        ax4.plot(df_sample.index, df_sample['Ethylene_ppm'], label='Ethylene', linewidth=1, alpha=0.7)
        ax4.set_xlabel('Índice de Muestra')
        ax4.set_ylabel('Concentración (ppm)')
        ax4.set_title(f'Concentraciones de {gas_type} y Ethylene (Muestra de {sample_size} puntos)', 
                     fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close()

    #performance del modelo
    with tab2:
        st.subheader("Performance del Modelo")
        
        #Métricas de test
        if 'metrics_test' in model_data:
            metrics = model_data['metrics_test']
            
            st.write("**Métricas en Conjunto de Test:**")
            
            for target_name, target_metrics in metrics.items():
                with st.expander(f"{target_name}", expanded=True):
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    
                    mae = target_metrics.get('MAE', 0)
                    rmse = target_metrics.get('RMSE', 0)
                    r2 = target_metrics.get('R2', 0)
                    mae_norm = target_metrics.get('MAE_norm_%', 0)
                    
                    col_m1.metric("MAE (ppm)", f"{mae:.3f}")
                    col_m2.metric("RMSE (ppm)", f"{rmse:.3f}")
                    col_m3.metric("R² Score", f"{r2:.4f}")
                    
                    if mae_norm < 7:
                        col_m4.success(f"**MAE Normalizado: {mae_norm:.2f}%**")
                    elif mae_norm < 15:
                        col_m4.warning(f"**MAE Normalizado: {mae_norm:.2f}%**")
                    else:
                        col_m4.error(f"**MAE Normalizado: {mae_norm:.2f}%** ")

        st.divider()
        
        # Feature Importance interactiva
        st.write("**Importancia de Variables**")
        
        try:
            feature_names = model_data['feature_cols']
            model_keys = list(model_data['models'].keys())
            
            #Selector de target
            selected_target = st.selectbox(
                "Selecciona el gas para analizar:",
                model_keys,
                help="Cada gas tiene su propio modelo con diferentes importancias"
            )
            
            model_obj = model_data['models'][selected_target]['model_stable']
            
            if hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_
                
                #Slider para número de features
                top_n = st.slider("Número de features a mostrar:", 5, 50, 20, 5)
                
                imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importancia': importances
                }).sort_values(by='Importancia', ascending=False).head(top_n)
                
                #Gráfico horizontal
                fig5, ax5 = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
                ax5.barh(range(top_n), imp_df['Importancia'].values, color='steelblue')
                ax5.set_yticks(range(top_n))
                ax5.set_yticklabels(imp_df['Feature'].values)
                ax5.invert_yaxis()
                ax5.set_xlabel('Importancia')
                ax5.set_title(f'Top {top_n} Features - {selected_target}', fontsize=14, fontweight='bold')
                ax5.grid(axis='x', alpha=0.3)
                st.pyplot(fig5)
                plt.close()
                
                # Mostrar tabla
                st.write(f"**Top {top_n} Features (tabla):**")
                st.dataframe(imp_df.reset_index(drop=True), height=300)
                
        except Exception as e:
            st.error(f"Error al generar feature importance: {e}")

    #anomalias
    with tab3:
        st.subheader("Detección de Anomalías")
        
        if 'anomalies' in analysis_results:
            anomaly_data = analysis_results['anomalies']
            
            st.markdown(f"""
            <div class="success-box">
            <strong>Análisis de Anomalías Completado</strong><br>
            Método: Isolation Forest (contamination=5%)<br>
            Dataset: Ethylene + {gas_type}
            </div>
            """, unsafe_allow_html=True)
            
            #Métricas de anomalías
            col_a1, col_a2, col_a3 = st.columns(3)
            
            n_anomalies = anomaly_data.get('n_anomalies', 0)
            pct_anomalies = anomaly_data.get('pct_anomalies', 0)
            
            col_a1.metric("Anomalías Detectadas", f"{n_anomalies:,}")
            col_a2.metric("Porcentaje", f"{pct_anomalies:.2f}%")
            col_a3.metric("Método", "Isolation Forest")
            
            st.divider()
            
            #Estadísticas de residuales
            st.write("**Estadísticas de Residuales:**")
            
            residual_stats = anomaly_data.get('residual_stats', {})
            
            if residual_stats:
                stats_df = pd.DataFrame(residual_stats).T
                stats_df.columns = ['Media', 'Desv. Estándar']
                st.dataframe(stats_df, use_container_width=True)
            
            #visualización si hay imagen
            anomaly_img_path = f'figures/anomalies_{gas_type.lower()}.png'
            if os.path.exists(anomaly_img_path):
                st.write("**Visualización de Residuales y Anomalías:**")
                st.image(anomaly_img_path, use_container_width=True)
        
        else:
            st.warning("No se encontraron resultados de análisis de anomalías. Ejecuta el notebook completo.")

    #robustex
    with tab4:
        st.subheader("Análisis de Robustez ante Fallos de Sensores")
        
        if 'robustness' in analysis_results:
            robustness_df = analysis_results['robustness']
            
            st.markdown(f"""
            <div class="warning-box">
            <strong>Simulación de Fallos de Sensores</strong><br>
            Se simularon fallos aleatorios en 0%, 10%, 20% y 30% de los sensores<br>
            Objetivo: Degradación ≤ 15% con 30% de fallos
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            #Mostrar resultados por tasa de fallo
            st.write("**Degradación del MAE por Tasa de Fallo:**")
            
            for failure_rate in [0.0, 0.1, 0.2, 0.3]:
                df_rate = robustness_df[robustness_df['failure_rate'] == failure_rate]
                
                with st.expander(f"Fallos al {int(failure_rate*100)}%", expanded=(failure_rate == 0.3)):
                    cols_rob = st.columns(len(df_rate))
                    
                    for idx, (_, row) in enumerate(df_rate.iterrows()):
                        with cols_rob[idx]:
                            target = row['target']
                            mae_mean = row['MAE_mean']
                            mae_std = row['MAE_std']
                            degradation = row['MAE_degradation_%']
                            
                            st.write(f"**{target}**")
                            st.metric("MAE", f"{mae_mean:.2f} ± {mae_std:.2f}")
                            
                            if degradation < 15:
                                st.success(f"Degradación: {degradation:+.1f}%")
                            elif degradation < 50:
                                st.warning(f"Degradación: {degradation:+.1f}%")
                            else:
                                st.error(f"Degradación: {degradation:+.1f}%")
            
            st.divider()
            
            #Gráfico de degradación
            st.write("**Gráfico de Degradación:**")
            
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            
            for target in robustness_df['target'].unique():
                df_target = robustness_df[robustness_df['target'] == target]
                ax6.plot(df_target['failure_rate'] * 100, 
                        df_target['MAE_degradation_%'],
                        marker='o', label=target, linewidth=2, markersize=8)
            
            ax6.axhline(15, color='red', linestyle='--', linewidth=2, label='Objetivo (≤15%)')
            ax6.set_xlabel('Tasa de Fallos (%)', fontsize=12)
            ax6.set_ylabel('Degradación del MAE (%)', fontsize=12)
            ax6.set_title('Robustez ante Fallos de Sensores', fontsize=14, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            st.pyplot(fig6)
            plt.close()
            
            #Evaluación final
            max_degradation = robustness_df['MAE_degradation_%'].max()
            
        else:
            st.warning("No se encontraron resultados de análisis de robustez. Ejecuta el notebook completo.")

    #comparativas
    with tab5:
        st.subheader("Comparativa de Modelos y Análisis")
        
        # Comparar ambos datasets
        st.write("**Comparativa entre Datasets (CO vs Methane):**")
        
        # Cargar resumen del otro dataset
        other_gas = "methane" if gas_type == "CO" else "co"
        other_summary_path = f'results/COMPLETE_SUMMARY_{other_gas}.json'
        
        current_summary = analysis_results.get('summary', {})
        
        if os.path.exists(other_summary_path):
            with open(other_summary_path, 'r') as f:
                other_summary = json.load(f)
            
            #Tabla comparativa
            comparison_data = []
            
            for dataset, summary in [("CO", current_summary if gas_type == "CO" else other_summary),
                                     ("Methane", other_summary if gas_type == "CO" else current_summary)]:
                if 'test_performance' in summary:
                    for target, metrics in summary['test_performance'].items():
                        comparison_data.append({
                            'Dataset': f"Ethylene + {dataset}",
                            'Target': target,
                            'MAE (ppm)': metrics['MAE'],
                            'RMSE (ppm)': metrics['RMSE'],
                            'R²': metrics['R2'],
                            'MAE_norm (%)': metrics['MAE_normalized_%']
                        })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                #Gráfico comparativo
                fig7, (ax7_1, ax7_2) = plt.subplots(1, 2, figsize=(14, 5))
                
                #R² comparison
                comparison_df_pivot = comparison_df.pivot(index='Target', columns='Dataset', values='R²')
                comparison_df_pivot.plot(kind='bar', ax=ax7_1, color=['steelblue', 'orange'])
                ax7_1.set_title('Comparación de R² Score', fontsize=12, fontweight='bold')
                ax7_1.set_ylabel('R² Score')
                ax7_1.set_ylim([0, 1])
                ax7_1.legend(title='Dataset')
                ax7_1.grid(axis='y', alpha=0.3)
                
                #MAE_norm comparison
                comparison_df_pivot_mae = comparison_df.pivot(index='Target', columns='Dataset', values='MAE_norm (%)')
                comparison_df_pivot_mae.plot(kind='bar', ax=ax7_2, color=['steelblue', 'orange'])
                ax7_2.set_title('Comparación de MAE Normalizado', fontsize=12, fontweight='bold')
                ax7_2.set_ylabel('MAE Normalizado (%)')
                ax7_2.axhline(7, color='red', linestyle='--', linewidth=2, label='Objetivo')
                ax7_2.legend(title='Dataset')
                ax7_2.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig7)
                plt.close()
        
        else:
            st.info("Ejecuta el notebook para ambos datasets (CO y Methane) para ver comparativas completas.")
        
        st.divider()
        
        #Resumen de versión del modelo
        st.write("**Información del Modelo:**")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.write("**Configuración:**")
            if 'version' in model_data:
                st.write(f"- Versión: `{model_data['version']}`")
            if 'train_time' in model_data:
                st.write(f"- Tiempo entrenamiento: {model_data['train_time']:.1f}s")
            st.write(f"- Features: {len(model_data.get('feature_cols', []))}")
            st.write(f"- Targets: {len(model_data.get('target_cols', []))}")
        
        with col_info2:
            st.write("**Arquitectura:**")
            st.write("- Tipo: Dual Model (Classifier + Regressors)")
            st.write("- Algoritmo: XGBoost")
            st.write("- Especialización: Transiciones + Estables")

    #hacer una prediccion en tiempo rezal
    with tab6:
        st.subheader("Predicción en Tiempo Real (API BentoML)")
        
        st.markdown(f"""
        <div class="success-box">
        <strong>Conectado con servicio BentoML</strong><br>
        Endpoint: <code>http://localhost:3000/predict_{gas_type.lower()}</code><br>
        Modelo: <code>dual_model_v2_{gas_type}</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("**Configurar Valores de los 16 Sensores:**")
        
        with st.form("input_form"):
            # Opción para auto-rellenar rápido
            col_preset1, col_preset2 = st.columns(2)
            
            with col_preset1:
                use_preset = st.checkbox("Usar valores por defecto (45.0 todos)", value=True)
            
            with col_preset2:
                if st.form_submit_button("Resetear a 45.0", use_container_width=True):
                    st.rerun()
            
            st.divider()
            
            # Crear sliders para los 16 sensores en formato 4x4
            st.write("**Sensores TGS-2600** (sensibles a H₂, CH₄, CO, alcoholes)")
            cols_row1 = st.columns(4)
            sensors_row1 = {}
            for i, col in enumerate(cols_row1, start=1):
                with col:
                    default_val = 45.0 if use_preset else 45.0
                    sensors_row1[f'Sensor_{i:02d}'] = st.slider(
                        f"Sensor_{i:02d}",
                        min_value=0.0,
                        max_value=100.0,
                        value=default_val,
                        step=0.5,
                        help=f"TGS-2600 #{i}"
                    )
            
            st.divider()
            
            st.write("**Sensores TGS-2602** (sensibles a NH₃, H₂S, VOCs)")
            cols_row2 = st.columns(4)
            sensors_row2 = {}
            for i, col in enumerate(cols_row2, start=5):
                with col:
                    default_val = 45.0 if use_preset else 45.0
                    sensors_row2[f'Sensor_{i:02d}'] = st.slider(
                        f"Sensor_{i:02d}",
                        min_value=0.0,
                        max_value=100.0,
                        value=default_val,
                        step=0.5,
                        help=f"TGS-2602 #{i}"
                    )
            
            st.divider()
            
            st.write("**Sensores TGS-2610** (sensibles a LP gas, butano)")
            cols_row3 = st.columns(4)
            sensors_row3 = {}
            for i, col in enumerate(cols_row3, start=9):
                with col:
                    default_val = 45.0 if use_preset else 45.0
                    sensors_row3[f'Sensor_{i:02d}'] = st.slider(
                        f"Sensor_{i:02d}",
                        min_value=0.0,
                        max_value=100.0,
                        value=default_val,
                        step=0.5,
                        help=f"TGS-2610 #{i}"
                    )
            
            st.divider()
            
            st.write("**Sensores TGS-2620** (sensibles a alcoholes, solventes orgánicos)")
            cols_row4 = st.columns(4)
            sensors_row4 = {}
            for i, col in enumerate(cols_row4, start=13):
                with col:
                    default_val = 45.0 if use_preset else 45.0
                    sensors_row4[f'Sensor_{i:02d}'] = st.slider(
                        f"Sensor_{i:02d}",
                        min_value=0.0,
                        max_value=100.0,
                        value=default_val,
                        step=0.5,
                        help=f"TGS-2620 #{i}"
                    )
            
            st.divider()
            
            # Botón de predicción
            predict_btn = st.form_submit_button("Realizar Predicción", use_container_width=True, type="primary")

        if predict_btn:
            # Combinar todos los valores de sensores
            sensor_values = {**sensors_row1, **sensors_row2, **sensors_row3, **sensors_row4}
            
            # Mostrar resumen de valores
            with st.expander("Ver valores de entrada", expanded=False):
                st.json(sensor_values)
            
            # Empaquetar payload
            payload = {"input_data": sensor_values}
            
            endpoint = f"http://localhost:3000/predict_{gas_type.lower()}"
            
            try:
                with st.spinner("Procesando predicción en el servidor..."):
                    response = requests.post(endpoint, json=payload, timeout=5)
                
                if response.status_code == 200:
                    data_res = response.json()
                    
                    st.markdown('<div class="success-box"><strong>Predicción Completada</strong></div>', 
                               unsafe_allow_html=True)
                    
                    # Mostrar resultados
                    col_result1, col_result2, col_result3 = st.columns(3)
                    
                    # Extraer valores
                    if gas_type == "CO":
                        ppm_main = data_res.get("CO_ppm", 0.0)
                        main_label = "CO"
                    else:
                        ppm_main = data_res.get("Methane_ppm", 0.0)
                        main_label = "Methane"
                    
                    ppm_eth = data_res.get("Ethylene_ppm", 0.0)
                    status = data_res.get("status", "unknown")
                    
                    with col_result1:
                        st.metric(
                            f"Concentración {main_label}", 
                            f"{ppm_main:.2f} ppm",
                            help="Predicción del modelo para el gas principal"
                        )
                    
                    with col_result2:
                        st.metric(
                            "Concentración Ethylene", 
                            f"{ppm_eth:.2f} ppm",
                            help="Predicción del modelo para Ethylene"
                        )
                    
                    with col_result3:
                        if status == "success":
                            st.success("Status: OK")
                        else:
                            st.error("Status: Error")
                    
                    st.divider()
                    
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar al servicio BentoML (localhost:3000)")
                st.markdown("""
                **Para iniciar el servicio:**
                ```bash
                bentoml serve service.py:GasAnalyticsService --reload
                ```
                """)
            except requests.exceptions.Timeout:
                st.error("Timeout: El servidor tardó demasiado en responder")
            except Exception as e:
                st.error(f"Error inesperado: {str(e)}")
        
        #información adicional
        st.divider()
        st.info("""
        **Consejos para la predicción:**
        - Los valores típicos de sensores están entre 20-80
        - Usa el checkbox "Usar valores por defecto" para pruebas rápidas
        - Los sensores del mismo tipo (TGS-xxxx) suelen tener valores similares
        - La predicción asume estado estable (sin historial temporal)
        """)

else:
    st.error("Faltan archivos")
    st.markdown(f"""
    **Archivos requeridos:**
    - Dataset: `{csv_path}` {'Ok' if df is not None else 'bad'}
    - Modelo: `{model_path}` {'Ok' if model_data is not None else 'bad'}
    
    **Solución:** Ejecuta el notebook completo primero.
    """)

#footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p><strong>Industrial Gas Analytics Dashboard</strong> | Proyecto Final: Análisis de Datos para la Industria</p>
    <p>Powered by Streamlit + BentoML + XGBoost</p>
</div>
""", unsafe_allow_html=True)