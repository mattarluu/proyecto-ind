import bentoml
import numpy as np
import pandas as pd
from bentoml.io import JSON
import pickle

for gas in ["co", "methane"]:
    path = f'models/dual_model_v2_{gas}.pkl'
    with open(path, 'rb') as f:
        model_data = pickle.load(f)

    bentoml.picklable_model.save_model(
        f"gasdetector{gas}",
        model_data,
        signatures={"predict": {"batchable": False}}
    )
print("Modelos registrados en el almacén de BentoML")
@bentoml.service(name="gas_analytics_service")
class GasAnalyticsService:
    def __init__(self):
        print("Cargando modelos del almacén...")
        # 1. Cargamos los diccionarios completos
        self.model_co_data = bentoml.picklable_model.load_model("gasdetectorco:latest")
        self.model_methane_data = bentoml.picklable_model.load_model("gasdetectormethane:latest")
        
        # 2. Recuperamos la lista exacta de columnas que el modelo espera
        # (Esto es vital para evitar el error de "Shape mismatch")
        self.feature_cols = self.model_co_data['feature_cols']
        print(f"¡Modelos cargados! Esperando {len(self.feature_cols)} variables por predicción.")

    def _prepare_features(self, input_df):
        """
        Adapta los 16 sensores crudos al formato complejo que espera el modelo.
        Rellena las features de tiempo (lags, derivadas) con 0 asumiendo estado estable.
        """
        # Creamos un DataFrame vacío con TODAS las columnas que el modelo quiere
        full_df = pd.DataFrame(index=input_df.index, columns=self.feature_cols)
        
        # Rellenamos con 0.0 (Asumimos estado estable para la demo instantánea)
        full_df = full_df.fillna(0.0)
        
        # Copiamos los valores de los sensores que sí tenemos
        # El modelo espera nombres como 'Sensor_01', 'Sensor_02'...
        for col in input_df.columns:
            if col in full_df.columns:
                full_df[col] = input_df[col]
                
        # Aseguramos que los tipos sean float
        return full_df.astype(float)

    @bentoml.api
    def predict_co(self, input_data: dict) -> dict:
        """Endpoint para predecir CO y Etileno"""
        # El input viene envuelto en "input_data" desde Streamlit
        sensor_data = input_data.get("input_data", input_data)
        
        # Convertimos a DataFrame
        df_raw = pd.DataFrame([sensor_data])
        
        # --- CORRECCIÓN 1: Generar las features extra ---
        df_ready = self._prepare_features(df_raw)
        
        # --- CORRECCIÓN 2: Usar las claves correctas ('CO_ppm' en vez de 'CO') ---
        # Accedemos a la estructura guardada en el notebook
        model_co = self.model_co_data['models']['CO_ppm']['model_stable']
        model_eth = self.model_co_data['models']['Ethylene_ppm']['model_stable']
        
        # Predecimos
        res_co = model_co.predict(df_ready)[0]
        res_eth = model_eth.predict(df_ready)[0]
        
        return {"CO_ppm": float(res_co), "Ethylene_ppm": float(res_eth)}

    @bentoml.api
    def predict_methane(self, input_data: dict) -> dict:
        """Endpoint para predecir Metano y Etileno"""
        sensor_data = input_data.get("input_data", input_data)
        df_raw = pd.DataFrame([sensor_data])
        
        # Preparar features
        df_ready = self._prepare_features(df_raw)
        
        # Usar claves correctas ('Methane_ppm')
        model_meth = self.model_methane_data['models']['Methane_ppm']['model_stable']
        model_eth = self.model_methane_data['models']['Ethylene_ppm']['model_stable']
        
        res_meth = model_meth.predict(df_ready)[0]
        res_eth = model_eth.predict(df_ready)[0]
        
        return {"Methane_ppm": float(res_meth), "Ethylene_ppm": float(res_eth)}