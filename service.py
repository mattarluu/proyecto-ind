import bentoml
import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, Any

#configuracion de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#registro de modelos en bentoML
try:
    for gas in ["co", "methane"]:
        path = f'models/dual_model_v2_{gas}.pkl'
        logger.info(f"Registrando modelo: {path}")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        bentoml.picklable_model.save_model(
            f"gasdetector{gas}",
            model_data,
            signatures={"predict": {"batchable": False}}
        )
        logger.info(f"Modelo {gas} registrado exitosamente")
    
    print("Todos los modelos registrados en el almacén de BentoML")
    
except Exception as e:
    logger.error(f"Error al registrar modelos: {e}")
    raise

#servicion bento
@bentoml.service(name="gas_analytics_service")
class GasAnalyticsService:
    """
    Servicio de inferencia para detección de gases industriales.
    
    Endpoints disponibles:
    - /predict_co: Predicción de CO y Ethylene
    - /predict_methane: Predicción de Methane y Ethylene
    - /health: Health check del servicio
    """
    
    def __init__(self):
        logger.info("Inicializando servicio de análisis de gases...")
        
        try:
            #cargar modelos
            self.model_co_data = bentoml.picklable_model.load_model("gasdetectorco:latest")
            self.model_methane_data = bentoml.picklable_model.load_model("gasdetectormethane:latest")
            
            #Recuperar lista de features esperadas
            self.feature_cols = self.model_co_data['feature_cols']
            self.n_features = len(self.feature_cols)
            
            #Validar que ambos modelos usan las mismas features
            assert self.feature_cols == self.model_methane_data['feature_cols'], \
                "Los modelos CO y Methane tienen diferentes conjuntos de features"
            
            logger.info(f"Modelos cargados. Esperando {self.n_features} features por predicción")
            
            # Guardar estadísticas para validación
            self._load_validation_stats()
            
        except Exception as e:
            logger.error(f"Error al cargar modelos: {e}")
            raise
    
    def _load_validation_stats(self):
        """Carga estadísticas de los sensores para validación de entrada"""
        #Rango razonable para sensores (valores típicos observados en entrenamiento)
        self.sensor_ranges = {
            'min': 0.0,
            'max': 150.0  
        }
        logger.info("Estadísticas de validación cargadas")
    
    def _validate_input(self, sensor_data: Dict[str, float]) -> tuple:
        """
        Valida que los datos de entrada sean correctos
        
        Returns:
            (is_valid, error_message)
        """
        #Verificar que tenga exactamente 16 sensores
        expected_sensors = [f'Sensor_{i:02d}' for i in range(1, 17)]
        
        if not all(sensor in sensor_data for sensor in expected_sensors):
            missing = [s for s in expected_sensors if s not in sensor_data]
            return False, f"Faltan sensores: {missing}"
        
        #Verificar tipos y rangos
        for sensor, value in sensor_data.items():
            if sensor.startswith('Sensor_'):
                try:
                    val = float(value)
                    if not (self.sensor_ranges['min'] <= val <= self.sensor_ranges['max']):
                        return False, f"{sensor}={val} fuera de rango [{self.sensor_ranges['min']}, {self.sensor_ranges['max']}]"
                except (ValueError, TypeError):
                    return False, f"{sensor} tiene valor no numérico: {value}"
        
        return True, ""
    
    def _prepare_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adapta los 16 sensores crudos al formato complejo que espera el modelo.
        
        IMPORTANTE: Las features temporales (lags, derivadas, rolling) se rellenan con 0,
        asumiendo estado estable. Para predicciones más precisas en producción, debería
        mantenerse un histórico de las últimas N muestras.
        
        Args:
            input_df: DataFrame con los 16 sensores
            
        Returns:
            DataFrame con todas las features esperadas por el modelo
        """
        #Crear DataFrame con todas las columnas que el modelo espera
        full_df = pd.DataFrame(index=input_df.index, columns=self.feature_cols)
        
        #Rellenar con 0.0 (asumimos estado estable para demo instantánea)
        full_df = full_df.fillna(0.0)
        
        #Copiar valores de sensores base
        for col in input_df.columns:
            if col in full_df.columns:
                full_df[col] = input_df[col]
        
        #Asegurar tipos float
        return full_df.astype(float)
    
    def _predict_with_model(self, model_data: Dict, df_ready: pd.DataFrame, target_names: list) -> Dict[str, float]:
        """
        Realiza la predicción usando los modelos duales
        
        Args:
            model_data: Diccionario con los modelos
            df_ready: DataFrame preparado con features
            target_names: Lista de nombres de targets a predecir
            
        Returns:
            Diccionario con las predicciones
        """
        predictions = {}
        
        for target in target_names:
            try:
                #Usar el modelo estable (en producción podríamos clasificar primero)
                model_stable = model_data['models'][target]['model_stable']
                pred = model_stable.predict(df_ready)[0]
                
                #Asegurar que la predicción sea no negativa (físicamente imposible)
                pred = max(0.0, float(pred))
                
                predictions[target] = pred
                
            except Exception as e:
                logger.error(f"Error al predecir {target}: {e}")
                predictions[target] = 0.0
        
        return predictions
    
    @bentoml.api
    def health(self) -> Dict[str, Any]:
        """
        Health check endpoint
        
        Returns:
            Estado del servicio y modelos cargados
        """
        return {
            "status": "healthy",
            "service": "gas_analytics_service",
            "models_loaded": {
                "co": bool(self.model_co_data),
                "methane": bool(self.model_methane_data)
            },
            "n_features_expected": self.n_features
        }
    
    @bentoml.api
    def predict_co(self, input_data: dict) -> dict:
        """
        Endpoint para predecir CO y Ethylene
        
        Args:
            input_data: Diccionario con estructura:
                {
                    "input_data": {
                        "Sensor_01": float,
                        "Sensor_02": float,
                        ...
                        "Sensor_16": float
                    }
                }
        
        Returns:
            {
                "CO_ppm": float,
                "Ethylene_ppm": float,
                "status": "success" | "error",
                "message": str (opcional)
            }
        """
        try:
            # Extraer datos de sensores
            sensor_data = input_data.get("input_data", input_data)
            
            # Validar entrada
            is_valid, error_msg = self._validate_input(sensor_data)
            if not is_valid:
                logger.warning(f"Entrada inválida: {error_msg}")
                return {
                    "CO_ppm": 0.0,
                    "Ethylene_ppm": 0.0,
                    "status": "error",
                    "message": f"Validación fallida: {error_msg}"
                }
            
            # Convertir a DataFrame
            df_raw = pd.DataFrame([sensor_data])
            
            # Preparar features
            df_ready = self._prepare_features(df_raw)
            
            # Predecir
            predictions = self._predict_with_model(
                self.model_co_data,
                df_ready,
                ['CO_ppm', 'Ethylene_ppm']
            )
            
            # Log de predicción exitosa
            logger.info(f"Predicción CO exitosa: CO={predictions['CO_ppm']:.2f}, Ethylene={predictions['Ethylene_ppm']:.2f}")
            
            return {
                "CO_ppm": predictions['CO_ppm'],
                "Ethylene_ppm": predictions['Ethylene_ppm'],
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Error en predict_co: {e}")
            return {
                "CO_ppm": 0.0,
                "Ethylene_ppm": 0.0,
                "status": "error",
                "message": str(e)
            }
    
    @bentoml.api
    def predict_methane(self, input_data: dict) -> dict:
        """
        Endpoint para predecir Methane y Ethylene
        
        Args:
            input_data: Diccionario con estructura:
                {
                    "input_data": {
                        "Sensor_01": float,
                        "Sensor_02": float,
                        ...
                        "Sensor_16": float
                    }
                }
        
        Returns:
            {
                "Methane_ppm": float,
                "Ethylene_ppm": float,
                "status": "success" | "error",
                "message": str (opcional)
            }
        """
        try:
            # Extraer datos de sensores
            sensor_data = input_data.get("input_data", input_data)
            
            # Validar entrada
            is_valid, error_msg = self._validate_input(sensor_data)
            if not is_valid:
                logger.warning(f"Entrada inválida: {error_msg}")
                return {
                    "Methane_ppm": 0.0,
                    "Ethylene_ppm": 0.0,
                    "status": "error",
                    "message": f"Validación fallida: {error_msg}"
                }
            
            # Convertir a DataFrame
            df_raw = pd.DataFrame([sensor_data])
            
            # Preparar features
            df_ready = self._prepare_features(df_raw)
            
            # Predecir
            predictions = self._predict_with_model(
                self.model_methane_data,
                df_ready,
                ['Methane_ppm', 'Ethylene_ppm']
            )
            
            # Log de predicción exitosa
            logger.info(f"Predicción Methane exitosa: Methane={predictions['Methane_ppm']:.2f}, Ethylene={predictions['Ethylene_ppm']:.2f}")
            
            return {
                "Methane_ppm": predictions['Methane_ppm'],
                "Ethylene_ppm": predictions['Ethylene_ppm'],
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Error en predict_methane: {e}")
            return {
                "Methane_ppm": 0.0,
                "Ethylene_ppm": 0.0,
                "status": "error",
                "message": str(e)
            }

#main para testing local
if __name__ == "__main__":
    print("Servicio de análisis de gases inicializado")
    print("Para iniciar el servidor:")
    print("  bentoml serve service.py:GasAnalyticsService --reload")