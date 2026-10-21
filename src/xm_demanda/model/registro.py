"""Envoltorio pyfunc, firma y registro en Unity Catalog.

Un modelo registrado recibe filas de la tabla de oro y devuelve kWh, sea cual sea el
estimador que lleva dentro. Quien lo consume (batch, endpoint, app) no necesita saberlo.
"""

import mlflow
import pandas as pd
from mlflow.models import infer_signature


class ModeloDemanda(mlflow.pyfunc.PythonModel):
    """Lleva dentro un estimador de modelos.py; predict devuelve la demanda en t+h en kWh."""

    def __init__(self, estimador=None):
        self.estimador = estimador

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return pd.Series(self.estimador.predict(model_input), name="demanda_estimada_kwh")


def registrar_en_run(
    estimador, ejemplo: pd.DataFrame, nombre_artefacto: str = "modelo", code_paths=None
):
    """Guarda el estimador envuelto como artefacto del run activo, con firma y ejemplo.

    `code_paths`: rutas que viajan con el modelo (la carpeta src/xm_demanda), para que el
    artefacto se pueda cargar en un endpoint o en otro job sin tener el repo a mano.
    """
    envuelto = ModeloDemanda(estimador)
    salida = envuelto.predict(None, ejemplo)
    firma = infer_signature(ejemplo, salida)
    return mlflow.pyfunc.log_model(
        artifact_path=nombre_artefacto,
        python_model=envuelto,
        signature=firma,
        input_example=ejemplo,
        code_paths=code_paths,
    )


def registrar_version(run_id: str, nombre_modelo: str, nombre_artefacto: str = "modelo"):
    """Crea una versión del modelo registrado en Unity Catalog a partir de un run."""
    mlflow.set_registry_uri("databricks-uc")
    return mlflow.register_model(f"runs:/{run_id}/{nombre_artefacto}", nombre_modelo)


def poner_alias(nombre_modelo: str, alias: str, version: int) -> None:
    """champion = el que sirve; challenger = el que compite. Mover el alias no despliega código."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.MlflowClient().set_registered_model_alias(nombre_modelo, alias, version)


def cargar(nombre_modelo: str, alias: str = "champion"):
    """El consumidor no sabe qué versión es: pide @champion y recibe un pyfunc listo."""
    mlflow.set_registry_uri("databricks-uc")
    return mlflow.pyfunc.load_model(f"models:/{nombre_modelo}@{alias}")
