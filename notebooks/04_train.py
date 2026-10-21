# Databricks notebook source
# MAGIC %md
# MAGIC # Entrenamiento reproducible y registro del modelo en Unity Catalog
# MAGIC Clases 9–10. Notebook delgado: recibe parámetros del job y llama a `xm_demanda.model`.
# MAGIC En producción lo ejecuta el job `train_model` (clase 12) con `catalog = dev | qa | prod`:
# MAGIC "deploy code, not models": cada ambiente entrena su versión con sus datos.
# MAGIC
# MAGIC Lee `gold_energia.features_demanda_diaria`, entrena el estimador elegido, registra el run (parámetros, métricas,
# MAGIC dataset con versión Delta, modelo con firma) y crea una versión de `<catalog>.models.demanda_7d` con alias `challenger`.
# MAGIC La promoción a `champion` es una decisión humana (o una regla explícita), no un efecto secundario del entrenamiento.

# COMMAND ----------

# MAGIC %pip install -q lightgbm

# COMMAND ----------

dbutils.widgets.text("catalog", "dev")
dbutils.widgets.dropdown("modelo", "lightgbm_global", ["lightgbm_global", "naive_festivo", "naive_estacional"])
dbutils.widgets.text("corte", "")
catalog = dbutils.widgets.get("catalog")
nombre_estimador = dbutils.widgets.get("modelo")

# COMMAND ----------

import os
import sys

import mlflow
import pandas as pd

raiz = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(raiz, "src"))
sys.dont_write_bytecode = True

from xm_demanda.features.build import columnas_contrato
from xm_demanda.model import datos, metricas, modelos, registro
from xm_demanda.utils.config import fqn, load_config, validar_config

cfg = load_config(catalog)
validar_config(cfg)
H = cfg["modelo"]["horizon_days"]
OBJETIVO = f"y_h{H}"
tabla = fqn(cfg, "gold", "gold_features")
nombre_modelo = f"{cfg['catalog']}.{cfg['schemas']['models']}.{cfg['modelo']['nombre']}"
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"{cfg['experimento']}/{cfg['modelo']['nombre']}")
print(tabla, "→", nombre_modelo)

# COMMAND ----------

version = spark.sql(f"DESCRIBE HISTORY {tabla}").agg({"version": "max"}).first()[0]
oro = spark.table(tabla).toPandas()
assert set(oro.columns) == set(columnas_contrato(cfg)), "La tabla de oro no cumple el contrato (clase 8)"
oro["fecha"] = pd.to_datetime(oro["fecha"])
oro = oro[oro["dias_con_datos"] >= 29]

corte = dbutils.widgets.get("corte") or str((oro["fecha"].max() - pd.Timedelta(days=H + 14)).date())
train, valid = datos.split_temporal(oro, corte, 14, OBJETIVO)
festivos = pd.read_csv(os.path.join(raiz, "data", "reference", "festivos_co.csv"), parse_dates=["fecha"])["fecha"].tolist()
estimador = {
    "lightgbm_global": lambda: modelos.GlobalLightGBM(),
    "naive_festivo": lambda: modelos.NaiveFestivo(festivos, H),
    "naive_estacional": lambda: modelos.NaiveEstacional(),
}[nombre_estimador]()

# COMMAND ----------

with mlflow.start_run(run_name=f"{nombre_estimador}_{catalog}") as run:
    estimador.fit(train, OBJETIVO)
    m = metricas.metricas(valid[OBJETIVO], estimador.predict(valid))
    m["wape_baseline"] = metricas.wape(valid[OBJETIVO], valid["demanda_real_kwh"])
    mlflow.log_params({"modelo": nombre_estimador, "catalog": catalog, "horizonte_dias": H, "corte": corte, **getattr(estimador, "params", {})})
    mlflow.log_metrics(m)
    mlflow.log_input(mlflow.data.load_delta(table_name=tabla, version=str(version)), context="training")
    # Reentrena con todo lo disponible para producción y registra ese artefacto
    estimador.fit(oro[oro[OBJETIVO].notna()], OBJETIVO)
    registro.registrar_en_run(estimador, valid.head(5).drop(columns=[OBJETIVO]), code_paths=[os.path.join(raiz, "src", "xm_demanda")])

v = registro.registrar_version(run.info.run_id, nombre_modelo)
registro.poner_alias(nombre_modelo, "challenger", v.version)
print(f"{nombre_modelo} versión {v.version} @challenger · wape {m['wape']:.4f} vs baseline {m['wape_baseline']:.4f}")
