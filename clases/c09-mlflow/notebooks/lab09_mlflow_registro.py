# Databricks notebook source
# MAGIC %md
# MAGIC # Clases 9 y 10 — MLOps en acción: experimentar, medir y registrar con MLflow
# MAGIC
# MAGIC Este notebook es la clase. La teoría de MLOps está en las diapositivas; aquí se **ve en acción** sobre tu tabla de oro:
# MAGIC varios modelos, de varios tipos, todos rastreados de la misma forma, comparados con honestidad y uno de ellos registrado
# MAGIC en Unity Catalog con un alias que la clase 11 va a consumir.
# MAGIC
# MAGIC ```
# MAGIC   features_demanda_diaria (oro, clase 7)          experimento MLflow · demanda_7d
# MAGIC   72.704 filas · 355 series · 24 columnas          ┌─────────────────────┬────────┬──────────────┐
# MAGIC            │                                       │ run                 │ wape   │ tipo         │
# MAGIC            │  corte temporal 30-jun                ├─────────────────────┼────────┼──────────────┤
# MAGIC            ▼                                       │ naive_estacional    │ 0.048  │ baseline     │
# MAGIC   train ≤ 30-jun · valid 1–18 jul                  │ naive_festivo       │ 0.046  │ baseline+regla│
# MAGIC            │                                       │ lightgbm_global     │ 0.050  │ modelo       │
# MAGIC   ┌────────┴──────────┬──────────────┐             │ lightgbm_fuga       │ 0.024 ⚠│ trampa       │
# MAGIC   ▼                   ▼              ▼             │ walk_forward (3)    │ …      │ validación   │
# MAGIC   baselines      LightGBM global   clasificador    │ clasificador_caida  │ auc .81│ otro tipo    │
# MAGIC   (1 línea)      (355 series)      + KMeans        │ kmeans_perfiles     │ sil .50│ otro tipo    │
# MAGIC                                                    └─────────────────────┴────────┴──────────────┘
# MAGIC            │  el mejor según walk-forward                       │
# MAGIC            ▼                                                    ▼
# MAGIC   workspace.c01_<usuario>.demanda_7d  ·  versión N @champion  ·  versión M @challenger  → clase 11
# MAGIC ```
# MAGIC
# MAGIC Dos `TODO` en `src/xm_demanda/model/`: `wape` en `metricas.py` y `NaiveFestivo.fit` en `modelos.py`. Dos preguntas por escrito.
# MAGIC Requiere el lab 8: `workspace.c01_<usuario>.features_demanda_diaria` con las 24 columnas del contrato.

# COMMAND ----------

# MAGIC %pip install -q lightgbm

# COMMAND ----------

dbutils.widgets.text("usuario", "")
usuario = dbutils.widgets.get("usuario").strip().lower()
assert usuario, "Escribe tu usuario (el sufijo de tu esquema c01_<usuario>)."

esquema = f"workspace.c01_{usuario}"
T_FEATURES = f"{esquema}.features_demanda_diaria"
NOMBRE_MODELO = f"{esquema}.demanda_7d"
print("Oro:", T_FEATURES, "| Modelo registrado:", NOMBRE_MODELO)

# COMMAND ----------

import importlib
import os
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd


def _raiz_repo():
    d = os.getcwd()
    for _ in range(8):
        if os.path.exists(os.path.join(d, "src", "xm_demanda")):
            return d
        d = os.path.dirname(d)
    raise FileNotFoundError("No encuentro src/xm_demanda: ejecuta este notebook desde tu Git folder.")


raiz = _raiz_repo()
sys.path.insert(0, os.path.join(raiz, "src"))
sys.dont_write_bytecode = True
import xm_demanda.model.datos as datos
import xm_demanda.model.metricas as met
import xm_demanda.model.modelos as mod
import xm_demanda.model.registro as reg
from xm_demanda.utils.config import load_config

for m in (datos, met, mod, reg):
    importlib.reload(m)

cfg = load_config("dev")
H = cfg["modelo"]["horizon_days"]
OBJETIVO = f"y_h{H}"
CODE_PATHS = [os.path.join(raiz, "src", "xm_demanda")]  # el modelo viaja con su código

usuario_ws = spark.sql("SELECT current_user()").first()[0]
EXPERIMENTO = f"/Users/{usuario_ws}/xm_demanda_{cfg['modelo']['nombre']}"
mlflow.set_experiment(EXPERIMENTO)
mlflow.set_registry_uri("databricks-uc")
print("Experimento:", EXPERIMENTO, "| horizonte:", H, "días")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Dónde estamos en el ciclo
# MAGIC
# MAGIC ![El ciclo MLOps](./../img/mlops_ciclo.png)
# MAGIC
# MAGIC Las clases 4 a 8 fueron el paso 1: datos con contrato. Hoy hacemos los pasos 2, 3 y 4, los tres con MLflow, y dejamos el
# MAGIC candidato listo para que la clase 11 lo despliegue. Lo que se repite en cada paso es la misma idea: **nada que no quede
# MAGIC anotado existe**. Un modelo que dio buena métrica en una celda y no está en un run es una anécdota.
# MAGIC
# MAGIC ![Qué ofrece Databricks](./../img/databricks_ml.png)
# MAGIC
# MAGIC De este mapa hoy usamos tres piezas: **Tracking** (experimentos y runs), **evaluación** (`search_runs`, métricas por
# MAGIC segmento, walk-forward) y **Registry en Unity Catalog** (versiones, alias, firma). Serving, Apps y Monitoring son las
# MAGIC clases 11 y 15. Todo corre en Free Edition.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Los datos para modelar y el corte temporal
# MAGIC
# MAGIC ![Corte temporal](./../img/split_temporal.png)
# MAGIC
# MAGIC Con series de tiempo no se baraja. Entrenamiento: filas con `fecha` hasta el **30 de junio**. Validación: del 1 al 18 de
# MAGIC julio (sus objetivos `y_h7` llegan hasta el 25, el último día con datos). Ninguna fecha de validación es anterior a una de
# MAGIC entrenamiento, y el entrenamiento no sabe nada de julio. La variante *walk-forward* de la sección 7 repite esto con tres
# MAGIC cortes, porque una sola validación es una muestra de tamaño uno.
# MAGIC
# MAGIC Dos detalles que vienen de la clase 7: se descartan las filas sin 28 días de historia (sus `lag_28` y `media_28` son nulos) y
# MAGIC la tabla se lee con su **versión Delta**, que vamos a anotar en cada run. Eso es el linaje datos ↔ modelo.

# COMMAND ----------

version_oro = spark.sql(f"DESCRIBE HISTORY {T_FEATURES}").agg({"version": "max"}).first()[0]
oro_spark = spark.table(T_FEATURES)
oro = oro_spark.toPandas()
oro["fecha"] = pd.to_datetime(oro["fecha"])
oro = oro[oro["dias_con_datos"] >= 29].reset_index(drop=True)

CORTE, DIAS_VALID = "2026-06-30", 18
train, valid = datos.split_temporal(oro, CORTE, DIAS_VALID, OBJETIVO)
print(f"Versión Delta de oro: {version_oro} | filas con historia: {len(oro):,}")
print(f"train: {len(train):,} filas hasta {train['fecha'].max().date()} | valid: {len(valid):,} filas, {valid['fecha'].min().date()} → {valid['fecha'].max().date()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### TODO 1 — `wape` en `src/xm_demanda/model/metricas.py`
# MAGIC
# MAGIC La métrica principal del proyecto. WAPE = suma de errores absolutos / suma de |y|. Dos razones para preferirla al MAPE:
# MAGIC pondera cada serie por su energía (un 10 % de error en Bogotá importa más que un 10 % en una serie de 30 kWh), y no explota
# MAGIC cuando `y = 0`, que en el mercado no regulado pasa. Devuelve una fracción; `0.05` se lee "5 %". Si `sum(|y|) == 0`, `nan`.
# MAGIC Los tests en `tests/unit/test_model.py` describen los tres casos. Tres líneas con numpy.

# COMMAND ----------

importlib.reload(met)
assert met.wape([100, 200], [110, 180]) == 0.1
assert met.wape([0, 100], [0, 90]) == 0.1
assert np.isnan(met.wape([0, 0], [1, 1]))
print("wape OK. Un 10 % de error en la serie grande vs. en la pequeña:", met.wape([10_000, 100], [9_000, 100]), "vs", met.wape([10_000, 100], [10_000, 90]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Tracking: el primer run, a mano, para ver qué se anota
# MAGIC
# MAGIC ![Anatomía de un run](./../img/tracking_anatomia.png)
# MAGIC
# MAGIC Un **experimento** es una carpeta de intentos comparables sobre la misma pregunta. Un **run** es un intento, y guarda seis
# MAGIC cosas: parámetros (lo que decidiste), métricas (lo que obtuviste), artefactos (archivos: gráficas, CSV, el modelo), el
# MAGIC dataset (qué tabla y qué versión), tags y origen (commit, notebook, usuario: MLflow los pone solo en un Git folder) y el
# MAGIC modelo con su **firma**.
# MAGIC
# MAGIC El primer modelo es el baseline que hay que vencer: `y(t+7) = y(t)`, el mismo día de la semana pasada. Una línea. Lo
# MAGIC registramos con todo el detalle para ver cada pieza; después lo encapsulamos en una función.

# COMMAND ----------

fest = pd.read_csv(os.path.join(raiz, "data", "reference", "festivos_co.csv"), parse_dates=["fecha"])["fecha"].tolist()
ejemplo = valid.head(5).drop(columns=[OBJETIVO])

with mlflow.start_run(run_name="naive_estacional") as run:
    estimador = mod.NaiveEstacional().fit(train, OBJETIVO)
    yhat = estimador.predict(valid)
    m = met.metricas(valid[OBJETIVO], yhat)

    por_mercado = met.wape_por_segmento(valid.assign(yhat=yhat), OBJETIVO, "yhat", "tipo_mercado")
    for _, fila in por_mercado.iterrows():
        m[f"wape_{fila['tipo_mercado'].lower().replace(' ', '_')}"] = fila["wape"]

    mlflow.log_params({"modelo": estimador.nombre, "horizonte_dias": H, "corte": CORTE, "dias_valid": DIAS_VALID, "tipo": "baseline"})
    mlflow.log_metrics(m)
    mlflow.log_input(mlflow.data.load_delta(table_name=T_FEATURES, version=str(version_oro)), context="training")
    mlflow.set_tags({"capa": "oro", "candidato": "no"})
    reg.registrar_en_run(estimador, ejemplo, code_paths=CODE_PATHS)
    print(run.info.run_id, m)

# COMMAND ----------

# MAGIC %md
# MAGIC Abre el experimento (**Experiments** en el menú lateral, o el enlace que aparece bajo la celda) y mira el run: pestaña
# MAGIC *Overview* con parámetros, métricas y el dataset `features_demanda_diaria` versión N; *Artifacts* con `modelo/` y dentro
# MAGIC `MLmodel`, la firma (23 columnas de entrada: el contrato sin `y_h7`; una salida en kWh), `input_example.json` y la carpeta `code/` con
# MAGIC `xm_demanda`. Ese run ya es reproducible por alguien que no esté en esta sala.
# MAGIC
# MAGIC La función de abajo hace lo mismo para cualquier estimador con `fit`/`predict`: así los runs quedan **comparables**, con los
# MAGIC mismos nombres de métricas y parámetros. Esa uniformidad es la que después permite ordenar por `wape` sin pensar.

# COMMAND ----------

def correr_experimento(estimador, train, valid, run_name=None, tags=None, params=None, nested=False, figura=None):
    """Entrena, evalúa en validación y registra todo en un run. Devuelve (run_id, yhat, metricas)."""
    with mlflow.start_run(run_name=run_name or estimador.nombre, nested=nested) as run:
        estimador.fit(train, OBJETIVO)
        yhat = estimador.predict(valid)
        m = met.metricas(valid[OBJETIVO], yhat)
        por_mercado = met.wape_por_segmento(valid.assign(yhat=yhat), OBJETIVO, "yhat", "tipo_mercado")
        for _, fila in por_mercado.iterrows():
            m[f"wape_{fila['tipo_mercado'].lower().replace(' ', '_')}"] = fila["wape"]
        mlflow.log_params({"modelo": estimador.nombre, "horizonte_dias": H, "corte": CORTE, "dias_valid": DIAS_VALID, **getattr(estimador, "params", {}), **(params or {})})
        mlflow.log_metrics(m)
        mlflow.log_input(mlflow.data.load_delta(table_name=T_FEATURES, version=str(version_oro)), context="training")
        mlflow.set_tags({"capa": "oro", **(tags or {})})
        if figura is not None:
            mlflow.log_figure(figura(estimador, valid, yhat), "graficas/diagnostico.png")
        reg.registrar_en_run(estimador, valid.head(5).drop(columns=[OBJETIVO]), code_paths=CODE_PATHS)
        print(f"{run_name or estimador.nombre:22s} wape {m['wape']:.4f}  mae {m['mae_gwh']:.4f} GWh  → run {run.info.run_id[:8]}")
        return run.info.run_id, yhat, m


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Un modelo global: LightGBM sobre 355 series
# MAGIC
# MAGIC En la clase 7 decidieron un **modelo global**: una sola tabla, un solo modelo que aprende de todas las series. Eso tiene un
# MAGIC requisito que no se ve en la teoría: las 355 series van de 30 kWh a 35 GWh por día. Un árbol que ve `demanda = 34.000.000`
# MAGIC y `demanda = 30` no puede compartir nada entre ellas. La solución está en `src/xm_demanda/model/datos.py`: cada magnitud se
# MAGIC divide por `media_28` de su propia serie (`lag_7_rel`, `media_7_rel`…), el objetivo también, y al predecir se multiplica de
# MAGIC vuelta. Así "la demanda de hoy es 1,05 veces mi media" significa lo mismo en Bogotá y en Guaviare.
# MAGIC
# MAGIC ![Importancia de features](./../img/importancia_features.png)
# MAGIC
# MAGIC Las categóricas (`mercado_comercializacion`, `ciiu_seccion`, `codigo_sic_agente`) entran como categorías nativas de
# MAGIC LightGBM, sin one-hot. Y la función objetivo es `l1` (error absoluto), coherente con la métrica WAPE que vamos a mirar: se
# MAGIC optimiza lo que se mide. El modelo tarda unos 30 s en serverless.

# COMMAND ----------

def fig_importancia(estimador, valid, yhat):
    imp = estimador.importancia().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, color=["#7C3AED" if c.endswith("_rel") else "#15803D" if c in datos.CALENDARIO else "#B45309" for c in imp.index])
    ax.set_xlabel("importancia (ganancia, %)")
    ax.set_title("LightGBM: qué usa el modelo")
    plt.tight_layout()
    return fig


run_lgbm, yhat_lgbm, m_lgbm = correr_experimento(mod.GlobalLightGBM(), train, valid, tags={"candidato": "si"}, figura=fig_importancia)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Comparar con honestidad: la tabla de runs y la métrica por segmento
# MAGIC
# MAGIC `mlflow.search_runs` devuelve el experimento como un DataFrame: es la misma tabla que ves en la interfaz, pero en código,
# MAGIC para ordenarla, filtrarla o ponerla en un PR. Mírala y fíjate en la primera sorpresa del día: el modelo global **no vence al
# MAGIC baseline**. Con seis meses de datos y una métrica ponderada por energía, "el mismo día de la semana pasada" es un rival
# MAGIC durísimo en demanda eléctrica. Esto es normal y es la razón por la que el baseline va primero: sin él, un WAPE de 5 %
# MAGIC parecería un logro.
# MAGIC
# MAGIC ![Error por segmento](./../img/error_por_segmento.png)
# MAGIC
# MAGIC La segunda sorpresa está en la métrica por segmento: el error del lunes es el doble que el del resto de la semana, y no
# MAGIC es casualidad: en la validación hay dos lunes festivos (13 y 20 de julio). Ninguno de los dos modelos sabe lo suficiente de
# MAGIC festivos: el naive copia el lunes anterior, que fue laboral, y LightGBM solo vio once festivos en el entrenamiento.
# MAGIC **La métrica global escondía exactamente el día que más le importa a XM.**

# COMMAND ----------

runs = mlflow.search_runs(experiment_names=[EXPERIMENTO], order_by=["metrics.wape ASC"])
cols = ["tags.mlflow.runName", "metrics.wape", "metrics.mae_gwh", "metrics.wape_regulado", "metrics.wape_no_regulado", "params.modelo", "tags.candidato", "start_time"]
display(runs[[c for c in cols if c in runs.columns]])

# COMMAND ----------

yhat_naive = mod.NaiveEstacional().predict(valid)
comp = valid.assign(naive=yhat_naive, lightgbm=yhat_lgbm, dia_obj=((valid["fecha"] + pd.Timedelta(days=H)).dt.dayofweek + 1) % 7 + 1)
dias = {1: "Dom", 2: "Lun", 3: "Mar", 4: "Mié", 5: "Jue", 6: "Vie", 7: "Sáb"}
por_dia = pd.DataFrame({
    "dia_objetivo": [dias[k] for k in sorted(comp["dia_obj"].unique())],
    "wape_naive": [met.wape(d[OBJETIVO], d["naive"]) for _, d in sorted(comp.groupby("dia_obj"))],
    "wape_lightgbm": [met.wape(d[OBJETIVO], d["lightgbm"]) for _, d in sorted(comp.groupby("dia_obj"))],
})
display(por_dia)
display(met.wape_por_segmento(comp, OBJETIVO, "lightgbm", "tipo_mercado"))

# COMMAND ----------

# Visto en una serie: la más grande del país, 8–25 de julio. Las franjas son los festivos del 13 y el 20.
serie_top = comp.groupby(datos.llave_serie(comp))[OBJETIVO].sum().idxmax()
s = comp[datos.llave_serie(comp) == serie_top].assign(fecha_obj=lambda d: d["fecha"] + pd.Timedelta(days=H)).sort_values("fecha_obj")
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(s["fecha_obj"], s[OBJETIVO] / 1e6, "o-", color="#1B1B2F", lw=2.5, label="observado")
ax.plot(s["fecha_obj"], s["naive"] / 1e6, "s--", color="#9CA3AF", label=f"naive (wape {met.wape(s[OBJETIVO], s['naive']):.3f})")
ax.plot(s["fecha_obj"], s["lightgbm"] / 1e6, "^-", color="#7C3AED", label=f"lightgbm (wape {met.wape(s[OBJETIVO], s['lightgbm']):.3f})")
for f in fest:
    if s["fecha_obj"].min() <= f <= s["fecha_obj"].max():
        ax.axvspan(f - pd.Timedelta(hours=12), f + pd.Timedelta(hours=12), color="#B45309", alpha=0.15)
ax.set_title(serie_top)
ax.set_ylabel("GWh/día")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · TODO 2 — una regla de negocio como modelo: `NaiveFestivo`
# MAGIC
# MAGIC Lo que la métrica por segmento acaba de decir es "el problema está en los festivos". La respuesta más honesta no es un modelo
# MAGIC más grande: es una **regla**. Si el día objetivo es festivo, la serie se comporta como un domingo. `NaiveFestivo` en
# MAGIC `src/xm_demanda/model/modelos.py` ya tiene el `predict`; falta el `fit`, que aprende **por serie** el factor
# MAGIC `media(domingos) / media(lunes a viernes)` sobre el entrenamiento, y `factor_global` (la mediana) para series nuevas.
# MAGIC
# MAGIC Fíjate en lo que esto enseña sobre MLOps: una regla de una línea también es un modelo. Tiene `fit`, tiene parámetros
# MAGIC aprendidos (355 factores), se registra con firma, se versiona y se puede servir. El registro no distingue entre "sofisticado"
# MAGIC y "simple"; distingue entre reproducible y no.

# COMMAND ----------

importlib.reload(mod)
run_festivo, yhat_festivo, m_festivo = correr_experimento(mod.NaiveFestivo(fest, H), train, valid, tags={"candidato": "si"}, params={"regla": "festivo=domingo", "n_series_con_factor": "aprendido"})
comp["naive_festivo"] = yhat_festivo
por_dia["wape_naive_festivo"] = [met.wape(d[OBJETIVO], d["naive_festivo"]) for _, d in sorted(comp.groupby("dia_obj"))]
display(por_dia)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Validación temporal: walk-forward con runs anidados
# MAGIC
# MAGIC ![Walk-forward](./../img/walk_forward.png)
# MAGIC
# MAGIC Un solo corte es una muestra de tamaño uno: julio tuvo dos lunes festivos; junio no. *Walk-forward* repite el experimento
# MAGIC con tres cortes sucesivos (4 y 18 de junio, 2 de julio), cada uno entrenando solo con lo anterior y validando 14 días. En
# MAGIC MLflow esto se anota con **runs anidados**: un run padre `walk_forward` con el promedio, y un hijo por fold. En la interfaz se
# MAGIC ven colapsados bajo el padre.
# MAGIC
# MAGIC Mira el resultado antes de leer la conclusión: con un solo corte ganaba la regla de festivo; en el promedio de tres cortes,
# MAGIC LightGBM. Ninguno de los dos resultados es "el verdadero": lo verdadero es que la diferencia entre ambos es pequeña y
# MAGIC depende del mes, y eso también es un hallazgo que hay que registrar.

# COMMAND ----------

CORTES = ["2026-06-04", "2026-06-18", "2026-07-02"]
estimadores = {"naive_estacional": lambda: mod.NaiveEstacional(), "naive_festivo": lambda: mod.NaiveFestivo(fest, H), "lightgbm_global": lambda: mod.GlobalLightGBM()}
resumen = {}
with mlflow.start_run(run_name="walk_forward") as padre:
    mlflow.log_params({"cortes": ",".join(CORTES), "dias_valid": 14, "tipo": "validacion_temporal"})
    for corte in CORTES:
        tr, va = datos.split_temporal(oro, corte, 14, OBJETIVO)
        with mlflow.start_run(run_name=f"fold_{corte}", nested=True):
            mlflow.log_params({"corte": corte, "n_train": len(tr), "n_valid": len(va)})
            for nombre, crear in estimadores.items():
                e = crear().fit(tr, OBJETIVO)
                w = met.wape(va[OBJETIVO], e.predict(va))
                mlflow.log_metric(f"wape_{nombre}", w)
                resumen.setdefault(nombre, []).append(w)
    promedios = {f"wape_promedio_{k}": float(np.mean(v)) for k, v in resumen.items()}
    mlflow.log_metrics(promedios)
display(pd.DataFrame(resumen, index=[f"fold {c}" for c in CORTES]).T.assign(promedio=lambda d: d.mean(axis=1)).round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · La fuga de información, vista en el tracking
# MAGIC
# MAGIC ![Baseline vs modelos](./../img/baseline_vs_modelos.png)
# MAGIC
# MAGIC En la clase 7 la fuga era una idea; hoy es un run. Un analista calcula una "tendencia suavizada" de la serie con una ventana
# MAGIC **centrada** de 3 días y la une por fecha objetivo. Parece una feature razonable y contiene literalmente `y(t+7)`. El modelo
# MAGIC con esa columna da un WAPE de la mitad. Sin un baseline al lado y sin saber cómo se construyó la feature, ese run sería
# MAGIC "el mejor modelo que hemos tenido". Lo registramos con el tag `fuga = true` para que quede como ejemplo en el experimento:
# MAGIC un run demasiado bueno es una alarma, no una celebración.

# COMMAND ----------

oro_fuga = oro.sort_values([*datos.LLAVE_SERIE, "fecha"]).copy()
oro_fuga["tendencia_suavizada"] = oro_fuga.groupby(datos.LLAVE_SERIE)["demanda_real_kwh"].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean().shift(-H))


class LightGBMConFuga(mod.GlobalLightGBM):
    nombre = "lightgbm_fuga"

    def fit(self, df, objetivo):
        import lightgbm as lgb
        X = datos.features_relativas(df)
        X["tendencia_rel"] = (df["tendencia_suavizada"] / df["media_28"].replace(0, np.nan)).clip(0, 5)
        y = datos.objetivo_relativo(df, objetivo)
        ok = y.notna() & np.isfinite(y) & X["tendencia_rel"].notna()
        self.modelo = lgb.LGBMRegressor(**self.params).fit(X[ok], y[ok], categorical_feature=datos.CATEGORICAS)
        self.columnas = list(X.columns)
        return self

    def predict(self, df):
        X = datos.features_relativas(df)
        X["tendencia_rel"] = (df["tendencia_suavizada"] / df["media_28"].replace(0, np.nan)).clip(0, 5)
        return datos.a_kwh(df, self.modelo.predict(X[self.columnas]))


tr_f, va_f = datos.split_temporal(oro_fuga, CORTE, DIAS_VALID, OBJETIVO)
run_fuga, _, m_fuga = correr_experimento(LightGBMConFuga(), tr_f, va_f, tags={"fuga": "true", "candidato": "no"}, params={"feature_extra": "tendencia_suavizada (ventana centrada, unida por fecha objetivo)"})

# COMMAND ----------

# MAGIC %md
# MAGIC **Pregunta 1 (respóndela aquí):** el run `lightgbm_fuga` tiene el mejor WAPE del experimento. Explica en dos o tres frases
# MAGIC por qué no se puede promover, y qué dos cosas del tracking (no de la métrica) te permitirían detectarlo dentro de seis meses
# MAGIC si no estuvieras en esta clase.
# MAGIC
# MAGIC > Tu respuesta:

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Otros tipos de modelo, el mismo tracking
# MAGIC
# MAGIC MLflow no sabe de pronóstico de demanda. Sabe de parámetros, métricas y artefactos, y por eso sirve igual para un
# MAGIC clasificador, un clúster, una red neuronal o un modelo de lenguaje. Dos ejemplos sobre los mismos datos, para que lo vean:
# MAGIC
# MAGIC | Modelo | Pregunta | Métricas que importan | Artefacto que se guarda |
# MAGIC |---|---|---|---|
# MAGIC | **Clasificador** (LightGBM) | ¿La demanda de esta serie caerá más de 10 % en 7 días? Útil para alertas comerciales | AUC, precisión, recall; la matriz de confusión | matriz de confusión como imagen |
# MAGIC | **Clustering** (KMeans) | ¿Cuántos "perfiles de semana" distintos hay entre las 355 series? Útil para segmentar | silhouette; tamaño de cada clúster | perfiles por clúster como imagen y CSV |
# MAGIC
# MAGIC ![Matriz de confusión](./../img/matriz_confusion.png)
# MAGIC
# MAGIC ![Perfiles por clúster](./../img/clusters_perfiles.png)
# MAGIC
# MAGIC Dos lecciones del clasificador: la tasa de positivos es 13 %, así que una "exactitud" de 87 % se consigue prediciendo
# MAGIC siempre "no cae"; por eso la métrica es AUC y recall, no accuracy. Y para KMeans no hay "verdad" contra la que medir:
# MAGIC silhouette mide cohesión, y el artefacto (los perfiles) es lo que un humano revisa. Ambos usan `mlflow.sklearn` /
# MAGIC `mlflow.lightgbm`, los *flavors* nativos; nuestro pronóstico usa `pyfunc` porque lleva preprocesamiento propio.

# COMMAND ----------

import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

# Clasificador: caída > 10 % en t+7 respecto a t
Xtr, Xva = datos.features_relativas(train), datos.features_relativas(valid)
ytr = (train[OBJETIVO] < 0.9 * train["demanda_real_kwh"]).astype(int)
yva = (valid[OBJETIVO] < 0.9 * valid["demanda_real_kwh"]).astype(int)
with mlflow.start_run(run_name="clasificador_caida") as run:
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=7).fit(Xtr, ytr, categorical_feature=datos.CATEGORICAS)
    proba = clf.predict_proba(Xva)[:, 1]
    pred = (proba >= 0.5).astype(int)
    mlflow.log_params({"modelo": "lightgbm_classifier", "objetivo": "caida_mayor_10pct_h7", "umbral": 0.5, "tipo": "clasificacion"})
    mlflow.log_metrics({"auc": roc_auc_score(yva, proba), "precision": precision_score(yva, pred), "recall": recall_score(yva, pred), "tasa_positivos": float(yva.mean())})
    cm = confusion_matrix(yva, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Purples")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred no cae", "pred cae"])
    ax.set_yticklabels(["real no cae", "real cae"])
    plt.tight_layout()
    mlflow.log_figure(fig, "graficas/matriz_confusion.png")
    mlflow.lightgbm.log_model(clf, artifact_path="clasificador", signature=mlflow.models.infer_signature(Xva, proba))
    print("clasificador: auc", round(roc_auc_score(yva, proba), 3), "| recall", round(recall_score(yva, pred), 3))

# Clustering: perfil semanal normalizado por serie
perfil = oro.assign(dow=oro["fecha"].dt.dayofweek, serie=datos.llave_serie(oro)).pivot_table(index="serie", columns="dow", values="demanda_real_kwh", aggfunc="mean")
perfil = perfil[perfil.sum(axis=1) > 0]
perfil_n = perfil.div(perfil.mean(axis=1), axis=0).fillna(1.0)
with mlflow.start_run(run_name="kmeans_perfiles") as run:
    km = KMeans(n_clusters=4, n_init=10, random_state=7).fit(perfil_n.values)
    mlflow.log_params({"modelo": "kmeans", "k": 4, "feature": "perfil_semanal_normalizado", "tipo": "clustering"})
    mlflow.log_metric("silhouette", silhouette_score(perfil_n.values, km.labels_))
    for c in range(4):
        mlflow.log_metric(f"tamano_cluster_{c}", int((km.labels_ == c).sum()))
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in range(4):
        ax.plot(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"], perfil_n.values[km.labels_ == c].mean(axis=0), marker="o", label=f"clúster {c} ({(km.labels_ == c).sum()} series)")
    ax.axhline(1, ls=":", color="gray")
    ax.legend()
    plt.tight_layout()
    mlflow.log_figure(fig, "graficas/perfiles.png")
    mlflow.log_text(perfil_n.assign(cluster=km.labels_).to_csv(), "tablas/series_por_cluster.csv")
    mlflow.sklearn.log_model(km, artifact_path="kmeans", input_example=perfil_n.head(3))
    print("kmeans: silhouette", round(silhouette_score(perfil_n.values, km.labels_), 3), "| tamaños", np.bincount(km.labels_).tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 · Del run al modelo registrado: versiones, alias y linaje en Unity Catalog
# MAGIC
# MAGIC ![Registro en UC](./../img/registro_uc.png)
# MAGIC
# MAGIC Un run es un intento; un **modelo registrado** es un activo: nombre de tres partes (`catálogo.esquema.modelo`), dueño,
# MAGIC permisos, versiones y alias. Registramos dos versiones, las dos candidatas, y ponemos los alias según el walk-forward:
# MAGIC `@champion` al mejor promedio, `@challenger` al otro. Quien consume (el job de la clase 11, un endpoint, una app) escribe
# MAGIC `models:/…demanda_7d@champion` y nunca sabe qué versión es. Promover un modelo nuevo es mover el alias; volver atrás, también.
# MAGIC
# MAGIC Y el linaje queda armado solo: la versión apunta al run, el run al dataset (`features_demanda_diaria` versión N), al commit
# MAGIC del Git folder y a los parámetros. Las cuatro versiones de la clase 8, enlazadas. En XM, además, el patrón es **"deploy code,
# MAGIC not models"**: el código de entrenamiento se promueve entre ambientes y cada ambiente entrena su versión con sus datos; el
# MAGIC artefacto de `dev` no viaja a `prod`.

# COMMAND ----------

candidatos = {"lightgbm_global": run_lgbm, "naive_festivo": run_festivo}
campeon = min(candidatos, key=lambda k: np.mean(resumen[k]))
retador = [k for k in candidatos if k != campeon][0]
runs_cand = candidatos
print("Según walk-forward → champion:", campeon, "| challenger:", retador)

v_champ = reg.registrar_version(runs_cand[campeon], NOMBRE_MODELO)
v_chall = reg.registrar_version(runs_cand[retador], NOMBRE_MODELO)
reg.poner_alias(NOMBRE_MODELO, "champion", v_champ.version)
reg.poner_alias(NOMBRE_MODELO, "challenger", v_chall.version)
print(f"{NOMBRE_MODELO}: versión {v_champ.version} @champion ({campeon}) · versión {v_chall.version} @challenger ({retador})")

# COMMAND ----------

# El consumidor: carga @champion sin saber qué versión es y predice sobre las filas de inferencia (y_h7 nulo = los últimos 7 días).
modelo = reg.cargar(NOMBRE_MODELO, "champion")
ultimos = oro["fecha"] > oro["fecha"].max() - pd.Timedelta(days=H)
inferencia = oro[oro[OBJETIVO].isna() & ultimos].drop(columns=[OBJETIVO])
pron = inferencia[["fecha", *datos.LLAVE_SERIE]].assign(fecha_pronostico=inferencia["fecha"] + pd.Timedelta(days=H), demanda_estimada_kwh=modelo.predict(inferencia).values)
print(f"{len(pron):,} pronósticos para {pron['fecha_pronostico'].min().date()} → {pron['fecha_pronostico'].max().date()} (la clase 11 los escribe en gold_energia.pronostico_demanda)")
display(pron.sort_values("demanda_estimada_kwh", ascending=False).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Abre **Catalog → workspace → c01_<usuario> → Models → demanda_7d**: las dos versiones, sus alias, y en cada una el enlace al
# MAGIC run de origen, la firma y el ejemplo. Desde la pestaña *Lineage* del modelo se llega a la tabla `features_demanda_diaria`.
# MAGIC
# MAGIC **Pregunta 2 (respóndela aquí):** con un solo corte ganaba `naive_festivo`; con walk-forward, `lightgbm_global`, por poco.
# MAGIC ¿Cuál dejarías como `@champion` para XM y por qué? Considera algo más que el WAPE: explicabilidad ante un auditor, costo de
# MAGIC mantener cada uno, qué pasa el próximo festivo, y qué harías para que la decisión sea más clara en tres meses.
# MAGIC
# MAGIC > Tu respuesta:

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificación
# MAGIC
# MAGIC Cuando todo esté en OK: commit `feat(c09): experimento demanda_7d, baselines, lightgbm y modelo registrado` (incluye
# MAGIC `src/xm_demanda/model/`), push, PR hacia `main` de tu fork, CI en verde, merge. Al cierre se libera el tag `s09`. **Hito 5 abierto**;
# MAGIC se cierra en la clase 11 con el pronóstico publicado.

# COMMAND ----------

def verificar():
    checks = []
    importlib.reload(met)
    importlib.reload(mod)
    try:
        checks.append(("wape implementada (3 casos)", met.wape([100, 200], [110, 180]) == 0.1 and met.wape([0, 100], [0, 90]) == 0.1 and np.isnan(met.wape([0, 0], [1, 1]))))
    except NotImplementedError:
        checks.append(("wape implementada", False))
    try:
        nf = mod.NaiveFestivo(fest, H).fit(train, OBJETIVO)
        checks.append(("NaiveFestivo.fit aprende un factor por serie (< 1 en promedio)", len(nf.factores) > 300 and 0.3 < np.median(list(nf.factores.values())) < 1.0))
    except NotImplementedError:
        checks.append(("NaiveFestivo.fit implementado", False))
    r = mlflow.search_runs(experiment_names=[EXPERIMENTO])
    nombres = set(r["tags.mlflow.runName"]) if "tags.mlflow.runName" in r.columns else set()
    esperados = {"naive_estacional", "lightgbm_global", "naive_festivo", "walk_forward", "lightgbm_fuga", "clasificador_caida", "kmeans_perfiles"}
    checks.append((f"Experimento con los runs {sorted(esperados)}", esperados <= nombres))
    checks.append(("Todos los runs de pronóstico registran wape y wape_regulado", r[r["tags.mlflow.runName"].isin(["naive_estacional", "lightgbm_global", "naive_festivo"])][["metrics.wape", "metrics.wape_regulado"]].notna().all().all() if "metrics.wape_regulado" in r.columns else False))
    checks.append(("Un run marcado con fuga = true", ("tags.fuga" in r.columns) and (r["tags.fuga"] == "true").any()))
    checks.append(("walk_forward con 3 runs hijos", ("tags.mlflow.parentRunId" in r.columns) and r["tags.mlflow.parentRunId"].notna().sum() >= 3))
    checks.append(("Runs de otro tipo con sus métricas (auc, silhouette)", {"metrics.auc", "metrics.silhouette"} <= set(r.columns)))
    try:
        cli = mlflow.MlflowClient()
        ch = cli.get_model_version_by_alias(NOMBRE_MODELO, "champion")
        cl = cli.get_model_version_by_alias(NOMBRE_MODELO, "challenger")
        checks.append(("Modelo registrado con @champion y @challenger en versiones distintas", ch.version != cl.version))
        mv = mlflow.models.get_model_info(f"models:/{NOMBRE_MODELO}@champion")
        checks.append(("@champion tiene firma (23 entradas: el contrato sin y_h7 → 1 salida)", mv.signature is not None and len(mv.signature.inputs.inputs) == 23))
        checks.append(("@champion predice sobre 5 filas de oro", len(reg.cargar(NOMBRE_MODELO).predict(valid.head(5).drop(columns=[OBJETIVO]))) == 5))
    except Exception as e:  # noqa: BLE001
        checks.append((f"Modelo registrado con alias ({type(e).__name__})", False))

    print()
    for nombre, ok in checks:
        print(("OK   " if ok else "FALTA") + "  " + nombre)
    print("\nListo: commit y push (incluye src/xm_demanda/model)." if all(ok for _, ok in checks) else "\nRevisa los puntos marcados FALTA.")


verificar()
