"""Entrenamiento, evaluación y registro (clases 9-11).

- metricas.py : métricas puras (WAPE, MAE) y por segmento. Sin Spark.
- datos.py    : corte temporal y features sin escala a partir de la tabla de oro (pandas).
- modelos.py  : estimadores con la misma interfaz fit/predict: baselines y LightGBM global.
- registro.py : envoltorio pyfunc, firma, registro en Unity Catalog y alias.
"""
