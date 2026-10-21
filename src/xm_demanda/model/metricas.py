"""Métricas de pronóstico. Funciones puras sobre arreglos y DataFrames de pandas."""

import numpy as np
import pandas as pd


def mae(y, yhat) -> float:
    """Error absoluto medio, en las unidades de y (kWh)."""
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def wape(y, yhat) -> float:
    """Weighted Absolute Percentage Error: sum(|y - yhat|) / sum(|y|).

    Es la métrica principal del proyecto porque pondera cada serie por su energía
    (una serie de 30 GWh pesa más que una de 30 kWh) y no explota cuando y = 0,
    a diferencia del MAPE. Devuelve una fracción (0.05 = 5 %).
    Si sum(|y|) == 0 devuelve nan.
    >>> wape([100, 200], [110, 180])
    0.1
    """
    # TODO (lab 9, sección 2): implementar en 3 líneas con numpy. Cuida el caso sum(|y|) == 0.
    raise NotImplementedError("wape: completa esta función (lab 9, sección 2)")


def metricas(y, yhat) -> dict[str, float]:
    """Las dos métricas que registra cada run: wape y mae en GWh."""
    return {"wape": wape(y, yhat), "mae_gwh": mae(y, yhat) / 1e6}


def wape_por_segmento(df: pd.DataFrame, col_y: str, col_yhat: str, segmento: str) -> pd.DataFrame:
    """WAPE por cada valor de `segmento`, ordenado de peor a mejor, con la energía de cada uno.

    Columnas: segmento, wape, energia_gwh, filas. Es lo que convierte "wape 0.05" en
    "wape 0.14 los lunes festivos": la métrica global esconde segmentos.
    """
    filas = []
    for valor, d in df.groupby(segmento):
        filas.append(
            {
                segmento: valor,
                "wape": wape(d[col_y], d[col_yhat]),
                "energia_gwh": float(d[col_y].abs().sum() / 1e6),
                "filas": int(len(d)),
            }
        )
    return pd.DataFrame(filas).sort_values("wape", ascending=False).reset_index(drop=True)
