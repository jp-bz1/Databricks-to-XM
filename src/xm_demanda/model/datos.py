"""De la tabla de oro (pandas) a lo que necesita un modelo: corte temporal y features sin escala.

Regla de escala: las 355 series van de 30 kWh a 35 GWh por día. Un modelo global solo puede
aprender de todas a la vez si las magnitudes se vuelven comparables: cada valor se divide por
`media_28` de su propia serie, y el objetivo también. Al predecir se vuelve a kWh multiplicando.
"""

import numpy as np
import pandas as pd

from xm_demanda.silver.transform import LLAVE_SERIE

ESCALA = "media_28"
RELATIVAS = [
    "demanda_real_kwh", "lag_1", "lag_7", "lag_14", "lag_28", "media_7", "desv_7", "desv_28"
]
CALENDARIO = [
    "dia_semana", "es_fin_de_semana", "es_festivo", "mes", "dia_del_anio",
    "es_cero", "dias_con_datos",
]
CATEGORICAS = ["tipo_mercado", "ciiu_seccion", "mercado_comercializacion", "codigo_sic_agente"]
TOPE_RELATIVO = 5.0  # un valor 5 veces la media de la serie ya es un atípico; se recorta


def columnas_features() -> list[str]:
    """Nombres de las columnas que entran al modelo global, en orden."""
    return [f"{c}_rel" for c in RELATIVAS] + CALENDARIO + CATEGORICAS


def split_temporal(
    df: pd.DataFrame, corte: str, dias_valid: int, objetivo: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrenamiento = fecha <= corte; validación = los `dias_valid` días siguientes.

    Solo filas con objetivo.

    Nunca se mezcla: ninguna fecha de validación es anterior a una de entrenamiento.
    """
    corte = pd.Timestamp(corte)
    con_objetivo = df[df[objetivo].notna()]
    train = con_objetivo[con_objetivo["fecha"] <= corte]
    fin = corte + pd.Timedelta(days=dias_valid)
    valid = con_objetivo[(con_objetivo["fecha"] > corte) & (con_objetivo["fecha"] <= fin)]
    return train, valid


def features_relativas(df: pd.DataFrame) -> pd.DataFrame:
    """Features del modelo global: magnitudes / media_28 (recortadas), calendario, categóricas."""
    base = df[ESCALA].replace(0, np.nan)
    rel = {f"{c}_rel": (df[c] / base).clip(0, TOPE_RELATIVO) for c in RELATIVAS}
    X = pd.DataFrame(rel, index=df.index)
    for c in CALENDARIO:
        X[c] = df[c].astype(int)
    for c in CATEGORICAS:
        X[c] = df[c].astype("category")
    return X


def objetivo_relativo(df: pd.DataFrame, objetivo: str) -> pd.Series:
    """y_h / media_28, recortado. Nan donde no hay base o no hay objetivo."""
    return (df[objetivo] / df[ESCALA].replace(0, np.nan)).clip(0, TOPE_RELATIVO)


def a_kwh(df: pd.DataFrame, ratio: np.ndarray) -> np.ndarray:
    """Vuelve de ratio a kWh. Si la serie no tiene base (media_28 = 0), cae al naive estacional."""
    base = df[ESCALA].to_numpy(dtype=float)
    return np.where(base > 0, ratio * base, df["demanda_real_kwh"].to_numpy(dtype=float))


def llave_serie(df: pd.DataFrame) -> pd.Series:
    """Identificador de serie como texto, para agrupar y para reportar."""
    return df[LLAVE_SERIE].astype(str).agg("·".join, axis=1)
