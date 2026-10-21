"""Estimadores con una misma interfaz: fit(df_oro) y predict(df_oro) -> kWh.

Todos reciben filas de la tabla de oro (el contrato de la clase 7) y devuelven la demanda
estimada en t+h en kWh. Así el envoltorio pyfunc, la firma y la inferencia son iguales para
un baseline de una línea y para LightGBM.
"""

import numpy as np
import pandas as pd

from xm_demanda.model.datos import (
    CATEGORICAS,
    a_kwh,
    features_relativas,
    llave_serie,
    objetivo_relativo,
)


class NaiveEstacional:
    """y(t+7) = y(t): el mismo día de la semana anterior. El baseline que hay que vencer."""

    nombre = "naive_estacional"

    def fit(self, df: pd.DataFrame, objetivo: str):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["demanda_real_kwh"].to_numpy(dtype=float)


class NaiveFestivo:
    """Naive estacional con una regla de negocio: si t+h es festivo, se comporta como domingo.

    fit() aprende por serie el factor domingo/día-laboral sobre el entrenamiento. predict()
    aplica y(t) * factor cuando la fecha objetivo (t + horizonte) es festivo y no es domingo.
    """

    nombre = "naive_festivo"

    def __init__(self, festivos: list, horizonte: int = 7):
        self.festivos = set(pd.to_datetime(list(festivos)))
        self.horizonte = horizonte
        self.factores: dict[str, float] = {}
        self.factor_global = 1.0

    def fit(self, df: pd.DataFrame, objetivo: str):
        # TODO (lab 9, sección 7): self.factores = {serie: media(domingos) / media(lun-vie)}
        # usando dia_semana (1 = domingo, 2..6 = lunes a viernes) y llave_serie(df); y
        # self.factor_global = mediana de los factores (para series nuevas). Devolver self.
        raise NotImplementedError("NaiveFestivo.fit: completa este método (lab 9, sección 7)")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        fecha_obj = pd.to_datetime(df["fecha"]) + pd.Timedelta(days=self.horizonte)
        es_festivo_obj = fecha_obj.isin(self.festivos) & (fecha_obj.dt.dayofweek != 6)
        factor = llave_serie(df).map(self.factores).fillna(self.factor_global).to_numpy(dtype=float)
        y_t = df["demanda_real_kwh"].to_numpy(dtype=float)
        return np.where(es_festivo_obj.to_numpy(), y_t * factor, y_t)


class GlobalLightGBM:
    """Un solo modelo para las 355 series, sobre features sin escala (ver datos.py)."""

    nombre = "lightgbm_global"
    PARAMS = dict(
        objective="l1",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=7,
    )

    def __init__(self, **params):
        self.params = {**self.PARAMS, **params}
        self.modelo = None

    def fit(self, df: pd.DataFrame, objetivo: str):
        import lightgbm as lgb

        X = features_relativas(df)
        y = objetivo_relativo(df, objetivo)
        relativas = [c for c in X.columns if c.endswith("_rel")]
        ok = y.notna() & np.isfinite(y) & X[relativas].notna().all(axis=1)
        self.modelo = lgb.LGBMRegressor(**self.params)
        self.modelo.fit(X[ok], y[ok], categorical_feature=CATEGORICAS)
        self.columnas = list(X.columns)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = features_relativas(df)[self.columnas]
        return a_kwh(df, self.modelo.predict(X))

    def importancia(self) -> pd.Series:
        """Importancia por ganancia, en % del total, de mayor a menor."""
        imp = pd.Series(self.modelo.booster_.feature_importance("gain"), index=self.columnas)
        return (imp / imp.sum() * 100).sort_values(ascending=False)
