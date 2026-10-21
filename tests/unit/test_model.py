import numpy as np
import pandas as pd
import pytest

from xm_demanda.model.datos import (
    CALENDARIO,
    CATEGORICAS,
    RELATIVAS,
    TOPE_RELATIVO,
    a_kwh,
    columnas_features,
    features_relativas,
    objetivo_relativo,
    split_temporal,
)
from xm_demanda.model.metricas import mae, wape, wape_por_segmento
from xm_demanda.model.modelos import NaiveEstacional, NaiveFestivo


def _wape(y, yhat):
    try:
        return wape(y, yhat)
    except NotImplementedError:
        pytest.skip("Pendiente: lab 9, sección 2")


def _oro_minima(n_dias=30, series=("A", "B")):
    """Tabla de oro diminuta: dos series, una 100 veces más grande que la otra."""
    filas = []
    for i, s in enumerate(series):
        escala = 100.0 ** i
        for d in range(n_dias):
            fecha = pd.Timestamp("2026-03-01") + pd.Timedelta(days=d)
            dia_semana = (fecha.dayofweek + 1) % 7 + 1
            base = 1000 * escala * (0.8 if dia_semana == 1 else 1.0)
            filas.append(
                {
                    "fecha": fecha, "codigo_sic_agente": s, "mercado_comercializacion": "M",
                    "tipo_mercado": "Regulado", "clasificacion_industrial": "SIN CLASIFICAR",
                    "demanda_real_kwh": base, "ciiu_seccion": "X", "dia_semana": dia_semana,
                    "es_fin_de_semana": dia_semana in (1, 7), "es_festivo": False, "mes": 3,
                    "dia_del_anio": fecha.dayofyear, "lag_1": base, "lag_7": base, "lag_14": base,
                    "lag_28": base, "media_7": base, "desv_7": 10.0 * escala,
                    "media_28": 1000 * escala, "desv_28": 10.0 * escala, "es_cero": False,
                    "dias_con_datos": d + 1,
                    "y_h7": base if d < n_dias - 7 else np.nan,
                }
            )
    return pd.DataFrame(filas)


def test_wape_basico_y_con_ceros_en_y():
    assert _wape([100, 200], [110, 180]) == pytest.approx(0.1)
    assert _wape([0, 100], [0, 90]) == pytest.approx(0.1)  # el MAPE explotaría en el 0
    assert np.isnan(_wape([0, 0], [1, 1]))


def test_wape_pondera_por_energia():
    """Un 10 % de error en la serie grande pesa 100 veces más que en la pequeña."""
    y = [10_000, 100]
    assert _wape(y, [9_000, 100]) == pytest.approx(1000 / 10_100)
    assert _wape(y, [10_000, 90]) == pytest.approx(10 / 10_100)


def test_mae_en_unidades_de_y():
    assert mae([1.0, 3.0], [2.0, 1.0]) == 1.5


def test_wape_por_segmento_ordena_de_peor_a_mejor():
    try:
        wape([1], [1])
    except NotImplementedError:
        pytest.skip("Pendiente: lab 9, sección 2")
    df = pd.DataFrame(
        {"y": [100, 100, 100, 100], "yhat": [100, 100, 50, 100], "seg": ["a", "a", "b", "b"]}
    )
    out = wape_por_segmento(df, "y", "yhat", "seg")
    assert list(out["seg"]) == ["b", "a"] and out.loc[0, "wape"] == pytest.approx(0.25)


def test_split_temporal_no_mezcla_fechas():
    df = _oro_minima()
    train, valid = split_temporal(df, "2026-03-15", 7, "y_h7")
    assert train["fecha"].max() < valid["fecha"].min()
    assert valid["fecha"].max() <= pd.Timestamp("2026-03-22")
    assert train["y_h7"].notna().all() and valid["y_h7"].notna().all()


def test_features_relativas_vuelven_comparables_las_series():
    df = _oro_minima()
    X = features_relativas(df)
    assert list(X.columns) == columnas_features()
    assert X.loc[df["codigo_sic_agente"] == "A", "demanda_real_kwh_rel"].max() <= TOPE_RELATIVO
    a = X.loc[df["codigo_sic_agente"] == "A", "lag_7_rel"].mean()
    b = X.loc[df["codigo_sic_agente"] == "B", "lag_7_rel"].mean()
    assert a == pytest.approx(b)  # misma forma, distinta escala → mismas features
    assert all(str(X[c].dtype) == "category" for c in CATEGORICAS)
    assert all(np.issubdtype(X[c].dtype, np.integer) for c in CALENDARIO)
    assert len(RELATIVAS) == 8


def test_objetivo_relativo_y_vuelta_a_kwh():
    df = _oro_minima().dropna(subset=["y_h7"])
    ratio = objetivo_relativo(df, "y_h7")
    assert np.allclose(a_kwh(df, ratio.to_numpy()), df["y_h7"])


def test_a_kwh_sin_base_cae_al_naive():
    df = _oro_minima().head(3).copy()
    df["media_28"] = 0.0
    assert np.allclose(a_kwh(df, np.array([2.0, 2.0, 2.0])), df["demanda_real_kwh"])


def test_naive_estacional_devuelve_y_de_hoy():
    df = _oro_minima()
    assert np.allclose(NaiveEstacional().fit(df, "y_h7").predict(df), df["demanda_real_kwh"])


def test_naive_festivo_aplica_factor_solo_en_festivos():
    df = _oro_minima()
    festivo = pd.Timestamp("2026-03-23")  # lunes
    modelo = NaiveFestivo(festivos=[festivo], horizonte=7)
    try:
        modelo.fit(df, "y_h7")
    except NotImplementedError:
        pytest.skip("Pendiente: lab 9, sección 7")
    assert modelo.factores["A·M·Regulado·SIN CLASIFICAR"] == pytest.approx(0.8)
    pred = modelo.predict(df)
    es_obj_festivo = (df["fecha"] + pd.Timedelta(days=7)) == festivo
    assert np.allclose(pred[es_obj_festivo], df.loc[es_obj_festivo, "demanda_real_kwh"] * 0.8)
    assert np.allclose(pred[~es_obj_festivo], df.loc[~es_obj_festivo, "demanda_real_kwh"])
