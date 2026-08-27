from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from replica_inpp.dominio.calculo.base import (
    _construir_diagnostico,
    _laspeyres_por_grupo,
    _recortar_series_fecha,
    _rellenar_dato_serie_faltante,
)
from replica_inpp.dominio.errores import ErrorCalculo
from replica_inpp.dominio.periodos import PeriodoMensual

_ANTES_2019 = PeriodoMensual(2019, 6)
_INICIO_2019 = PeriodoMensual(2019, 7)
_FIN_2019 = PeriodoMensual(2025, 7)
_DESPUES_2019 = PeriodoMensual(2025, 8)

_P1 = PeriodoMensual(2019, 7)
_P2 = PeriodoMensual(2019, 8)
_P3 = PeriodoMensual(2019, 9)

_COLUMNAS_DIAGNOSTICO = [
    "version",
    "agregacion",
    "rubro",
    "periodo",
    "generico",
    "nivel_faltante",
    "tipo_faltante",
    "detalle",
]


def _serie(datos: dict[str, list[object]], periodos: list[PeriodoMensual]) -> pd.DataFrame:
    return pd.DataFrame(datos, index=periodos).T


# -- _recortar_series_fecha --


def test_recortar_incluye_ambos_extremos_del_rango() -> None:
    df = _serie(
        {"001": [1.0, 2.0, 3.0, 4.0]},
        [_ANTES_2019, _INICIO_2019, _FIN_2019, _DESPUES_2019],
    )
    resultado = _recortar_series_fecha(df, 2019)
    assert list(resultado.columns) == [_INICIO_2019, _FIN_2019]


def test_recortar_version_sin_fin_incluye_todo_lo_posterior_al_inicio() -> None:
    # 2025: RANGOS_CANASTAS[2025] = (Jul 2025, None) -- sin límite superior
    inicio_2025 = PeriodoMensual(2025, 7)
    lejano = PeriodoMensual(2030, 1)
    df = _serie({"001": [1.0, 2.0, 3.0]}, [_ANTES_2019, inicio_2025, lejano])
    resultado = _recortar_series_fecha(df, 2025)
    assert list(resultado.columns) == [inicio_2025, lejano]


def test_recortar_sin_periodos_en_rango_lanza_error_calculo() -> None:
    # serie de una versión distinta, sin ningún periodo dentro del rango de 2019
    df = _serie({"001": [1.0, 2.0]}, [PeriodoMensual(2012, 1), PeriodoMensual(2012, 2)])
    with pytest.raises(ErrorCalculo):
        _recortar_series_fecha(df, 2019)


# -- _rellenar_dato_serie_faltante --


def test_rellenar_hueco_interior_marca_rellenado_con_periodo_fuente() -> None:
    # 001 sin dato en Ago 2019 -- bfill toma el dato del periodo SIGUIENTE
    # (Sep 2019, que corre primero); ffill solo entra si bfill no alcanza
    df = _serie({"001": [100.0, None, 102.0], "002": [50.0, 51.0, 52.0]}, [_P1, _P2, _P3])
    rellenada, diagnostico, periodos_rel = _rellenar_dato_serie_faltante(
        df, 2019, "INPP", "produccion_total"
    )
    assert rellenada.at["001", _P2] == pytest.approx(102.0)
    assert periodos_rel == {_P2}
    fila = diagnostico.iloc[0]
    assert fila["generico"] == "001"
    assert fila["agregacion"] == "INPP"
    assert fila["rubro"] == "produccion_total"
    assert fila["tipo_faltante"] == "rellenado"
    assert str(_P3) in fila["detalle"]


def test_rellenar_fila_totalmente_faltante_no_se_rellena_ni_se_marca() -> None:
    # 003 sin dato en NINGÚN periodo -- bfill/ffill no tiene de dónde tomar
    df = _serie(
        {"001": [100.0, 101.0], "003": [float("nan"), float("nan")]},
        [_P1, _P2],
    )
    rellenada, diagnostico, periodos_rel = _rellenar_dato_serie_faltante(
        df, 2019, "INPP", "produccion_total"
    )
    assert cast("pd.Series[bool]", rellenada.loc["003"]).isna().all()
    assert "003" not in diagnostico["generico"].values
    assert periodos_rel == set()


def test_rellenar_sin_faltantes_devuelve_serie_intacta() -> None:
    df = _serie({"001": [100.0, 101.0]}, [_P1, _P2])
    rellenada, diagnostico, periodos_rel = _rellenar_dato_serie_faltante(
        df, 2019, "INPP", "produccion_total"
    )
    assert rellenada.equals(df)
    assert diagnostico.empty
    assert list(diagnostico.columns) == _COLUMNAS_DIAGNOSTICO
    assert periodos_rel == set()


# -- _laspeyres_por_grupo --


def test_laspeyres_por_grupo_valores_correctos() -> None:
    numerador = _serie({"a": [100.0], "b": [200.0], "c": [50.0], "d": [150.0]}, [_P1])
    ponderador = pd.Series({"a": 10.0, "b": 30.0, "c": 20.0, "d": 40.0})
    cat_por_gen = pd.Series({"a": "X", "b": "X", "c": "Y", "d": "Y"})
    resultado = _laspeyres_por_grupo(numerador, ponderador, cat_por_gen)
    # a mano: X=(10*100+30*200)/40=175; Y=(20*50+40*150)/60=116.6667
    assert resultado.at["X", _P1] == pytest.approx(175.0)
    assert resultado.at["Y", _P1] == pytest.approx(116.6667, abs=1e-3)


def test_laspeyres_por_grupo_un_solo_elemento_devuelve_su_propio_valor() -> None:
    numerador = _serie({"a": [80.0]}, [_P1])
    ponderador = pd.Series({"a": 15.0})
    cat_por_gen = pd.Series({"a": "X"})
    resultado = _laspeyres_por_grupo(numerador, ponderador, cat_por_gen)
    assert resultado.at["X", _P1] == pytest.approx(80.0)


def test_laspeyres_por_grupo_desbordamiento_al_ponderar_lanza_error_calculo() -> None:
    # numerador finito (1e308, pasa SerieNormalizada) pero ponderar desborda a inf
    numerador = _serie({"a": [1e308]}, [_P1])
    ponderador = pd.Series({"a": 40.0})
    cat_por_gen = pd.Series({"a": "X"})
    with pytest.raises(ErrorCalculo):
        _laspeyres_por_grupo(numerador, ponderador, cat_por_gen)


# -- _construir_diagnostico --


def test_construir_diagnostico_celda_nan_produce_fila_con_schema_correcto() -> None:
    df = _serie({"001": [100.0, None]}, [_P1, _P2])
    diagnostico = _construir_diagnostico(df, 2019, "INPP", "produccion_total")
    assert len(diagnostico) == 1
    fila = diagnostico.iloc[0]
    assert fila["version"] == 2019
    assert fila["agregacion"] == "INPP"
    assert fila["rubro"] == "produccion_total"
    assert fila["generico"] == "001"
    assert fila["periodo"] == _P2
    assert fila["tipo_faltante"] == "indice"
    assert "NaN" in fila["detalle"]


def test_construir_diagnostico_sin_nan_devuelve_df_vacio_con_columnas_correctas() -> None:
    df = _serie({"001": [100.0, 101.0]}, [_P1, _P2])
    diagnostico = _construir_diagnostico(df, 2019, "INPP", "produccion_total")
    assert diagnostico.empty
    assert list(diagnostico.columns) == _COLUMNAS_DIAGNOSTICO
