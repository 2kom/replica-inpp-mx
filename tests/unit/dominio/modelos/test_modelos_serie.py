from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.periodos import PeriodoMensual

_CODIGOS = ("001", "002", "003")
_NOMBRES = ("soya", "frijol", "garbanzo")
_PERIODOS = (
    PeriodoMensual(2019, 1),
    PeriodoMensual(2019, 2),
    PeriodoMensual(2019, 3),
)
_VALORES = (
    (100.0, 101.0, 102.0),
    (100.0, 102.0, 104.0),
    (100.0, None, 106.0),
)


def _df(
    codigos: tuple[str, ...] = _CODIGOS,
    nombres: tuple[str, ...] = _NOMBRES,
    periodos: tuple[PeriodoMensual, ...] = _PERIODOS,
    valores: tuple[tuple[float | None, ...], ...] = _VALORES,
    con_generico: bool = True,
) -> pd.DataFrame:
    df = pd.DataFrame(
        list(valores),
        index=pd.Index(list(codigos), name="codigo"),
        columns=list(periodos),
    )
    if con_generico:
        df.insert(0, "generico", list(nombres))
    return df


# ---------- Construcción ----------


def test_construccion_valida() -> None:
    s = SerieNormalizada(_df(), recorte="mercado_nacional")
    assert list(s.df.index) == list(_CODIGOS)


def test_recorte_property() -> None:
    s = SerieNormalizada(_df(), recorte="mercado_exportacion")
    assert s.recorte == "mercado_exportacion"


def test_columna_generico_presente_y_primero() -> None:
    s = SerieNormalizada(_df(), recorte="mercado_nacional")
    assert s.df.columns[0] == "generico"
    assert list(s.df["generico"]) == list(_NOMBRES)


# ---------- Invariantes ----------


def test_indice_duplicado_falla() -> None:
    df = _df(codigos=("001", "001", "003"))
    with pytest.raises(InvarianteViolado):
        SerieNormalizada(df, recorte="mercado_nacional")


def test_indice_con_cadena_vacia_falla() -> None:
    df = _df(codigos=("001", "", "003"))
    with pytest.raises(InvarianteViolado):
        SerieNormalizada(df, recorte="mercado_nacional")


def test_falta_columna_generico_falla() -> None:
    df = _df(con_generico=False)
    with pytest.raises(InvarianteViolado):
        SerieNormalizada(df, recorte="mercado_nacional")


def test_sin_columnas_de_periodo_falla() -> None:
    df = pd.DataFrame({"generico": list(_NOMBRES)}, index=pd.Index(list(_CODIGOS), name="codigo"))
    with pytest.raises(InvarianteViolado):
        SerieNormalizada(df, recorte="mercado_nacional")


def test_columna_no_periodo_mensual_falla() -> None:
    df = _df()
    df.columns = cast(Any, ["generico", "no_es_periodo", *list(_PERIODOS[1:])])
    with pytest.raises(InvarianteViolado):
        SerieNormalizada(df, recorte="mercado_nacional")


def test_columnas_periodo_duplicadas_falla() -> None:
    df = _df(periodos=(_PERIODOS[0], _PERIODOS[0], _PERIODOS[2]))
    with pytest.raises(InvarianteViolado):
        SerieNormalizada(df, recorte="mercado_nacional")


def test_valores_negativos_falla() -> None:
    df = _df(valores=((100.0, -1.0, 102.0), (100.0, 102.0, 104.0), (100.0, None, 106.0)))
    with pytest.raises(InvarianteViolado):
        SerieNormalizada(df, recorte="mercado_nacional")


def test_valores_con_nan_no_falla() -> None:
    SerieNormalizada(_df(), recorte="mercado_nacional")


@pytest.mark.parametrize("valor_no_finito", [float("inf"), float("-inf")])
def test_valores_no_finitos_falla(valor_no_finito: float) -> None:
    df = _df(valores=((100.0, valor_no_finito, 102.0), (100.0, 102.0, 104.0), (100.0, None, 106.0)))
    with pytest.raises(InvarianteViolado):
        SerieNormalizada(df, recorte="mercado_nacional")


# ---------- Properties ----------


def test_columnas_de_periodo_desordenadas_se_ordenan_cronologicamente() -> None:
    # el relleno bfill/ffill futuro opera por posición física de columna -- si
    # SerieNormalizada no ordena, propagaría el dato del vecino físico equivocado.
    periodos_desordenados = (_PERIODOS[0], _PERIODOS[2], _PERIODOS[1])
    df = _df(periodos=periodos_desordenados)
    s = SerieNormalizada(df, recorte="mercado_nacional")
    assert list(s.df.columns) == ["generico", *sorted(periodos_desordenados)]


def test_columna_generico_no_se_reordena_por_valor() -> None:
    # "generico" es texto -- no debe intentar compararse con PeriodoMensual al ordenar.
    df = _df(periodos=(_PERIODOS[2], _PERIODOS[0], _PERIODOS[1]))
    s = SerieNormalizada(df, recorte="mercado_nacional")
    assert s.df.columns[0] == "generico"


def test_repr_html_devuelve_string() -> None:
    s = SerieNormalizada(_df(), recorte="mercado_nacional")
    assert isinstance(s._repr_html_(), str)
