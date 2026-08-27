from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.base import Resultado, Vista
from replica_inpp.dominio.periodos import PeriodoMensual


def _df_largo_1col() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [(PeriodoMensual(2019, 7), "001"), (PeriodoMensual(2019, 8), "001")],
        names=["periodo", "indice"],
    )
    return pd.DataFrame({"x": [1.0, 2.0]}, index=idx)


def _df_largo_ncols(con_nan: bool = False) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [
            (PeriodoMensual(2019, 7), "001"),
            (PeriodoMensual(2019, 8), "001"),
            (PeriodoMensual(2019, 7), "002"),
            (PeriodoMensual(2019, 8), "002"),
        ],
        names=["periodo", "indice"],
    )
    if con_nan:
        return pd.DataFrame(
            {"a": [1.0, 2.0, np.nan, 4.0], "b": [10.0, np.nan, 30.0, 40.0]},
            index=idx,
        )
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]}, index=idx)


class _ResultadoMinimo(Resultado):
    @property
    def resultado(self) -> Vista:
        return Vista(self.df, ["x"])

    @property
    def resumen(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def reporte(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def diagnostico(self) -> pd.DataFrame:
        return pd.DataFrame()

    def _repr_html_(self) -> str:
        return ""


# ---------- Vista ----------


def test_vista_largo_retorna_df_sin_transformar() -> None:
    df = _df_largo_1col()
    assert Vista(df, ["x"]).largo is df


def test_vista_ancho_1col_filas_indice_cols_periodo() -> None:
    ancho = Vista(_df_largo_1col(), ["x"]).ancho
    assert ancho.index.name == "indice"
    assert ancho.columns.name == "periodo"
    assert ancho.shape == (1, 2)


def test_vista_ancho_ncols_filas_multiindex() -> None:
    ancho = Vista(_df_largo_ncols(), ["a", "b"]).ancho
    assert isinstance(ancho.index, pd.MultiIndex)
    assert ancho.index.names == ["indice", "metrica"]
    assert ancho.columns.name == "periodo"


def test_vista_ancho_ncols_nivel_metrica_es_indexable_por_nombre() -> None:
    ancho = Vista(_df_largo_ncols(), ["a", "b"]).ancho
    fila_a = ancho.xs("a", level="metrica")
    assert list(fila_a.index) == ["001", "002"]


def test_vista_repr_html_devuelve_string() -> None:
    html = Vista(_df_largo_1col(), ["x"])._repr_html_()
    assert isinstance(html, str)
    assert len(html) > 0


def test_vista_ancho_ncols_preserva_nan() -> None:
    df = _df_largo_ncols(con_nan=True)
    ancho = Vista(df, ["a", "b"]).ancho
    p1 = PeriodoMensual(2019, 7)
    p2 = PeriodoMensual(2019, 8)
    assert ancho.shape == (4, 2)
    assert set(ancho.index) == {("001", "a"), ("001", "b"), ("002", "a"), ("002", "b")}
    assert pd.isna(ancho.loc[("002", "a"), cast(Any, p1)])
    assert pd.isna(ancho.loc[("001", "b"), cast(Any, p2)])
    assert ancho.loc[("001", "a"), cast(Any, p1)] == 1.0
    assert ancho.loc[("002", "b"), cast(Any, p2)] == 40.0


def test_vista_ancho_metrica_nan_en_todos_los_periodos_no_desaparece_fila() -> None:
    idx = pd.MultiIndex.from_tuples(
        [
            (PeriodoMensual(2019, 7), "001"),
            (PeriodoMensual(2019, 8), "001"),
        ],
        names=["periodo", "indice"],
    )
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [10.0, 20.0]}, index=idx)
    ancho = Vista(df, ["a", "b"]).ancho
    assert ("001", "a") in ancho.index
    assert ancho.loc[cast(Any, ("001", "a"))].isna().all()


# ---------- Resultado ----------


def test_resultado_no_instanciable_directamente() -> None:
    with pytest.raises(TypeError):
        Resultado(_df_largo_1col())  # type: ignore[abstract]


def test_resultado_subclase_minima_construye() -> None:
    r = _ResultadoMinimo(_df_largo_1col())
    assert r.df.shape == (2, 1)


def test_resultado_df_vacio_falla() -> None:
    idx = pd.MultiIndex.from_tuples([], names=["periodo", "indice"])
    df_vacio = pd.DataFrame({"x": []}, index=idx)
    with pytest.raises(InvarianteViolado):
        _ResultadoMinimo(df_vacio)


def test_resultado_df_sin_multiindex_falla() -> None:
    df_plano = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(InvarianteViolado):
        _ResultadoMinimo(df_plano)


def test_resultado_df_multiindex_nombres_incorrectos_falla() -> None:
    idx = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["foo", "bar"])
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=idx)
    with pytest.raises(InvarianteViolado):
        _ResultadoMinimo(df)


def test_resultado_df_multiindex_nlevels_distinto_de_2_falla() -> None:
    idx = pd.MultiIndex.from_tuples(
        [("p", "i", "extra"), ("p2", "i2", "extra2")],
        names=["periodo", "indice", "extra"],
    )
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=idx)
    with pytest.raises(InvarianteViolado):
        _ResultadoMinimo(df)


def test_resultado_df_multicolumna_falla() -> None:
    df = _df_largo_ncols()
    with pytest.raises(InvarianteViolado):
        _ResultadoMinimo(df)


def test_resultado_df_con_duplicados_falla() -> None:
    p = PeriodoMensual(2019, 7)
    idx = pd.MultiIndex.from_tuples([(p, "001"), (p, "001")], names=["periodo", "indice"])
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=idx)
    with pytest.raises(InvarianteViolado):
        _ResultadoMinimo(df)


def test_resultado_df_property_retorna_lo_guardado() -> None:
    df = _df_largo_1col()
    r = _ResultadoMinimo(df)
    assert r.df is df


def test_resultado_pipe_aplica_funcion() -> None:
    r = _ResultadoMinimo(_df_largo_1col())
    assert r.pipe(lambda res: res.df.shape) == (2, 1)


def test_resultado_pipe_reenvia_args_y_kwargs() -> None:
    r = _ResultadoMinimo(_df_largo_1col())

    def fn(res: Resultado, factor: int, *, offset: int) -> int:
        return res.df.shape[0] * factor + offset

    assert r.pipe(fn, 3, offset=1) == 7
