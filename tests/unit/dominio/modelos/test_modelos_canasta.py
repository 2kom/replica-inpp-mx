from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.canasta import CanastaINPP

_CODIGOS = ("001", "002", "003")

_COLUMNAS_PESO = (
    "produccion total",
    "bienes intermedios",
    "bienes finales",
    "demanda interna total",
    "demanda interna consumo",
    "demanda interna capital",
    "exportaciones",
)


def _columnas_base() -> dict[str, list[Any]]:
    return {
        "generico": ["soya", "frijol", "garbanzo"],
        "codigo sector": ["11", "11", "12"],
        "sector": ["11 agricultura", "11 agricultura", "12 mineria"],
        "codigo subsector": ["111", "111", "121"],
        "subsector": ["111 agricultura", "111 agricultura", "121 mineria de carbon"],
        "codigo rama": ["1111", "1111", "1211"],
        "rama": ["1111 cultivo", "1111 cultivo", "1211 extraccion de carbon"],
        "codigo subrama": ["11111", "11111", "12111"],
        "subrama": [
            "11111 cultivo de soya",
            "11111 cultivo de soya",
            "12111 extraccion de carbon mineral",
        ],
        "codigo clase": ["111110", "111131", "121110"],
        "clase": [
            "111110 cultivo de soya",
            "111131 cultivo de frijol",
            "121110 extraccion de carbon mineral",
        ],
        **{col: [50.0, 30.0, 20.0] for col in _COLUMNAS_PESO},
        "encadenamiento total": [None, None, None],
        "encadenamiento produccion nacional": [None, None, None],
        "encadenamiento exportacion": [1.1, None, 1.3],
        "encadenamiento uso final": [1.1, None, 1.3],
    }


def _df(
    codigos: tuple[str, ...] = _CODIGOS, overrides: dict[str, list[Any]] | None = None
) -> pd.DataFrame:
    columnas = _columnas_base()
    if overrides:
        columnas.update(overrides)
    return pd.DataFrame(columnas, index=pd.Index(list(codigos), name="codigo"))


# ---------- Construcción ----------


def test_construccion_valida() -> None:
    c = CanastaINPP(_df(), version=2019)
    assert list(c.df.index) == list(_CODIGOS)


def test_version_property() -> None:
    c = CanastaINPP(_df(), version=2025)
    assert c.version == 2025


@pytest.mark.parametrize("version", [2010, 2013, 2018, 2024, 0, 9999])
def test_version_invalida_falla(version: int) -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(), version=version)  # type: ignore[arg-type]


# ---------- Invariantes: índice ----------


def test_indice_duplicado_falla() -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(codigos=("001", "001", "003")), version=2019)


def test_indice_con_cadena_vacia_falla() -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(codigos=("001", "", "003")), version=2019)


# -- regla de ausencia: None/NaN, independiente de la regla de formato --


def test_indice_ausente_falla() -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(codigos=cast(Any, ("001", None, "003"))), version=2019)


# -- regla de formato: valor PRESENTE que no matchea 3 dígitos exactos --


@pytest.mark.parametrize("codigo_invalido", ["12", "00A", "1234", " 001", "001 "])
def test_indice_formato_invalido_falla(codigo_invalido: str) -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(codigos=("001", codigo_invalido, "003")), version=2019)


def test_indice_formato_3_digitos_no_falla() -> None:
    CanastaINPP(_df(codigos=("000", "001", "999")), version=2019)


# ---------- Invariantes: columnas de peso ----------


@pytest.mark.parametrize("col", _COLUMNAS_PESO)
def test_peso_columna_no_numerica_falla(col: str) -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(overrides={col: ["cincuenta", 30.0, 20.0]}), version=2019)


def test_peso_columna_valor_negativo_falla() -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(overrides={"exportaciones": [-1.0, 81.0, 20.0]}), version=2019)


def test_peso_columna_cero_real_no_falla() -> None:
    # cero es un valor real -- genérico que no participa nada en ese recorte
    CanastaINPP(_df(overrides={"bienes intermedios": [0.0, 80.0, 20.0]}), version=2019)


def test_peso_columna_suma_distinta_de_100_falla() -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(overrides={"exportaciones": [50.0, 30.0, 30.0]}), version=2019)


def test_peso_columna_nan_se_ignora_en_la_suma() -> None:
    # 2 filas no nulas suman 100 -- la fila NaN no participa en ese recorte
    CanastaINPP(_df(overrides={"bienes finales": [70.0, 30.0, None]}), version=2019)


# ---------- Invariantes: encadenamiento todo-o-nada ----------


@pytest.mark.parametrize("col", ["encadenamiento total", "encadenamiento produccion nacional"])
def test_encadenamiento_todo_o_nada_parcial_falla(col: str) -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(overrides={col: [1.0, None, 1.2]}), version=2025)


@pytest.mark.parametrize("col", ["encadenamiento total", "encadenamiento produccion nacional"])
def test_encadenamiento_todo_o_nada_completamente_nan_no_falla(col: str) -> None:
    CanastaINPP(_df(overrides={col: [None, None, None]}), version=2019)


@pytest.mark.parametrize("col", ["encadenamiento total", "encadenamiento produccion nacional"])
def test_encadenamiento_todo_o_nada_completamente_lleno_no_falla(col: str) -> None:
    CanastaINPP(_df(overrides={col: [1.0, 1.1, 1.2]}), version=2025)


@pytest.mark.parametrize("col", ["encadenamiento total", "encadenamiento produccion nacional"])
def test_encadenamiento_todo_o_nada_valor_no_positivo_falla(col: str) -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(overrides={col: [1.0, 0.0, 1.2]}), version=2025)


@pytest.mark.parametrize("col", ["encadenamiento total", "encadenamiento produccion nacional"])
def test_encadenamiento_todo_o_nada_texto_no_numerico_falla(col: str) -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(overrides={col: [1.0, "-", 1.2]}), version=2025)


# ---------- Invariantes: encadenamiento parcial ----------


@pytest.mark.parametrize("col", ["encadenamiento exportacion", "encadenamiento uso final"])
def test_encadenamiento_parcial_permite_nan_parcial_no_falla(col: str) -> None:
    CanastaINPP(_df(overrides={col: [1.1, None, 1.3]}), version=2025)


@pytest.mark.parametrize("col", ["encadenamiento exportacion", "encadenamiento uso final"])
def test_encadenamiento_parcial_completamente_nan_no_falla(col: str) -> None:
    CanastaINPP(_df(overrides={col: [None, None, None]}), version=2019)


@pytest.mark.parametrize("col", ["encadenamiento exportacion", "encadenamiento uso final"])
def test_encadenamiento_parcial_valor_no_positivo_falla(col: str) -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(overrides={col: [1.1, 0.0, None]}), version=2025)


@pytest.mark.parametrize("col", ["encadenamiento exportacion", "encadenamiento uso final"])
def test_encadenamiento_parcial_texto_no_numerico_falla(col: str) -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(overrides={col: ["-", None, 1.3]}), version=2025)


# ---------- Invariantes: encadenamiento infinito (las 4 columnas) ----------


@pytest.mark.parametrize(
    "col",
    [
        "encadenamiento total",
        "encadenamiento produccion nacional",
        "encadenamiento exportacion",
        "encadenamiento uso final",
    ],
)
def test_encadenamiento_valor_infinito_falla(col: str) -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(overrides={col: [1.0, float("inf"), 1.2]}), version=2025)


# ---------- Invariantes: columnas core nunca vacías ----------


@pytest.mark.parametrize(
    "col",
    [
        "generico",
        "codigo sector",
        "sector",
        "codigo subsector",
        "subsector",
        "codigo rama",
        "rama",
        "codigo subrama",
        "subrama",
        "codigo clase",
        "clase",
    ],
)
def test_columna_core_vacia_falla(col: str) -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(overrides={col: ["soya", "", "garbanzo"]}), version=2019)


@pytest.mark.parametrize("col", ["generico", "sector", "codigo clase"])
def test_columna_core_nan_falla(col: str) -> None:
    with pytest.raises(InvarianteViolado):
        CanastaINPP(_df(overrides={col: ["soya", None, "garbanzo"]}), version=2019)


# ---------- Properties ----------


def test_df_property() -> None:
    df = _df()
    c = CanastaINPP(df, version=2019)
    assert c.df is df


def test_repr_html_devuelve_string() -> None:
    c = CanastaINPP(_df(), version=2019)
    assert isinstance(c._repr_html_(), str)
