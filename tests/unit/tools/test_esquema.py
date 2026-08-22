from __future__ import annotations

import dataclasses

import pytest
from canasta_inpp.esquema import (
    COL_ENCADENAMIENTO_EXPORTACION,
    COL_ENCADENAMIENTO_PRODUCCION_NACIONAL,
    COL_ENCADENAMIENTO_TOTAL,
    COL_ENCADENAMIENTO_USO_FINAL,
    COLUMNAS_BASE,
    HOJA_ENCADENAMIENTO,
    LAYOUTS_XLSX,
    LayoutXlsx,
    VersionCanastaScian,
)

_VERSIONES: tuple[VersionCanastaScian, ...] = (2012, 2019, 2025)

# snapshot congelado de LAYOUTS_XLSX, verificado fila por fila contra los xlsx
# reales en esta sesión -- protege contra edición accidental (ej. swap de
# col_peso_demanda_consumo/col_peso_demanda_capital) que un chequeo de solo
# tipo/cantidad no detectaría.
_LAYOUTS_ESPERADOS: dict[VersionCanastaScian, LayoutXlsx] = {
    2012: LayoutXlsx(
        hoja_produccion_total="Producción Total",
        hoja_bienes_intermedios="Bienes Intermedios ",
        hoja_bienes_finales="Bienes Finales",
        hoja_demanda_interna="Demanda Interna",
        hoja_exportaciones="Exportaciones",
        col_s=1,
        col_sb=2,
        col_r=3,
        col_sr=4,
        col_c=5,
        col_g=6,
        col_actividad=7,
        col_peso_simple=10,
        col_peso_demanda_total=10,
        col_peso_demanda_consumo=11,
        col_peso_demanda_capital=12,
    ),
    2019: LayoutXlsx(
        hoja_produccion_total="ProduccionTotal",
        hoja_bienes_intermedios="BienesIntermedios",
        hoja_bienes_finales="BienesFinales",
        hoja_demanda_interna="DemandaInterna",
        hoja_exportaciones="Exportaciones",
        col_s=1,
        col_sb=2,
        col_r=3,
        col_sr=4,
        col_c=5,
        col_g=6,
        col_actividad=7,
        col_peso_simple=8,
        col_peso_demanda_total=8,
        col_peso_demanda_consumo=9,
        col_peso_demanda_capital=10,
    ),
    2025: LayoutXlsx(
        hoja_produccion_total="ProducciónTotal",
        hoja_bienes_intermedios="BienesIntermedios",
        hoja_bienes_finales="BienesFinales",
        hoja_demanda_interna="DemandaInterna",
        hoja_exportaciones="Exportaciones",
        col_s=1,
        col_sb=2,
        col_r=3,
        col_sr=4,
        col_c=5,
        col_g=6,
        col_actividad=7,
        col_peso_simple=8,
        col_peso_demanda_total=8,
        col_peso_demanda_consumo=9,
        col_peso_demanda_capital=10,
    ),
}


# -- COLUMNAS_BASE -------------------------------------------------------


def test_columnas_base_orden_y_cantidad() -> None:
    assert COLUMNAS_BASE == (
        "generico",
        "codigo",
        "sector",
        "subsector",
        "rama",
        "subrama",
        "clase",
        "produccion_total",
        "bienes_intermedios",
        "bienes_finales",
        "demanda_interna_total",
        "demanda_interna_consumo",
        "demanda_interna_capital",
        "exportaciones",
        "encadenamiento_total",
        "encadenamiento_produccion_nacional",
        "encadenamiento_exportacion",
        "encadenamiento_uso_final",
    )


def test_columnas_base_sin_columna_version() -> None:
    # la versión va como parámetro de carga, no como columna repetida --
    # mismo criterio que replica-inpc-mx
    assert "version" not in COLUMNAS_BASE


# -- LayoutXlsx ------------------------------------------------------------


def test_layouts_xlsx_cubre_exactamente_las_3_versiones() -> None:
    assert set(LAYOUTS_XLSX) == set(_VERSIONES)


def test_layout_xlsx_bloquea_reasignar_un_campo() -> None:
    # setattr(...) en vez de `LAYOUTS_XLSX[2019].col_g = 99`: la asignación
    # directa a un campo de un dataclass frozen es un error ESTÁTICO real
    # (mypy: "Property col_g ... is read-only [misc]"), no solo algo que
    # falla en runtime -- setattr conserva el mismo chequeo en runtime sin
    # introducir ese error de tipo.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(LAYOUTS_XLSX[2019], "col_g", 99)


@pytest.mark.parametrize("version", _VERSIONES)
def test_layout_xlsx_coincide_con_snapshot_esperado(version: VersionCanastaScian) -> None:
    assert LAYOUTS_XLSX[version] == _LAYOUTS_ESPERADOS[version]


@pytest.mark.parametrize("version", _VERSIONES)
def test_columnas_s_a_actividad_son_1_a_7_consecutivas(version: VersionCanastaScian) -> None:
    # S/SB/R/SR/C/G/ACTIVIDAD ECONÓMICA cae en la misma posición en las 3
    # versiones -- confirmado con los xlsx reales, no asumido por similitud
    layout = LAYOUTS_XLSX[version]
    assert [
        layout.col_s,
        layout.col_sb,
        layout.col_r,
        layout.col_sr,
        layout.col_c,
        layout.col_g,
        layout.col_actividad,
    ] == list(range(1, 8))


@pytest.mark.parametrize("version", _VERSIONES)
def test_columnas_demanda_interna_son_consecutivas_al_peso_simple(
    version: VersionCanastaScian,
) -> None:
    # DEMANDA INTERNA trae TOTAL/CONSUMO/FORMACIÓN DE CAPITAL en 3 columnas
    # seguidas, arrancando en col_peso_simple -- confirmado en 2012/2019/2025
    layout = LAYOUTS_XLSX[version]
    assert layout.col_peso_demanda_total == layout.col_peso_simple
    assert layout.col_peso_demanda_consumo == layout.col_peso_simple + 1
    assert layout.col_peso_demanda_capital == layout.col_peso_simple + 2


def test_col_peso_simple_distinto_en_2012_vs_2019_2025() -> None:
    # el hallazgo concreto que motivó tener LayoutXlsx por versión en vez de
    # una posición fija: 2012 parte el header en 2 filas con 2 columnas de
    # separación, 2019/2025 no
    assert LAYOUTS_XLSX[2012].col_peso_simple == 10
    assert LAYOUTS_XLSX[2019].col_peso_simple == 8
    assert LAYOUTS_XLSX[2025].col_peso_simple == 8


# -- archivo de encadenamiento (solo 2025) --------------------------------


def test_hoja_y_columnas_de_encadenamiento() -> None:
    assert HOJA_ENCADENAMIENTO == "FACTOR DE ENCADENAMIENTO"
    assert COL_ENCADENAMIENTO_TOTAL == 8
    assert COL_ENCADENAMIENTO_PRODUCCION_NACIONAL == 10
    assert COL_ENCADENAMIENTO_EXPORTACION == 12
    assert COL_ENCADENAMIENTO_USO_FINAL == 14


def test_columnas_de_encadenamiento_son_pares_consecutivos() -> None:
    # cada columna de factor va seguida de una columna vacía en el xlsx
    # real (confirmado) -- de ahí el salto de 2 en 2 en vez de 1 en 1
    columnas = [
        COL_ENCADENAMIENTO_TOTAL,
        COL_ENCADENAMIENTO_PRODUCCION_NACIONAL,
        COL_ENCADENAMIENTO_EXPORTACION,
        COL_ENCADENAMIENTO_USO_FINAL,
    ]
    assert columnas == list(range(8, 15, 2))


# -- 2003 queda fuera de este módulo, a propósito ---------------------------


def test_2003_no_esta_en_layouts_xlsx() -> None:
    # 2003 usa P/GD/DIV/R/SG/G-03 (previo a SCIAN), no un layout más de este
    # módulo -- VersionCanastaScian ya lo excluye por tipo, esto confirma que
    # tampoco se cuela en tiempo de ejecución (ej. por un typo al ampliar el
    # dict a mano). Diferido a v1.1 -- requiere PDF además de xlsx.
    assert 2003 not in LAYOUTS_XLSX
