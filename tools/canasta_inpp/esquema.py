from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Solo cubre las versiones que usan clasificación SCIAN (S/SB/R/SR/C). 2003
# queda deliberadamente fuera de este tipo -- usa P/GD/DIV/R/SG/G-03, un
# esquema previo a SCIAN, no un layout más de este mismo módulo. Ver
# memoria de proyecto: 2003 diferido a v1.1 (requiere PDF, no solo xlsx).
VersionCanastaScian = Literal[2012, 2019, 2025]

# Esquema final del CSV intermedio de ponderadores. Sin columna "version" --
# mismo criterio que replica-inpc-mx (la versión va como parámetro de carga,
# no como dato repetido en cada fila).
COLUMNAS_BASE: tuple[str, ...] = (
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


@dataclass(frozen=True)
class LayoutXlsx:
    """Nombres de hoja y posiciones de columna del xlsx de ponderadores, por versión.

    Las columnas están 0-indexadas sobre la tupla de fila que devuelve
    `Worksheet.iter_rows(values_only=True)` (el índice 0 siempre es la columna
    A del xlsx, vacía en las 3 versiones).

    S/SB/R/SR/C/G/ACTIVIDAD ECONÓMICA caen en la misma posición (1-7) en las
    3 versiones -- confirmado con los xlsx reales -- pero se dejan explícitas
    acá en vez de hardcodeadas, por si alguna versión futura las corre.

    La posición del peso SÍ varía por versión: en 2012 el peso arranca en la
    columna 10 (header partido en 2 filas, con 2 columnas de separación); en
    2019/2025 arranca en la 8, pegado a ACTIVIDAD ECONÓMICA. Confirmado
    revisando los 3 xlsx reales fila por fila, no asumido por similitud.
    """

    hoja_produccion_total: str
    hoja_bienes_intermedios: str
    hoja_bienes_finales: str
    hoja_demanda_interna: str
    hoja_exportaciones: str

    col_s: int
    col_sb: int
    col_r: int
    col_sr: int
    col_c: int
    col_g: int
    col_actividad: int

    col_peso_simple: int  # produccion_total / bienes_intermedios / bienes_finales / exportaciones
    col_peso_demanda_total: int
    col_peso_demanda_consumo: int
    col_peso_demanda_capital: int


LAYOUTS_XLSX: dict[VersionCanastaScian, LayoutXlsx] = {
    2012: LayoutXlsx(
        hoja_produccion_total="Producción Total",
        hoja_bienes_intermedios="Bienes Intermedios ",  # espacio final real en el xlsx
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

# Archivo de factor de encadenamiento -- solo existe para 2025, es un xlsx
# aparte del de ponderadores (no vive dentro de LayoutXlsx porque no es una
# hoja del mismo archivo). Posiciones confirmadas contra
# docs/requerimientos/xlsx/2025/factor_de_encadenamiento_ti.xlsx.
HOJA_ENCADENAMIENTO = "FACTOR DE ENCADENAMIENTO"
COL_ENCADENAMIENTO_TOTAL = 8
COL_ENCADENAMIENTO_PRODUCCION_NACIONAL = 10
COL_ENCADENAMIENTO_EXPORTACION = 12
COL_ENCADENAMIENTO_USO_FINAL = 14
