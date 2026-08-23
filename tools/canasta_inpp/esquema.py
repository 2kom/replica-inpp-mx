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
# data/tests/xlsx/2025/factor_de_encadenamiento_ti.xlsx.
HOJA_ENCADENAMIENTO = "FACTOR DE ENCADENAMIENTO"
# código de genérico -- misma posición que col_g de LAYOUTS_XLSX[2025] (este
# xlsx repite el mismo layout S/SB/R/SR/C/G/ACTIVIDAD en columnas 1-7).
COL_ENCADENAMIENTO_GENERICO = 6
COL_ENCADENAMIENTO_TOTAL = 8
COL_ENCADENAMIENTO_PRODUCCION_NACIONAL = 10
COL_ENCADENAMIENTO_EXPORTACION = 12
COL_ENCADENAMIENTO_USO_FINAL = 14


@dataclass(frozen=True)
class LayoutCanasta:
    """Nombre de hoja y posiciones de columna del xlsx de árbol SCIAN (canasta), por versión.

    Layout estructuralmente distinto de `LayoutXlsx` (ponderadores): acá cada
    fila trae el nivel jerárquico vigente (Sector, Subsector, Rama, Subrama o
    Clase) UNA sola columna llena a la vez -- hay que arrastrar el valor
    vigente de cada nivel fila a fila (`extraer_canasta` hace ese state
    machine), no leerlo directo como en ponderadores. Confirmado fila por
    fila contra los 3 xlsx reales, no asumido por similitud entre versiones.

    `col_<nivel>_nombre`: en 2012/2019 el nombre completo del nivel ya viene
    pegado al código en la misma celda de texto (ej. `"11 Agricultura, cría
    y explotación..."`) -- en ese caso este campo es `None` y se usa
    `col_<nivel>` tal cual. En 2025 código y nombre vienen en columnas
    separadas (`col_<nivel>`=código entero, `col_<nivel>_nombre`=texto) y hay
    que unirlos (`"{codigo} {nombre}"`) para mantener el mismo formato de
    texto combinado en las 3 versiones.

    `codigo_generico_en_columna_clase`: True únicamente en 2012 -- ese xlsx
    NO tiene columna propia para el código de genérico, reusa `col_clase`
    (la fila de Clase trae ahí un string, la fila de Genérico un `int` puro;
    se distinguen por tipo, no por posición). Confirmado contra las 567
    filas de genérico de `data/tests/xlsx/2012/canasta.xlsx`: código y
    nombre YA están en celdas separadas (código como `int`, nombre como
    `str`) -- a diferencia de INPC, acá no hace falta parsear un string
    combinado tipo `"01 alimentos"` para separar código de nombre.
    """

    hoja: str
    fila_datos_inicio: int  # primera fila (1-indexed) con datos reales de la jerarquía

    col_sector: int
    col_sector_nombre: int | None
    col_subsector: int
    col_subsector_nombre: int | None
    col_rama: int
    col_rama_nombre: int | None
    col_subrama: int
    col_subrama_nombre: int | None
    col_clase: int
    col_clase_nombre: int | None

    col_codigo_generico: int
    col_nombre_generico: int
    codigo_generico_en_columna_clase: bool


LAYOUTS_CANASTA: dict[VersionCanastaScian, LayoutCanasta] = {
    2012: LayoutCanasta(
        hoja="CANASTA",
        fila_datos_inicio=10,
        col_sector=1,
        col_sector_nombre=None,
        col_subsector=2,
        col_subsector_nombre=None,
        col_rama=3,
        col_rama_nombre=None,
        col_subrama=4,
        col_subrama_nombre=None,
        col_clase=5,
        col_clase_nombre=None,
        col_codigo_generico=5,
        col_nombre_generico=6,
        codigo_generico_en_columna_clase=True,
    ),
    2019: LayoutCanasta(
        hoja="Canasta Julio 2019=100.0",
        fila_datos_inicio=12,
        col_sector=1,
        col_sector_nombre=None,
        col_subsector=2,
        col_subsector_nombre=None,
        col_rama=3,
        col_rama_nombre=None,
        col_subrama=4,
        col_subrama_nombre=None,
        col_clase=5,
        col_clase_nombre=None,
        col_codigo_generico=6,
        col_nombre_generico=7,
        codigo_generico_en_columna_clase=False,
    ),
    2025: LayoutCanasta(
        hoja="Canasta INPP",
        fila_datos_inicio=9,
        col_sector=1,
        col_sector_nombre=2,
        col_subsector=3,
        col_subsector_nombre=4,
        col_rama=5,
        col_rama_nombre=6,
        col_subrama=7,
        col_subrama_nombre=8,
        col_clase=9,
        col_clase_nombre=10,
        col_codigo_generico=11,
        col_nombre_generico=12,
        codigo_generico_en_columna_clase=False,
    ),
}
