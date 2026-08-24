from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VersionCanastaScian = Literal[2012, 2019, 2025]

COLUMNAS_BASE: tuple[str, ...] = (
    "generico",
    "codigo",
    "sector",
    "subsector",
    "rama",
    "subrama",
    "clase",
    "produccion total",
    "bienes intermedios",
    "bienes finales",
    "demanda interna total",
    "demanda interna consumo",
    "demanda interna capital",
    "exportaciones",
    "encadenamiento total",
    "encadenamiento produccion nacional",
    "encadenamiento exportacion",
    "encadenamiento uso final",
)

# agrupación semántica de las 7 columnas de peso (participación en VBP) --
# complemento de COLUMNAS_ENCADENAMIENTO, usado en registro.py.
COLUMNAS_PESO: tuple[str, ...] = (
    "produccion total",
    "bienes intermedios",
    "bienes finales",
    "demanda interna total",
    "demanda interna consumo",
    "demanda interna capital",
    "exportaciones",
)

# agrupación semántica de las 4 columnas de encadenamiento -- no todas
# admiten N/A, ver COLUMNAS_ENCADENAMIENTO_NA_PERMITIDO.
COLUMNAS_ENCADENAMIENTO: tuple[str, ...] = (
    "encadenamiento total",
    "encadenamiento produccion nacional",
    "encadenamiento exportacion",
    "encadenamiento uso final",
)

# único subconjunto donde NaN es N/A legítimo de INEGI -- las otras 2 nunca
# traen N/A (ver extraer_encadenamiento).
COLUMNAS_ENCADENAMIENTO_NA_PERMITIDO: tuple[str, ...] = (
    "encadenamiento exportacion",
    "encadenamiento uso final",
)


@dataclass(frozen=True)
class LayoutXlsx:
    """Nombres de hoja y posiciones de columna del xlsx de ponderadores, por versión.

    Columnas 0-indexadas. La posición del peso varía por versión: 2012 usa la
    columna 10, 2019/2025 usan la 8.
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

    col_peso_simple: int
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

# solo existe para 2025, archivo aparte del de ponderadores.
HOJA_ENCADENAMIENTO = "FACTOR DE ENCADENAMIENTO"
COL_ENCADENAMIENTO_GENERICO = 6
COL_ENCADENAMIENTO_TOTAL = 8
COL_ENCADENAMIENTO_PRODUCCION_NACIONAL = 10
COL_ENCADENAMIENTO_EXPORTACION = 12
COL_ENCADENAMIENTO_USO_FINAL = 14


@dataclass(frozen=True)
class LayoutCanasta:
    """Nombre de hoja y posiciones de columna del xlsx de árbol SCIAN (canasta), por versión.

    Cada fila trae un solo nivel jerárquico vigente a la vez (state machine en
    `extraer_canasta`). `col_<nivel>_nombre` es `None` cuando código+nombre ya
    vienen combinados en una sola celda (2012/2019); en 2025 vienen separados y
    hay que unirlos. `codigo_generico_en_columna_clase` es True solo en 2012
    (ese xlsx reusa `col_clase` para el código de genérico, distinguible por tipo).
    """

    hoja: str
    fila_datos_inicio: int

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
