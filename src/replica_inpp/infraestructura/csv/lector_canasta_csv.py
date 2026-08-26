"""Lector de la canasta de ponderadores del INPP (CSV generado por
`tools/generar_canasta.py`, no un xlsx crudo de INEGI)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from replica_inpp.dominio.errores import (
    ArchivoCorrupto,
    ArchivoNoEncontrado,
    ArchivoVacio,
    ColumnasMinFaltantes,
    EncodingNoLegible,
)

# Columnas esperadas tras fijar `codigo` como índice. Lista propia de esta capa,
# independiente de `tools/canasta_inpp/esquema.py::COLUMNAS_BASE` (ese esquema
# es del CLI generador, capa `tools/`, fuera del dominio) aunque hoy coincidan.
COLUMNAS_REQUERIDAS = [
    "generico",
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
]


class LectorCanastaCsv:
    def leer(self, ruta: Path) -> pd.DataFrame:
        try:
            canasta = pd.read_csv(ruta, index_col="codigo", dtype={"codigo": str})
        except FileNotFoundError:
            raise ArchivoNoEncontrado(f"No se encontró el archivo: {ruta}")
        except pd.errors.EmptyDataError:
            raise ArchivoVacio(f"El archivo está vacío: {ruta}")
        except pd.errors.ParserError:
            raise ArchivoCorrupto(f"El archivo está corrupto o no es un CSV válido: {ruta}")
        except UnicodeDecodeError:
            raise EncodingNoLegible(
                f"No se pudo leer el archivo debido a un problema de encoding: {ruta}"
            )

        columnas_faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in canasta.columns]
        if columnas_faltantes:
            raise ColumnasMinFaltantes(
                f"Faltan columnas requeridas: {', '.join(columnas_faltantes)}"
            )

        canasta.attrs["origen"] = ruta
        return canasta
