"""Lector de la canasta de ponderadores del INPP (CSV generado por
`tools/generar_canasta.py`, no un xlsx crudo de INEGI)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from replica_inpp.dominio.errores import (
    ArchivoCorrupto,
    ArchivoNoEncontrado,
    ArchivoVacio,
    ColumnasMinFaltantes,
    EncodingNoLegible,
)
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.tipos import VersionCanasta

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

# Columnas jerárquicas SCIAN que pueden venir de dos formas según la corrida de
# `tools/generar_canasta.py`: combinadas ("11 agricultura...", cuando se pasó
# --canasta, mezclando código+nombre) o bare (solo "11", cuando se cargó nomás
# --ponderadores, sin nombres disponibles).
COLUMNAS_JERARQUIA = ("sector", "subsector", "rama", "subrama", "clase")

# Código al inicio + espacio + nombre. Igual que `_PATRON_HOJA` de
# lector_series_csv.py pero sin fijar cantidad de dígitos (el nivel jerárquico
# determina el largo del código: 2 para sector, hasta 6 para clase).
_PATRON_CODIGO_NOMBRE = re.compile(r"^(\d+)\s+(.+)$")
_PATRON_BARE = re.compile(r"^\d+$")


class LectorCanastaCsv:
    def leer(self, ruta: Path, version: VersionCanasta) -> CanastaINPP:
        try:
            canasta = pd.read_csv(
                ruta,
                index_col="codigo",
                dtype={"codigo": str, **{c: str for c in COLUMNAS_JERARQUIA}},
            )
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

        canasta = self._separar_codigo_jerarquia(canasta, ruta)

        canasta.attrs["origen"] = ruta
        return CanastaINPP(canasta, version)

    def _separar_codigo_jerarquia(self, canasta: pd.DataFrame, ruta: Path) -> pd.DataFrame:
        """Agrega `codigo <nivel>` a la izquierda de cada columna en
        `COLUMNAS_JERARQUIA`, siempre -- en las dos variantes de origen.

        Columna combinada ("11 agricultura..."): el código se extrae por regex.
        Columna bare (ya es solo "11"): se copia tal cual. La columna original
        nunca se modifica -- `codigo <nivel>` es puramente aditiva, para que el
        código de cada nivel esté disponible en el mismo nombre de columna sin
        importar qué variante de archivo se cargó.
        """
        for col in COLUMNAS_JERARQUIA:
            valores = canasta[col].dropna()

            combinados = valores.str.match(_PATRON_CODIGO_NOMBRE)
            bares = valores.str.match(_PATRON_BARE)

            if combinados.all():
                codigos = canasta[col].str.extract(_PATRON_CODIGO_NOMBRE)[0]
            elif bares.all():
                codigos = canasta[col]
            else:
                ejemplos = sorted(valores[~(combinados | bares)].unique())[:3]
                raise ArchivoCorrupto(
                    f"La columna '{col}' mezcla formato código+nombre y formato bare, "
                    f"o trae valores irreconocibles: {ejemplos} ({ruta})"
                )

            posicion = list(canasta.columns).index(col)
            canasta.insert(posicion, f"codigo {col}", codigos)

        return canasta
