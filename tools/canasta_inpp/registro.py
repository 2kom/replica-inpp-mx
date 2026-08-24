"""Genera el JSON de registro de una corrida de generar_canasta.py."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from canasta_inpp.esquema import COLUMNAS_ENCADENAMIENTO, COLUMNAS_PESO, VersionCanastaScian

_COLUMNAS_JERARQUIA = ("sector", "subsector", "rama", "subrama", "clase")


def escribir_registro(
    df: pd.DataFrame,
    *,
    version: VersionCanastaScian,
    xlsx_ponderadores: Path,
    xlsx_canasta: Path | None,
    xlsx_encadenamientos: Path | None,
    ruta_csv: Path,
    ruta_salida: Path,
) -> Path:
    """Genera el JSON de registro de una corrida de `generar_canasta.py::main()`.

    Funciona con o sin `--canasta`/`--encadenamientos` -- los campos correspondientes
    quedan en `None` si el flag no vino. Escribe junto a `ruta_csv` (mismo
    directorio que `-o`).

    Args:
        df: el mismo df que se le pasó a `guardar_csv` -- los valores faltantes
            siguen en NaN real (`guardar_csv` no muta `df` in place, su
            reindex/relleno vive en una copia local dentro de esa función).
        version: versión de canasta de la corrida.
        xlsx_ponderadores: ruta de `--ponderadores` (siempre presente).
        xlsx_canasta: ruta de `--canasta`, o `None` si no vino el flag.
        xlsx_encadenamientos: ruta de `--encadenamientos`, o `None` si no vino el flag.
        ruta_csv: ruta del CSV que ya escribió `guardar_csv`.
        ruta_salida: directorio de salida (`-o`) -- ahí se escribe el JSON.

    Returns:
        La ruta del JSON escrito.
    """
    ahora = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sufijo = uuid.uuid4().hex[:8]
    ruta_json = ruta_salida / f"canasta_{version}_{ahora}_{sufijo}.json"

    registro: dict = {
        "xlsx_ponderadores": str(xlsx_ponderadores),
        "xlsx_canasta": str(xlsx_canasta) if xlsx_canasta is not None else None,
        "xlsx_encadenamientos": (
            str(xlsx_encadenamientos) if xlsx_encadenamientos is not None else None
        ),
        "csv": str(ruta_csv),
        "version": version,
        "genericos": len(df),
        "pesos": _contar_no_vacios(df, COLUMNAS_PESO),
        "encadenamiento": (
            _contar_no_vacios(df, COLUMNAS_ENCADENAMIENTO)
            if xlsx_encadenamientos is not None
            else None
        ),
        "clasificaciones": {col: _clasificar(df, col) for col in _COLUMNAS_JERARQUIA},
    }

    ruta_json.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    _imprimir_resumen(registro, ruta_csv, ruta_json)
    return ruta_json


def _contar_no_vacios(df: pd.DataFrame, columnas: tuple[str, ...]) -> dict[str, int]:
    """Cuenta, por columna, cuántos genéricos tienen valor real (no NaN)."""
    return {columna: int(df[columna].notna().sum()) for columna in columnas}


def _clasificar(df: pd.DataFrame, columna: str) -> dict[str, dict[str, int]]:
    """Cuenta genéricos por categoría de una columna jerárquica (sector/.../clase).

    Ignora NaN y cadenas vacías/solo-espacios -- una jerarquía ausente (ej. el
    `""` con el que `extraer_canasta` inicializa su máquina de estados si un
    genérico aparece antes de su fila de sector/subsector/etc.) no debe
    aparecer como si fuera una categoría real.
    """
    valores = df[columna].dropna().astype(str).str.strip()
    valores = valores[valores != ""]
    categorias = sorted(valores.unique())
    return {categoria: {"genericos": int((valores == categoria).sum())} for categoria in categorias}


def _imprimir_resumen(registro: dict, ruta_csv: Path, ruta_json: Path) -> None:
    """Imprime a stdout el mismo resumen que queda en el JSON."""
    print(f"\n  version {registro['version']}: {registro['genericos']} genericos extraidos")
    for columna, conteo in registro["pesos"].items():
        print(f"  peso '{columna}': {conteo}")
    if registro["encadenamiento"] is not None:
        for columna, conteo in registro["encadenamiento"].items():
            print(f"  {columna}: {conteo}")
    for columna, categorias in registro["clasificaciones"].items():
        print(f"  {columna}: {len(categorias)} categorias")
    print(f"\n  csv:      {ruta_csv}")
    print(f"  registro: {ruta_json}\n")
