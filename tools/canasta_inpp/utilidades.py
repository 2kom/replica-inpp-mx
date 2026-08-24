# aqui van las funciones pequeñas que no ameritan un archivo aparte, pero que se usan en varios lugares

import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from canasta_inpp.esquema import (
    COLUMNAS_BASE,
    COLUMNAS_ENCADENAMIENTO_NA_PERMITIDO,
    VersionCanastaScian,
)

_TRANS_TILDES = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")
_PATRON_ESPACIOS = re.compile(r"\s+")


def normalizar_texto(texto: str) -> str:
    """Minúsculas, sin tildes (conserva la ñ), sin puntuación, sin espacios laterales ni dobles."""
    texto = texto.translate(_TRANS_TILDES).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    return _PATRON_ESPACIOS.sub(" ", texto).strip()


def normalizar_columnas_texto(df: pd.DataFrame, columnas: Sequence[str]) -> pd.DataFrame:
    """Aplica `normalizar_texto` a `columnas`. Usar solo en texto libre sin código (ej. `generico`)."""
    df = df.copy()
    for columna in columnas:
        df[columna] = df[columna].apply(normalizar_texto)
    return df


def normalizar_texto_con_codigo(texto: str) -> str:
    """Normaliza preservando intacto el código al inicio del texto (ej. `"31-33 Industrias..."`)."""
    codigo, _, nombre = texto.partition(" ")
    if not nombre:
        return codigo
    return f"{codigo} {normalizar_texto(nombre)}"


def normalizar_columnas_con_codigo(df: pd.DataFrame, columnas: Sequence[str]) -> pd.DataFrame:
    """Aplica `normalizar_texto_con_codigo` a `columnas` (sector/subsector/rama/subrama/clase)."""
    df = df.copy()
    for columna in columnas:
        df[columna] = df[columna].apply(normalizar_texto_con_codigo)
    return df


def resolver_sector_agrupado(df: pd.DataFrame) -> pd.DataFrame:
    """Resuelve `sector` agrupado (rango SCIAN, ej. `"31-33"`) al código concreto vía `subsector`.

    Lanza `ValueError` si la jerarquía es inconsistente (subsector sin código de 3
    dígitos, o fuera del rango declarado por sector).
    """
    df = df.copy()

    def _resolver(fila: pd.Series) -> str:
        codigo, separador, nombre = str(fila["sector"]).partition(" ")
        if "-" not in codigo:
            return str(fila["sector"])

        identificador = (
            f"código {fila['codigo']}" if "codigo" in fila.index else f"fila {fila.name}"
        )

        codigo_subsector = str(fila["subsector"]).partition(" ")[0]
        if not (len(codigo_subsector) == 3 and codigo_subsector.isdigit()):
            raise ValueError(
                f"No se puede resolver sector agrupado '{fila['sector']}' ({identificador}): "
                f"subsector '{fila['subsector']}' no trae un código de 3 dígitos."
            )

        lo_str, _, hi_str = codigo.partition("-")
        if not (len(lo_str) == 2 and lo_str.isdigit() and len(hi_str) == 2 and hi_str.isdigit()):
            raise ValueError(
                f"Rango de sector '{codigo}' ({identificador}) con formato inesperado -- se "
                f"esperaban 2 códigos de 2 dígitos separados por guion."
            )

        lo, hi = int(lo_str), int(hi_str)
        codigo_resuelto = codigo_subsector[:2]
        if not (lo <= int(codigo_resuelto) <= hi):
            raise ValueError(
                f"Subsector '{fila['subsector']}' (código {codigo_resuelto}, {identificador}) no "
                f"pertenece al rango de sector '{fila['sector']}' ({lo}-{hi})."
            )

        return f"{codigo_resuelto}{separador}{nombre}"

    df["sector"] = df.apply(_resolver, axis=1)
    return df


def guardar_csv(df: pd.DataFrame, ruta: Path, version: VersionCanastaScian) -> None:
    """Completa el esquema fijo de columnas (`COLUMNAS_BASE`) y escribe el CSV final.

    Distingue 3 semánticas de "sin valor": valor real (incluido cero, se preserva tal
    cual), columna entera ausente en `df` (se rellena con `""`), y celda `NaN` dentro
    de una columna presente (`"-"` si la columna está en
    `COLUMNAS_ENCADENAMIENTO_NA_PERMITIDO`, `ValueError` en cualquier otra columna --
    ver Raises).

    Args:
        df: Datos a guardar. Puede traer un subconjunto de `COLUMNAS_BASE`; las
            columnas faltantes se agregan vacías. Columnas fuera de `COLUMNAS_BASE`
            se descartan con una advertencia impresa (no lanzan).
        ruta: Ruta del CSV de salida.
        version: Versión de canasta (2012/2019/2025). No se usa en el cuerpo de la
            función -- el nombre de archivo con la versión lo arma quien llama.

    Returns:
        None.

    Raises:
        ValueError: Si alguna columna fuera de `COLUMNAS_ENCADENAMIENTO_NA_PERMITIDO`
            trae una celda `NaN` -- se interpreta como dato requerido faltante, no
            como N/A legítimo.
    """
    sobrantes = set(df.columns) - set(COLUMNAS_BASE)
    if sobrantes:
        print(
            f"[canasta_inpp] Advertencia: columnas fuera de esquema descartadas: {sorted(sobrantes)}"
        )

    # capturado ANTES del reindex -- después, "codigo" siempre existe
    # (`fill_value=""`), así que el chequeo de presencia perdería sentido
    # si se hiciera sobre el df ya reindexado.
    codigo_original = df["codigo"] if "codigo" in df.columns else None

    df = df.reindex(columns=COLUMNAS_BASE, fill_value="")

    columnas_sin_na_permitido = [
        c for c in COLUMNAS_BASE if c not in COLUMNAS_ENCADENAMIENTO_NA_PERMITIDO
    ]
    for columna in columnas_sin_na_permitido:
        mask = df[columna].isna()
        if mask.any():
            # posición dentro del df (0-indexed), NO el índice de `df` --
            # decisión de diseño, no un descuido: `guardar_csv` reporta por
            # posición SIEMPRE, por contrato, sin importar qué índice traiga
            # `df`. Los 3 extractores (extraer_ponderadores/extraer_canasta/
            # extraer_encadenamiento) siempre devuelven índice fresco, y
            # `df.merge(...)` (el llamador real en `generar_canasta.py::main()`)
            # también resetea a un RangeIndex fresco -- pero reportar por
            # posición evita de raíz toda la clase de bugs de índice (etiquetas
            # duplicadas rompiendo `.loc`, tipos numpy en el mensaje) sin
            # depender de esa invariante.
            posiciones = [pos for pos, es_nan in enumerate(mask) if es_nan]
            identificadores: list[object] = [
                codigo_original.iloc[pos]
                if codigo_original is not None
                and pd.notna(codigo_original.iloc[pos])
                and codigo_original.iloc[pos] != ""
                else pos
                for pos in posiciones
            ]
            raise ValueError(
                f"Columna '{columna}' trae {len(posiciones)} celda(s) sin valor -- solo "
                f"encadenamiento exportacion/uso final permiten N/A. Códigos/posiciones "
                f"afectadas: {identificadores}."
            )

    for columna in COLUMNAS_ENCADENAMIENTO_NA_PERMITIDO:
        df[columna] = df[columna].fillna("-")

    df.to_csv(ruta, index=False)
