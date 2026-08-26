from __future__ import annotations

import numpy as np
import pandas as pd

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.tipos import VersionCanasta

# Código de genérico: exactamente 3 dígitos, sin excepción -- verificado contra
# los 6 CSV reales (2012/2019/2025 x 2 variantes de origen).
_PATRON_CODIGO = r"^\d{3}$"

# Participación en VBP (Valor Bruto de Producción) por recorte — no en gasto de
# consumo como en INPC. Una columna por recorte, a diferencia de INPC que trae
# un solo `ponderador` escalar: acá el mismo genérico pesa distinto según
# destino/etapa (ver `RecorteINPP`, sin equivalente en INPC).
_COLUMNAS_PESO = (
    "produccion total",
    "bienes intermedios",
    "bienes finales",
    "demanda interna total",
    "demanda interna consumo",
    "demanda interna capital",
    "exportaciones",
)

# Estas 2 nunca traen NaN parcial cuando la fuente sí se extrajo (ver
# tools/canasta_inpp/extraccion_xlsx.py::extraer_encadenamiento) — la columna
# entera es NaN si no se corrió `--encadenamientos` (solo existe para 2025),
# o está completamente poblada si sí.
_COLUMNAS_ENCADENAMIENTO_TODO_O_NADA = (
    "encadenamiento total",
    "encadenamiento produccion nacional",
)

# Estas 2 sí admiten NaN por genérico aun con fuente extraída — INEGI mismo
# marca "N/A" para algunos genéricos en export/uso final.
_COLUMNAS_ENCADENAMIENTO_PARCIAL = (
    "encadenamiento exportacion",
    "encadenamiento uso final",
)

_COLUMNAS_CORE = (
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
)


class CanastaINPP:
    """Representa la canasta de ponderadores usada para el cálculo del índice.

    Args:
        df: DataFrame con `codigo` como índice y columnas según el esquema de
            `infraestructura.csv.lector_canasta_csv` — incluye `codigo <nivel>`
            por cada nivel jerárquico SCIAN (agregada por el lector, siempre
            presente sin importar si el archivo fuente traía nombre o no).
        version: Versión de la canasta/ponderadores. Debe ser 2012, 2019 o 2025.

    Raises:
        InvarianteViolado: Si la versión no es válida, si el índice contiene
            duplicados, valores ausentes, o códigos que no sean exactamente 3
            dígitos; si alguna columna de `_COLUMNAS_PESO` contiene texto no
            numérico, algún valor negativo (cero sí es válido — participación
            nula de un genérico en ese recorte, ej. "edificación residencial"
            no participa nada en `bienes intermedios`), o su suma de valores
            no nulos no es 100; si `encadenamiento total`/`encadenamiento
            produccion nacional` están parcialmente pobladas (deben ser 100%
            NaN o 100% no nulas) o contienen texto no numérico o valores no
            positivos/no finitos donde no son nulas; si `encadenamiento
            exportacion`/`encadenamiento uso final` contienen texto no
            numérico o valores no positivos/no finitos donde no son nulas; o
            si alguna columna de `_COLUMNAS_CORE` tiene valores vacíos.

    Esquema del DataFrame (índice: `codigo`):
        generico (str): nombre del genérico, decorativo.
        codigo sector/subsector/rama/subrama/clase (str): código SCIAN de cada
            nivel, siempre presente — extraído por regex si la columna del
            nivel viene combinada con nombre, copiado tal cual si ya venía bare.
        sector/subsector/rama/subrama/clase (str): tal como vino del archivo
            fuente — combinado ("11 agricultura...") o bare ("11") según si el
            CSV se generó con `--canasta` o solo con `--ponderadores`.
        produccion total/bienes intermedios/bienes finales/demanda interna
            total/demanda interna consumo/demanda interna capital/exportaciones
            (float/NaN): participación en VBP del recorte, una columna por
            recorte — a diferencia de INPC, no hay un solo `ponderador` escalar.
        encadenamiento total/produccion nacional/exportacion/uso final
            (float/NaN): factor de encadenamiento (solo 2025) — `NaN` en las 4
            si el CSV se generó sin `--encadenamientos`.

    Ver: CLAUDE.md, sección "Dominio: INPP vs INPC".
    """

    def __init__(self, df: pd.DataFrame, version: VersionCanasta) -> None:
        if version not in (2012, 2019, 2025):
            raise InvarianteViolado("La versión de la canasta debe ser 2012, 2019 o 2025.")
        if df.index.duplicated().any():
            raise InvarianteViolado(
                "El índice del DataFrame de la canasta no puede tener valores duplicados."
            )
        if pd.isna(df.index).any():
            raise InvarianteViolado("El índice del DataFrame no puede tener valores ausentes.")
        if not df.index.astype(str).str.match(_PATRON_CODIGO).all():
            raise InvarianteViolado(
                "El índice del DataFrame debe ser un código de genérico de exactamente 3 dígitos."
            )

        for col in _COLUMNAS_PESO:
            try:
                valores = df[col].astype(float)
            except ValueError as e:
                raise InvarianteViolado(
                    f"La columna '{col}' contiene un valor no numérico: {e}"
                ) from e
            no_nulos = valores.dropna()
            if (no_nulos < 0).any():
                raise InvarianteViolado(f"La columna '{col}' no puede contener valores negativos.")
            if abs(no_nulos.sum() - 100) > 1e-5:  # Permitir una pequeña tolerancia numérica
                raise InvarianteViolado(
                    f"La suma de los valores no nulos de la columna '{col}' debe ser igual a 100."
                )

        for col in _COLUMNAS_ENCADENAMIENTO_TODO_O_NADA:
            nulos = df[col].isna()
            if nulos.any() and not nulos.all():
                raise InvarianteViolado(
                    f"La columna '{col}' debe estar completamente vacía o completamente llena, "
                    "no puede tener NaN parcial."
                )
            if not nulos.all():
                self._validar_encadenamiento_positivo(df, col)

        for col in _COLUMNAS_ENCADENAMIENTO_PARCIAL:
            if df[col].notna().any():
                self._validar_encadenamiento_positivo(df, col)

        columnas_vacias = [
            col
            for col in _COLUMNAS_CORE
            if col in df.columns
            and (df[col].isna() | (df[col].astype(str).str.strip() == "")).any()
        ]
        if columnas_vacias:
            raise InvarianteViolado(
                f"Las columnas {columnas_vacias} no pueden tener valores vacíos."
            )

        self._df = df
        self._version: VersionCanasta = version

    def _validar_encadenamiento_positivo(self, df: pd.DataFrame, col: str) -> None:
        try:
            valores = df[col].astype(float)
        except ValueError as e:
            raise InvarianteViolado(f"La columna '{col}' contiene un valor no numérico: {e}") from e
        no_nulos = valores.dropna()
        if not ((no_nulos > 0) & np.isfinite(no_nulos)).all():
            raise InvarianteViolado(
                f"La columna '{col}' debe contener solo valores positivos y finitos donde no es nula."
            )

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def version(self) -> VersionCanasta:
        """Devuelve la versión de la canasta/ponderadores."""
        return self._version

    def _repr_html_(self) -> str:
        """Renderiza la canasta como tabla HTML en entornos interactivos."""
        return self._df._repr_html_()  # type: ignore[operator]
