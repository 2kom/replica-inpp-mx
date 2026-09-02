"""Núcleo compartido de las funciones de consulta.

`consulta/variaciones.py` es un envoltorio thin sobre estas operaciones
genéricas, parametrizadas por nombre de columna (`variacion_pp`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.periodos import PeriodoMensual

Periodo = PeriodoMensual


def _verificar_periodo(df: pd.DataFrame, periodo: Periodo) -> None:
    if periodo not in set(df.index.get_level_values("periodo")):
        raise InvarianteViolado(f"El periodo {periodo} no existe en el resultado.")


def _verificar_indice(df: pd.DataFrame, indice: str) -> None:
    if indice not in set(df.index.get_level_values("indice")):
        raise InvarianteViolado(f"El índice '{indice}' no existe en el resultado.")


def _verificar_rango(periodo_desde: Periodo | None, periodo_hasta: Periodo | None) -> None:
    if periodo_desde is None or periodo_hasta is None:
        return
    if periodo_hasta < periodo_desde:
        raise InvarianteViolado(
            f"'desde' ({periodo_desde}) no puede ser posterior a 'hasta' ({periodo_hasta})."
        )


def valor_en(df: pd.DataFrame, columna: str, periodo: Periodo) -> pd.DataFrame:
    """Devuelve todas las categorías en `periodo`; índice = `indice`."""
    _verificar_periodo(df, periodo)
    return df.xs(periodo, level="periodo")[[columna]].copy()  # type: ignore[return-value]


def serie_en_rango(
    df: pd.DataFrame,
    columna: str,
    desde: Periodo | None,
    hasta: Periodo | None,
    indice: str,
) -> pd.Series:
    """Serie de `columna` para `indice` dentro de `[desde, hasta]`.

    `desde`/`hasta` `None` se resuelven al primer/último periodo de ese
    `indice`. Rango sin filas → `InvarianteViolado`.
    """
    _verificar_indice(df, indice)
    if desde is not None:
        _verificar_periodo(df, desde)
    if hasta is not None:
        _verificar_periodo(df, hasta)
    _verificar_rango(desde, hasta)

    serie = df.xs(indice, level="indice")[columna]
    inicio = desde if desde is not None else serie.index.min()
    fin = hasta if hasta is not None else serie.index.max()
    en_rango = serie[(serie.index >= inicio) & (serie.index <= fin)]
    if en_rango.empty:
        raise InvarianteViolado(f"Sin filas de '{indice}' en el rango [{inicio}, {fin}].")
    return en_rango


def acumulada(
    df: pd.DataFrame,
    columna: str,
    desde: Periodo,
    hasta: Periodo | None,
    indice: str,
) -> float:
    """Suma de `columna` en `[desde, hasta]` para `indice`."""
    return float(serie_en_rango(df, columna, desde, hasta, indice).sum())


def promedio_simple(
    df: pd.DataFrame,
    columna: str,
    desde: Periodo | None,
    hasta: Periodo | None,
    indice: str,
) -> float:
    """Media aritmética de `columna` en `[desde, hasta]` para `indice`."""
    return float(serie_en_rango(df, columna, desde, hasta, indice).mean())


def extremo(
    df: pd.DataFrame,
    columna: str,
    desde: Periodo | None,
    hasta: Periodo | None,
    indice: str | None,
    mayor: bool,
) -> tuple[Periodo, str, float]:
    """`(periodo, indice, valor)` del máximo (`mayor=True`) o mínimo del rango.

    `indice=None` busca entre todos los índices. Desempate: primer
    `(periodo, indice)` en orden del índice.
    """
    if indice is not None:
        _verificar_indice(df, indice)
    if desde is not None:
        _verificar_periodo(df, desde)
    if hasta is not None:
        _verificar_periodo(df, hasta)
    _verificar_rango(desde, hasta)

    sub = df
    if indice is not None:
        sub = sub[sub.index.get_level_values("indice") == indice]

    periodos = sub.index.get_level_values("periodo")
    keep = np.ones(len(sub), dtype=bool)
    if desde is not None:
        keep &= periodos >= desde
    if hasta is not None:
        keep &= periodos <= hasta
    sub = sub[keep]
    if sub.empty:
        raise InvarianteViolado(
            f"Sin filas en el rango [{desde}, {hasta}]"
            + (f" para el índice '{indice}'" if indice is not None else "")
            + "."
        )

    col = sub[columna]
    etiqueta = col.idxmax() if mayor else col.idxmin()
    periodo, ind = etiqueta  # type: ignore[misc, str-unpack]
    return periodo, str(ind), float(col.loc[etiqueta])  # type: ignore[return-value]
