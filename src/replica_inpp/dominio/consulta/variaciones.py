"""Consulta de variaciones sobre un `ResultadoVariacion`.

Funciones thin sin estado ni IO; operan sobre la columna `variacion_pp`.
Devuelven escalares, pares o `DataFrame` — nunca un `ResultadoX`.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from replica_inpp.dominio.consulta import _comun
from replica_inpp.dominio.consulta._comun import Periodo
from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion

_COL = "variacion_pp"


def _verificar_periodos_consecutivos(
    serie: pd.Series, indice: str, contexto: str, desde: Periodo | None, hasta: Periodo | None
) -> None:
    """Exige que el índice de `serie` (ya ordenado, ver llamadores) sea una
    secuencia mensual sin huecos que además cubra los extremos pedidos.

    Dos causas distintas del mismo síntoma (negociadas juntas en
    `/negociar-hallazgos`, ronda 4, 2026-09-01):

    1. **Hueco interno** (ronda 3, hallazgo A): `serie_en_rango` filtra por
       `[desde, hasta]` sobre las filas que EXISTEN en `resultado.df` -- pero
       `variacion_periodica` ya excluyó del `.df` los periodos no computables
       (`sin_datos`, ver `.reporte`), así que un hueco en medio del rango pasa
       desapercibido: dos periodos que en realidad no son consecutivos (ej.
       feb y may, con mar/abr faltantes) parecerían serlo si solo se mira su
       posición en la serie filtrada.
    2. **Extremo ausente** (ronda 4, hallazgo A): `_comun._verificar_periodo`
       valida que `desde`/`hasta` existan en ALGÚN índice de `df`, no
       específicamente en `indice` -- si otro índice sí tiene ese periodo pero
       `indice` no (fue `sin_datos` y quedó fuera del `.df`), la serie
       filtrada arranca/termina más adentro del rango pedido sin que nada lo
       marque, y el hueco 1 no lo detecta si el sobrante es internamente
       consecutivo.

    Componer o elevar a potencia una secuencia con cualquiera de los dos
    problemas trata transiciones desconectadas -- o un rango más angosto que
    el pedido -- como si fueran válidas de punta a punta.
    """
    periodos = list(serie.index)
    ordinales = [p.año * 12 + p.mes for p in periodos]  # type: ignore[attr-defined]
    esperados = list(range(ordinales[0], ordinales[0] + len(ordinales)))
    if ordinales != esperados:
        raise InvarianteViolado(
            f"{contexto}: periodos no consecutivos para '{indice}' -- "
            f"{periodos}. Requiere una secuencia mensual continua, sin huecos "
            "(revisar .reporte para los periodos sin_datos)."
        )
    if desde is not None and periodos[0] != desde:
        raise InvarianteViolado(
            f"{contexto}: falta el extremo 'desde' ({desde}) para '{indice}' -- "
            f"el primer periodo disponible en el rango es {periodos[0]}."
        )
    if hasta is not None and periodos[-1] != hasta:
        raise InvarianteViolado(
            f"{contexto}: falta el extremo 'hasta' ({hasta}) para '{indice}' -- "
            f"el último periodo disponible en el rango es {periodos[-1]}."
        )


def _factores_validados(serie: pd.Series, indice: str, contexto: str) -> np.ndarray:
    """`1 + v/100` por fila, validado finito y `>= 0` (rechaza `v < -100`).

    Un factor negativo (índice implícito negativo) no tiene sentido económico
    y produce un número complejo al elevarlo a una potencia fraccionaria
    (TCAC) -- `-100%` exacto queda permitido (factor `0`, el índice llegó a
    cero, matemáticamente válido).
    """
    factores = (1.0 + serie / 100.0).to_numpy(dtype=float)
    if not np.all(np.isfinite(factores)):
        raise InvarianteViolado(f"{contexto}: variacion_pp no finita en el rango para '{indice}'.")
    if np.any(factores < 0):
        raise InvarianteViolado(
            f"{contexto}: variacion_pp < -100% (factor negativo) para '{indice}' -- "
            "un índice no puede implicar una variación menor a -100%."
        )
    return factores


def inflacion_en(resultado: ResultadoVariacion, periodo: Periodo) -> pd.DataFrame:
    """Variación de todas las categorías en `periodo`; índice = `indice`."""
    return _comun.valor_en(resultado.df, _COL, periodo)


def inflacion_acumulada(
    resultado: ResultadoVariacion,
    desde: Periodo,
    hasta: Periodo | None = None,
    *,
    indice: str,
) -> float:
    """Variación total del rango para `indice`, componiendo las tasas periódicas.

    `100 * (Π(1 + v/100) - 1)` — composición geométrica exacta, no una suma de
    puntos porcentuales (que solo aproxima linealmente y diverge en rangos
    largos o tasas grandes). Requiere `resultado.manifiesto.clase ==
    "periodica_mensual"`: componer tasas de ventana móvil traslapada
    (bimestral, anual, ...) contaría meses de más de una vez.

    Raises:
        InvarianteViolado: `resultado.manifiesto.clase != "periodica_mensual"`;
            si los periodos `[desde, hasta]` no son mensuales consecutivos
            (un hueco -- ej. un periodo `sin_datos` en medio -- compondría
            transiciones desconectadas como si fueran una sola); si `indice`
            no tiene dato exactamente en `desde` o en `hasta` (aunque ese
            periodo exista para otro índice del mismo `resultado`); si algún
            `variacion_pp` implica `< -100%` (factor negativo); o si algún
            factor intermedio o el resultado final no son finitos (overflow).
    """
    if resultado.manifiesto.clase != "periodica_mensual":
        raise InvarianteViolado(
            "inflacion_acumulada requiere resultado.manifiesto.clase == "
            f"'periodica_mensual' (componer tasas de ventana móvil traslapada "
            f"cuenta periodos de más); recibió '{resultado.manifiesto.clase}'."
        )
    return _acumulada_compuesta(resultado.df, desde, hasta, indice)


def _acumulada_compuesta(
    df: pd.DataFrame, desde: Periodo | None, hasta: Periodo | None, indice: str
) -> float:
    """`100 * (Π(1 + v/100) - 1)` sobre las `variacion_pp` del rango para `indice`."""
    serie = _comun.serie_en_rango(df, _COL, desde, hasta, indice).sort_index()
    _verificar_periodos_consecutivos(serie, indice, "inflacion_acumulada", desde, hasta)
    factores = _factores_validados(serie, indice, "inflacion_acumulada")
    with np.errstate(over="ignore", invalid="ignore"):
        factor = float(np.prod(factores))
    if not math.isfinite(factor):
        raise InvarianteViolado(
            f"inflacion_acumulada: factor compuesto no finito (overflow) para '{indice}'."
        )
    variacion_total = (factor - 1.0) * 100.0
    if not math.isfinite(variacion_total):
        raise InvarianteViolado(
            f"inflacion_acumulada: resultado no finito (overflow) para '{indice}'."
        )
    return variacion_total


def inflacion_promedio(
    resultado: ResultadoVariacion,
    desde: Periodo | None = None,
    hasta: Periodo | None = None,
    *,
    indice: str,
    metodo: Literal["tcac", "simple"] = "tcac",
) -> float:
    """Inflación promedio del rango para `indice`.

    `metodo="simple"` → media aritmética de `variacion_pp`; admite cualquier
    `resultado.manifiesto.clase` que empiece con `"periodica_"` (estadística
    descriptiva válida incluso sobre tasas de ventana móvil, ej. "promedio de
    la inflación anual observada en el rango" no pretende componer periodos)
    pero rechaza `acumulada_anual`/`desde`, cuyos valores ya son totales.
    `metodo="tcac"` → tasa de crecimiento anual compuesta; exige exclusivamente
    `"periodica_mensual"` (componer ventanas móviles traslapadas cuenta meses
    de más).

    Raises:
        InvarianteViolado: `metodo` fuera de `{"tcac", "simple"}`;
            `metodo="simple"` con `resultado.manifiesto.clase` fuera de
            `"periodica_*"`; `metodo="tcac"` con `resultado.manifiesto.clase !=
            "periodica_mensual"`, con periodos `[desde, hasta]` no mensuales
            consecutivos (un hueco compondría transiciones desconectadas), con
            `indice` sin dato exactamente en `desde`/`hasta` (aunque ese
            periodo exista para otro índice), con algún `variacion_pp <
            -100%` (factor negativo), o con factor/resultado no finitos
            (overflow, incluida la potencia de anualización).
    """
    if metodo == "simple":
        if not resultado.manifiesto.clase.startswith("periodica_"):
            raise InvarianteViolado(
                "inflacion_promedio(metodo='simple') requiere "
                "resultado.manifiesto.clase == 'periodica_*' (los valores de "
                f"'acumulada_anual'/'desde' ya son totales); recibió "
                f"'{resultado.manifiesto.clase}'."
            )
        return _comun.promedio_simple(resultado.df, _COL, desde, hasta, indice)
    if metodo == "tcac":
        if resultado.manifiesto.clase != "periodica_mensual":
            raise InvarianteViolado(
                "inflacion_promedio(metodo='tcac') requiere resultado.manifiesto.clase "
                "== 'periodica_mensual' (componer tasas de ventana móvil traslapada "
                f"cuenta periodos de más); recibió '{resultado.manifiesto.clase}'."
            )
        return _tcac(resultado.df, desde, hasta, indice)
    raise InvarianteViolado(f"metodo '{metodo}' inválido; usa 'tcac' o 'simple'.")


def _tcac(df: pd.DataFrame, desde: Periodo | None, hasta: Periodo | None, indice: str) -> float:
    """Tasa de crecimiento anual compuesta sobre las variaciones del rango.

    `factor = Π(1 + v/100)`; se anualiza suponiendo que cada fila representa
    `1/12` de año — el INPP siempre es mensual, a diferencia del INPC
    (`ppy = 24` para quincenal) no hay periodicidad que distinguir acá.
    Precondición (verificada por el llamador): `resultado.manifiesto.clase ==
    "periodica_mensual"`.

    Raises:
        InvarianteViolado: si los periodos del rango no son mensuales
            consecutivos, si `indice` no tiene dato exactamente en
            `desde`/`hasta`, si algún factor no es finito o es negativo
            (`variacion_pp < -100%`), o si el factor compuesto o el resultado
            final no son finitos (overflow) -- incluido el caso en que el
            producto sí es finito pero la potencia de anualización desborda.
    """
    serie = _comun.serie_en_rango(df, _COL, desde, hasta, indice).sort_index()
    _verificar_periodos_consecutivos(serie, indice, "_tcac", desde, hasta)
    factores = _factores_validados(serie, indice, "_tcac")
    with np.errstate(over="ignore", invalid="ignore"):
        factor = float(np.prod(factores))
    if not math.isfinite(factor):
        raise InvarianteViolado(f"_tcac: factor compuesto no finito (overflow) para '{indice}'.")
    with np.errstate(over="ignore", invalid="ignore"):
        # `np.power` (no el `**` de Python sobre float nativo) es necesario acá:
        # una potencia sobre `float` de Python que desborda lanza `OverflowError`
        # incluso dentro de `np.errstate` -- verificado, `np.power` sí respeta
        # `errstate` y devuelve `inf`, traducible después a `InvarianteViolado`.
        tcac = float((np.power(np.float64(factor), 12 / len(serie)) - 1.0) * 100.0)
    if not math.isfinite(tcac):
        raise InvarianteViolado(f"_tcac: resultado no finito (overflow) para '{indice}'.")
    return tcac


def inflacion_maxima(
    resultado: ResultadoVariacion,
    desde: Periodo | None = None,
    hasta: Periodo | None = None,
    indice: str | None = None,
) -> tuple[Periodo, str, float]:
    """`(periodo, indice, variacion_pp)` del máximo en el rango."""
    return _comun.extremo(resultado.df, _COL, desde, hasta, indice, mayor=True)


def inflacion_minima(
    resultado: ResultadoVariacion,
    desde: Periodo | None = None,
    hasta: Periodo | None = None,
    indice: str | None = None,
) -> tuple[Periodo, str, float]:
    """`(periodo, indice, variacion_pp)` del mínimo en el rango."""
    return _comun.extremo(resultado.df, _COL, desde, hasta, indice, mayor=False)
