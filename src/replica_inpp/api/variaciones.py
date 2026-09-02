"""Cálculo y análisis de variaciones (inflación)."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from replica_inpp.dominio.calculo.variaciones import (
    variacion_acumulada_anual as _variacion_acumulada_anual,
)
from replica_inpp.dominio.calculo.variaciones import (
    variacion_desde as _variacion_desde,
)
from replica_inpp.dominio.calculo.variaciones import (
    variacion_periodica as _variacion_periodica,
)
from replica_inpp.dominio.consulta import variaciones as _consulta
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion
from replica_inpp.dominio.periodos import periodo_desde_str

# -- series --------------------------------------------------------------------


def variacion_periodica(resultado: ResultadoIndice, frecuencia: str) -> ResultadoVariacion:
    """Variación de cada periodo contra N periodos anteriores según `frecuencia`.

    Args:
        resultado: resultado de índices.
        frecuencia: `"mensual"` (1M), `"bimestral"` (2M), `"trimestral"` (3M),
            `"cuatrimestral"` (4M), `"semestral"` (6M) o `"anual"` (12M).

    Raises:
        InvarianteViolado: `frecuencia` fuera del conjunto válido.
    """
    return _variacion_periodica(resultado, frecuencia)  # type: ignore[arg-type]


def variacion_acumulada_anual(resultado: ResultadoIndice) -> ResultadoVariacion:
    """Variación acumulada del año en curso (ene→periodo vs dic año anterior)."""
    return _variacion_acumulada_anual(resultado)


def variacion_desde(
    resultado: ResultadoIndice,
    desde: str,
    hasta: str | None = None,
    incluir_parciales: bool = True,
) -> ResultadoVariacion:
    """Variación total del rango `[desde, hasta]`; una fila por índice.

    Args:
        resultado: resultado de índices.
        desde: periodo inicial.
        hasta: periodo final; `None` = último disponible.
        incluir_parciales: si `False`, excluye periodos con
            `estado_calculo = parcial`. Default `True`.

    Raises:
        PeriodoNoInterpretable: `desde`/`hasta` con formato inválido.
        InvarianteViolado: `desde`/`hasta` interpretable pero fuera de rango,
            ausente en `resultado`, o `desde` posterior a `hasta`.
    """
    return _variacion_desde(
        resultado,
        periodo_desde_str(desde),
        periodo_desde_str(hasta) if hasta is not None else None,
        incluir_parciales,
    )


# -- análisis ------------------------------------------------------------------


def inflacion_en(resultado: ResultadoVariacion, periodo: str) -> pd.DataFrame:
    """Variación de todas las categorías en `periodo`; índice = `indice`.

    Raises:
        PeriodoNoInterpretable: `periodo` con formato inválido.
        InvarianteViolado: `periodo` fuera de rango o ausente en `resultado`.
    """
    return _consulta.inflacion_en(resultado, periodo_desde_str(periodo))


def inflacion_acumulada(
    resultado: ResultadoVariacion,
    desde: str,
    hasta: str | None = None,
    *,
    indice: str,
) -> float:
    """Variación total del rango para `indice`, componiendo las tasas periódicas.

    `100 * (Π(1 + v/100) - 1)` — composición geométrica exacta, no una suma de
    puntos porcentuales. Requiere que `resultado` venga de
    `variacion_periodica(..., "mensual")`: rechaza `variacion_desde`/
    `variacion_acumulada_anual` (valores ya totales) y también rechaza
    `variacion_periodica` con cualquier otra frecuencia (bimestral,
    trimestral, ...) — componer tasas de ventana móvil traslapada contaría
    meses de más.

    Args:
        desde: periodo inicial.
        hasta: periodo final; `None` = último disponible.
        indice: índice a consultar, debe existir en `resultado`. Keyword-only.

    Raises:
        PeriodoNoInterpretable: `desde`/`hasta` con formato inválido.
        InvarianteViolado: `desde`/`hasta` fuera de rango o ausente en
            `resultado`, `desde` posterior a `hasta`, `indice` inexistente;
            `resultado` no viene de `variacion_periodica(..., "mensual")`;
            los periodos `[desde, hasta]` no son mensuales consecutivos (un
            hueco -- ej. un periodo `sin_datos` -- compondría transiciones
            desconectadas); `indice` sin dato exactamente en `desde`/`hasta`
            (aunque ese periodo exista para otro índice de `resultado`);
            algún `variacion_pp < -100%` (factor negativo); o algún factor
            intermedio o el resultado final no son finitos (overflow).
    """
    return _consulta.inflacion_acumulada(
        resultado,
        periodo_desde_str(desde),
        periodo_desde_str(hasta) if hasta is not None else None,
        indice=indice,
    )


def inflacion_promedio(
    resultado: ResultadoVariacion,
    desde: str | None = None,
    hasta: str | None = None,
    *,
    indice: str,
    metodo: Literal["tcac", "simple"] = "tcac",
) -> float:
    """Inflación promedio del rango para `indice` (TCAC o media simple).

    `metodo="simple"` (media aritmética) admite `resultado` de
    `variacion_periodica` con CUALQUIER frecuencia (mensual, bimestral,
    anual, ...) — es estadística descriptiva sobre observaciones disponibles,
    no compone periodos; rechaza `variacion_desde`/`variacion_acumulada_anual`
    (valores ya totales). `metodo="tcac"` (tasa de crecimiento anual
    compuesta) exige exclusivamente `variacion_periodica(..., "mensual")` —
    componer ventanas móviles traslapadas contaría meses de más.

    Args:
        desde: `None` = primer periodo disponible.
        hasta: `None` = último disponible.
        indice: índice a consultar. Keyword-only.
        metodo: `"tcac"` = tasa de crecimiento anual compuesta (default);
            `"simple"` = media aritmética.

    Raises:
        PeriodoNoInterpretable: `desde`/`hasta` con formato inválido.
        InvarianteViolado: `desde`/`hasta` fuera de rango o ausente en
            `resultado`, o `indice` inexistente; `metodo="simple"` con
            `resultado` que no viene de `variacion_periodica` (cualquier
            frecuencia); `metodo="tcac"` con `resultado` que no viene de
            `variacion_periodica(..., "mensual")`, con periodos `[desde,
            hasta]` no mensuales consecutivos, con `indice` sin dato
            exactamente en `desde`/`hasta` (aunque ese periodo exista para
            otro índice), con algún `variacion_pp < -100%` (factor negativo),
            o con factor/resultado no finitos (overflow, incluida la potencia
            de anualización).
    """
    return _consulta.inflacion_promedio(
        resultado,
        periodo_desde_str(desde) if desde is not None else None,
        periodo_desde_str(hasta) if hasta is not None else None,
        indice=indice,
        metodo=metodo,
    )


def inflacion_maxima(
    resultado: ResultadoVariacion,
    desde: str | None = None,
    hasta: str | None = None,
    indice: str | None = None,
) -> tuple[str, str, float]:
    """`(periodo, indice, variacion_pp)` del máximo en el rango.

    `periodo` se devuelve como `str`. `indice=None` busca el máximo global
    entre todos los índices y periodos, en vez de uno específico.

    Raises:
        PeriodoNoInterpretable: `desde`/`hasta` con formato inválido.
        InvarianteViolado: `desde`/`hasta` fuera de rango o ausente en
            `resultado`, o `indice` inexistente.
    """
    periodo, idx, valor = _consulta.inflacion_maxima(
        resultado,
        periodo_desde_str(desde) if desde is not None else None,
        periodo_desde_str(hasta) if hasta is not None else None,
        indice,
    )
    return str(periodo), idx, valor


def inflacion_minima(
    resultado: ResultadoVariacion,
    desde: str | None = None,
    hasta: str | None = None,
    indice: str | None = None,
) -> tuple[str, str, float]:
    """`(periodo, indice, variacion_pp)` del mínimo en el rango.

    Mismo contrato que `inflacion_maxima`, en la dirección opuesta.
    """
    periodo, idx, valor = _consulta.inflacion_minima(
        resultado,
        periodo_desde_str(desde) if desde is not None else None,
        periodo_desde_str(hasta) if hasta is not None else None,
        indice,
    )
    return str(periodo), idx, valor
