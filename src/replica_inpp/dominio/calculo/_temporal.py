"""Utilidades temporales compartidas por `variaciones.py`.

Puerto podado de `replica-inpc-mx/.../dominio/calculo/_temporal.py`: INPP es
mensual-only (`PeriodoMensual` único, sin `PeriodoQuincenal`), así que no hace
falta `es_mensual`, `LAG_QUINCENAL` ni `restar_quincenas` -- solo sobrevive la
mitad mensual del original.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from replica_inpp.dominio.periodos import PeriodoMensual

Frecuencia = Literal[
    "mensual",
    "bimestral",
    "trimestral",
    "cuatrimestral",
    "semestral",
    "anual",
]

# Lag en número de meses según la frecuencia solicitada.
LAG_MENSUAL: dict[str, int] = {
    "mensual": 1,
    "bimestral": 2,
    "trimestral": 3,
    "cuatrimestral": 4,
    "semestral": 6,
    "anual": 12,
}


def restar_meses(periodo: PeriodoMensual, n: int) -> PeriodoMensual:
    """Resta `n` meses a `periodo`, cruzando el cambio de año.

    Convierte a un ordinal de meses desde el año 0 para no tener que manejar
    el acarreo a mano. `n` no se valida: el único llamador lo saca de
    `LAG_MENSUAL`, que solo tiene enteros positivos.
    """
    ordinal = periodo.año * 12 + (periodo.mes - 1)
    ordinal -= n
    return PeriodoMensual(ordinal // 12, ordinal % 12 + 1)


def resolver_extremo(
    exacto: PeriodoMensual,
    validos: Sequence[PeriodoMensual],
    *,
    incluir_parciales: bool,
    primero: bool,
) -> PeriodoMensual | None:
    """Resuelve el periodo real de un extremo de rango para un índice/genérico.

    Devuelve `exacto` si tiene dato; si no y `incluir_parciales`, el periodo
    válido más temprano (`primero`) o más tardío; `None` si no es computable.

    Toma el mínimo y el máximo en vez del primer y último elemento: así el
    resultado no depende de que `validos` venga ordenado.

    `validos` es `Sequence` y no `list` porque la función solo la recorre: con
    `list` invariante, pasarle una `list[PeriodoMensual]` concreta no tipa
    contra `Sequence[PeriodoMensual]` en algunos llamadores.
    """
    if exacto in validos:
        return exacto
    if not incluir_parciales or not validos:
        return None
    return min(validos) if primero else max(validos)
