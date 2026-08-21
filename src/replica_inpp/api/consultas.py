"""Consulta directa de series publicadas por INEGI (sin comparación).

Devuelven `pd.DataFrame` indexado por `periodo` listo para inspeccionar.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from replica_inpp.api import config
from replica_inpp.infraestructura.inegi.fuente_validacion_api import FuenteValidacionApi


def _a_dataframe(series: Mapping[str, Mapping[Any, float | None]]) -> pd.DataFrame:
    df = pd.DataFrame(series)
    df.index.name = "periodo"
    df.sort_index(inplace=True)
    return df


def consultar_indice(tipo: str) -> pd.DataFrame:
    """Devuelve el histórico de índices publicados por INEGI para `tipo`.

    Cubre desde el primer hasta el último periodo que INEGI tiene en su serie.
    Un periodo intermedio sin dato aparece como `NaN` (gap visible); un
    periodo anterior al inicio de la serie simplemente no existe en el
    resultado — son dos ausencias distintas, no confundirlas.

    El INPP se publica solo mensual — a diferencia de `consultar_indice` en
    `replica-inpc-mx`, acá no hay parámetro `periodicidad`.

    Args:
        tipo: se normaliza con `.upper()`. Valores soportados: ver
            `docs/requerimientos/indicadores_bie_inpp.md`. Por ahora solo
            `"PRODUCCION TOTAL"`.

    Raises:
        ErrorConfiguracion: `tipo` sin indicador INEGI, o no hay token
            configurado (`rep.set_token(...)` o `INEGI_TOKEN`).
        FuenteNoDisponible: la API de INEGI no responde o devuelve error HTTP.
        RespuestaInvalida: la respuesta de INEGI tiene formato inesperado.
    """
    tipo = tipo.upper()
    fuente = FuenteValidacionApi(config.get_token(), tipo, timeout=config.timeout_api)
    return _a_dataframe(fuente.historico_indices())
