"""Consulta directa de series publicadas por INEGI (sin comparación).

Devuelven `pd.DataFrame` indexado por `periodo` listo para inspeccionar.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from replica_inpp.api import config
from replica_inpp.infraestructura.inegi.fuente_validacion_api import (
    FrecuenciaVariacion,
    historico_indicador,
    resolver_indicador,
    resolver_indicador_variacion,
)


def _a_dataframe(series: Mapping[str, Mapping[Any, float | None]]) -> pd.DataFrame:
    df = pd.DataFrame(series)
    df.index.name = "periodo"
    df.sort_index(inplace=True)
    return df


def consultar_indice(tipo: str, *, incluir_petroleo: bool | None = None) -> pd.DataFrame:
    """Devuelve el histórico publicado por INEGI para una serie puntual.

    Cubre 2 ejes: sectores SCIAN (`tipo` = código de 2 dígitos, más
    `"31-33"`/`"48-49"`, que el BIE publica combinados) y `rubro`/headline
    (`tipo` = `"INPP"`, `"bienes_intermedios"`, `"demanda_interna_total"`,
    `"demanda_interna_consumo"`, `"demanda_interna_capital"`,
    `"exportaciones"`). **`"bienes_finales"` NO está** — probado exhaustivo
    contra las 23 series de nivel disponibles (con/sin petróleo, ninguna
    calza), no hay ID BIE real conocido para ese rubro. Ver
    `infraestructura.inegi.fuente_validacion_api._NIVELES` para la lista
    completa soportada hoy.

    A diferencia de `consultar_indice` en `replica-inpc-mx`, acá `tipo`
    identifica UNA serie puntual, no una familia — el resultado es un
    `DataFrame` de 1 sola columna (nombre = `tipo`), nunca un subconjunto ni
    una familia completa. Tampoco hay parámetro `periodicidad`: el INPP se
    publica solo mensual.

    Cubre desde el primer hasta el último periodo que INEGI tiene en su serie.
    Un periodo intermedio sin dato aparece como `NaN` (gap visible); un
    periodo anterior al inicio de la serie simplemente no existe en el
    resultado — son dos ausencias distintas, no confundirlas.

    Args:
        tipo: código de sector SCIAN (ej. `"11"`, `"21"`, `"31-33"`), o
            `"INPP"`/nombre de `rubro` (ver arriba).
        incluir_petroleo: solo distingue algo en `tipo` con 2 IDs publicados
            por separado (hoy: `"21"` y `"INPP"`). En el resto, dejar `None`
            (default) — pasar `True`/`False` explícito ahí lanza error porque
            no hay una segunda serie que elegir. En `"21"`/`"INPP"`, `None` es
            ambiguo (hay 2 series reales) y también lanza error — hay que
            especificar cuál.

    Raises:
        ErrorConfiguracion: `tipo` no soportado, `incluir_petroleo` no aplica
            o es ambiguo para ese `tipo`, no hay token configurado
            (`rep.set_token(...)` o `INEGI_TOKEN`), o `rep.timeout_api` es
            inválido (no positivo o no finito).
        FuenteNoDisponible: la API de INEGI no responde o devuelve error HTTP.
        RespuestaInvalida: la respuesta de INEGI tiene formato inesperado.
    """
    indicador = resolver_indicador(tipo, incluir_petroleo)
    historico = historico_indicador(config.get_token(), indicador, timeout=config.timeout_api)
    return _a_dataframe({tipo: historico})


def consultar_variacion(
    tipo: str, frecuencia: FrecuenciaVariacion, *, incluir_petroleo: bool | None = None
) -> pd.DataFrame:
    """Devuelve la variación publicada por INEGI para una serie puntual.

    Mismo mecanismo que `consultar_indice` (`tipo`/`incluir_petroleo` con
    idéntico significado), pero contra el catálogo de variaciones — sin
    default, `frecuencia` selecciona la tabla:

    - `"mensual"`: mes vs mes anterior.
    - `"anual"`: mes de hoy vs mismo mes año anterior (interanual).
    - `"acumulada"`: suma de variaciones mensuales desde enero (en enero es
      dic→ene; cada mes siguiente le suma la variación mensual de ese mes;
      se reinicia cada año).

    Mismo universo de 18 series (`"INPP"` + sectores SCIAN) en las 3
    frecuencias.

    Cubre desde el primer hasta el último periodo que INEGI tiene en su serie.
    Un periodo intermedio sin dato aparece como `NaN`; un periodo anterior al
    inicio de la serie simplemente no existe en el resultado.

    Args:
        tipo: `"INPP"`, código de sector SCIAN (ej. `"11"`, `"21"`), o
            `"31-33"`/`"48-49"`.
        frecuencia: `"mensual"`, `"anual"` o `"acumulada"`.
        incluir_petroleo: solo distingue algo en `tipo` con 2 IDs publicados
            por separado (hoy: `"INPP"` y `"21"`, en cualquier `frecuencia`).
            En el resto, dejar `None` (default) — pasar `True`/`False`
            explícito ahí lanza error. En `"INPP"`/`"21"`, `None` es ambiguo
            y también lanza error.

    Raises:
        ErrorConfiguracion: `frecuencia` no es una de las 3 soportadas, `tipo`
            no soportado, `incluir_petroleo` no aplica o es ambiguo para ese
            `tipo`, no hay token configurado (`rep.set_token(...)` o
            `INEGI_TOKEN`), o `rep.timeout_api` es inválido (no positivo o no
            finito).
        FuenteNoDisponible: la API de INEGI no responde o devuelve error HTTP.
        RespuestaInvalida: la respuesta de INEGI tiene formato inesperado.
    """
    indicador = resolver_indicador_variacion(tipo, frecuencia, incluir_petroleo)
    historico = historico_indicador(config.get_token(), indicador, timeout=config.timeout_api)
    return _a_dataframe({tipo: historico})
