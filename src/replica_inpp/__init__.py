"""replica_inpp — réplica del INPP de México.

Superficie pública flat estilo pandas: `import replica_inpp as rep` y luego
`rep.<func>(...)`.
"""

from __future__ import annotations

import sys
from types import ModuleType

from replica_inpp.api import config as _config
from replica_inpp.api.config import limpiar_cache, mostrar_config, reset_config, set_token
from replica_inpp.api.consultas import consultar_indice
from replica_inpp.dominio.errores import (
    ArchivoCorrupto,
    ArchivoNoEncontrado,
    ArchivoVacio,
    CanastaNoSoportada,
    CanastaSinGenericos,
    ColumnasMinFaltantes,
    EncodingNoLegible,
    ErrorCalculo,
    ErrorConfiguracion,
    ErrorDominio,
    ErrorImportacion,
    ErrorValidacion,
    FuenteNoDisponible,
    InvarianteViolado,
    OrientacionNoDetectable,
    PeriodoNoDisponible,
    PeriodoNoInterpretable,
    PeriodosInsuficientes,
    PonderadorFaltante,
    ReplicaInppError,
    RespuestaInvalida,
    SerieVacia,
    VersionNoCoincide,
)
from replica_inpp.dominio.periodos import PeriodoMensual, periodo_desde_str

# Declaración para el type checker — runtime manejado por _ReplicaModule proxy.
timeout_api: int

__all__ = [
    # periodos
    "PeriodoMensual",
    "periodo_desde_str",
    # config
    "limpiar_cache",
    "mostrar_config",
    "reset_config",
    "set_token",
    # consultas INEGI
    "consultar_indice",
    # errores
    "ArchivoCorrupto",
    "ArchivoNoEncontrado",
    "ArchivoVacio",
    "CanastaNoSoportada",
    "CanastaSinGenericos",
    "ColumnasMinFaltantes",
    "EncodingNoLegible",
    "ErrorCalculo",
    "ErrorConfiguracion",
    "ErrorDominio",
    "ErrorImportacion",
    "ErrorValidacion",
    "FuenteNoDisponible",
    "InvarianteViolado",
    "OrientacionNoDetectable",
    "PeriodoNoDisponible",
    "PeriodoNoInterpretable",
    "PeriodosInsuficientes",
    "PonderadorFaltante",
    "ReplicaInppError",
    "RespuestaInvalida",
    "SerieVacia",
    "VersionNoCoincide",
    # variables configurables (vía proxy de módulo)
    "timeout_api",
]


class _ReplicaModule(ModuleType):
    """Módulo paquete con proxy de las variables configurables.

    `rep.timeout_api = X` se redirige a `api/config.py` para que las funciones
    de consulta lean siempre el valor vigente — un nombre re-exportado por
    valor no propagaría la reasignación.
    """

    _PROXY = ("timeout_api",)

    def __getattr__(self, name: str) -> object:
        if name in type(self)._PROXY:
            return getattr(_config, name)
        raise AttributeError(f"module 'replica_inpp' has no attribute '{name}'")

    def __setattr__(self, name: str, value: object) -> None:
        if name in type(self)._PROXY:
            setattr(_config, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ReplicaModule
