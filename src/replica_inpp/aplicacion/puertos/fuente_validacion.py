from __future__ import annotations

from typing import Protocol

from replica_inpp.dominio.periodos import PeriodoMensual


class FuenteValidacion(Protocol):
    """Contrato para obtener series publicadas por INEGI para validación.

    Por ahora cubre solo niveles de índice — el INPP se publica únicamente mensual
    (no hay eje quincenal como en el INPC), por lo que `PeriodoMensual` es el único
    tipo de periodo. `variaciones`/`incidencias` se agregan cuando haya IDs BIE
    confirmados para ellas (ver `docs/requerimientos/indicadores_bie_inpp.md`).

    El `tipo` (p. ej. `"PRODUCCION TOTAL"`) se fija en el constructor del
    implementador, no en el método — mismo patrón que `replica-inpc-mx`.

    Implementado por `FuenteValidacionApi`
    (`infraestructura/inegi/fuente_validacion_api.py`), de forma estructural: el
    adaptador no importa este Protocol.

    Esquema de retorno — `dict[str, dict[PeriodoMensual, float | None]]`:

    - clave exterior: nombre del índice publicado por INEGI (p. ej. `"Demanda
      interna"`, `"11 Agricultura, cría y explotación de animales, aprovechamiento
      forestal, pesca y caza"`).
    - clave interior: el `PeriodoMensual` consultado.
    - valor `float`: valor publicado por INEGI.
    - valor `None`: INEGI tiene el periodo en rango pero sin dato.
    - periodo ausente del dict interior: fuera del histórico INEGI por cualquiera
      de sus dos extremos.

    Errores comunes:

    - `len(periodos) == 0` → `InvarianteViolado`.
    - `tipo` sin indicador INEGI → `ErrorConfiguracion`.
    - API no responde / HTTP error → `FuenteNoDisponible`.
    - respuesta INEGI con formato inesperado → `RespuestaInvalida`.
    """

    def obtener_indices(
        self,
        periodos: list[PeriodoMensual],
    ) -> dict[str, dict[PeriodoMensual, float | None]]:
        """Niveles de índice publicados por INEGI (series BIE de nivel)."""
        ...
