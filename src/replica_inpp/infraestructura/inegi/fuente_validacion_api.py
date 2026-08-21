from __future__ import annotations

import math

import requests

from replica_inpp.dominio.errores import (
    ErrorConfiguracion,
    FuenteNoDisponible,
    InvarianteViolado,
    RespuestaInvalida,
)
from replica_inpp.dominio.periodos import PeriodoMensual

# IDs BIE — ver docs/requerimientos/indicadores_bie_inpp.md para procedencia,
# verificación y notas. Conjunto "Ruta B" (superconjunto verificado contra API real).
_INDICADORES: dict[str, dict[str, str]] = {
    "PRODUCCION TOTAL": {
        "INPP sin Petróleo y con Servicios": "910491",
        "INPP con Petróleo y con Servicios": "1700002",
        "Índice General Excluyendo Petróleo": "910493",
        "INPP Mercancías y Servicios Finales": "1700001",
        "INPP Mercancías y Servicios Intermedios": "1750002",
        "Demanda interna": "1380015",
        "Consumo": "1380016",
        "Formación de capital": "1380017",
        "Exportaciones": "1380018",
        "Actividades primarias": "1700003",
        "11 Agricultura, cría y explotación de animales, aprovechamiento forestal, "
        "pesca y caza": "1700004",
        "Actividades secundarias sin petróleo": "1700160",
        "Actividades secundarias con petróleo": "1700161",
        "21 Minería sin petróleo": "1700162",
        "21 Minería con Petróleo": "1700163",
        "22 Generación, transmisión y distribución de energía eléctrica, "
        "suministro de agua y de gas por ductos al consumidor final": "1700211",
        "23 Construcción": "1700226",
        "31-33 Industrias manufactureras": "1700244",
        "Actividades terciarias": "1701070",
        "48-49 Transportes, correos y almacenamiento": "1701071",
        "51 Información en medios masivos": "1701146",
        "53 Servicios inmobiliarios y de alquiler de bienes muebles e intangibles": "1701192",
        "54 Servicios profesionales, científicos y técnicos": "1701215",
        "56 Servicios de apoyo a los negocios y manejo de desechos y servicios "
        "de remediación": "1701264",
        "61 Servicios educativos": "1701306",
        "62 Servicios de salud y de asistencia social": "1701329",
        "71 Servicios de esparcimiento culturales y deportivos, y otros "
        "servicios recreativos": "1701362",
        "72 Servicios de alojamiento temporal y de preparación de alimentos y bebidas": "1701381",
        "81 Otros servicios excepto actividades gubernamentales": "1701404",
    },
}

_URL = (
    "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml"
    "/INDICATOR/{indicador}/es/00/false/BIE-BISE/2.0/{token}?type=json"
)


def _rango_completo(historico: dict[PeriodoMensual, float | None]) -> list[PeriodoMensual]:
    """Genera todos los periodos entre min y max del histórico, inclusive.

    Periodos faltantes en `historico` dentro del rango se incluyen con `None`,
    haciendo visibles los gaps en el DataFrame resultante.
    """
    if not historico:
        return []
    min_p = min(historico)
    max_p = max(historico)
    return [
        PeriodoMensual(a, m)
        for a in range(min_p.año, max_p.año + 1)
        for m in range(1, 13)
        if min_p <= PeriodoMensual(a, m) <= max_p
    ]


def _recortar_al_historico(
    periodos: list[PeriodoMensual], historico: dict[PeriodoMensual, float | None]
) -> dict[PeriodoMensual, float | None]:
    """Deja solo los periodos que caen dentro del histórico publicado.

    La ausencia de una clave significa `fuera_rango_inegi` para el comparador, y
    `None` significa `no_disponible` (INEGI cubre el periodo pero no publicó
    valor). Recortar por AMBOS extremos es lo que mantiene esa distinción: un
    periodo posterior al último publicado —el caso corriente cuando la réplica
    llega más lejos que la publicación oficial— no es un hueco de la serie, es
    territorio que INEGI todavía no cubre.
    """
    if not historico:
        return {}
    min_p, max_p = min(historico), max(historico)
    return {p: historico.get(p) for p in periodos if min_p <= p <= max_p}


def _exigir_periodos(periodos: list[PeriodoMensual], metodo: str) -> None:
    """Rechaza una lista de periodos vacía antes de tocar caché o red."""
    if not periodos:
        raise InvarianteViolado(
            f"FuenteValidacionApi.{metodo}: 'periodos' no puede estar vacío; "
            f"sin periodos no hay nada que consultar."
        )


class FuenteValidacionApi:
    """Implementa `FuenteValidacion` sobre la API del BIE del INEGI.

    Por ahora solo índices de nivel — variaciones e incidencias se agregan
    cuando haya IDs BIE confirmados (ver
    `docs/requerimientos/indicadores_bie_inpp.md`).
    """

    _cache: dict[str, dict[PeriodoMensual, float | None]] = {}

    @classmethod
    def indicadores_en_cache(cls) -> int:
        """Cuántos indicadores tiene descargados el cache de clase."""
        return len(cls._cache)

    @classmethod
    def limpiar_cache(cls) -> None:
        """Vacía el cache de clase; la siguiente consulta vuelve a descargar."""
        cls._cache.clear()

    def __init__(self, token: str, tipo: str, timeout: int = 10) -> None:
        if tipo not in _INDICADORES:
            raise ErrorConfiguracion(
                f"tipo '{tipo}' no tiene indicador INEGI disponible. "
                f"Tipos soportados: {list(_INDICADORES)}"
            )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ErrorConfiguracion(
                f"timeout {timeout!r} inválido; debe ser un valor finito mayor a 0 segundos."
            )
        self._token = token
        self._tipo = tipo
        self._timeout = timeout

    def obtener_indices(
        self, periodos: list[PeriodoMensual]
    ) -> dict[str, dict[PeriodoMensual, float | None]]:
        """Devuelve el valor publicado por el INEGI por índice y por periodo.

        Usa cache de clase — la primera llamada descarga el histórico completo;
        las siguientes lo reutilizan sin hacer requests adicionales.

        Raises:
            InvarianteViolado: Si `periodos` está vacío.
        """
        _exigir_periodos(periodos, "obtener_indices")
        indicadores = _INDICADORES[self._tipo]
        resultado: dict[str, dict[PeriodoMensual, float | None]] = {}
        for nombre, indicador in indicadores.items():
            if indicador not in self._cache:
                self._cache[indicador] = self._fetch(indicador)
            historico = self._cache[indicador]
            resultado[nombre] = _recortar_al_historico(periodos, historico)
        return resultado

    def historico_indices(self) -> dict[str, dict[PeriodoMensual, float | None]]:
        """Devuelve el histórico completo de índices sin filtro de periodo.

        Cubre desde el primer hasta el último periodo que INEGI tiene en su
        serie. Periodos intermedios sin dato aparecen con valor `None`.
        """
        indicadores = _INDICADORES[self._tipo]
        resultado: dict[str, dict[PeriodoMensual, float | None]] = {}
        for nombre, indicador in indicadores.items():
            if indicador not in self._cache:
                self._cache[indicador] = self._fetch(indicador)
            historico = self._cache[indicador]
            rango = _rango_completo(historico)
            resultado[nombre] = {p: historico.get(p) for p in rango}
        return resultado

    def _fetch(self, indicador: str) -> dict[PeriodoMensual, float | None]:
        # La URL lleva el token en texto plano (formato fijo de la API del BIE) —
        # nunca incluir `str(exc)` en el mensaje. Tampoco alcanza con `from None`
        # DENTRO del except: __context__ sigue apuntando a la excepción original
        # (con el token) aunque se suprima su impresión en el traceback por
        # defecto. Levantar la excepción sanitizada FUERA del try/except es lo
        # que evita que __context__ se fije en absoluto.
        url = _URL.format(indicador=indicador, token=self._token)
        resp: requests.Response | None = None
        error_msg: str | None = None
        try:
            resp = requests.get(url, timeout=self._timeout)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "desconocido"
            error_msg = (
                f"La API del INEGI respondió {status} para el indicador {indicador!r}. "
                f"Verifica el token INEGI configurado."
            )
        except requests.exceptions.RequestException as exc:
            error_msg = (
                f"No se pudo conectar a la API del INEGI ({type(exc).__name__}) para "
                f"el indicador {indicador!r}."
            )
        if error_msg is not None:
            raise FuenteNoDisponible(error_msg)
        assert resp is not None  # error_msg es None solo si el try completó sin excepción

        try:
            data = resp.json()
        except ValueError as exc:
            raise RespuestaInvalida(f"Respuesta del INEGI no es JSON válido: {exc}") from exc
        if not isinstance(data, dict):
            raise RespuestaInvalida(
                f"La API devolvió un JSON con forma inesperada para el indicador "
                f"{indicador!r} (esperaba un objeto, llegó {type(data).__name__})."
            )

        series = data.get("Series")
        if not series:
            raise RespuestaInvalida("La API devolvió 'Series' vacío o ausente.")
        if not isinstance(series, list) or not isinstance(series[0], dict):
            raise RespuestaInvalida(
                f"'Series' del indicador {indicador!r} tiene forma inesperada: {series!r}"
            )
        observations = series[0].get("OBSERVATIONS")
        if not observations:
            raise RespuestaInvalida(
                f"La API devolvió 'OBSERVATIONS' vacío o ausente para el indicador {indicador!r}."
            )
        if not isinstance(observations, list):
            raise RespuestaInvalida(
                f"'OBSERVATIONS' del indicador {indicador!r} esperaba una lista, "
                f"llegó {type(observations).__name__}."
            )

        resultado: dict[PeriodoMensual, float | None] = {}
        for obs in observations:
            if not isinstance(obs, dict):
                raise RespuestaInvalida(f"Observación con formato inesperado: {obs!r}")
            time_period = obs.get("TIME_PERIOD")
            if not isinstance(time_period, str):
                raise RespuestaInvalida(
                    f"Indicador {indicador!r}: TIME_PERIOD esperaba texto, llegó "
                    f"{type(time_period).__name__} ({time_period!r})."
                )
            partes = time_period.split("/")
            if len(partes) != 2:
                raise RespuestaInvalida(
                    f"Indicador {indicador!r} esperaba periodo mensual ('AAAA/MM'), "
                    f"pero TIME_PERIOD={time_period!r} tiene {len(partes)} partes."
                )
            try:
                periodo = PeriodoMensual(int(partes[0]), int(partes[1]))
            except ValueError as exc:
                raise RespuestaInvalida(
                    f"TIME_PERIOD={time_period!r} no tiene año/mes numéricos: {exc}"
                ) from exc
            except InvarianteViolado as exc:
                raise RespuestaInvalida(
                    f"TIME_PERIOD={time_period!r} tiene año/mes fuera de rango: {exc}"
                ) from exc

            if "OBS_VALUE" not in obs:
                raise RespuestaInvalida(f"Observación sin 'OBS_VALUE': {obs!r}")
            raw = obs["OBS_VALUE"]
            valor: float | None
            if raw is None:
                valor = None
            else:
                if isinstance(raw, bool):
                    raise RespuestaInvalida(
                        f"Indicador {indicador!r}, periodo {periodo}: "
                        f"OBS_VALUE={raw!r} es booleano, no un valor numérico."
                    )
                try:
                    valor = float(raw)
                except (TypeError, ValueError) as exc:
                    raise RespuestaInvalida(f"OBS_VALUE={raw!r} no es numérico: {exc}") from exc
                if not math.isfinite(valor):
                    raise RespuestaInvalida(
                        f"Indicador {indicador!r}, periodo {periodo}: "
                        f"OBS_VALUE={raw!r} no es un valor finito."
                    )
            resultado[periodo] = valor

        return resultado
