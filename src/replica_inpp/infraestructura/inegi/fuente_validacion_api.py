from __future__ import annotations

import math
from typing import Literal

import requests

from replica_inpp.dominio.errores import (
    ErrorConfiguracion,
    FuenteNoDisponible,
    InvarianteViolado,
    RespuestaInvalida,
)
from replica_inpp.dominio.periodos import PeriodoMensual

# IDs BIE — conjunto "Ruta B" (superconjunto verificado contra API real).
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

# Catálogo por serie individual, para `consultar_indice(tipo, incluir_petroleo=...)` —
# a diferencia de `_INDICADORES` arriba (familia completa de 27 series bajo un solo
# "tipo"), acá cada clave es la serie puntual que se pide. Eje SCIAN — sectores (más
# "31-33"/"48-49", que el BIE publica combinados aunque `CanastaINPP` los traiga
# separados).
#
# Valor `str` = 1 solo ID publicado (sin variante `incluir_petroleo` separada —
# verificado 2026-09-06 que el genérico 070 Petróleo crudo pertenece únicamente al
# sector 21, así que filtrarlo de cualquier otro sector no cambia nada real; INEGI
# nunca publicó una segunda serie ahí). Valor `dict[bool, str]` = 2 IDs reales
# distintos, uno por variante (hoy: solo sector 21, Minería).
_SECTORES: dict[str, str | dict[bool, str]] = {
    "11": "1700004",
    "21": {True: "1700163", False: "1700162"},
    "22": "1700211",
    "23": "1700226",
    "31-33": "1700244",
    "48-49": "1701071",
    "51": "1701146",
    "53": "1701192",
    "54": "1701215",
    "56": "1701264",
    "61": "1701306",
    "62": "1701329",
    "71": "1701362",
    "72": "1701381",
    "81": "1701404",
}

# Eje destino/etapa (`rubro`, ortogonal a SCIAN — ver "Dominio: INPP vs INPC" en
# CLAUDE.md), mismo formato que `_SECTORES`. `"INPP"` es el headline sin desglose
# (`agregacion="INPP"`, `rubro="produccion_total"` en `calcular_indice`); el resto son
# nombres de `rubro` tal cual los conoce `RUBRO_A_COLUMNA_PESO` (`dominio/tipos.py`).
# IDs verificados exactos contra `LaspeyresDirecto` (canasta 2019, 73 obs
# jul2019-jul2025): `"INPP"` con `910491`/`910492`, `bienes_intermedios` con `910493`
# (`incluir_petroleo=False` — único match tras probar 10 combos, ver
# docs/requerimientos/investigacion_bienes_finales_intermedios.md §7), `demanda_interna_
# */exportaciones` con `910494`/`1380016`/`1380017`/`1380018`.
#
# NO incluye `"bienes_finales"` — probado exhaustivo (con/sin petróleo) contra las 23
# series de nivel disponibles hoy (este dict + `_SECTORES`), ningún ID calza (más
# cercano: `max_abs≈1.18` contra el headline sin petróleo, que ya es otra cosa). No hay
# ID BIE real para `bienes_finales` — no es que falte buscar, ya se buscó en todo el
# universo conocido.
_RUBROS: dict[str, str | dict[bool, str]] = {
    "INPP": {True: "1700002", False: "1700001"},
    "bienes_intermedios": "1750002",
    "demanda_interna_total": "1380015",
    "demanda_interna_consumo": "1380016",
    "demanda_interna_capital": "1380017",
    "exportaciones": "1380018",
}

# Catálogo combinado para `resolver_indicador` — unión de `_SECTORES` (eje SCIAN) y
# `_RUBROS` (eje destino/etapa); las claves no se pisan entre sí (sectores son
# numéricos, rubros son nombres).
_NIVELES: dict[str, str | dict[bool, str]] = {**_SECTORES, **_RUBROS}


# Igual estructura que `_SECTORES`, para `consultar_variacion(tipo, frecuencia, ...)` —
# una tabla por `frecuencia` (rama BIE "Variaciones > Mensual"/"Interanual"/"Acumulada",
# esta última sin IDs todavía). A diferencia de `_SECTORES` (niveles), acá SÍ incluyen
# `"INPP"` (headline con/sin petróleo) porque el usuario lo trajo como parte del mismo
# lote de IDs — no se agregó a `_SECTORES` porque esa migración se acotó a sectores nada
# más. Sin re-verificar contra la API real todavía (a diferencia de `_SECTORES`, que sí
# se confirmó con pull en vivo).
_VARIACION_MENSUAL: dict[str, str | dict[bool, str]] = {
    "INPP": {True: "1800002", False: "1800001"},
    "11": "1800003",
    "21": {True: "1800005", False: "1800004"},
    "22": "1800006",
    "23": "1800007",
    "31-33": "1800008",
    "48-49": "1800009",
    "51": "1800010",
    "53": "1800011",
    "54": "1800012",
    "56": "1800013",
    "61": "1800014",
    "62": "1800015",
    "71": "1800016",
    "72": "1800017",
    "81": "1800018",
}

# Variación INTERANUAL (mes de hoy vs mismo mes año anterior) — mismo universo de 18
# series que `_VARIACION_MENSUAL`, IDs distintos.
_VARIACION_INTERANUAL: dict[str, str | dict[bool, str]] = {
    "INPP": {True: "1801002", False: "1801001"},
    "11": "1801003",
    "21": {True: "1801005", False: "1801004"},
    "22": "1801006",
    "23": "1801007",
    "31-33": "1801008",
    "48-49": "1801009",
    "51": "1801010",
    "53": "1801011",
    "54": "1801012",
    "56": "1801013",
    "61": "1801014",
    "62": "1801015",
    "71": "1801016",
    "72": "1801017",
    "81": "1801018",
}

# Variación ACUMULADA (suma de variaciones mensuales desde enero: en enero es dic→ene, cada
# mes siguiente le suma la variación mensual de ese mes, se reinicia cada año) — mismo
# universo de 18 series que `_VARIACION_MENSUAL`/`_VARIACION_INTERANUAL`, IDs distintos.
_VARIACION_ACUMULADA: dict[str, str | dict[bool, str]] = {
    "INPP": {True: "1802002", False: "1802001"},
    "11": "1802003",
    "21": {True: "1802005", False: "1802004"},
    "22": "1802006",
    "23": "1802007",
    "31-33": "1802008",
    "48-49": "1802009",
    "51": "1802010",
    "53": "1802011",
    "54": "1802012",
    "56": "1802013",
    "61": "1802014",
    "62": "1802015",
    "71": "1802016",
    "72": "1802017",
    "81": "1802018",
}

FrecuenciaVariacion = Literal["mensual", "anual", "acumulada"]

_VARIACION_POR_FRECUENCIA: dict[str, dict[str, str | dict[bool, str]]] = {
    "mensual": _VARIACION_MENSUAL,
    "anual": _VARIACION_INTERANUAL,
    "acumulada": _VARIACION_ACUMULADA,
}


def _resolver_indicador_en(
    catalogo: dict[str, str | dict[bool, str]], tipo: str, incluir_petroleo: bool | None
) -> str:
    """Resuelve `tipo`/`incluir_petroleo` al ID BIE concreto dentro de `catalogo`.

    Reglas puramente de catálogo (cuántos IDs hay publicados para `tipo`), sin
    ningún cálculo de pesos de por medio:

    - `tipo` con 1 solo ID publicado: `incluir_petroleo=None` resuelve directo;
      pasar `True`/`False` explícito lanza error porque no hay una segunda
      serie distinta que elegir.
    - `tipo` con 2 IDs publicados (con/sin petróleo): `incluir_petroleo=None`
      es ambiguo y lanza error pidiendo que se especifique; `True`/`False`
      resuelve al ID correspondiente.

    Raises:
        ErrorConfiguracion: `tipo` no soportado en `catalogo`, o
            `incluir_petroleo` no aplica (tipo de 1 sola variante) o es
            ambiguo (tipo de 2 variantes, sin especificar cuál).
    """
    if tipo not in catalogo:
        raise ErrorConfiguracion(
            f"tipo {tipo!r} no tiene indicador INEGI disponible. Tipos soportados: {list(catalogo)}"
        )
    entrada = catalogo[tipo]
    if isinstance(entrada, str):
        if incluir_petroleo is not None:
            raise ErrorConfiguracion(
                f"tipo {tipo!r} no tiene 2 variantes de 'incluir_petroleo' publicadas "
                f"por separado; usa incluir_petroleo=None."
            )
        return entrada
    if incluir_petroleo is None:
        raise ErrorConfiguracion(
            f"tipo {tipo!r} tiene 2 variantes publicadas (con y sin petróleo); "
            f"especifica incluir_petroleo=True o incluir_petroleo=False."
        )
    return entrada[incluir_petroleo]


def resolver_indicador(tipo: str, incluir_petroleo: bool | None) -> str:
    """`_resolver_indicador_en` sobre `_NIVELES` (`_SECTORES` ∪ `_RUBROS`) — niveles de índice."""
    return _resolver_indicador_en(_NIVELES, tipo, incluir_petroleo)


def resolver_indicador_variacion(
    tipo: str, frecuencia: FrecuenciaVariacion, incluir_petroleo: bool | None
) -> str:
    """Resuelve `tipo`/`frecuencia`/`incluir_petroleo` al ID BIE de variación.

    `frecuencia` selecciona la tabla (`_VARIACION_POR_FRECUENCIA`: `"mensual"`,
    `"anual"`, `"acumulada"`, mismo universo de 18 series en las 3); dentro de
    ella, mismas reglas que `_resolver_indicador_en`. La validación de
    `frecuencia` es defensiva (`FrecuenciaVariacion` ya la limita a las 3 en
    tiempo de tipado) — cubre una llamada sin chequeo de tipos con un string
    fuera de esas 3.

    Raises:
        ErrorConfiguracion: `frecuencia` no es una de las 3 soportadas, o lo
            que ya lanza `_resolver_indicador_en` para `tipo`/`incluir_petroleo`
            dentro de esa frecuencia.
    """
    if frecuencia not in _VARIACION_POR_FRECUENCIA:
        raise ErrorConfiguracion(
            f"frecuencia {frecuencia!r} no válida. "
            f"Frecuencias soportadas: {list(_VARIACION_POR_FRECUENCIA)}"
        )
    return _resolver_indicador_en(_VARIACION_POR_FRECUENCIA[frecuencia], tipo, incluir_petroleo)


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
    cuando haya IDs BIE confirmados.
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
        return _fetch_crudo(self._token, indicador, self._timeout)


def _fetch_crudo(token: str, indicador: str, timeout: int) -> dict[PeriodoMensual, float | None]:
    """Descarga y parsea el histórico crudo de UN indicador BIE (sin cache).

    Función libre (no ligada a `self`) para que tanto `FuenteValidacionApi`
    (familia completa, `_INDICADORES`) como `historico_indicador` (serie
    puntual, `_SECTORES`) compartan la misma lógica de red/parseo sin
    necesitar una instancia con `tipo` fijo.
    """
    # La URL lleva el token en texto plano (formato fijo de la API del BIE) —
    # nunca incluir `str(exc)` en el mensaje. Tampoco alcanza con `from None`
    # DENTRO del except: __context__ sigue apuntando a la excepción original
    # (con el token) aunque se suprima su impresión en el traceback por
    # defecto. Levantar la excepción sanitizada FUERA del try/except es lo
    # que evita que __context__ se fije en absoluto.
    url = _URL.format(indicador=indicador, token=token)
    resp: requests.Response | None = None
    error_msg: str | None = None
    try:
        resp = requests.get(url, timeout=timeout)
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


def historico_indicador(
    token: str, indicador: str, timeout: int = 10
) -> dict[PeriodoMensual, float | None]:
    """Histórico completo (rango min-max, huecos como `None`) de UN indicador
    BIE puntual, ya resuelto por `resolver_indicador` — sin agrupar por
    familia, a diferencia de `FuenteValidacionApi.historico_indices()`.

    Comparte el cache de clase de `FuenteValidacionApi` (mismo `indicador` ya
    descargado por cualquiera de los 2 caminos no repite el request).

    Raises:
        ErrorConfiguracion: `timeout` no positivo o no finito.
        FuenteNoDisponible: la API de INEGI no responde o devuelve error HTTP.
        RespuestaInvalida: la respuesta de INEGI tiene formato inesperado.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ErrorConfiguracion(
            f"timeout {timeout!r} inválido; debe ser un valor finito mayor a 0 segundos."
        )
    if indicador not in FuenteValidacionApi._cache:
        FuenteValidacionApi._cache[indicador] = _fetch_crudo(token, indicador, timeout)
    historico = FuenteValidacionApi._cache[indicador]
    rango = _rango_completo(historico)
    return {p: historico.get(p) for p in rango}
