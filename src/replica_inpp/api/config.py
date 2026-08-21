"""Configuración global de la API: token INEGI y timeout.

`timeout_api` es la fuente de verdad; `replica_inpp/__init__.py` instala un
proxy de módulo para que `rep.timeout_api = X` la actualice acá.
"""

from __future__ import annotations

import os

from replica_inpp.dominio.errores import ErrorConfiguracion
from replica_inpp.infraestructura.inegi.fuente_validacion_api import FuenteValidacionApi

_token: str | None = None

# Variable configurable.
timeout_api: int = 10


def set_token(token: str) -> None:
    """Almacena el token INEGI para la sesión actual.

    Cualquier string se acepta aquí — la validez recién se pone a prueba cuando
    una llamada de `consultar_*` dispara una petición real a la API. Si el
    indicador ya está en cache (`FuenteValidacionApi._cache`), ni siquiera
    entonces: la respuesta se sirve desde ahí sin tocar la red, así que un token
    inválido puede pasar desapercibido toda la sesión si los datos ya se
    descargaron antes con un token válido.
    """
    global _token
    _token = token


def get_token() -> str:
    """Devuelve el token INEGI configurado (uso interno).

    Busca primero la variable de entorno `INEGI_TOKEN` y, si no existe, el
    valor fijado con `set_token`. El orden no es arbitrario: en CI y en CLI el
    token se fija por entorno sin escribir código, y ese contexto debe ganar
    sobre un `set_token` dejado por error en una celda de notebook. En un
    notebook interactivo, donde no hay variable de entorno, `set_token` sigue
    siendo el único mecanismo.

    Raises:
        ErrorConfiguracion: Si no hay token por ninguna de las dos vías.
    """
    token = os.environ.get("INEGI_TOKEN") or _token
    if not token:
        raise ErrorConfiguracion(
            "No hay token INEGI configurado. Usa rep.set_token('...') o exporta "
            "la variable de entorno INEGI_TOKEN."
        )
    return token


def reset_config() -> None:
    """Restaura el timeout a su valor por defecto."""
    global timeout_api
    timeout_api = 10


def mostrar_config() -> None:
    """Imprime el estado actual de la configuración en stdout."""
    if os.environ.get("INEGI_TOKEN"):
        estado_token = "configurado (INEGI_TOKEN)"
    elif _token:
        estado_token = "configurado (set_token)"
    else:
        estado_token = "no configurado"
    n = FuenteValidacionApi.indicadores_en_cache()
    print(
        f"timeout_api: {timeout_api}\n"
        f"token INEGI: {estado_token}\n"
        f"cache:       {n} indicador{'es' if n != 1 else ''}"
    )


def limpiar_cache() -> None:
    """Vacía el cache de respuestas INEGI.

    El cache es de clase (`FuenteValidacionApi._cache`), compartido por todas
    las instancias — la siguiente llamada a `consultar_*` vuelve a descargar
    el indicador.
    """
    FuenteValidacionApi.limpiar_cache()
