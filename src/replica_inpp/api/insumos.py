"""IO de insumos: carga de series desde CSV."""

from __future__ import annotations

from pathlib import Path

from replica_inpp.dominio.errores import InvarianteViolado, VersionNoCoincide
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.infraestructura.csv.lector_series_csv import LectorSeriesCsv

_VERSIONES_VALIDAS = (2012, 2019, 2025)


def _validar_version(version: int) -> None:
    if version not in _VERSIONES_VALIDAS:
        raise InvarianteViolado(f"version fuera de {_VERSIONES_VALIDAS}: {version}")


def cargar_serie(ruta: str, version: int) -> SerieNormalizada:
    """Carga una serie de índices del INPP desde un CSV del BIE.

    `recorte` no es parámetro: se detecta solo a partir del propio archivo
    (prefijo numérico embebido en `Título` para los archivos planos; fijo a
    `produccion_total` si el archivo es jerárquico — solo esa carpeta trae
    variante `ae`).

    `version` SÍ se valida contra el propio archivo: el año de la base
    ("Base <mes> <AAAA>=100" en el `Título`) coincide con la versión de
    canasta. Si no coincide con el `version` recibido, truena
    `VersionNoCoincide` — no valida todavía cobertura ni tramo de periodos
    contra una Canasta (no existe aún una Canasta de INPP para eso).

    Args:
        ruta: CSV de series del BIE, orientación horizontal o vertical
            (se detecta sola), con o sin columnas de metadata (variante
            `m`/`nm` — el preámbulo de 5 líneas del BIE es igual en ambas),
            plano o jerárquico (`ae`).
        version: 2012, 2019 o 2025.

    Raises:
        InvarianteViolado: la versión no es una de las tres válidas, o el
            DataFrame extraído viola alguna invariante de `SerieNormalizada`.
        VersionNoCoincide: la versión recibida no coincide con la base
            detectada en el `Título` del archivo.
        ArchivoNoEncontrado: la ruta no existe.
        ArchivoVacio: el CSV no tiene contenido.
        ArchivoCorrupto: el CSV no se puede parsear, mezcla prefijos de
            recorte distintos en una misma tabla, o mezcla años de base distintos.
        EncodingNoLegible: el archivo no es legible con los encodings soportados.
        OrientacionNoDetectable: no se pudo determinar si es horizontal o vertical.
        SerieVacia: no se encontraron genéricos en el archivo.
    """
    _validar_version(version)
    df = LectorSeriesCsv().leer(Path(ruta))

    version_detectada = df.attrs.get("version_detectada")
    if version_detectada is not None and version_detectada != version:
        raise VersionNoCoincide(
            f"version={version} no coincide con la base detectada en el archivo "
            f"({version_detectada}): {ruta}"
        )

    df.attrs["version"] = version
    return SerieNormalizada(df, recorte=df.attrs["recorte"])
