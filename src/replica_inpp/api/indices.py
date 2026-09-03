"""Cálculo de índices del INPP."""

from __future__ import annotations

from replica_inpp.dominio.calculo.laspeyres_directo import LaspeyresDirecto
from replica_inpp.dominio.calculo.laspeyres_encadenado import LaspeyresEncadenado
from replica_inpp.dominio.conversion import empalmar as _empalmar
from replica_inpp.dominio.conversion import rebasar as _rebasar
from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.periodos import periodo_desde_str
from replica_inpp.dominio.tipos import VersionCanasta


def calcular_indice(
    canasta: CanastaINPP,
    serie: SerieNormalizada,
    agregacion: str,
    rubro: str | None = None,
    *,
    incluir_petroleo: bool = True,
    referencia: ResultadoIndice | None = None,
) -> ResultadoIndice:
    """Calcula el índice de una agregación/rubro de la canasta del INPP.

    Calcula sin encadenar para canasta 2012/2019. Para canasta 2025 sí
    encadena, y para eso exige `referencia`: el mismo índice ya calculado
    con canasta 2019 (misma `agregacion`/`rubro`/`incluir_petroleo`). Sin
    `referencia`, rechaza el cálculo en vez de dar un número mal calculado
    sin avisar.

    Args:
        canasta: canasta ya cargada (`cargar_canasta`).
        serie: serie ya cargada (`cargar_serie`), misma versión que `canasta`.
            El indice calculado se infiere por el tipo de serie (`recorte`).
        agregacion: `"INPP"` (general), o nivel SCIAN (`"SECTOR"`,
            `"SUBSECTOR"`, `"RAMA"`, `"SUBRAMA"`, `"CLASE"`) o `"MERCANCIAS_SERVICIOS"`.
        rubro: parámetro específico para cuando serie cargada es del recorte
            `mercado_nacional` — ese recorte alimenta 4 índices distintos y
            hay que elegir cuál calcular: `demanda_interna_total`,
            `demanda_interna_consumo`, `demanda_interna_capital` o
            `bienes_intermedios`. Para los demás recortes
            (`produccion_total`, `bienes_finales`, `mercado_exportacion`) no
            hace falta — se infiere solo.
        incluir_petroleo: si `False`, excluye el genérico `070` (Petróleo
            crudo) antes de calcular el indice.
        referencia: solo si `canasta.version == 2025` — `ResultadoIndice` de
            la MISMA `agregacion`/`rubro`/`incluir_petroleo`, con
            `canasta.version=2019` y que cubra el periodo de traslape
            (jul-2025). Ignorado si `canasta.version != 2025`.

    Raises:
        InvarianteViolado: canasta 2025 sin `referencia`; `referencia` de
            otra `agregacion`/`rubro`/`incluir_petroleo` o que no sea de
            canasta 2019; `agregacion` inválida; o `rubro` inválido o
            ambiguo para el recorte de `serie`.
        VersionNoCoincide: `serie` y `canasta` son de versiones distintas
            (solo se detecta si `serie` viene de `cargar_serie`).
        CanastaSinGenericos: tras filtrar pesos vacíos (y el 070 si
            `incluir_petroleo=False`), no queda ningún genérico utilizable.
        ErrorCalculo: a la serie le faltan genéricos que el grupo necesita,
            no cubre ningún periodo válido, hay desbordamiento al ponderar,
            o (solo canasta 2025) falta el factor de encadenamiento,
            `referencia` no cubre el periodo de traslape, o algún grupo
            quedó sin factor por reclasificación entre versiones.

    Examples:
        >>> calcular_indice(canasta, serie, "INPP")
        >>> calcular_indice(canasta, serie, "INPP", "bienes_intermedios")
        >>> calcular_indice(canasta_25, serie_25, "INPP", referencia=res_2019)
    """
    if canasta.version == 2025:
        if referencia is None:
            raise InvarianteViolado(
                "calcular_indice con canasta.version=2025 requiere 'referencia' -- el "
                "ResultadoIndice de la misma agregacion/rubro/incluir_petroleo ya "
                "calculado con canasta.version=2019."
            )
        return LaspeyresEncadenado(referencia).calcular(
            canasta,
            serie,
            agregacion,
            rubro=rubro,
            incluir_petroleo=incluir_petroleo,
        )
    return LaspeyresDirecto().calcular(
        canasta, serie, agregacion, rubro=rubro, incluir_petroleo=incluir_petroleo
    )


def rebasar(
    resultado: ResultadoIndice,
    periodo_referencia: str,
    valor_base: float = 100.0,
) -> ResultadoIndice:
    """Reexpresa `resultado` para que `periodo_referencia` valga `valor_base`.

    Endógeno: reescala la serie propia de `resultado` (cada `indice` -- "INPP",
    código de sector, etc. -- por separado), sin comparar contra otra versión
    de canasta ni requerir un módulo de correspondencia de códigos entre
    versiones. Útil para llevar el resultado de una canasta más vieja (ej.
    2012, base Jun2012=100) a la referencia de otra (ej. Jul2019=100) antes de
    compararlos -- a diferencia del encadenamiento 2019→2025, este NO absorbe
    reclasificación de genéricos: solo reescala números ya agregados por
    `agregacion`/`rubro`.

    Args:
        resultado: índice ya calculado (`calcular_indice`) a reexpresar.
        periodo_referencia: `"Jul 2019"` -- texto canónico `"Mes AAAA"`.
        valor_base: cuánto valdrá `periodo_referencia`. 100.0 por convención.

    Raises:
        PeriodoNoInterpretable: `periodo_referencia` no es un periodo mensual
            reconocible.
        InvarianteViolado: `valor_base` no es finito y positivo; el periodo no
            existe en `resultado` para ningún `indice`; o el valor base de
            algún `indice` ahí es NaN o exactamente 0.
        ErrorCalculo: `valor_base` es finito pero produce overflow (`inf`) al
            aplicarlo contra algún `indice_replicado` real.

    Examples:
        >>> rebasar(resultado_2012, "Jul 2019")
        >>> rebasar(resultado, "Ene 2020", valor_base=50.0)
    """
    return _rebasar(resultado, periodo_desde_str(periodo_referencia), valor_base)


def empalmar(
    resultados: list[ResultadoIndice],
    forzar: bool = False,
    version_nombres: VersionCanasta | None = None,
) -> ResultadoIndice:
    """Concatena tramos de distinta versión de canasta en un único `ResultadoIndice`.

    En la frontera (periodo de traslape entre versiones contiguas, ver
    `RANGOS_CANASTAS`), el tramo anterior posee `(periodo, indice)` si esa fila
    existe en él; si no, el posterior la aporta. Para que la frontera sea
    numéricamente coherente, el tramo anterior normalmente necesita `rebasar`
    antes a la escala del posterior (ej. `rebasar(resultado_2012, "Jul 2019")`
    antes de `empalmar([resultado_2012, resultado_2019])`) -- `empalmar`
    valida esto: compara `indice_replicado` de ambos tramos en la frontera con
    una tolerancia interna fija, y rechaza (o advierte con `forzar=True`) si
    no coinciden.

    No normaliza `indice` (nombre/código de categoría) entre versiones -- a
    diferencia de `replica-inpc-mx`, este repo no tiene módulo de
    correspondencia de códigos entre versiones. Solo es seguro para
    agregaciones donde `indice` ya es estable entre las versiones que se
    empalman: `"INPP"` y `"SECTOR"` (código de 2 dígitos) lo son;
    SUBSECTOR/RAMA/SUBRAMA/CLASE o genérico no están garantizados si hay
    reclasificación SCIAN real ahí.

    Args:
        resultados: al menos 2 `ResultadoIndice`, cada uno de una versión de
            canasta distinta, mismos `agregacion`/`rubro`/`incluir_petroleo`.
        forzar: permite empalmar cuando `periodo_referencia` de un tramo no
            coincide con la frontera con el siguiente, o cuando `indice_replicado`
            difiere entre tramos en la frontera más allá de la tolerancia
            interna (en ambos casos, `UserWarning` en vez de rechazar). Los
            tramos recién calculados (sin rebasar) tienen
            `periodo_referencia=None` y no disparan la primera guardia.
        version_nombres: versión de canasta cuyo `nombres` original tiene
            precedencia MÁXIMA en la columna `nombre` combinada (gana incluso
            sobre tramos más nuevos) -- ej. `empalmar([r2012, r2019, r2025],
            version_nombres=2012)` conserva los nombres de 2012 aunque 2019/2025
            también los traigan. `None` (default) = precedencia del más
            reciente. Para un `indice` que esa versión no nombra, sigue
            aplicando el fallback normal entre el resto de los tramos. Robusto
            a empalme incremental: `empalmar([empalmar([r2012, r2019]), r2025],
            version_nombres=2012)` da el mismo resultado que empalmar los 3 de
            una -- no importa si `r2012` llegó directo o ya pasó por un
            `empalmar` anterior.

    Raises:
        InvarianteViolado: menos de 2 `resultados`; no todos comparten
            `(agregacion, rubro, incluir_petroleo)`; los periodos no forman una
            topología PATH (cada par consecutivo comparte exactamente 1
            periodo, ninguno no consecutivo comparte alguno); un tramo trae
            `periodo_referencia` que no coincide con la frontera sin `forzar`;
            `indice_replicado` de algún `indice` compartido en la frontera
            difiere entre tramos más allá de la tolerancia interna (escala no
            coherente -- falta `rebasar()` antes) sin `forzar`; o
            `version_nombres` no corresponde a ningún tramo de `resultados`
            (ni a su historial de empalmes previos), o ese tramo no tiene
            `nombres` asignados.

    Examples:
        >>> empalmar([resultado_2012, resultado_2019, resultado_2025])
        >>> empalmar([r2012, r2019, r2025], version_nombres=2012)
    """
    return _empalmar(resultados, forzar=forzar, version_nombres=version_nombres)
