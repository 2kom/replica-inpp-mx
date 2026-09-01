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

    Envuelve `LaspeyresDirecto` (Etapa 2, sin encadenar) para `canasta.version`
    2012/2019. Para `canasta.version == 2025` despacha a `LaspeyresEncadenado`
    (Etapa 3, encadenamiento) en su lugar — requiere `referencia`, el
    `ResultadoIndice` de la MISMA `agregacion`/`rubro`/`incluir_petroleo` ya
    calculado con `canasta.version=2019` (típicamente con una llamada previa
    a `calcular_indice`), del que sale `factor_h`. Sin `referencia`, rechaza
    el cálculo en vez de calcular con `LaspeyresDirecto` sobre 2025: eso daría
    un número mal calculado sin avisar (mezcla la serie 2025, que sigue en
    escala absoluta continua de 2019, con ponderadores 2025).

    Args:
        canasta: canasta ya cargada (`cargar_canasta`).
        serie: serie ya cargada (`cargar_serie`), de la misma versión que
            `canasta` y del `recorte` correspondiente a `rubro`.
        agregacion: `"INPP"` (general), un nivel SCIAN (`"SECTOR"`,
            `"SUBSECTOR"`, `"RAMA"`, `"SUBRAMA"`, `"CLASE"` — nombre completo o
            abreviatura oficial) o `"MERCANCIAS_SERVICIOS"`.
        rubro: columna de peso/destino de producción. Se infiere solo del
            `recorte` de `serie` — `produccion_total`/`bienes_finales` se
            llaman igual que su recorte, `mercado_exportacion` infiere
            `"exportaciones"` (único caso con nombre distinto al recorte).
            Único caso ambiguo: `mercado_nacional` acepta 4 rubros
            (`demanda_interna_total`, `demanda_interna_consumo`,
            `demanda_interna_capital`, `bienes_intermedios`) y exige
            indicarlo a mano — ver `dominio/tipos.py::RUBROS_POR_RECORTE`
            para el resto del mapeo.
        incluir_petroleo: si `False`, excluye el genérico `070` (Petróleo
            crudo) antes de agrupar, sin normalizar aparte — la propia
            división `Σ(w·índice)/Σw` ya renormaliza sobre los genéricos
            restantes.
        referencia: solo si `canasta.version == 2025` — `ResultadoIndice` de
            la MISMA `agregacion`/`rubro`/`incluir_petroleo`, con
            `canasta.version=2019` y que cubra el periodo de traslape
            (jul-2025). Ignorado si `canasta.version != 2025`.

    Raises:
        InvarianteViolado: `canasta.version == 2025` sin `referencia` (es
            obligatoria en ese caso); `referencia` sin un manifiesto con
            `version=2019` y la MISMA `agregacion`/`rubro`/`incluir_petroleo`
            que se pide acá (ver
            `dominio/calculo/laspeyres_encadenado.py::LaspeyresEncadenado`);
            `agregacion` no es válida; o `rubro` no es válido para el
            `recorte` de `serie` (o es ambiguo y no se indicó).
        VersionNoCoincide: `serie` viene de `cargar_serie` (trae `version` en
            metadata) y su versión no coincide con `canasta.version`. No se
            valida si `serie` se construyó a mano, sin esa metadata.
        CanastaSinGenericos: tras filtrar pesos NaN/0 (y el 070 si
            `incluir_petroleo=False`), no queda ningún genérico utilizable.
        ErrorCalculo: a la serie le faltan genéricos que el grupo necesita, no
            tiene ningún periodo dentro del rango vigente de la versión, hay
            desbordamiento al ponderar la serie, o (solo `canasta.version ==
            2025`) falta el factor de encadenamiento, `referencia` no cubre
            el periodo de traslape, o algún grupo no tiene `factor_h` en el
            tramo anterior (reclasificación de agrupación entre versiones).
    """
    if canasta.version == 2025:
        if referencia is None:
            raise InvarianteViolado(
                "calcular_indice con canasta.version=2025 requiere 'referencia' -- el "
                "ResultadoIndice de la misma agregacion/rubro/incluir_petroleo ya calculado con "
                "canasta.version=2019, del que LaspeyresEncadenado saca factor_h (ver "
                "dominio/calculo/laspeyres_encadenado.py)."
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
    compararlos -- a diferencia del encadenamiento 2019→2025
    (`LaspeyresEncadenado`), este NO absorbe reclasificación de genéricos: solo
    reescala números ya agregados por `agregacion`/`rubro`.

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
    """
    return _rebasar(resultado, periodo_desde_str(periodo_referencia), valor_base)


def empalmar(resultados: list[ResultadoIndice], forzar: bool = False) -> ResultadoIndice:
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

    Raises:
        InvarianteViolado: menos de 2 `resultados`; no todos comparten
            `(agregacion, rubro, incluir_petroleo)`; los periodos no forman una
            topología PATH (cada par consecutivo comparte exactamente 1
            periodo, ninguno no consecutivo comparte alguno); un tramo trae
            `periodo_referencia` que no coincide con la frontera sin `forzar`;
            o `indice_replicado` de algún `indice` compartido en la frontera
            difiere entre tramos más allá de la tolerancia interna (escala no
            coherente -- falta `rebasar()` antes) sin `forzar`.
    """
    return _empalmar(resultados, forzar=forzar)
