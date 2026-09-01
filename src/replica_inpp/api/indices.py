"""Cálculo de índices del INPP."""

from __future__ import annotations

from replica_inpp.dominio.calculo.laspeyres_directo import LaspeyresDirecto
from replica_inpp.dominio.calculo.laspeyres_encadenado import LaspeyresEncadenado
from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada


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
