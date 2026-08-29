"""Cálculo de índices del INPP."""

from __future__ import annotations

from replica_inpp.dominio.calculo.laspeyres_directo import LaspeyresDirecto
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada


def calcular_indice(
    canasta: CanastaINPP,
    serie: SerieNormalizada,
    agregacion: str,
    rubro: str | None = None,
    sin_petroleo: bool = False,
) -> ResultadoIndice:
    """Calcula el índice de una agregación/rubro de la canasta del INPP.

    Envuelve `LaspeyresDirecto` — única estrategia implementada hasta ahora
    (Etapa 2 sin encadenar; el encadenamiento 2025 todavía no existe).

    Args:
        canasta: canasta ya cargada (`cargar_canasta`).
        serie: serie ya cargada (`cargar_serie`), de la misma versión que
            `canasta` y del `recorte` correspondiente a `rubro`.
        agregacion: `"INPP"` (general), un nivel SCIAN (`"SECTOR"`,
            `"SUBSECTOR"`, `"RAMA"`, `"SUBRAMA"`, `"CLASE"` — nombre completo o
            abreviatura oficial) o `"MERCANCIAS_SERVICIOS"`. Se normaliza a
            mayúsculas.
        rubro: columna de peso/destino de producción (`"produccion_total"`,
            `"bienes_intermedios"`, `"bienes_finales"`, `"demanda_interna_total"`,
            `"demanda_interna_consumo"`, `"demanda_interna_capital"` o
            `"exportaciones"`). Si se omite, se infiere solo cuando el
            `recorte` de `serie` admite un único rubro válido — con
            `produccion_total` o `mercado_nacional` (recortes ambiguos, ver
            tabla abajo) hay que indicarlo.
        sin_petroleo: excluye el genérico `070` (Petróleo crudo) antes de
            agrupar — reproduce "INPP sin Petróleo y con Servicios" (BIE
            `910491`), verificado exacto contra el BIE real. NO es "Índice
            General Excluyendo Petróleo" (BIE `910493`) — esa serie diverge
            (~3.3 puntos).

    Combinaciones — `rubro` (filas) × `recorte` de `serie` (columnas). `✓` =
    válido, `—` = no aplica. `agregacion` no aparece en la matriz porque es
    ortogonal: combina libre con cualquier celda `✓` (opera sobre `codigo
    <nivel>`/`codigo sector` de `CanastaINPP`, columna siempre presente sin
    importar el rubro). Emparejamiento verificado contra la nota b/ al pie de
    cada hoja del xlsx de ponderadores de 2019 y 2025 (coinciden exacto entre
    sí), que dice explícito con qué precios se combina cada columna de peso —
    no siempre es `produccion_total`, como se asumía antes. 2012 trae una
    anomalía de copy-paste en 2 de sus 5 notas (`bienes_finales`/
    `exportaciones` mencionan "demanda interna", tema ajeno a esas hojas) que
    no se toma como señal confiable — ver `dominio/tipos.py::RUBROS_POR_RECORTE`::

        rubro \\ recorte          produccion_total  mercado_nacional  bienes_finales  mercado_exportacion
        produccion_total                 ✓                 —                —                 —
        bienes_intermedios               ✓                 —                —                 —
        demanda_interna_total            —                 ✓                —                 —
        demanda_interna_consumo          —                 ✓                —                 —
        demanda_interna_capital          —                 ✓                —                 —
        bienes_finales                   —                 —                ✓                 —
        exportaciones                    —                 —                —                 ✓

    Columnas `produccion_total` (2 `✓`) y `mercado_nacional` (3 `✓`) son
    ambiguas — ahí hay que indicar `rubro` a mano; `bienes_finales` y
    `mercado_exportacion` tienen un único `✓` y se infieren solas.
    `demanda_interna_total/consumo/capital` con `mercado_nacional` coincide,
    dentro del error de punto flotante (max_abs ≤ 3.27e-13; 73/73 dentro de
    tolerancia 0.0009), con los indicadores del BIE (validado 2026-08-27);
    `bienes_intermedios` también trae nota b/ "mercado nacional" pero no
    reproduce exacto con ninguna de las dos series probadas (mecanismo real
    sin identificar) — se deja en
    `produccion_total`. Mismo caso sin resolver para `bienes_finales`.

    Raises:
        VersionNoCoincide: `serie` viene de `cargar_serie` (trae `version` en
            metadata) y su versión no coincide con `canasta.version`. No se
            valida si `serie` se construyó a mano, sin esa metadata.
        InvarianteViolado: `agregacion` no es válida, o `rubro` no es válido
            para el `recorte` de `serie` (o es ambiguo y no se indicó).
        CanastaSinGenericos: tras filtrar pesos NaN/0 (y el 070 si
            `sin_petroleo=True`), no queda ningún genérico utilizable.
        ErrorCalculo: a la serie le faltan genéricos que el grupo necesita, no
            tiene ningún periodo dentro del rango vigente de la versión, o hay
            desbordamiento al ponderar la serie.
    """
    return LaspeyresDirecto().calcular(canasta, serie, agregacion, rubro, sin_petroleo)
