"""Graficación de `ResultadoIndice`/`ResultadoVariacion` sobre plotnine.

Sin `Protocol` en `aplicacion/puertos/`: igual que `replica-inpc-mx`, ningún
componente de `dominio/` o `aplicacion/` consume un graficador internamente -- no
hay abstracción que enforzar.

Solo pipeline de línea -- INPP no tiene `ResultadoIncidencia` (barras apiladas)
todavía, y es mensual-only (sin rama quincenal que portar de INPC).
"""

from __future__ import annotations

import pandas as pd
from plotnine import (
    aes,
    annotate,
    element_blank,
    element_text,
    geom_hline,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_linetype_manual,
    scale_x_datetime,
    scale_y_continuous,
    theme,
    theme_bw,
)

from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.infraestructura.graficacion._prepocesamiento import (
    _ETIQUETA_Y_VARIACION,
    _VALOR_BASE,
    _VALOR_BASE_VARIACION,
    _aplanar_resultado,
    _breaks_y,
    _breaks_y_etiquetas_x,
    _colores_y_etiquetas,
    _datos_para_puntos,
    _descartar_grupos_no_graficables,
    _etiqueta_y_indice,
    _etiquetas_linetype,
    _grupos_por_tamano,
    _indices_comparacion_repetidos,
    _ordenar_series_dibujo,
    _particionar_series,
    _primero_y_ultimo_para_anotar,
    _recortar_tramo,
    _reservar_colores_comparacion,
    _titulo,
)


def _construir_grafica_linea(
    datos: pd.DataFrame,
    resultado: ResultadoIndice | ResultadoVariacion,
    *,
    columna_valor: str = "indice_replicado",
    valor_base: float = _VALOR_BASE,
    etiqueta_y: str | None = None,
    colores_reservados: dict[str, str] | None = None,
) -> ggplot:
    """Arma el `ggplot` completo a partir de `datos` ya aplanados (y opcionalmente particionados).

    `resultado` solo se usa para la etiqueta del eje Y cuando `etiqueta_y` no
    viene explícita (índices: `periodo_referencia` si fue rebasado) -- los
    datos a dibujar salen enteros de `datos`. `columna_valor`, `valor_base` y
    `etiqueta_y` permiten reusar el mismo armado para variaciones
    (`variacion_pp`, base 0, `"Variación (pp)"`).

    `colores_reservados` (de `_reservar_colores_comparacion`, calculado UNA
    vez por `_graficar_indice`/`_graficar_variacion` sobre `datos` completo,
    antes de particionar) fija el color de los índices de `comparacion` que
    se repiten en varios paneles -- sin esto, cada panel recalculaba su
    propia paleta local y la MISMA referencia cambiaba de color entre
    imágenes. El resto de colores se sigue calculando LOCAL a este panel
    (`_colores_y_etiquetas`, con `reservados` como desplazamiento) -- nunca
    global sobre todos los índices a la vez, que reintroduciría colisiones
    cuando la unión total supera la capacidad de la paleta.
    """
    datos = datos.copy()
    datos["indice"] = _ordenar_series_dibujo(datos["indice"], datos["linetype"])
    titulo = _titulo(datos)

    series = list(pd.unique(datos["indice"]))
    # `comparacion` puede compartir color con `resultado` (mismo rubro de
    # agregación INPP, ej. con/sin petróleo) -- ahí el color deja de distinguir
    # y hace falta la leyenda de `linetype` para saber cuál línea es cuál. Se
    # muestra SOLO cuando de verdad colisionan (algún `indice` aparece en
    # ambos grupos): si color ya distingue (ej. SECTOR desglosado + INPP de
    # comparación), una leyenda genérica "Resultado"/"Comparación" no aporta
    # nada nuevo -- el color y su propia etiqueta ya lo dicen.
    solidas = set(datos.loc[datos["linetype"] == "solid", "indice"])
    punteadas = set(datos.loc[datos["linetype"] == "dashed", "indice"])
    colisiona = bool(solidas & punteadas)
    mostrar_leyenda = len(series) > 1 or colisiona
    breaks_x, etiquetas_x = _breaks_y_etiquetas_x(datos)
    breaks_y = _breaks_y(datos, columna_valor, valor_base)
    etiqueta_y_final = etiqueta_y if etiqueta_y is not None else _etiqueta_y_indice(resultado)
    colores, etiquetas_leyenda = _colores_y_etiquetas(series, colores_reservados)

    grafica = ggplot(
        datos, aes(x="periodo_ts", y=columna_valor, color="indice", linetype="linetype")
    )
    if breaks_y[0] <= valor_base <= breaks_y[-1]:
        # Fuera de rango (ej. tramo recortado donde todo el valor quedó por
        # encima o por debajo de la base): la línea de base ya no aporta nada,
        # se omite en vez de dibujarla fuera del panel visible.
        grafica = grafica + geom_hline(
            yintercept=valor_base, linetype="dashed", color="grey", size=0.3
        )
    # La capa lineal recibe solo los grupos que puede dibujar: con una única
    # observación no traza nada y además emite `PlotnineWarning`, así que
    # pasarle esos grupos solo produce ruido. `na_rm=True` calla el aviso de
    # plotnine al recortar un NaN terminal (Ene sin_datos, ej.) -- no cambia
    # qué se dibuja: el NaN interior sigue intacto para el hueco (matplotlib
    # lo corta solo), el terminal se recorta igual con o sin esto.
    grafica = (
        grafica
        + geom_line(
            data=_grupos_por_tamano(datos, columna_valor, solitarios=False), size=0.5, na_rm=True
        )
        + scale_linetype_manual(
            values={"solid": "solid", "dashed": "dashed"},
            labels=_etiquetas_linetype(datos),
            guide="legend" if colisiona else None,
        )
    )
    # Los puntos no van siempre: en un tramo largo, uno por periodo satura la
    # línea. `_datos_para_puntos` decide qué filas los llevan -- todas si el
    # tramo cabe en un año, o solo las series de una única observación, que
    # `geom_line` deja invisibles.
    datos_puntos = _datos_para_puntos(datos, columna_valor)
    if datos_puntos is not None:
        grafica = grafica + geom_point(data=datos_puntos, size=0.5)

    primero_ultimo = _primero_y_ultimo_para_anotar(datos, series, columna_valor)
    if primero_ultimo is not None:
        # Serie única (sin leyenda): hay espacio para el numerito, y el margen
        # más ancho existe justo para que ese texto no se corte contra el borde.
        expand_x, expand_y = (0.045, 0.045), (0.02, 0.02)
        primero, ultimo = primero_ultimo
        grafica = (
            grafica
            + annotate(
                "text",
                x=primero["periodo_ts"],
                y=primero[columna_valor],
                label=f"{primero[columna_valor]:.2f}",
                ha="right",
                va="center",
                size=6,
                color=colores[primero["indice"]],
            )
            + annotate(
                "text",
                x=ultimo["periodo_ts"],
                y=ultimo[columna_valor],
                label=f"{ultimo[columna_valor]:.2f}",
                ha="left",
                va="center",
                size=6,
                color=colores[ultimo["indice"]],
            )
        )
    else:
        # Varias series: sin numerito que proteger, margen mínimo.
        expand_x, expand_y = (0.01, 0.01), (0.01, 0.01)

    return (
        grafica
        # `limits` fuerza el rango completo de `breaks_x` -- sin esto, cuando
        # lo único graficable (ej. un punto solitario anotado en sus propias
        # coordenadas reales) cae angosto, plotnine encoge el rango de
        # coordenadas al de las geometrías y censura los `breaks`/`labels`
        # que quedan afuera, con conteos que ya no coinciden
        # (`PlotnineError: Breaks and labels are different lengths`).
        + scale_x_datetime(
            breaks=breaks_x,  # type: ignore[arg-type]
            labels=etiquetas_x,
            expand=expand_x,  # type: ignore[arg-type]
            limits=(min(breaks_x), max(breaks_x)),  # type: ignore[arg-type]
        )
        + scale_y_continuous(breaks=breaks_y, expand=expand_y)
        + scale_color_manual(
            values=colores,
            labels=etiquetas_leyenda,
            guide="legend" if len(series) > 1 else None,
        )
        + labs(title=titulo, x="Periodo", y=etiqueta_y_final)
        + theme_bw()
        + theme(
            axis_title=element_text(size=8),
            axis_text_x=element_text(rotation=30, ha="right", size=6),
            axis_text_y=element_text(size=6),
            legend_position="bottom" if mostrar_leyenda else "none",
            legend_box="horizontal",
            legend_title=element_blank(),
            legend_text=element_text(size=6),
            legend_key_size=8,
            legend_box_spacing=0.01,
            plot_margin=0.005,
            figure_size=(8, 4),
            dpi=300,
        )
    )


def _graficar_indice(
    resultado: ResultadoIndice,
    comparacion: ResultadoIndice | None,
    desde: PeriodoMensual | None,
    hasta: PeriodoMensual | None,
) -> None:
    """Arma y dibuja uno o varios `ggplot` de `resultado` (+ `comparacion` si viene).

    `desde`/`hasta`, si vienen, recortan el tramo ANTES de particionar -- el
    primer/último valor anotado (serie única) y las particiones (varias series)
    ya reflejan solo el tramo pedido, no el histórico completo.

    Descarta categorías sin ningún valor finito en el tramo recortado ANTES de
    particionar (ver `_descartar_grupos_no_graficables`) -- así no consumen un
    lugar en el reparto de colores/imágenes por algo que no va a dibujar nada,
    y si NINGUNA categoría conserva un valor finito, falla explícito en vez de
    producir una imagen vacía.

    Más de una sola imagen cuando el espacio de colores no alcanza para todos
    los índices (ver `_particionar_series`). Cuando eso pasa por repetir
    `comparacion` completa en cada panel, sus colores se reservan una sola
    vez acá (`_reservar_colores_comparacion`) y se pasan a cada panel -- así
    la misma referencia no cambia de color entre imágenes.
    """
    datos = _aplanar_resultado(resultado, comparacion)
    datos = _recortar_tramo(datos, desde, hasta)
    datos = _descartar_grupos_no_graficables(datos, "indice_replicado")
    colores_reservados = _reservar_colores_comparacion(_indices_comparacion_repetidos(datos) or [])
    for parte in _particionar_series(datos):
        _construir_grafica_linea(parte, resultado, colores_reservados=colores_reservados).draw(
            show=True
        )


def _graficar_variacion(
    resultado: ResultadoVariacion,
    comparacion: ResultadoVariacion | None,
    desde: PeriodoMensual | None,
    hasta: PeriodoMensual | None,
) -> None:
    """Ídem `_graficar_indice` para un `ResultadoVariacion`: `variacion_pp` en el eje Y, base en 0."""
    datos = _aplanar_resultado(resultado, comparacion)
    datos = _recortar_tramo(datos, desde, hasta)
    datos = _descartar_grupos_no_graficables(datos, "variacion_pp")
    colores_reservados = _reservar_colores_comparacion(_indices_comparacion_repetidos(datos) or [])
    for parte in _particionar_series(datos):
        _construir_grafica_linea(
            parte,
            resultado,
            columna_valor="variacion_pp",
            valor_base=_VALOR_BASE_VARIACION,
            etiqueta_y=_ETIQUETA_Y_VARIACION,
            colores_reservados=colores_reservados,
        ).draw(show=True)


def graficar(
    resultado: ResultadoIndice | ResultadoVariacion,
    comparacion: ResultadoIndice | ResultadoVariacion | None = None,
    desde: PeriodoMensual | None = None,
    hasta: PeriodoMensual | None = None,
) -> None:
    """Grafica un `ResultadoIndice` o `ResultadoVariacion`; no devuelve nada, dibuja directo.

    Detecta el tipo de `resultado` y dispara el pipeline correspondiente --
    línea para índice (base 100, eje Y "Indice") y variación (base 0, eje Y
    "Variación (pp)"), mismo punto de entrada para ambos.

    Cada `indice` distinto que trae `resultado` (ej. cada código SCIAN de una
    `agregacion`) se dibuja como su propia línea de color. Cuando `agregacion ==
    "INPP"`, `indice` no distingue nada (es constante) -- se colorea por `rubro`
    en su lugar, con colores fijos reservados (ver `_colores_y_etiquetas`). Si
    `resultado` (+ `comparacion`) trae más de 8 categorías propias, se genera
    más de una imagen -- lo que venga de `comparacion` se repite completo en
    todas.

    Args:
        resultado: Resultado principal a graficar -- `ResultadoIndice` o
            `ResultadoVariacion`.
        comparacion: Un segundo resultado opcional, del MISMO tipo que
            `resultado`. Si ambos son `ResultadoVariacion`, además deben
            compartir `manifiesto.clase` (ej. no se puede comparar una
            variación mensual contra una anual). Se superpone en el mismo panel
            con línea PUNTEADA para distinguirse de `resultado` (línea sólida)
            aunque comparta color. Si algo de esto no se cumple, no se levanta
            excepción: se imprime un mensaje de error y no se dibuja nada.
        desde: Periodo inicial del tramo a graficar. `None` = desde el primer
            periodo disponible.
        hasta: Periodo final del tramo a graficar. `None` = hasta el último
            periodo disponible.
    """
    if isinstance(resultado, ResultadoIndice):
        if comparacion is not None:
            if not isinstance(comparacion, ResultadoIndice):
                print("Error, comparacion debe ser del mismo tipo que resultado (ResultadoIndice).")
                return
        _graficar_indice(resultado, comparacion, desde, hasta)
    elif isinstance(resultado, ResultadoVariacion):
        if comparacion is not None:
            if not isinstance(comparacion, ResultadoVariacion):
                print(
                    "Error, comparacion debe ser del mismo tipo que resultado (ResultadoVariacion)."
                )
                return
            if comparacion.manifiesto.clase != resultado.manifiesto.clase:
                print(
                    "Error, comparacion debe tener la misma clase_variacion (frecuencia) que "
                    f"resultado ('{resultado.manifiesto.clase}' != '{comparacion.manifiesto.clase}')."
                )
                return
        _graficar_variacion(resultado, comparacion, desde, hasta)
    else:
        print("Error, se esperaba un ResultadoIndice o ResultadoVariacion.")
