"""Graficación de resultados."""

from __future__ import annotations

from replica_inpp.dominio.errores import PeriodoNoDisponible
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion
from replica_inpp.dominio.periodos import PeriodoMensual, periodo_desde_str
from replica_inpp.infraestructura.graficacion.graficador import graficar as _graficar


def _periodos_disponibles(
    resultado: ResultadoIndice | ResultadoVariacion,
    comparacion: ResultadoIndice | ResultadoVariacion | None,
) -> set[PeriodoMensual]:
    """Periodos que pueden usarse como `desde`/`hasta`: unión de `resultado` y `comparacion`.

    `resultado` y `comparacion` se concatenan en un solo DataFrame y se dibujan
    en el mismo panel, así que un límite presente solo en `comparacion` sigue
    recortando algo real.
    """
    periodos = set(resultado.resultado.largo.index.get_level_values("periodo"))
    if comparacion is not None:
        periodos |= set(comparacion.resultado.largo.index.get_level_values("periodo"))
    return periodos


def graficar(
    resultado: ResultadoIndice | ResultadoVariacion,
    comparacion: ResultadoIndice | ResultadoVariacion | None = None,
    desde: str | None = None,
    hasta: str | None = None,
) -> None:
    """Grafica `resultado` (índice o variación); no devuelve nada, muestra la(s) imagen(es) directo.

    Cada `indice` distinto que trae `resultado` (ej. cada código SCIAN de una
    `agregacion`) se dibuja como su propia línea de color. Cuando `agregacion ==
    "INPP"` (el índice general, sin subcategoría), la línea se colorea por
    `rubro` en su lugar -- `indice` ahí no distingue nada. Si `resultado` (+
    `comparacion`) trae más de 8 categorías propias, se generan varias imágenes.
    Si `resultado` es un `ResultadoVariacion`, el eje Y muestra `variacion_pp`
    con base en 0 en vez del índice con base 100.

    Args:
        resultado: Resultado principal a graficar -- `ResultadoIndice` o
            `ResultadoVariacion`.
        comparacion: Un segundo resultado opcional. Si algo de lo que sigue no
            se cumple, no se levanta excepción: se imprime un mensaje de error
            y no se dibuja nada.

            Debe ser del MISMO tipo que `resultado`, y si ambos son
            `ResultadoVariacion` además deben compartir `manifiesto.clase` (ej.
            no se puede comparar una variación mensual contra una anual). Se
            superpone en el mismo panel con línea PUNTEADA para distinguirse de
            `resultado` (línea sólida) aunque comparta color -- y siempre se
            dibuja por encima de las demás líneas, sea cual sea el parámetro por
            el que entre.
        desde: Periodo inicial del tramo a mostrar (ej. `"Ene 2018"`) -- recorta
            el eje X. Tiene que existir de verdad en los datos (en `resultado` o
            en `comparacion`); no basta con que el texto tenga formato válido.
            `None` = desde el primer periodo disponible.
        hasta: Igual que `desde`, pero el límite final del tramo. `None` =
            hasta el último periodo disponible.

    Raises:
        PeriodoNoInterpretable: `desde` o `hasta` no son un periodo mensual
            reconocible (texto canónico `"Mes AAAA"`).
        PeriodoNoDisponible: `desde` o `hasta` tienen formato válido pero no
            están presentes en los datos a graficar.
    """
    periodo_desde = periodo_desde_str(desde) if desde is not None else None
    periodo_hasta = periodo_desde_str(hasta) if hasta is not None else None

    if periodo_desde is not None or periodo_hasta is not None:
        disponibles = _periodos_disponibles(resultado, comparacion)
        for periodo, nombre in ((periodo_desde, "desde"), (periodo_hasta, "hasta")):
            if periodo is not None and periodo not in disponibles:
                raise PeriodoNoDisponible(
                    f"'{nombre}' ({periodo}) no está presente en los datos a graficar."
                )

    _graficar(resultado, comparacion, periodo_desde, periodo_hasta)
