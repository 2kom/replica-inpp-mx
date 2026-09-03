"""Helpers puros para armar los `ggplot` de `graficador.py` -- sin dibujar nada acá.

Port de `replica-inpc-mx/infraestructura/graficacion/_prepocesamiento.py`, podado
al pipeline de línea (`ResultadoIndice`/`ResultadoVariacion`) -- sin barras
apiladas (INPP no tiene `ResultadoIncidencia` todavía) y sin rama quincenal (INPP
es mensual-only). Dos adaptaciones reales de dominio, no solo poda:

- INPP agrupa por 2 ejes (`agregacion`, `rubro`) en vez de 1 (`tipo` en INPC) --
  `_titulo` arma la clave compuesta.
- Cuando `agregacion == "INPP"`, `indice` es constante (sin subcategoría SCIAN) y
  no sirve para distinguir series -- `_aplanar_resultado` lo sobrescribe con
  `rubro`, y `_colores_y_etiquetas` reserva colores fijos para los 7 valores
  conocidos (`RUBRO_A_COLUMNA_PESO`), en vez del color negro fijo único que INPC
  reserva para `"INPC"` (sin análogo acá -- no hay una serie de referencia
  universal, cada rubro es un agregado distinto).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from mizani.breaks import breaks_extended

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import RUBRO_A_COLUMNA_PESO

_MAX_ETIQUETAS_EJE_X = 20
_N_ETIQUETAS_EJE_Y = 10
# Distancia mínima entre un break intermedio y un extremo forzado, como
# fracción del rango del eje — por debajo de esto las etiquetas se enciman.
_SEPARACION_MINIMA_BREAKS = 0.035
_VALOR_BASE = 100.0
_VALOR_BASE_VARIACION = 0.0
_ETIQUETA_Y_VARIACION = "Variación (pp)"
_LINETYPE_PRINCIPAL = "solid"
_LINETYPE_COMPARACION = "dashed"
_PALETA_OTROS_TIPOS = (
    "#2a78d6",  # azul
    "#eb6834",  # naranja
    "#1baf7a",  # aqua
    "#eda100",  # amarillo
    "#e87ba4",  # magenta
    "#008300",  # verde
    "#4a3aa7",  # violeta
    "#e34948",  # rojo
)
# Mismo tope que colores validados en _PALETA_OTROS_TIPOS -- pasado esto, no hay
# color distinguible que asignar, toca partir en varias imágenes.
_MAX_SERIES_POR_IMAGEN = len(_PALETA_OTROS_TIPOS)
_MAX_CARACTERES_LEYENDA = 34
# Un año de periodos mensuales: hasta acá los puntos caben sobre la línea sin
# saturarla, y marcan cada observación real (útil en tramos cortos). INPP es
# mensual-only -- a diferencia de INPC no hay tope quincenal que distinguir.
# Ene 2024 a Ene 2025 son 13 meses, no 12 (un año va de extremo a extremo).
_MAX_PERIODOS_CON_PUNTOS = 13

# Color negro fijo, reservado para cuando `agregacion == "INPP"` -- ahí `indice`
# ya no distingue nada (ver `_aplanar_resultado`, sobrescrito con `rubro`) y NO
# hay variedad de color por rubro: sea cual sea el rubro o el `incluir_petroleo`,
# la línea siempre es negra -- única condición es que `agregacion == "INPP"`.
# Sin esto, dos gráficas que comparan el mismo par de rubros mostrarían colores
# distintos según el orden de aparición, y el color dejaría de significar
# "índice general" -- la ambigüedad que forzó esta regla (ver `_titulo`: la
# leyenda pasa a depender de `linetype`, no de color, cuando colisiona).
_COLOR_INPP = "black"
_RUBROS_CONOCIDOS: frozenset[str] = frozenset(RUBRO_A_COLUMNA_PESO)
# Etiqueta legible por rubro (ej. "demanda_interna_consumo" -> "Demanda interna
# consumo") -- el valor crudo de `rubro` es un identificador interno, no texto
# para una persona. Se usa tanto en el título como en la leyenda.
_ETIQUETAS_RUBRO: dict[str, str] = {
    rubro: rubro.replace("_", " ").capitalize() for rubro in RUBRO_A_COLUMNA_PESO
}
# Etiqueta legible por `linetype` -- distingue `resultado` de `comparacion`
# cuando el color no alcanza (ej. mismo rubro, con/sin petróleo: ambos negros,
# solo el trazo los distingue). Se muestra en su propia leyenda solo cuando hay
# `comparacion` (ver `graficador._construir_grafica_linea`).
_ETIQUETAS_LINETYPE: dict[str, str] = {
    _LINETYPE_PRINCIPAL: "Resultado",
    _LINETYPE_COMPARACION: "Comparación",
}


def _segmento(agregacion: str, rubro: str, incluir_petroleo: bool | None) -> str:
    """Texto de una combinación `(agregacion, rubro, incluir_petroleo)` -- usado en título y leyenda.

    `incluir_petroleo=None` (caso `ResultadoVariacion`, que no lo trae en su
    manifiesto) omite ese segmento en vez de forzar un texto sin sentido.
    """
    rubro_legible = _ETIQUETAS_RUBRO.get(rubro, rubro)
    segmento = f"{agregacion} {rubro_legible}"
    if incluir_petroleo is not None:
        segmento += " con petróleo" if incluir_petroleo else " sin petróleo"
    return segmento


def _titulo(datos: pd.DataFrame) -> str:
    """Título: une cada combinación única `(agregacion, rubro, incluir_petroleo)` presente, en orden de aparición."""
    combinaciones = datos[["agregacion", "rubro", "incluir_petroleo"]].drop_duplicates()
    partes = [
        _segmento(
            fila["agregacion"],
            fila["rubro"],
            fila["incluir_petroleo"] if pd.notna(fila["incluir_petroleo"]) else None,
        )
        for _, fila in combinaciones.iterrows()
    ]
    return " + ".join(partes)


def _etiquetas_linetype(datos: pd.DataFrame) -> dict[str, str]:
    """Etiqueta de leyenda por `linetype`: la combinación propia de ese grupo (agregación+rubro+petróleo).

    Se usa en vez del rol genérico `_ETIQUETAS_LINETYPE` ("Resultado"/
    "Comparación") cuando `resultado` y `comparacion` COLISIONAN en `indice`
    (mismo color, ej. mismo rubro de `agregacion=="INPP"` con/sin petróleo) --
    ahí un rol genérico no dice cuál línea es cuál, pero la combinación real sí
    (ej. "INPP Produccion total con petróleo" vs "... sin petróleo"). Si algún
    `linetype` no está presente en `datos` (sin `comparacion`), conserva el
    genérico de respaldo -- no importa, esa leyenda no se muestra en ese caso
    (ver `graficador._construir_grafica_linea`).
    """
    etiquetas = dict(_ETIQUETAS_LINETYPE)
    for valor in (_LINETYPE_PRINCIPAL, _LINETYPE_COMPARACION):
        grupo = datos[datos["linetype"] == valor]
        if not grupo.empty:
            fila = grupo.iloc[0]
            etiquetas[valor] = _segmento(
                fila["agregacion"], fila["rubro"], fila["incluir_petroleo"]
            )
    return etiquetas


def _desambiguar_medicion_heterogenea(datos: pd.DataFrame) -> pd.DataFrame:
    """Si el mismo `indice` aparece bajo más de una combinación `(agregacion, rubro, incluir_petroleo)` DENTRO DEL MISMO `linetype`, le agrega un sufijo que distingue la medición.

    Acotado a un solo `linetype` a la vez -- cuando el mismo `indice` aparece
    en `solid` Y en `dashed` (la comparación legítima entre `resultado` y
    `comparacion`), el propio `linetype` ya distingue cuál es cuál; cruzar la
    desambiguación ahí sería redundante y rompería el color/identificador
    compartido que esa comparación necesita para mostrarse (ver
    `_etiquetas_linetype`). `version` queda fuera de la clave a propósito -- un
    `ResultadoIndice` empalmado varía `version` por tramo sin cambiar de
    medición, y desambiguar por eso cortaría la continuidad visual de una
    serie real en la frontera del empalme.

    Un `ResultadoIndice` heterogéneo (construido a mano, fuera del flujo
    guiado -- ver `_mapa_incluir_petroleo`) puede repetir el mismo código de
    `indice` bajo agregaciones o rubros distintos EN EL MISMO `linetype` (ej.
    "11" como SECTOR en enero y como SUBSECTOR en febrero, ambos parte de
    `resultado`, mediciones sin relación). Sin esto, `geom_line` las agrupa
    como una sola serie visual (mismo `indice`, mismo `linetype`) y conecta
    100 con 200 como si fueran el mismo dato continuo.

    No afecta el caso normal (todo el flujo guiado -- `calcular_indice`,
    `empalmar`): ahí `agregacion`/`rubro`/`incluir_petroleo` son homogéneos
    por construcción dentro de cada `linetype` (`empalmar` lo exige), así que
    ningún `indice` colisiona y esta función no toca nada.
    """
    combos = datos[["agregacion", "rubro", "incluir_petroleo"]].astype(str).agg("|".join, axis=1)
    conteos = combos.groupby([datos["indice"], datos["linetype"]], observed=True).transform(
        "nunique"
    )
    colisiona: pd.Series = conteos > 1
    if not colisiona.any():
        return datos
    datos = datos.copy()
    # Arma la etiqueta desambiguada entera en un solo `apply` (en vez de
    # concatenar columnas de texto por separado) -- evita el tipo ambiguo que
    # deja la suma encadenada de varias `Series[str]`.
    datos.loc[colisiona, "indice"] = datos.loc[colisiona].apply(
        lambda fila: (
            f"{fila['indice']} "
            f"[{_segmento(fila['agregacion'], fila['rubro'], fila['incluir_petroleo'])}]"
        ),
        axis=1,
    )
    return datos


def _mapa_incluir_petroleo(
    resultado: ResultadoIndice | ResultadoVariacion,
) -> dict[tuple[int, str, str], bool] | None:
    """Mapa `(version, agregacion, rubro) -> incluir_petroleo` de TODO el manifiesto, o `None` si `resultado` es una `ResultadoVariacion` (no lo trae).

    Un `ResultadoIndice` puede traer manifiesto con MÁS de una combinación
    `(version, agregacion, rubro)` -- `empalmar` produce eso al fusionar
    versiones, y el constructor de `ResultadoIndice` tampoco impide construir
    uno a mano con combinaciones que difieran en `agregacion`/`rubro` (a
    diferencia de `ResultadoVariacion`, que sí valida homogeneidad). Tomar
    `manifiesto[0]` a secas asumía un único valor válido para todo el objeto;
    el mapa por combinación es correcto sin importar cuántas traiga.
    """
    if not isinstance(resultado, ResultadoIndice):
        return None
    return {(m.version, m.agregacion, m.rubro): m.incluir_petroleo for m in resultado.manifiesto}


def _aplanar_resultado(
    resultado: ResultadoIndice | ResultadoVariacion,
    comparacion: ResultadoIndice | ResultadoVariacion | None = None,
) -> pd.DataFrame:
    """Aplana `.resultado.largo` a un DataFrame único, una fila por `(periodo, indice)`.

    Sirve igual para índices y variaciones -- el aplanado no toca la columna de
    valor (`indice_replicado`/`variacion_pp`), solo la estructura común del
    índice. Agrega `periodo_ts` (timestamp del `periodo`, para el eje X) e
    `incluir_petroleo` -- por FILA, vía `_mapa_incluir_petroleo` (`None` en
    toda la columna si `resultado_actual` es una `ResultadoVariacion`).

    `comparacion`, si viene, se concatena marcando `linetype="dashed"`
    (`resultado` queda `linetype="solid"`) -- se distingue visualmente aunque
    comparta color.

    Cuando `agregacion == "INPP"` (el índice general, sin subcategoría SCIAN),
    `indice` trae el mismo valor constante en esas filas y no sirve para
    distinguir series -- se sobrescribe con `rubro`, que sí distingue (ej.
    comparar `produccion_total` contra `exportaciones` del índice general). El
    reemplazo es por MÁSCARA (`agregacion == "INPP"` fila por fila), no por un
    chequeo de la primera fila del objeto -- un `ResultadoIndice` puede traer
    filas de más de una `agregacion` a la vez (mismo motivo que
    `_mapa_incluir_petroleo`: el constructor no exige homogeneidad), y decidir
    por la primera fila swapeaba el objeto entero o ninguna fila según el orden
    físico, perdiendo códigos SCIAN reales en el proceso.
    """
    resultados = [(resultado, _LINETYPE_PRINCIPAL)]
    if comparacion is not None:
        resultados.append((comparacion, _LINETYPE_COMPARACION))

    partes = []
    for resultado_actual, linetype in resultados:
        df = resultado_actual.resultado.largo.reset_index()
        df["periodo_ts"] = pd.to_datetime(df["periodo"].map(lambda p: p.to_timestamp()))
        df["linetype"] = linetype
        mapa_petroleo = _mapa_incluir_petroleo(resultado_actual)
        if mapa_petroleo is None:
            df["incluir_petroleo"] = None
        else:
            claves = list(zip(df["version"], df["agregacion"], df["rubro"], strict=True))
            df["incluir_petroleo"] = [mapa_petroleo[clave] for clave in claves]
        mascara_inpp = df["agregacion"] == "INPP"
        df.loc[mascara_inpp, "indice"] = df.loc[mascara_inpp, "rubro"]
        partes.append(df)

    datos = pd.concat(partes, ignore_index=True) if len(partes) > 1 else partes[0]
    return _desambiguar_medicion_heterogenea(datos)


def _recortar_tramo(
    datos: pd.DataFrame,
    desde: PeriodoMensual | None,
    hasta: PeriodoMensual | None,
) -> pd.DataFrame:
    """Recorta `datos` a `[desde, hasta]` sobre la columna `periodo`. `None` en un lado = sin límite ahí.

    Sin exigir que `desde`/`hasta` existan exacto en los datos -- acá es un
    recorte visual (zoom), no una consulta puntual.
    """
    if desde is not None and hasta is not None and hasta < desde:
        raise InvarianteViolado(f"'desde' ({desde}) no puede ser posterior a 'hasta' ({hasta}).")

    filtrado = datos[_mascara_tramo(datos, desde, hasta)]
    if filtrado.empty:
        raise InvarianteViolado(f"Sin datos en el rango [{desde}, {hasta}].")
    return filtrado


def _mascara_tramo(
    datos: pd.DataFrame,
    desde: PeriodoMensual | None,
    hasta: PeriodoMensual | None,
) -> pd.Series:
    """Filas de `datos` dentro de `[desde, hasta]`. `None` en un lado = sin límite ahí."""
    mascara = pd.Series(True, index=datos.index)
    if desde is not None:
        mascara &= datos["periodo"] >= desde
    if hasta is not None:
        mascara &= datos["periodo"] <= hasta
    return mascara


def _finito(datos: pd.DataFrame, columna_valor: str) -> pd.Series:
    """Máscara booleana de valores finitos en `columna_valor`, tipada como `pd.Series`.

    `np.isfinite` sobre una `Series` ya devuelve una `Series` en tiempo de
    ejecución (pandas sobrescribe el ufunc) -- pero los stubs de numpy la
    tipan como `NDArray`, sin `.groupby`. Envolverla acá una sola vez evita
    repetir el mismo `pd.Series(..., index=...)` en cada llamador.
    """
    return pd.Series(np.isfinite(datos[columna_valor]), index=datos.index)


def _descartar_grupos_no_graficables(datos: pd.DataFrame, columna_valor: str) -> pd.DataFrame:
    """Descarta los grupos visuales `(indice, linetype)` sin NINGÚN valor finito en `columna_valor`.

    Un grupo con `estado_calculo` `sin_datos`/`fallida` en TODAS sus filas (ej.
    tres periodos consecutivos sin cobertura) no tiene nada que dibujar --
    dejarlo pasar hasta plotnine produce una capa vacía: 0 líneas, 0 puntos,
    límites de eje Y arbitrarios. Los NaN sueltos dentro de un grupo que sí
    conserva algún valor finito NO se tocan acá -- quedan intactos para que
    `geom_line` corte el trazo donde falta el dato (matplotlib ya rompe la
    línea en un NaN interior sin ayuda; filtrar esas filas conectaría directo
    el valor anterior con el siguiente y ocultaría el hueco real).

    Raises:
        InvarianteViolado: ningún grupo conserva un valor finito -- no queda
            nada que graficar.
    """
    finito = _finito(datos, columna_valor)
    grupo_tiene_dato = finito.groupby(
        [datos["indice"], datos["linetype"]], observed=True
    ).transform("any")
    graficable = datos[grupo_tiene_dato]
    if graficable.empty:
        raise InvarianteViolado(
            f"Ningún valor de '{columna_valor}' es finito -- no hay nada que graficar."
        )
    return graficable


def _indices_comparacion_repetidos(
    datos: pd.DataFrame, capacidad: int = _MAX_SERIES_POR_IMAGEN
) -> list[str] | None:
    """Índices de `comparacion` que `_particionar_series` repite en cada panel con la estrategia de comparación fija, o `None` si esa estrategia no aplica.

    Mismo criterio que la rama `_particionar_con_comparacion_fija` de
    `_particionar_series` -- factorizado acá para que el color estable de
    esos índices (`_reservar_colores_comparacion`, en `graficador.py`) se
    calcule ANTES de particionar, con la MISMA condición exacta que decide si
    de verdad se repiten -- dos lugares decidiendo esto por separado podrían
    desincronizarse.

    Devuelve `None` (no la lista vacía) cuando la estrategia no aplica --
    SIN `comparacion` la lista de índices repetidos también está vacía
    (`[]`) pero la estrategia de partición SIGUE aplicando si hay más
    índices propios que capacidad; `[]` sería indistinguible de "no aplica"
    para quien solo mira si la lista tiene contenido.
    """
    indices_comparacion = list(pd.unique(datos.loc[datos["linetype"] == "dashed", "indice"]))
    indices_propios = list(pd.unique(datos.loc[datos["linetype"] == "solid", "indice"]))
    comparacion_set = set(indices_comparacion)
    propios_fuera_de_comparacion = [v for v in indices_propios if v not in comparacion_set]
    if len(indices_comparacion) < capacidad:
        capacidad_extra = capacidad - len(indices_comparacion)
        if len(propios_fuera_de_comparacion) > capacidad_extra:
            return indices_comparacion
    return None


def _particionar_series(
    datos: pd.DataFrame, capacidad: int = _MAX_SERIES_POR_IMAGEN
) -> list[pd.DataFrame]:
    """Parte `datos` en varios DataFrames si el espacio de colores no alcanza para todos los índices.

    El espacio de colores es `nunique(indice)`, no filas ni conteo por
    `linetype` -- `resultado` y `comparacion` comparten la misma paleta (ver
    `_colores_y_etiquetas`), así que un índice que aparece en ambos (mismo
    color) cuenta UNA vez, no dos. Sea `C` el conjunto de índices únicos de
    `comparacion` (`linetype == "dashed"`):

    - `len(C) < capacidad`: `C` se repite COMPLETO en cada imagen (mismo trato
      que INPC le daba al nombre fijo `"INPC"`, generalizado por estructura --
      `linetype`, no un string mágico) y los índices propios que NO estén ya en
      `C` (agregarlos no sumaría color nuevo) se reparten en grupos de a lo
      sumo `capacidad - len(C)`, así cada imagen nunca pasa de `capacidad`
      colores en total.
    - `len(C) == capacidad`: `C` ya agota la paleta -- solo cabe repetirla
      completa si NINGÚN índice propio queda afuera de `C` (nada que agregar).
      Si sobra algún índice propio ajeno a `C`, no hay lugar para repetir `C`
      Y agregar nada sin pasar `capacidad` -- cae al caso general.
    - Cualquier otro caso (`len(C) > capacidad`, o `== capacidad` con sobrantes):
      se particiona la UNIÓN completa de índices (propios + comparación) en
      grupos de a lo sumo `capacidad`, manteniendo juntas en la misma partición
      las filas sólida y punteada de un mismo índice -- ya no hay "referencia
      fija" que repetir, todo se reparte por igual.

    En los tres casos, cada imagen resultante cumple `nunique(indice) <=
    capacidad` -- antes, copiar `comparacion` completa sin contarla dejaba
    paneles con hasta el doble de series que colores en la paleta (colores
    repetidos entre categorías distintas, visualmente ambiguo).
    """
    indices_comparacion = set(pd.unique(datos.loc[datos["linetype"] == "dashed", "indice"]))
    indices_propios = list(pd.unique(datos.loc[datos["linetype"] == "solid", "indice"]))
    propios_fuera_de_comparacion = [v for v in indices_propios if v not in indices_comparacion]

    fijos_repetidos = _indices_comparacion_repetidos(datos, capacidad)
    if fijos_repetidos is not None:
        capacidad_extra = capacidad - len(indices_comparacion)
        return _particionar_con_comparacion_fija(
            datos, indices_comparacion, propios_fuera_de_comparacion, capacidad_extra
        )
    if len(indices_comparacion) < capacidad:
        return [datos]
    if len(indices_comparacion) == capacidad and not propios_fuera_de_comparacion:
        return [datos]
    return _particionar_union_de_indices(datos, capacidad)


def _particionar_con_comparacion_fija(
    datos: pd.DataFrame,
    fijos: set[str],
    extra: list[str],
    capacidad_extra: int,
) -> list[pd.DataFrame]:
    """Reparte `extra` en grupos de a lo sumo `capacidad_extra`, repitiendo `fijos` completo en cada uno.

    `fijos` (índices de `comparacion`, y cualquier fila propia que comparta ese
    mismo índice -- ver `_particionar_series`) se incluye ENTERO en cada
    partición mediante `datos["indice"].isin(fijos)`, sin importar `linetype`:
    así una fila sólida que colisiona en color con `comparacion` queda pegada a
    su contraparte punteada en la misma imagen. El resto de categorías propias
    se reparte lo más parejo posible (`ceil(N / capacidad_extra)` particiones)
    -- ej. 13 categorías con capacidad 8 da particiones de 7 y 6, no 8 y 5
    (evita una segunda imagen casi vacía).
    """
    filas_fijas = datos[datos["indice"].isin(fijos)]
    n_particiones = math.ceil(len(extra) / capacidad_extra)
    base, resto = divmod(len(extra), n_particiones)

    particiones = []
    inicio = 0
    for i in range(n_particiones):
        tamano = base + (1 if i < resto else 0)
        grupo = extra[inicio : inicio + tamano]
        inicio += tamano
        parte = datos[(datos["linetype"] == "solid") & (datos["indice"].isin(grupo))]
        if not filas_fijas.empty:
            parte = pd.concat([parte, filas_fijas], ignore_index=True)
        particiones.append(parte)
    return particiones


def _particionar_union_de_indices(datos: pd.DataFrame, capacidad: int) -> list[pd.DataFrame]:
    """Reparte TODOS los índices (propios + comparación) en grupos de a lo sumo `capacidad`.

    Sin referencia fija que repetir -- `comparacion` ya no cabe entera sin
    pasar el tope de colores, así que se reparte como una categoría más.
    `datos["indice"].isin(grupo)` mantiene juntas, en la misma partición, las
    filas sólida y punteada de un mismo índice (si las hay).
    """
    todos = list(pd.unique(datos["indice"]))
    n_particiones = math.ceil(len(todos) / capacidad)
    base, resto = divmod(len(todos), n_particiones)

    particiones = []
    inicio = 0
    for i in range(n_particiones):
        tamano = base + (1 if i < resto else 0)
        grupo = todos[inicio : inicio + tamano]
        inicio += tamano
        particiones.append(datos[datos["indice"].isin(grupo)])
    return particiones


def _ordenar_series_dibujo(
    valores: pd.Series, linetype: pd.Series, orden: list[str] | None = None
) -> pd.Categorical:
    """Categórico ordenado con las series de `comparacion` al final -- se dibujan últimas, quedan por encima.

    `geom_line` agrupa y dibuja según el orden categórico de `indice`, no el
    orden de fila del DataFrame -- sin esto, una serie de `comparacion` podría
    quedar tapada por otra que se dibuje después. Generaliza el trato que INPC
    da al nombre fijo `"INPC"` (siempre encima, sea cual sea el parámetro por el
    que entre): acá no hay un nombre universal que hardcodear, así que el
    criterio es estructural -- toda fila con `linetype == "dashed"` viene de
    `comparacion`, sin importar su valor de `indice`. Se aplica DESPUÉS de
    particionar (`_particionar_series`) para que las categorías reflejen solo lo
    presente en cada imagen.

    `orden`, si viene, reemplaza el orden de aparición de las series PROPIAS
    (`linetype == "solid"`) por uno explícito -- las que no estén en `orden` se
    agregan al final respetando su aparición, así una lista parcial nunca hace
    desaparecer categorías. Las de `comparacion` siempre van al final, sin
    importar `orden`.
    """
    siempre_encima = set(valores[linetype == _LINETYPE_COMPARACION])
    presentes = list(pd.unique(valores))
    propias_presentes = [v for v in presentes if v not in siempre_encima]
    if orden is None:
        secuencia = propias_presentes
    else:
        pedidos = [v for v in orden if v in set(propias_presentes)]
        resto = [v for v in propias_presentes if v not in set(pedidos)]
        secuencia = pedidos + resto
    secuencia += [v for v in presentes if v in siempre_encima]
    return pd.Categorical(valores, categories=secuencia, ordered=True)


def _breaks_y_etiquetas_x(datos: pd.DataFrame) -> tuple[list[pd.Timestamp], list[str]]:
    """Elige un subconjunto de periodos reales, repartido parejo, para las marcas del eje X.

    Usa `str(periodo)` (`"Ene 2024"`) de los periodos que ya están en los datos
    -- no reconstruye un periodo a partir de una fecha arbitraria elegida por el
    algoritmo de breaks. El primer y último break son siempre el primer y
    último periodo real.

    Las posiciones se reparten entre el primer y el último índice, así que
    ambos extremos caen en la grilla POR CONSTRUCCIÓN y los saltos difieren a lo
    sumo en un periodo.
    """
    pares = datos[["periodo", "periodo_ts"]].drop_duplicates().sort_values("periodo_ts")
    n = len(pares)
    cuantas = min(n, _MAX_ETIQUETAS_EJE_X)
    if cuantas <= 1:
        posiciones = [0]
    else:
        posiciones = sorted({round(i * (n - 1) / (cuantas - 1)) for i in range(cuantas)})
    seleccion = pares.iloc[posiciones]
    return list(seleccion["periodo_ts"]), [str(p) for p in seleccion["periodo"]]


def _breaks_desde_extremos(minimo: float, maximo: float, valor_base: float) -> list[float]:
    """Breaks del eje Y entre dos extremos ya calculados; `minimo`/`maximo` siempre son breaks.

    Ningún break intermedio puede quedar fuera de `[minimo, maximo]` ni a menos
    de `_SEPARACION_MINIMA_BREAKS` del rango respecto de un extremo: el mínimo y
    el máximo reales se fuerzan siempre (para que el lector vea el valor exacto
    de los extremos), y un break automático que caiga pegado a uno de ellos se
    dibuja encima. Se descarta el automático, no el extremo forzado.

    La base (0 para variaciones, 100 para índices) pasa por el mismo filtro: si
    cae pegada a un extremo, tampoco aporta una marca legible.
    """
    rango = maximo - minimo
    if rango <= 0:
        return [minimo]
    holgura = rango * _SEPARACION_MINIMA_BREAKS
    piso, techo = minimo + holgura, maximo - holgura
    extendidos = breaks_extended(n=_N_ETIQUETAS_EJE_Y)((minimo, maximo))
    intermedios = {b for b in extendidos.tolist() if piso < b < techo}
    if piso < valor_base < techo:
        intermedios.add(valor_base)
    return sorted({minimo, maximo, *intermedios})


def _breaks_y(datos: pd.DataFrame, columna_valor: str, valor_base: float) -> list[float]:
    """Breaks del eje Y: extremos = mínimo y máximo de la columna.

    Comparte el algoritmo entre índices (`indice_replicado`, base 100) y
    variaciones (`variacion_pp`, base 0).
    """
    return _breaks_desde_extremos(
        float(datos[columna_valor].min()), float(datos[columna_valor].max()), valor_base
    )


def _etiqueta_y_indice(resultado: ResultadoIndice | ResultadoVariacion) -> str:
    """Etiqueta del eje Y para índices: `"Indice"`, o `"Indice (periodo_referencia = 100)"` si fue rebasado."""
    if isinstance(resultado, ResultadoIndice) and resultado.periodo_referencia is not None:
        return f"Indice ({resultado.periodo_referencia} = 100)"
    return "Indice"


def _reservar_colores_comparacion(indices_comparacion: list[str]) -> dict[str, str]:
    """Colores fijos y estables para índices de `comparacion` que `_particionar_series` repite en varios paneles.

    Reservados PRIMERO, en orden de aparición, antes de que cada panel asigne
    color a sus propios índices -- así ocupan siempre los mismos primeros
    turnos de la paleta sin importar qué panel se construya después (ver
    `_colores_y_etiquetas`, parámetro `reservados`). Sin esto, la MISMA
    referencia de `comparacion` cambiaba de color entre imágenes -- cada
    panel volvía a contar desde cero con su propia lista local de series.

    `_RUBROS_CONOCIDOS` (negro fijo, `_COLOR_INPP`) no consume turno acá
    tampoco -- se excluyen de la reserva igual que del resto de la paleta.
    Pasar `indices_comparacion=[]` (nada que repetir entre paneles, caso
    normal) devuelve `{}` sin reservar nada.
    """
    reservados: dict[str, str] = {}
    i = 0
    for indice in indices_comparacion:
        if indice in _RUBROS_CONOCIDOS or indice in reservados:
            continue
        reservados[indice] = _PALETA_OTROS_TIPOS[i]
        i += 1
    return reservados


def _colores_y_etiquetas(
    series: list[str], reservados: dict[str, str] | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """Color y etiqueta de leyenda por serie -- mismas claves (`indice`), siempre se usan juntos.

    Color: cualquier rubro conocido (`_RUBROS_CONOCIDOS`) siempre toma
    `_COLOR_INPP` (negro), sin variedad por rubro ni por orden de aparición --
    relevante cuando `agregacion == "INPP"`, donde `indice` ya fue sobrescrito
    con `rubro` (ver `_aplanar_resultado`): la condición para ser negro es
    únicamente que `agregacion == "INPP"`, nunca cuál rubro ni si incluye
    petróleo. No consume turno de la paleta genérica -- gastar en él uno de los
    8 colores distinguibles dejaría fuera a una categoría real cuando SÍ hay
    código SCIAN de por medio (ej. `comparacion` trae el índice general
    superpuesto sobre un desglose por SECTOR).

    `reservados` (de `_reservar_colores_comparacion`, cuando `comparacion` se
    repite en varios paneles) fija de antemano el color de esas series -- se
    reusan tal cual, sin consumir un turno nuevo. El resto de `series` (ni
    conocidas ni reservadas) toma color de paleta en el orden en que aparece,
    arrancando DESPUÉS de los turnos ya reservados (`i = len(reservados)`),
    así nunca coincide con uno de ellos. Sin este desplazamiento, calcular una
    paleta única sobre TODOS los índices (propios + comparación) de una vez
    -- en vez de reservar y dejar el resto local a cada panel -- hace que el
    contador cicle más allá de los 8 colores cuando la unión total supera la
    capacidad, y dos índices AJENOS entre sí (nunca comparten panel) terminan
    con el mismo color por casualidad de orden, sin que eso distinga nada.

    Comparar 2 rubros de `agregacion == "INPP"` (ambos negros) o el mismo rubro
    con/sin petróleo (mismo `indice`, un solo color en la leyenda) depende
    entonces de `linetype` para distinguirse, no de color -- ver la leyenda de
    `linetype` en `graficador._construir_grafica_linea`.

    Etiqueta: nombre legible (`_ETIQUETAS_RUBRO` para rubros, o el valor crudo
    de `indice` para códigos SCIAN), o truncado a `_MAX_CARACTERES_LEYENDA` con
    `"..."` si no cabe -- nombres largos de categoría desbordarían el ancho de
    la leyenda. La etiqueta NO afecta color ni agrupación -- ambos dicts quedan
    keyed por el valor completo de `indice`, para
    `scale_color_manual(values=colores, labels=etiquetas)`. No depende de
    `reservados` ni del orden de `series` -- es una función pura de cada
    `serie` por separado, así que da lo mismo calcularla local (por panel) o
    global.
    """
    reservados = reservados or {}
    colores: dict[str, str] = {}
    etiquetas: dict[str, str] = {}
    i = len(reservados)
    for serie in series:
        if serie in _RUBROS_CONOCIDOS:
            colores[serie] = _COLOR_INPP
        elif serie in reservados:
            colores[serie] = reservados[serie]
        else:
            colores[serie] = _PALETA_OTROS_TIPOS[i % len(_PALETA_OTROS_TIPOS)]
            i += 1
        nombre = _ETIQUETAS_RUBRO.get(serie, serie)
        if len(nombre) > _MAX_CARACTERES_LEYENDA:
            etiquetas[serie] = nombre[: _MAX_CARACTERES_LEYENDA - 3] + "..."
        else:
            etiquetas[serie] = nombre
    return colores, etiquetas


def _datos_para_puntos(datos: pd.DataFrame, columna_valor: str) -> pd.DataFrame | None:
    """Filas FINITAS que llevan `geom_point`, o `None` si ninguna lo lleva.

    Dos casos, en ese orden:

    1. **Tramo corto** -- hasta `_MAX_PERIODOS_CON_PUNTOS` periodos distintos:
       todas las filas con valor finito. Con pocas observaciones los puntos
       marcan cada dato real sin competir con la línea.
    2. **Tramo largo** -- solo los grupos visuales `(indice, linetype)` con una
       única observación FINITA. `geom_line` no dibuja nada con un solo punto
       finito por grupo (emite `PlotnineWarning` y la categoría queda
       invisible), así que el punto es lo único que la hace visible -- una
       serie con 23 periodos `sin_datos` y 1 solo con valor real cuenta como
       "una observación", no como 24.

    Siempre se filtra a valores finitos: un NaN pasado a `geom_point` no se
    dibuja igual (plotnine lo descarta con una `PlotnineWarning` propia), así
    que filtrarlo acá evita el ruido y dice explícito qué se está omitiendo.

    El conteo de periodos (tramo corto/largo) es de periodos DISTINTOS
    presentes en `datos`, no de cuántos tienen valor -- el span temporal del
    eje X no cambia porque falte un dato puntual.

    Agrupa por `(indice, linetype)` y no solo por `indice` porque `resultado` y
    `comparacion` pueden compartir `indice` (ej. mismo rubro en ambos) y son
    grupos visuales distintos.
    """
    finitos = datos[_finito(datos, columna_valor)]
    if datos["periodo"].nunique() <= _MAX_PERIODOS_CON_PUNTOS:
        return finitos if not finitos.empty else None

    # `_grupos_por_tamano` devuelve el grupo COMPLETO (incluye los NaN del
    # grupo, a propósito -- son los que necesita `geom_line` para el hueco
    # interior). Acá hace falta solo la fila finita del grupo, no todas.
    solitarias = _grupos_por_tamano(datos, columna_valor, solitarios=True)
    solitarias_finitas = solitarias[_finito(solitarias, columna_valor)]
    return solitarias_finitas if not solitarias_finitas.empty else None


def _grupos_por_tamano(
    datos: pd.DataFrame, columna_valor: str, *, solitarios: bool
) -> pd.DataFrame:
    """Filas de los grupos visuales `(indice, linetype)` con una sola observación FINITA, o del resto.

    Cuenta valores FINITOS de `columna_valor`, no filas crudas: un grupo con 23
    `NaN` (`sin_datos`) y 1 valor real es "una observación" a efectos de
    `geom_point`/`geom_line`, no 24 -- contar filas crudas dejaría esa serie
    sin punto (`geom_line` tampoco dibuja nada con un solo valor real disperso
    entre NaN, así que sin el punto quedaría invisible).

    `geom_line` no dibuja nada con un solo punto finito por grupo: emite
    `PlotnineWarning` y la serie queda invisible. Las dos mitades se usan juntas
    -- los solitarios van a `geom_point` (única forma de verlos) y el resto a
    `geom_line` (así la capa lineal no recibe grupos que no puede dibujar, que
    es lo que provoca el warning). Las filas devueltas para `geom_line`
    conservan sus NaN interiores intactos -- ver `_descartar_grupos_no_graficables`.
    """
    finito = _finito(datos, columna_valor)
    conteos = finito.groupby([datos["indice"], datos["linetype"]], observed=True).transform("sum")
    return datos[conteos == 1] if solitarios else datos[conteos > 1]


def _primero_y_ultimo_para_anotar(
    datos: pd.DataFrame, series: list[str], columna_valor: str
) -> tuple[pd.Series, pd.Series] | None:
    """Primer y último punto FINITO a anotar con su valor, o `None` si no aplica.

    Solo con un único grupo visual `(indice, linetype)` -- el mismo caso donde
    no hay leyenda. `resultado` y `comparacion` pueden compartir `indice` (ej.
    mismo rubro en ambos, uno sólido y otro punteado): son 2 grupos visuales
    aunque `series` (que agrupa solo por `indice`) reporte 1 -- anotar ahí
    mezclaría el primer punto de un grupo con el último del otro. Con varias
    series (>1 `indice`), tampoco se anota: con N líneas cruzadas el numerito
    compite con la leyenda por espacio y no aporta.

    Los extremos se toman entre las filas FINITAS -- anotar un `NaN` en el
    borde (ej. el último periodo es `sin_datos`) mostraría "nan" en vez del
    último valor real.
    """
    if len(series) != 1:
        return None
    grupos = datos[["indice", "linetype"]].drop_duplicates()
    if len(grupos) != 1:
        return None
    finitos = datos[_finito(datos, columna_valor)]
    if finitos.empty:
        return None
    ordenado = finitos.sort_values("periodo_ts")
    return ordenado.iloc[0], ordenado.iloc[-1]
