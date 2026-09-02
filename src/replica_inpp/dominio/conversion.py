"""Reexpresión y unión de `ResultadoIndice` ya calculados.

Adaptado desde `rebasar()`/`empalmar()` de `replica-inpc-mx/dominio/conversion.py`
(mismo algoritmo base), con una validación propia que ese repo no tiene:
`empalmar()` acá también compara `indice_replicado` de ambos tramos en la
frontera (tolerancia interna, ver `_TOLERANCIA_FRONTERA_ABS`) -- INPC no lo
hace, confía en que `rebasar()`/`empalmar()` se llamen en el orden correcto.
Este repo NO porta `a_mensual()` -- el INPP siempre es mensual, no existe la
conversión quincenal->mensual que la motiva -- y por eso tampoco existe
`._frontera` en `ResultadoIndice` (esa existe en INPC solo para no perder
precisión al promediar 2 quincenas; acá no hay promedio que perder).
`empalmar()` tampoco normaliza nombres/códigos de categoría entre versiones (a
diferencia de INPC, que usa `correspondencia_canastas` para eso) -- el usuario
decidió que ese módulo de correspondencia no hace falta para este repo; ver
limitación documentada en `empalmar()` abajo.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from replica_inpp.dominio.errores import ErrorCalculo, InvarianteViolado
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import VersionCanasta

_ESTADOS_CON_VALOR = frozenset({"ok", "parcial", "rellenado"})

# Tolerancia de coherencia numérica en la frontera de `empalmar` -- interna, no
# expuesta como parámetro: compara el MISMO dato calculado 2 veces (una por
# tramo), así que cualquier discrepancia real por encima de precisión de punto
# flotante señala una escala mal alineada (falta `rebasar()` antes), no ruido
# de redondeo de INEGI (eso usa una tolerancia mucho más floja, ver
# `data/comprobacion.py::_TOLERANCIA`).
_TOLERANCIA_FRONTERA_ABS = 1e-6


def _validar_topologia(ordenados: list[ResultadoIndice]) -> list[PeriodoMensual]:
    """Valida topología PATH y devuelve la lista de periodos frontera entre pares consecutivos."""
    conjuntos = [set(r._df_resultado.index.get_level_values("periodo")) for r in ordenados]
    fronteras: list[PeriodoMensual] = []
    for i in range(len(ordenados) - 1):
        compartidos = conjuntos[i] & conjuntos[i + 1]
        if len(compartidos) == 0:
            raise InvarianteViolado(
                f"empalmar: par consecutivo [{i}, {i + 1}] no comparte ningún periodo — "
                "no hay frontera válida para empalmar."
            )
        if len(compartidos) > 1:
            raise InvarianteViolado(
                f"empalmar: par consecutivo [{i}, {i + 1}] comparte {len(compartidos)} periodos "
                f"({sorted(map(str, compartidos))}); se requiere exactamente 1 (topología PATH)."
            )
        fronteras.append(next(iter(compartidos)))
        for j in range(i + 2, len(ordenados)):
            no_consecutivos = conjuntos[i] & conjuntos[j]
            if no_consecutivos:
                raise InvarianteViolado(
                    f"empalmar: par no-consecutivo [{i}, {j}] comparte periodos "
                    f"({sorted(map(str, no_consecutivos))}); topología debe ser PATH lineal."
                )
    return fronteras


def _valores_en_periodo(resultado: ResultadoIndice, periodo: PeriodoMensual) -> dict[object, float]:
    """`indice_replicado` por `indice` en `periodo`, solo filas con estado con valor."""
    df = resultado._df_resultado
    mask = df.index.get_level_values("periodo") == periodo
    sub = df[mask]
    valores: dict[object, float] = {}
    for indice, valor, estado in zip(
        sub.index.get_level_values("indice"), sub["indice_replicado"], sub["estado_calculo"]
    ):
        if estado in _ESTADOS_CON_VALOR and pd.notna(valor):
            valores[indice] = float(valor)
    return valores


def _combinar_nombres(
    ordenados: list[ResultadoIndice], version_nombres: VersionCanasta | None = None
) -> tuple[pd.Series | None, dict[VersionCanasta, pd.Series]]:
    """Combina nombres por versión de los tramos para mostrar, y fusiona su historial.

    Devuelve `(nombres_planos, nombres_por_version)`:

    - `nombres_planos`: el que se muestra en `.resultado.ancho`. Se reconstruye
      en 2 capas:
      1. **Huérfanos**: por cada tramo, las entradas de su `_nombres` plano
         cuyo `indice` NO aparece en ninguna serie del propio
         `_nombres_por_version` de ESE tramo -- sin información de a qué
         versión pertenecen (puede ser un tramo ya empalmado sin historial, o
         un `indice` suelto dentro de un tramo por lo demás trazable). Se
         aplican en el orden de `ordenados` (cronológico por tramo, no por
         versión); ceden ante cualquier nombre trazable.
      2. **Registro por versión**: orden numérico ascendente de `registro`
         (versión = año en este dominio, así que asc = cronológico), el
         último (mayor) sobreescribe -- da la MISMA respuesta sin importar si
         un tramo llegó directo o ya pasó por un `empalmar` anterior con
         `version_nombres` explícito, porque nunca depende del `_nombres` ya
         colapsado de un tramo compuesto.
      Si se da `version_nombres`, esa versión tiene precedencia máxima sobre
      ambas capas; para un `indice` que esa versión nunca nombró, sigue
      aplicando lo de arriba.
    - `nombres_por_version`: fusión de `_nombres_por_version` de TODOS los
      tramos -- se pasa a `ResultadoIndice(..., nombres_por_version=...)` para
      que un empalme posterior (incremental) siga pudiendo resolver
      `version_nombres` de cualquier versión ya incluida, sin importar cuántos
      empalmes hubo antes. Los huérfanos NUNCA entran acá (no hay versión real
      que atribuirles) -- sobreviven a un empalme más porque se recalculan
      igual en cada llamada, pero no acumulan `nombres_por_version` propio.
    """
    registro: dict[VersionCanasta, pd.Series] = {}
    for r in ordenados:
        registro.update(r._nombres_por_version)

    if version_nombres is not None and version_nombres not in registro:
        raise InvarianteViolado(
            f"empalmar: version_nombres={version_nombres} no corresponde a ningún tramo de "
            "'resultados' (ni a su historial de empalmes previos), o ese tramo no tiene "
            "'nombres' asignados."
        )

    combinado: dict[object, str] = {}
    for r in ordenados:
        if r._nombres is None:
            continue
        indices_trazables_del_tramo: set[object] = set()
        for serie in r._nombres_por_version.values():
            indices_trazables_del_tramo.update(serie.index)
        huerfanos = {
            indice: nombre
            for indice, nombre in r._nombres.to_dict().items()
            if indice not in indices_trazables_del_tramo
        }
        combinado.update(huerfanos)
    for version in sorted(registro):  # cronológico ascendente: el último (mayor) sobreescribe
        combinado.update(registro[version].to_dict())
    if version_nombres is not None:
        combinado.update(registro[version_nombres].to_dict())  # precedencia máxima, va al final

    nombres_planos = pd.Series(combinado) if combinado else None
    return nombres_planos, registro


def empalmar(
    resultados: list[ResultadoIndice],
    forzar: bool = False,
    version_nombres: VersionCanasta | None = None,
) -> ResultadoIndice:
    """Concatena tramos de distinta versión de canasta en un único `ResultadoIndice`.

    En la frontera entre 2 tramos consecutivos (el periodo de traslape que
    `RANGOS_CANASTAS` comparte entre versiones contiguas), el tramo anterior
    posee `(periodo, indice)` si esa fila existe en él; si no, el tramo
    posterior la aporta. Para que la frontera sea numéricamente coherente
    (mismo valor en ambos tramos, no solo mismo periodo), el tramo anterior
    normalmente necesita `rebasar()` antes a la escala del posterior -- ver
    `rebasar()`.

    **Limitación real, a diferencia de `replica-inpc-mx`**: no normaliza
    `indice` (nombre/código de categoría) entre versiones -- ese repo usa
    `correspondencia_canastas` para eso, que este repo no tiene (decisión
    explícita: sin módulo de correspondencia de códigos entre versiones). Por
    eso `empalmar` solo es seguro para agregaciones donde `indice` YA es
    estable entre las versiones que se empalman -- `"INPP"` (agregación
    general, un solo `indice`) y `"SECTOR"` (código de 2 dígitos, estable
    entre SCIAN 2007/2013/2018) lo son; SUBSECTOR/RAMA/SUBRAMA/CLASE o
    genérico no están garantizados si hay reclasificación real ahí.

    Args:
        resultados: al menos 2 `ResultadoIndice`, cada uno de una versión de
            canasta distinta, mismos `agregacion`/`rubro`/`incluir_petroleo`.
        forzar: permite empalmar cuando `periodo_referencia` de un tramo no
            coincide con la frontera con el tramo siguiente, o cuando
            `indice_replicado` difiere entre tramos en la frontera más allá de
            la tolerancia interna (en ambos casos, `UserWarning` en vez de
            rechazar). Los tramos recién calculados (sin rebasar) tienen
            `periodo_referencia=None` y no disparan la primera guardia.
        version_nombres: versión de canasta cuyo `nombres` original tiene
            precedencia MÁXIMA en la columna `nombre` combinada (gana incluso
            sobre tramos más nuevos). `None` (default) = precedencia del más
            reciente. Para un `indice` que esa versión no nombra, sigue
            aplicando el fallback normal entre el resto de los tramos. Robusto
            a empalme incremental: resuelve contra `nombres_por_version` (que
            se fusiona en cada `empalmar`, ver `_combinar_nombres`), no contra
            el `nombres` ya mezclado de un tramo que a su vez viene de un
            `empalmar` anterior -- `empalmar([empalmar([r2012, r2019]), r2025],
            version_nombres=2012)` da el mismo resultado que
            `empalmar([r2012, r2019, r2025], version_nombres=2012)`.

    Combina `nombres` de todos los tramos para que el resultado combinado
    conserve la columna pública `nombre` de `.resultado.ancho` cuando algún
    tramo la trae -- ver `version_nombres` arriba para elegir cuál manda.

    Raises:
        InvarianteViolado: menos de 2 `resultados`; no todos comparten
            `(agregacion, rubro, incluir_petroleo)`; algún par consecutivo (una
            vez ordenados por periodo mínimo) no comparte exactamente 1
            periodo, o algún par no consecutivo comparte alguno (topología
            distinta de PATH); un tramo trae `periodo_referencia` que no
            coincide con la frontera sin `forzar`; `indice_replicado` de algún
            `indice` compartido en la frontera difiere entre tramos más allá
            de la tolerancia interna (escala no coherente — falta `rebasar()`
            antes) sin `forzar`; o `version_nombres` no corresponde a ningún
            tramo de `resultados`.
    """
    if len(resultados) < 2:
        raise InvarianteViolado("empalmar requiere al menos 2 ResultadoIndice.")

    combos = {(m.agregacion, m.rubro, m.incluir_petroleo) for r in resultados for m in r.manifiesto}
    if len(combos) != 1:
        raise InvarianteViolado(
            "empalmar requiere la misma (agregacion, rubro, incluir_petroleo) entre todos los "
            f"inputs; recibió {sorted(combos)}"
        )

    ordenados = sorted(
        resultados, key=lambda r: r._df_resultado.index.get_level_values("periodo").min()
    )

    fronteras = _validar_topologia(ordenados)

    for i, frontera in enumerate(fronteras):
        anterior, posterior = ordenados[i], ordenados[i + 1]

        ref_i = anterior.periodo_referencia
        if ref_i is not None and ref_i != frontera:
            msg = (
                f"empalmar: tramo {i} tiene periodo_referencia={ref_i} "
                f"pero la frontera con el siguiente tramo es {frontera}; "
                "la juntura puede ser discontinua — usa rebasar() antes o forzar=True."
            )
            if not forzar:
                raise InvarianteViolado(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

        valores_anterior = _valores_en_periodo(anterior, frontera)
        valores_posterior = _valores_en_periodo(posterior, frontera)
        for indice in sorted(set(valores_anterior) & set(valores_posterior), key=str):
            va, vp = valores_anterior[indice], valores_posterior[indice]
            if not math.isclose(va, vp, rel_tol=0.0, abs_tol=_TOLERANCIA_FRONTERA_ABS):
                msg = (
                    f"empalmar: en la frontera {frontera}, '{indice}' vale {va} en el tramo "
                    f"{i} y {vp} en el tramo {i + 1} (diferencia {abs(va - vp):.6g} > "
                    f"tolerancia {_TOLERANCIA_FRONTERA_ABS}); la escala no es coherente — "
                    "usa rebasar() antes o forzar=True."
                )
                if not forzar:
                    raise InvarianteViolado(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)

    df_combinado = pd.concat([r._df_resultado for r in ordenados])
    df_combinado = df_combinado[~df_combinado.index.duplicated(keep="first")]
    df_combinado.sort_index(level="periodo", sort_remaining=False, inplace=True)

    reporte_combinado = pd.concat([r.reporte for r in ordenados])
    reporte_combinado = reporte_combinado[~reporte_combinado.index.duplicated(keep="first")]
    reporte_combinado.sort_index(level="periodo", sort_remaining=False, inplace=True)

    diag_combinado = pd.concat([r.diagnostico for r in ordenados], ignore_index=True)
    manifiesto_combinado = [m for r in ordenados for m in r.manifiesto]

    refs_explicitas = [r.periodo_referencia for r in ordenados if r.periodo_referencia is not None]
    periodo_referencia_out = refs_explicitas[-1] if refs_explicitas else None

    nombres_out, nombres_por_version_out = _combinar_nombres(ordenados, version_nombres)

    return ResultadoIndice(
        df_combinado,
        manifiesto_combinado,
        reporte_combinado,
        diag_combinado,
        nombres=nombres_out,
        periodo_referencia=periodo_referencia_out,
        nombres_por_version=nombres_por_version_out,
    )


def rebasar(
    resultado: ResultadoIndice,
    periodo_referencia: PeriodoMensual,
    valor_base: float = 100.0,
) -> ResultadoIndice:
    """Reexpresa cada `indice` de `resultado` a una nueva referencia.

    Divide la serie de cada `indice` (agregación: `"INPP"`, código de sector,
    etc.) entre su propio valor replicado en `periodo_referencia` y la
    multiplica por `valor_base`, así ese periodo pasa a valer exactamente eso.
    Un `indice` sin dato válido ahí queda sin rebasar (`UserWarning`); si
    NINGÚN `indice` tiene dato en `periodo_referencia`, la operación entera
    falla en vez de devolver un resultado sin reescalar. No toca
    `indice_incidencia` (columna interna previa a cualquier rebase). El
    `ResultadoIndice` devuelto trae `periodo_referencia` seteado a este mismo
    valor (`empalmar()` lo usa para validar coherencia de frontera).

    Args:
        resultado: índice ya calculado a reexpresar.
        periodo_referencia: mes que pasará a valer `valor_base`.
        valor_base: cuánto valdrá `periodo_referencia`. 100.0 por convención.

    Raises:
        InvarianteViolado: si `valor_base` no es finito y positivo; si el
            valor base de algún `indice` en `periodo_referencia` es NaN o
            exactamente 0 (`inf` o negativo no se validan -- el dato ya está
            garantizado finito y positivo aguas arriba, en `dominio/calculo/`);
            o si ningún `indice` tiene dato en `periodo_referencia`.
        ErrorCalculo: `valor_base` es finito pero produce overflow (`inf`) al
            multiplicar contra algún `indice_replicado` real -- `valor_base`
            en sí pasa la validación de arriba, el desbordamiento ocurre recién
            al aplicarlo.

    Ver: replica-inpc-mx/dominio/conversion.py::rebasar (mismo algoritmo base,
    sin `._frontera` -- este repo no la tiene, ver módulo).
    """
    if not (math.isfinite(valor_base) and valor_base > 0):
        raise InvarianteViolado(f"rebasar: valor_base={valor_base} debe ser finito y positivo.")

    df = resultado._df_resultado.copy()
    indices_unicos = df.index.get_level_values("indice").unique()
    indices_sin_referencia: list[str] = []

    # Aislar, por índice, la fila en el periodo de referencia (el futuro denominador).
    mask_periodo_referencia = df.index.get_level_values("periodo") == periodo_referencia
    df_en_referencia = df[mask_periodo_referencia].copy()
    df_en_referencia.index = df_en_referencia.index.droplevel("periodo")

    factores_por_indice: dict[object, float] = {}
    for indice in indices_unicos:
        if indice not in df_en_referencia.index:
            indices_sin_referencia.append(str(indice))
            continue
        fila_referencia: pd.Series = df_en_referencia.loc[indice]
        estado_en_referencia = fila_referencia["estado_calculo"]
        if estado_en_referencia not in _ESTADOS_CON_VALOR:
            raise InvarianteViolado(
                f"El valor base de '{indice}' en {periodo_referencia} no está disponible "
                f"(estado_calculo='{estado_en_referencia}')."
            )
        valor_en_referencia_raw = fila_referencia["indice_replicado"]
        if pd.isna(valor_en_referencia_raw):
            raise InvarianteViolado(
                f"indice_replicado de '{indice}' en {periodo_referencia} es NaN; "
                f"estado_calculo='{estado_en_referencia}' es inconsistente."
            )
        valor_en_referencia = float(valor_en_referencia_raw)
        if valor_en_referencia == 0:
            raise InvarianteViolado(
                f"indice_replicado de '{indice}' en {periodo_referencia} es 0; no rebasable."
            )
        factores_por_indice[indice] = valor_base / valor_en_referencia

    if not factores_por_indice:
        raise InvarianteViolado(
            f"rebasar: ningún 'indice' tiene dato en {periodo_referencia}; no se puede "
            "rebasar (periodo inexistente en el resultado)."
        )

    # Solo se reescalan filas cuyo estado ya trae un valor confiable (`sin_datos`/
    # `fallida` quedan como NaN, intactas); índices sin referencia (huérfanos) no
    # tienen entrada en `factores_por_indice`, así que `factor_por_fila` les da NaN
    # y `mask_aplicar_factor` los excluye -- quedan en su escala original.
    mask_estado_rebasable = df["estado_calculo"].isin(_ESTADOS_CON_VALOR)
    indice_por_fila = df.index.get_level_values("indice")
    factor_por_fila = pd.Series(
        indice_por_fila.map(factores_por_indice),
        index=df.index,
        dtype=float,
    )
    mask_aplicar_factor = mask_estado_rebasable & factor_por_fila.notna()
    with np.errstate(over="ignore"):
        valores_rebasados = (
            df.loc[mask_aplicar_factor, "indice_replicado"].astype(float).to_numpy()
            * factor_por_fila.loc[mask_aplicar_factor].to_numpy()
        )
    if not np.isfinite(valores_rebasados).all():
        raise ErrorCalculo(
            f"rebasar: valor_base={valor_base} produce overflow (no finito) al reescalar "
            "uno o más 'indice' -- usa un valor_base más chico."
        )
    df.loc[mask_aplicar_factor, "indice_replicado"] = valores_rebasados

    if indices_sin_referencia:
        warnings.warn(
            f"rebasar: {len(indices_sin_referencia)} índice(s) sin dato en {periodo_referencia} "
            f"quedan sin rebasar (base original): {indices_sin_referencia}",
            UserWarning,
            stacklevel=2,
        )

    return ResultadoIndice(
        df,
        resultado.manifiesto,
        resultado.reporte,
        resultado.diagnostico,
        nombres=resultado._nombres,
        periodo_referencia=periodo_referencia,
        nombres_por_version=resultado._nombres_por_version,
    )
