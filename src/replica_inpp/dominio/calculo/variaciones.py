"""Cálculo de variaciones a partir de un `ResultadoIndice`.

Tres funciones producen `ResultadoVariacion`:

- `variacion_periodica` — una variación por periodo contra N periodos atrás.
- `variacion_acumulada_anual` — enero..periodo vs diciembre del año anterior.
- `variacion_desde` — variación total de un rango; una fila por índice.

Puerto de `replica-inpc-mx/.../dominio/calculo/variaciones.py`: mismo truco
central (reindexar `largo` con un MultiIndex `(periodo_base, indice)` desplazado)
sin cambio, porque no depende de qué agrupa. Dos adaptaciones reales: INPP es
mensual-only (sin rama quincenal) y agrupa por 2 ejes uniformes
(`agregacion`, `rubro`) en vez de 1 (`tipo`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from replica_inpp.dominio.calculo._temporal import (
    LAG_MENSUAL,
    Frecuencia,
    resolver_extremo,
    restar_meses,
)
from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import ManifestDerivado, VersionCanasta

Periodo = PeriodoMensual

# "error" (default): comportamiento histórico -- una fila computable con base
# 0, indice_replicado no finito, o variacion_pp resultante no finita revienta
# TODA la corrida. "marcar": esas filas se excluyen de `.df` (mismo trato que
# una fila sin dato crudo) y quedan documentadas en `.diagnostico` con
# estado_calculo="indefinido" -- no elige por su cuenta si el dato es
# real o basura, solo evita que un grupo degenerado tumbe el resto de
# `agregacion`/`rubro` que sí calculan bien. Ver `excluir_desde` (`dominio/
# conversion.py`) para recortar quirúrgicamente un `indice` puntual después.
EnIndefinido = Literal["error", "marcar"]

_COLS_REPORTE = [
    "estado_calculo",
    "motivo_error",
    "periodo_lag",
    "indice_t",
    "indice_lag",
    "version_t",
    "version_lag",
    "cobertura_pct_t",
    "cobertura_pct_lag",
]
_COLS_DIAGNOSTICO = [
    "versiones",
    "agregacion",
    "rubro",
    "clase_variacion",
    "periodo",
    "indice",
    "estado_calculo",
    "motivo_error",
    "periodo_lag",
    "version_t",
    "version_lag",
]


def _estado_derivado(estado_t: str, estado_base: object) -> str:
    """Estado de una fila derivada: `parcial` si el fuente en `t` o en la base lo es."""
    return "parcial" if estado_t == "parcial" or estado_base == "parcial" else "ok"


def _motivo_faltante(valor_t: float, valor_base: float) -> str:
    if pd.isna(valor_t) and pd.isna(valor_base):
        return "sin valor replicado en t ni en periodo base"
    if pd.isna(valor_base):
        return "sin valor replicado en periodo base"
    return "sin valor replicado en t"


def _motivo_indefinido(valor_t: float, valor_base: float) -> str:
    """Explica por qué una fila con dato en `t` y en la base quedó `indefinido`.

    Mismas 3 causas que revientan la corrida con `en_indefinido="error"`
    (`_variacion_contra_periodo_base`/`variacion_desde`), en el orden en que
    se evalúan ahí.
    """
    if not np.isfinite(valor_t) or not np.isfinite(valor_base):
        return "indice_replicado no finito en t o en periodo base"
    if valor_base == 0:
        return "base=0 en periodo base"
    return "variacion_pp resultante no finita (overflow)"


def _extraer_columna_cobertura(df_reporte_fuente: pd.DataFrame) -> pd.Series | None:
    if "cobertura_genericos_pct" in df_reporte_fuente.columns:
        return df_reporte_fuente["cobertura_genericos_pct"]
    return None


def _validar_en_indefinido(en_indefinido: str) -> None:
    """Rechaza cualquier valor fuera de `{"error", "marcar"}`.

    `EnIndefinido` es un `Literal` -- no se aplica en runtime, así que un typo
    (`en_indefinido="marcar_"`) pasaría por la rama implícita "no es 'error'"
    y excluiría filas en silencio sin que nadie lo pida. Valida incondicional
    al entrar, no solo dentro del `if invalido.any()` -- si no, el typo pasa
    desapercibido en cualquier corrida donde no haya filas indefinidas.
    """
    if en_indefinido not in ("error", "marcar"):
        raise InvarianteViolado(
            f"en_indefinido='{en_indefinido}' inválido; usa 'error' o 'marcar'."
        )


def _verificar_combinacion_unica(largo: pd.DataFrame) -> None:
    """Exige que `largo` describa una sola combinación `(agregacion, rubro)`.

    `resultado.manifiesto` puede traer varias entradas (varias `version`, ej.
    tras un `empalmar`), pero todas deben compartir la misma `agregacion`/
    `rubro` -- si no, tomar `agregacion`/`rubro` de `largo.iloc[0]` (como hacen
    `_variacion_contra_periodo_base` y `variacion_desde`) mezclaría en
    silencio filas de corridas distintas bajo una etiqueta que solo describe
    la primera. Negociado en `/negociar-hallazgos` (hallazgo #3, 2026-09-01).
    """
    combinaciones = largo[["agregacion", "rubro"]].drop_duplicates()
    if len(combinaciones) != 1:
        vistas = list(combinaciones.itertuples(index=False, name=None))
        raise InvarianteViolado(
            f"El resultado combina más de una (agregacion, rubro): {vistas}. "
            "variaciones requiere una sola combinación por corrida."
        )


def variacion_periodica(
    resultado: ResultadoIndice,
    frecuencia: Frecuencia,
    en_indefinido: EnIndefinido = "error",
) -> ResultadoVariacion:
    """Variación de cada periodo contra N periodos anteriores según `frecuencia`.

    Args:
        en_indefinido: `"error"` (default) preserva el comportamiento
            histórico -- ver `Raises`. `"marcar"` excluye esas filas de `.df`
            en vez de fallar toda la corrida; quedan en `.diagnostico` con
            `estado_calculo="indefinido"` y el motivo puntual (base=0, no
            finito, u overflow), el resto de `agregacion`/`rubro` calcula
            normal. Ver `EnIndefinido` arriba.

    Raises:
        InvarianteViolado: Si `en_indefinido` no está en `{"error", "marcar"}`;
            si `frecuencia` no está en el catálogo válido; si ningún periodo
            resulta computable; o si con `en_indefinido="error"` (default)
            alguna fila computable (con dato en `t` y en la base) tiene
            `indice_replicado` no finito, base exactamente 0, o `variacion_pp`
            resultante no finita (overflow: extremos finitos que producen un
            cociente infinito). Una fila NO computable (`t` o base sin dato)
            puede conservar un valor no finito en `.reporte` sin disparar esta
            validación — no afecta `variacion_pp`, que ya la excluye por no
            ser computable.
    """
    if frecuencia not in LAG_MENSUAL:
        raise InvarianteViolado(
            f"Frecuencia '{frecuencia}' no aplica a periodos mensuales. "
            f"Válidas: {sorted(LAG_MENSUAL)}."
        )
    periodos_atras = LAG_MENSUAL[frecuencia]

    def calcular_periodo_base(periodo_t: Periodo) -> Periodo:
        return restar_meses(periodo_t, periodos_atras)

    return _variacion_contra_periodo_base(
        resultado, calcular_periodo_base, f"periodica_{frecuencia}", "", en_indefinido
    )


def variacion_acumulada_anual(
    resultado: ResultadoIndice, en_indefinido: EnIndefinido = "error"
) -> ResultadoVariacion:
    """Variación de cada periodo contra diciembre del año anterior.

    Args:
        en_indefinido: ver `variacion_periodica` -- mismo contrato.

    Raises:
        InvarianteViolado: Si `en_indefinido` no está en `{"error", "marcar"}`;
            si ningún periodo resulta computable; o si con
            `en_indefinido="error"` (default) alguna fila computable (con dato
            en `t` y en la base) tiene `indice_replicado` no finito, base
            exactamente 0, o `variacion_pp` resultante no finita (overflow:
            extremos finitos que producen un cociente infinito). Una fila NO
            computable (`t` o base sin dato) puede conservar un valor no
            finito en `.reporte` sin disparar esta validación — no afecta
            `variacion_pp`, que ya la excluye por no ser computable.
    """

    def calcular_periodo_base(periodo_t: Periodo) -> Periodo:
        return PeriodoMensual(periodo_t.año - 1, 12)

    return _variacion_contra_periodo_base(
        resultado, calcular_periodo_base, "acumulada_anual", "", en_indefinido
    )


def _variacion_contra_periodo_base(
    resultado: ResultadoIndice,
    calcular_periodo_base: Callable[[Periodo], Periodo],
    clase_variacion: str,
    descripcion: str,
    en_indefinido: EnIndefinido = "error",
) -> ResultadoVariacion:
    """Núcleo de `variacion_periodica` y `variacion_acumulada_anual`.

    `calcular_periodo_base` mapea cada periodo `t` a su periodo base.
    """
    _validar_en_indefinido(en_indefinido)
    largo = resultado.resultado.largo
    _verificar_combinacion_unica(largo)
    versiones: list[VersionCanasta] = [m.version for m in resultado.manifiesto]
    agregacion = str(largo["agregacion"].iloc[0])
    rubro = str(largo["rubro"].iloc[0])

    indices_por_fila = largo.index.get_level_values("indice")
    periodos_por_fila = largo.index.get_level_values("periodo")
    valores_en_t = largo["indice_replicado"]

    periodos_base = [calcular_periodo_base(p) for p in periodos_por_fila]
    multiindice_base = pd.MultiIndex.from_arrays(
        [periodos_base, indices_por_fila], names=["periodo", "indice"]
    )
    valores_en_base = pd.Series(
        valores_en_t.reindex(multiindice_base).to_numpy(), index=largo.index
    )
    estados_en_base = pd.Series(
        largo["estado_calculo"].reindex(multiindice_base).to_numpy(), index=largo.index
    )
    versiones_en_base = pd.Series(
        largo["version"].reindex(multiindice_base).to_numpy(), index=largo.index
    )
    periodos_base_series = pd.Series(periodos_base, index=largo.index, dtype=object)

    variacion_pp = (valores_en_t / valores_en_base - 1.0) * 100.0
    computable = valores_en_t.notna() & valores_en_base.notna()

    invalido = computable & (
        ~np.isfinite(valores_en_t.astype(float))
        | ~np.isfinite(valores_en_base.astype(float))
        | (valores_en_base == 0)
        | ~np.isfinite(variacion_pp.astype(float))
    )
    indefinido = pd.Series(False, index=largo.index)
    if invalido.any():
        if en_indefinido == "error":
            primera_fila_invalida = largo.index[invalido][0]
            raise InvarianteViolado(
                f"variaciones: indice_replicado no finito, base=0, o variacion_pp resultante "
                f"no finita (overflow) en {int(invalido.sum())} fila(s) computable(s); "
                f"ejemplo {primera_fila_invalida}. Usa en_indefinido='marcar' para excluir "
                "estas filas del resultado en vez de fallar toda la corrida."
            )
        # en_indefinido == "marcar": estas filas salen de `.df` (mismo trato
        # que una fila sin dato crudo) y quedan documentadas en `.diagnostico`
        # con estado_calculo="indefinido" -- ver `_construir_reporte_y_diagnostico`.
        indefinido = invalido
        computable = computable & ~invalido

    estados_derivados = pd.Series(
        [
            _estado_derivado(estado_t, estado_base)
            for estado_t, estado_base in zip(largo["estado_calculo"], estados_en_base)
        ],
        index=largo.index,
    )

    df_out = pd.DataFrame(
        {
            "agregacion": agregacion,
            "rubro": rubro,
            "clase_variacion": clase_variacion,
            "variacion_pp": variacion_pp,
            "estado_calculo": estados_derivados,
            "version_t": largo["version"],
        },
        index=largo.index,
    )[computable].sort_index()
    if df_out.empty:
        raise InvarianteViolado(
            f"Sin periodos computables para clase '{clase_variacion}'. "
            "Se requieren datos suficientes en el periodo base."
        )

    df_reporte, df_diagnostico = _construir_reporte_y_diagnostico(
        largo,
        _extraer_columna_cobertura(resultado.reporte),
        valores_en_t,
        valores_en_base,
        versiones_en_base,
        periodos_base_series,
        multiindice_base,
        estados_derivados,
        computable,
        indefinido,
        versiones,
        agregacion,
        rubro,
        clase_variacion,
    )
    manifiesto = ManifestDerivado(
        versiones=versiones,
        agregacion=agregacion,
        rubro=rubro,
        clase=clase_variacion,
        descripcion=descripcion,
        fecha=datetime.now(),
    )
    return ResultadoVariacion(df_out, manifiesto, df_reporte, df_diagnostico)


def _construir_reporte_y_diagnostico(
    largo: pd.DataFrame,
    cobertura: pd.Series | None,
    valores_en_t: pd.Series,
    valores_en_base: pd.Series,
    versiones_en_base: pd.Series,
    periodos_base: pd.Series,
    multiindice_base: pd.MultiIndex,
    estados_derivados: pd.Series,
    computable: pd.Series,
    indefinido: pd.Series,
    versiones: list[VersionCanasta],
    agregacion: str,
    rubro: str,
    clase_variacion: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye `df_reporte` (todas las filas) y `df_diagnostico` (no computables).

    `computable` ya excluye las filas `indefinido` (`en_indefinido="marcar"`,
    ver `_variacion_contra_periodo_base`) -- acá solo se usa para distinguir su
    motivo (`"indefinido"`, dato presente pero matemáticamente indefinido) del
    de una fila sin dato crudo (`"sin_datos"`).
    """
    if cobertura is not None:
        cobertura_t = cobertura.reindex(largo.index)
        cobertura_base = pd.Series(
            cobertura.reindex(multiindice_base).to_numpy(), index=largo.index
        )
    else:
        cobertura_t = pd.Series(float("nan"), index=largo.index)
        cobertura_base = pd.Series(float("nan"), index=largo.index)

    estados_reporte: list[str] = []
    motivos: list[object] = []
    for es_computable, es_indefinido, estado, valor_t, valor_base in zip(
        computable, indefinido, estados_derivados, valores_en_t, valores_en_base
    ):
        if es_computable:
            estados_reporte.append(estado)
            motivos.append(float("nan"))
        elif es_indefinido:
            estados_reporte.append("indefinido")
            motivos.append(_motivo_indefinido(valor_t, valor_base))
        else:
            estados_reporte.append("sin_datos")
            motivos.append(_motivo_faltante(valor_t, valor_base))

    df_reporte = pd.DataFrame(
        {
            "estado_calculo": estados_reporte,
            "motivo_error": motivos,
            "periodo_lag": periodos_base,
            "indice_t": valores_en_t.to_numpy(),
            "indice_lag": valores_en_base.to_numpy(),
            "version_t": largo["version"].to_numpy(),
            "version_lag": versiones_en_base.to_numpy(),
            "cobertura_pct_t": cobertura_t.to_numpy(),
            "cobertura_pct_lag": cobertura_base.to_numpy(),
        },
        index=largo.index,
        columns=_COLS_REPORTE,
    ).sort_index()

    no_computable = ~computable
    df_diagnostico = pd.DataFrame(
        {
            "versiones": ",".join(str(v) for v in versiones),
            "agregacion": agregacion,
            "rubro": rubro,
            "clase_variacion": clase_variacion,
            "periodo": largo.index.get_level_values("periodo"),
            "indice": largo.index.get_level_values("indice"),
            "estado_calculo": estados_reporte,
            "motivo_error": motivos,
            "periodo_lag": periodos_base.to_numpy(),
            "version_t": largo["version"].to_numpy(),
            "version_lag": versiones_en_base.to_numpy(),
        },
        index=largo.index,
        columns=_COLS_DIAGNOSTICO,
    )[no_computable].reset_index(drop=True)

    return df_reporte, df_diagnostico


def variacion_desde(
    resultado: ResultadoIndice,
    desde: Periodo,
    hasta: Periodo | None = None,
    incluir_parciales: bool = True,
    en_indefinido: EnIndefinido = "error",
) -> ResultadoVariacion:
    """Variación total del rango `[desde, hasta]`; una fila por índice.

    Con `incluir_parciales=True`, un índice sin dato exacto en `desde`/`hasta`
    usa el primer/último periodo válido del rango; el periodo real usado se
    registra en `indices_parciales`.

    Args:
        en_indefinido: `"error"` (default) preserva el comportamiento
            histórico -- ver `Raises`. `"marcar"` excluye ese `indice` de
            `.df` en vez de fallar toda la corrida (los demás índices del
            rango calculan normal); queda en `.diagnostico` con
            `estado_calculo="indefinido"` y el motivo puntual.

    Raises:
        InvarianteViolado: Si `en_indefinido` no está en `{"error", "marcar"}`;
            si `desde`/`hasta` no existen en el resultado; si `hasta` es
            anterior a `desde`; si ningún índice tiene datos computables en el
            rango; o si con `en_indefinido="error"` (default) algún
            `indice_replicado` en alguno de los dos extremos no es finito, el
            extremo `desde` (la base) es exactamente 0 (el extremo `hasta` sí
            puede ser 0: produce `variacion_pp = -100`), o la `variacion_pp`
            resultante no es finita (overflow).
    """
    _validar_en_indefinido(en_indefinido)
    largo = resultado.resultado.largo
    _verificar_combinacion_unica(largo)
    versiones_manifiesto: list[VersionCanasta] = [m.version for m in resultado.manifiesto]
    agregacion = str(largo["agregacion"].iloc[0])
    rubro = str(largo["rubro"].iloc[0])

    periodos_todos = sorted(set(largo.index.get_level_values("periodo")))
    if desde not in periodos_todos:
        raise InvarianteViolado(f"El periodo 'desde' ({desde}) no existe en el resultado.")
    hasta_efectivo: Periodo = hasta if hasta is not None else periodos_todos[-1]
    if hasta is not None and hasta not in periodos_todos:
        raise InvarianteViolado(f"El periodo 'hasta' ({hasta}) no existe en el resultado.")
    if hasta_efectivo < desde:
        raise InvarianteViolado(
            f"'hasta' ({hasta_efectivo}) no puede ser anterior a 'desde' ({desde})."
        )

    rango = [p for p in periodos_todos if desde <= p <= hasta_efectivo]
    valores_replicados = largo["indice_replicado"]
    estados_calculo = largo["estado_calculo"]
    version_por_fila = largo["version"]
    cobertura = _extraer_columna_cobertura(resultado.reporte)
    indices = sorted(set(largo.index.get_level_values("indice")))
    versiones_str = ",".join(str(v) for v in versiones_manifiesto)

    filas_resultado: list[dict[str, object]] = []
    filas_reporte: list[dict[str, object]] = []
    filas_diagnostico: list[dict[str, object]] = []
    filas_parciales: list[dict[str, object]] = []

    for indice in indices:
        periodos_validos = [p for p in rango if pd.notna(valores_replicados.get((p, indice)))]
        desde_real = resolver_extremo(
            desde, periodos_validos, incluir_parciales=incluir_parciales, primero=True
        )
        hasta_real = resolver_extremo(
            hasta_efectivo, periodos_validos, incluir_parciales=incluir_parciales, primero=False
        )

        if desde_real is None or hasta_real is None:
            valor_desde = float(valores_replicados.get((desde, indice), float("nan")))
            valor_hasta = float(valores_replicados.get((hasta_efectivo, indice), float("nan")))
            motivo = _motivo_faltante(valor_hasta, valor_desde)
            filas_reporte.append(
                _construir_fila_reporte(
                    hasta_efectivo,
                    indice,
                    desde,
                    "sin_datos",
                    motivo,
                    valor_hasta,
                    valor_desde,
                    version_por_fila,
                    cobertura,
                )
            )
            filas_diagnostico.append(
                {
                    "versiones": versiones_str,
                    "agregacion": agregacion,
                    "rubro": rubro,
                    "clase_variacion": "desde",
                    "periodo": hasta_efectivo,
                    "indice": indice,
                    "estado_calculo": "sin_datos",
                    "motivo_error": motivo,
                    "periodo_lag": desde,
                    "version_t": version_por_fila.get((hasta_efectivo, indice), float("nan")),
                    "version_lag": version_por_fila.get((desde, indice), float("nan")),
                }
            )
            continue

        valor_desde = float(valores_replicados.at[(desde_real, indice)])  # type: ignore[arg-type]
        valor_hasta = float(valores_replicados.at[(hasta_real, indice)])  # type: ignore[arg-type]
        if not (math.isfinite(valor_desde) and math.isfinite(valor_hasta) and valor_desde != 0):
            if en_indefinido == "error":
                raise InvarianteViolado(
                    f"variacion_desde: indice_replicado no finito, o base (desde)=0, para "
                    f"'{indice}' entre {desde_real} y {hasta_real}. Usa en_indefinido='marcar' "
                    "para excluir este índice del resultado en vez de fallar toda la corrida."
                )
            _marcar_indefinido_desde(
                filas_reporte,
                filas_diagnostico,
                hasta_real,
                indice,
                desde_real,
                valor_hasta,
                valor_desde,
                version_por_fila,
                cobertura,
                versiones_str,
                agregacion,
                rubro,
            )
            continue
        variacion_pp = (valor_hasta / valor_desde - 1.0) * 100.0
        if not math.isfinite(variacion_pp):
            if en_indefinido == "error":
                raise InvarianteViolado(
                    f"variacion_desde: variacion_pp resultante no finita (overflow) para "
                    f"'{indice}' entre {desde_real} y {hasta_real}. Usa en_indefinido='marcar' "
                    "para excluir este índice del resultado en vez de fallar toda la corrida."
                )
            _marcar_indefinido_desde(
                filas_reporte,
                filas_diagnostico,
                hasta_real,
                indice,
                desde_real,
                valor_hasta,
                valor_desde,
                version_por_fila,
                cobertura,
                versiones_str,
                agregacion,
                rubro,
            )
            continue
        estado = _estado_derivado(
            str(estados_calculo.at[(hasta_real, indice)]),
            str(estados_calculo.at[(desde_real, indice)]),
        )
        # incluir_parciales=False excluye índices con estado derivado parcial.
        if incluir_parciales or estado != "parcial":
            filas_resultado.append(
                {
                    "periodo": hasta_real,
                    "indice": indice,
                    "agregacion": agregacion,
                    "rubro": rubro,
                    "clase_variacion": "desde",
                    "variacion_pp": variacion_pp,
                    "estado_calculo": estado,
                    "version_t": int(version_por_fila.at[(hasta_real, indice)]),  # type: ignore[arg-type]
                }
            )
        filas_reporte.append(
            _construir_fila_reporte(
                hasta_real,
                indice,
                desde_real,
                estado,
                float("nan"),
                valor_hasta,
                valor_desde,
                version_por_fila,
                cobertura,
            )
        )
        if desde_real != desde or hasta_real != hasta_efectivo:
            filas_parciales.append(
                {
                    "indice": indice,
                    "periodo_desde_real": desde_real,
                    "periodo_hasta_real": hasta_real,
                }
            )

    if not filas_resultado:
        raise InvarianteViolado(
            f"Ningún índice tiene datos computables en el rango [{desde}, {hasta_efectivo}]."
        )

    df_out = pd.DataFrame(filas_resultado).set_index(["periodo", "indice"]).sort_index()
    df_reporte = (
        pd.DataFrame(filas_reporte, columns=["periodo", "indice", *_COLS_REPORTE])
        .set_index(["periodo", "indice"])
        .sort_index()
    )
    df_diagnostico = pd.DataFrame(filas_diagnostico, columns=_COLS_DIAGNOSTICO)
    indices_parciales = pd.DataFrame(
        filas_parciales,
        columns=["indice", "periodo_desde_real", "periodo_hasta_real"],
    ).set_index("indice")

    manifiesto = ManifestDerivado(
        versiones=versiones_manifiesto,
        agregacion=agregacion,
        rubro=rubro,
        clase="desde",
        descripcion=f"desde {desde} hasta {hasta_efectivo}",
        fecha=datetime.now(),
    )
    return ResultadoVariacion(df_out, manifiesto, df_reporte, df_diagnostico, indices_parciales)


def _marcar_indefinido_desde(
    filas_reporte: list[dict[str, object]],
    filas_diagnostico: list[dict[str, object]],
    periodo: Periodo,
    indice: str,
    periodo_base: Periodo,
    valor_t: float,
    valor_base: float,
    version_por_fila: pd.Series,
    cobertura: pd.Series | None,
    versiones_str: str,
    agregacion: str,
    rubro: str,
) -> None:
    """Registra un `indice` como `indefinido` en reporte y diagnóstico (`en_indefinido="marcar"`).

    No agrega fila a `filas_resultado` -- el índice queda fuera de `.df` para
    este rango, mismo trato que un índice sin dato computable.
    """
    motivo = _motivo_indefinido(valor_t, valor_base)
    filas_reporte.append(
        _construir_fila_reporte(
            periodo,
            indice,
            periodo_base,
            "indefinido",
            motivo,
            valor_t,
            valor_base,
            version_por_fila,
            cobertura,
        )
    )
    filas_diagnostico.append(
        {
            "versiones": versiones_str,
            "agregacion": agregacion,
            "rubro": rubro,
            "clase_variacion": "desde",
            "periodo": periodo,
            "indice": indice,
            "estado_calculo": "indefinido",
            "motivo_error": motivo,
            "periodo_lag": periodo_base,
            "version_t": version_por_fila.get((periodo, indice), float("nan")),
            "version_lag": version_por_fila.get((periodo_base, indice), float("nan")),
        }
    )


def _construir_fila_reporte(
    periodo: Periodo,
    indice: str,
    periodo_base: Periodo,
    estado: str,
    motivo: object,
    valor_t: float,
    valor_base: float,
    version_por_fila: pd.Series,
    cobertura: pd.Series | None,
) -> dict[str, object]:
    """Construye una fila del `df_reporte` de `variacion_desde`."""

    def obtener_cobertura(periodo_objetivo: Periodo) -> float:
        if cobertura is None:
            return float("nan")
        try:
            return float(cobertura.at[(periodo_objetivo, indice)])  # type: ignore[arg-type]
        except KeyError:
            return float("nan")

    return {
        "periodo": periodo,
        "indice": indice,
        "estado_calculo": estado,
        "motivo_error": motivo,
        "periodo_lag": periodo_base,
        "indice_t": valor_t,
        "indice_lag": valor_base,
        "version_t": version_por_fila.get((periodo, indice), float("nan")),
        "version_lag": version_por_fila.get((periodo_base, indice), float("nan")),
        "cobertura_pct_t": obtener_cobertura(periodo),
        "cobertura_pct_lag": obtener_cobertura(periodo_base),
    }
