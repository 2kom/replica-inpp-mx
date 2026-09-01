from __future__ import annotations

import pandas as pd

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.base import Resultado, Vista
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import ManifestCalculo

_COLUMNAS_MINIMAS = {"version", "agregacion", "rubro", "indice_replicado", "estado_calculo"}
_ORDEN_SEVERIDAD = {"ok": 0, "rellenado": 1, "parcial": 2, "sin_datos": 3, "fallida": 4}
_ESTADOS_VALIDOS = frozenset(_ORDEN_SEVERIDAD)


class ResultadoIndice(Resultado):
    """Resultado de un cálculo de índice elemental o agregado sobre una canasta del INPP.

    Args:
        df_resultado: DataFrame con MultiIndex `(periodo, indice)` — ver Esquema abajo.
        manifiesto: Un `ManifestCalculo` por corrida elemental que compone este
            resultado; `empalmar` (`dominio/conversion.py`) concatena listas sin
            colapsar, igual que en `replica-inpc-mx`.
        df_reporte: DataFrame paralelo a `df_resultado` (mismo MultiIndex) con
            columnas de cobertura/calidad por fila.
        df_diagnostico: DataFrame plano, una fila por celda `(periodo, generico)`
            sin dato — no comparte índice con `df_resultado`.
        nombres: nombre legible por valor de `indice` (índice: mismos valores que
            el nivel `indice` de `df_resultado`), opcional. Solo tiene sentido con
            agregación por nivel SCIAN y cuando la canasta trae nombre real (no
            bare) — con `"INPP"`/`"MERCANCIAS_SERVICIOS"` o canasta solo con
            códigos, se omite (`None`): `.resultado.ancho` no agrega columna
            `nombre` en ese caso, en vez de agregarla vacía.
        periodo_referencia: Ancla de escala — el periodo cuyo valor se fijó en
            `valor_base` (default 100) al rebasar, y respecto del cual está
            expresada toda la serie. `None` = resultado en escala natural del
            cálculo (recién salido de `calcular_indice`); lo setea `rebasar()`.
            A diferencia de `replica-inpc-mx`, este repo no tiene `_frontera`
            (esa existe ahí para no perder precisión al promediar 2 quincenas
            en `a_mensual` — el INPP siempre es mensual, no hay promedio que
            perder precisión).

    Raises:
        InvarianteViolado: Si `manifiesto` está vacío, si `df_resultado` no trae
            las columnas mínimas (`version`, `agregacion`, `rubro`,
            `indice_replicado`, `estado_calculo`), si `estado_calculo` tiene
            valores fuera de `{ok, rellenado, parcial, sin_datos, fallida}`, si
            dos entradas de `manifiesto` repiten la misma combinación
            `(version, agregacion, rubro)`, o si el conjunto de combinaciones
            `(version, agregacion, rubro)` de `manifiesto` no coincide EXACTO
            (en ambas direcciones) con el de `df_resultado` — ni manifiesto sin
            filas que lo respalden, ni filas huérfanas sin manifiesto que las
            declare.

    Esquema de `df_resultado` (MultiIndex: `(periodo, indice)`):
        version (int): versión de canasta de la corrida (2012, 2019 o 2025).
        agregacion (str): nivel de agrupación (`"INPP"`, `"SECTOR"`, `"SUBSECTOR"`,
            `"RAMA"`, `"SUBRAMA"`, `"CLASE"` o `"MERCANCIAS_SERVICIOS"`).
        rubro (str): columna de peso usada (`"produccion_total"`,
            `"bienes_intermedios"`, `"bienes_finales"`, `"demanda_interna_total"`,
            `"demanda_interna_consumo"`, `"demanda_interna_capital"` o
            `"exportaciones"`).
        indice_replicado (float): valor del índice; `NaN` si `estado_calculo` es
            `sin_datos` o `fallida`.
        indice_incidencia (float): columna interna, valor antes de cualquier
            rebase/encadenamiento posterior; no se expone en ninguna vista
            pública. No es obligatoria en `_COLUMNAS_MINIMAS` — `.resultado` la
            dropea con `errors="ignore"`, tolera que no esté.
        estado_calculo (str): `{ok, rellenado, parcial, sin_datos, fallida}`.
        motivo_error (str | None): motivo cuando `estado_calculo` no es `ok`,
            `parcial` ni `rellenado`.

    Nota sobre `incluir_petroleo`: es un campo de `ManifestCalculo`, no una
    columna de `df_resultado` — una sola corrida siempre tiene un valor fijo de
    `incluir_petroleo` (es un filtro aplicado antes de agrupar, no una
    categoría más que coexista con otras en el mismo resultado), así que no
    hace falta repetirlo por fila. Como `incluir_petroleo` no vive en
    `df_resultado`, no hay columna contra la cual validarlo — por eso
    `manifiesto` no puede repetir la misma combinación `(version, agregacion,
    rubro)` en dos entradas (aunque difieran en `incluir_petroleo`): dos
    manifiestos así apuntarían a las mismas filas sin que nada los distinga, y
    `.resumen` reportaría dos variantes que en realidad comparten un único
    cálculo.

    Example:
        `.resumen`:
        | version_agregacion_rubro_petroleo | estado_calculo | periodo_inicio | periodo_fin |
        | ---------------------------------- | -------------- | -------------- | ----------- |
        | 2019:SECTOR:produccion_total:con_petroleo | ok | Jul 2019 | Jun 2025 |
    """

    def __init__(
        self,
        df_resultado: pd.DataFrame,
        manifiesto: list[ManifestCalculo],
        df_reporte: pd.DataFrame,
        df_diagnostico: pd.DataFrame,
        nombres: pd.Series | None = None,
        periodo_referencia: PeriodoMensual | None = None,
    ) -> None:
        if not manifiesto:
            raise InvarianteViolado("ResultadoIndice.manifiesto no puede estar vacío")
        faltantes = _COLUMNAS_MINIMAS - set(df_resultado.columns)
        if faltantes:
            raise InvarianteViolado(
                f"ResultadoIndice.df_resultado requiere columnas mínimas {sorted(faltantes)}"
            )
        estados_invalidos = set(df_resultado["estado_calculo"].unique()) - _ESTADOS_VALIDOS
        if estados_invalidos:
            raise InvarianteViolado(
                f"ResultadoIndice.df_resultado.estado_calculo admite solo "
                f"{sorted(_ESTADOS_VALIDOS)}; recibió {sorted(estados_invalidos, key=repr)}"
            )
        combos_manifiesto = [(m.version, m.agregacion, m.rubro) for m in manifiesto]
        if len(set(combos_manifiesto)) != len(combos_manifiesto):
            raise InvarianteViolado(
                "ResultadoIndice.manifiesto no puede tener combinaciones "
                "(version, agregacion, rubro) repetidas — incluir_petroleo no distingue "
                "manifiestos porque no vive en df_resultado."
            )
        combos_df = set(
            df_resultado[["version", "agregacion", "rubro"]].itertuples(index=False, name=None)
        )
        if set(combos_manifiesto) != combos_df:
            solo_en_manifiesto = sorted(set(combos_manifiesto) - combos_df)
            solo_en_df = sorted(combos_df - set(combos_manifiesto))
            raise InvarianteViolado(
                "ResultadoIndice.manifiesto y df_resultado deben describir exactamente las "
                f"mismas combinaciones (version, agregacion, rubro); solo en manifiesto: "
                f"{solo_en_manifiesto}, solo en df_resultado: {solo_en_df}"
            )
        super().__init__(df_resultado[["indice_replicado"]])
        self._df_resultado = df_resultado
        self._manifiesto = manifiesto
        self._df_reporte = df_reporte
        self._df_diagnostico = df_diagnostico
        self._nombres = nombres
        self._periodo_referencia = periodo_referencia

    @property
    def manifiesto(self) -> list[ManifestCalculo]:
        return self._manifiesto

    @property
    def periodo_referencia(self) -> PeriodoMensual | None:
        return self._periodo_referencia

    @property
    def resultado(self) -> Vista:
        # `indice_incidencia` es columna interna; no se expone en la vista pública.
        return Vista(
            self._df_resultado.drop(columns=["indice_incidencia"], errors="ignore"),
            ["indice_replicado"],
            nombres=self._nombres,
        )

    @property
    def _completo(self) -> pd.DataFrame:
        """Vista interna con todas las columnas, incl. `indice_incidencia`.

        No es API pública — reservada para cuando exista el motor de incidencias
        (todavía sin implementar en este repo).
        """
        return self._df_resultado

    @property
    def reporte(self) -> pd.DataFrame:
        return self._df_reporte

    @property
    def diagnostico(self) -> pd.DataFrame:
        return self._df_diagnostico

    @property
    def resumen(self) -> pd.DataFrame:
        df = self._df_resultado
        filas = []
        for m in self._manifiesto:
            mascara = (
                (df["version"] == m.version)
                & (df["agregacion"] == m.agregacion)
                & (df["rubro"] == m.rubro)
            )
            subset = df[mascara]
            estados = subset["estado_calculo"].unique()
            estado = max(estados, key=lambda e: _ORDEN_SEVERIDAD[e])
            periodos = subset.index.get_level_values("periodo")
            periodo_inicio = min(periodos)
            periodo_fin = max(periodos)
            clave = (
                f"{m.version}:{m.agregacion}:{m.rubro}:"
                f"{'con_petroleo' if m.incluir_petroleo else 'sin_petroleo'}"
            )
            filas.append(
                {
                    "version_agregacion_rubro_petroleo": clave,
                    "estado_calculo": estado,
                    "periodo_inicio": periodo_inicio,
                    "periodo_fin": periodo_fin,
                    "fecha": m.fecha,
                }
            )
        return pd.DataFrame(filas).set_index("version_agregacion_rubro_petroleo")

    def _repr_html_(self) -> str:
        return self.resumen._repr_html_()  # type: ignore[operator]
