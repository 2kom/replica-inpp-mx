from __future__ import annotations

import pandas as pd

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.base import Resultado, Vista
from replica_inpp.dominio.tipos import ManifestDerivado

_CLASES_VALIDAS = frozenset(
    {
        "periodica_mensual",
        "periodica_bimestral",
        "periodica_trimestral",
        "periodica_cuatrimestral",
        "periodica_semestral",
        "periodica_anual",
        "acumulada_anual",
        "desde",
    }
)
_COLUMNAS_MINIMAS = {"agregacion", "rubro", "clase_variacion", "variacion_pp", "estado_calculo"}
_ESTADOS_VALIDOS = frozenset({"ok", "parcial"})
_ORDEN_SEVERIDAD = {"ok": 0, "parcial": 1}


class ResultadoVariacion(Resultado):
    """Resultado de una variación (inflación) calculada sobre un `ResultadoIndice`.

    Args:
        df_resultado: DataFrame con MultiIndex `(periodo, indice)` — ver Esquema abajo.
        manifiesto: Proveniencia de la corrida (agregación, rubro, clase, descripción, fecha).
        df_reporte: DataFrame con el mismo MultiIndex `(periodo, indice)` y columnas
            de cobertura/calidad por fila. Es un **superconjunto** de
            `df_resultado`: incluye las filas no computables que el largo omite —
            por eso las validaciones de derivados extraen de acá los periodos a
            consultar.
        df_diagnostico: DataFrame plano de faltantes, esquema `DiagnosticoFaltantes`.
        indices_parciales: Solo cuando `clase_variacion == "desde"` — índices
            intermedios usados en el cálculo acumulado; `None` en el resto de clases.

    Raises:
        InvarianteViolado: Si `df_resultado` no trae las columnas mínimas
            (`agregacion`, `rubro`, `clase_variacion`, `variacion_pp`,
            `estado_calculo`), si `clase_variacion` no es homogénea o no está en
            el catálogo válido, si `indices_parciales is not None` no coincide
            exactamente con `clase_variacion == "desde"`, si
            `manifiesto.agregacion`/`manifiesto.rubro`/`manifiesto.clase` no
            coinciden con los de `df_resultado`, o si `estado_calculo` tiene
            valores fuera de `{ok, parcial}`.

    Esquema de `df_resultado` (MultiIndex: `(periodo, indice)`):
        agregacion (str): nivel de agrupación del índice sobre el que se calculó
            la variación (`"INPP"`, `"SECTOR"`, ...).
        rubro (str): columna de peso del índice sobre el que se calculó la
            variación (`"produccion_total"`, ...).
        clase_variacion (str): periodicidad de la variación, ver catálogo en
            `_CLASES_VALIDAS` (ej. `periodica_mensual`, `acumulada_anual`, `desde`).
        variacion_pp (float): variación en puntos porcentuales; finito cuando
            `estado_calculo` es `ok`/`parcial`. La finitud la garantiza el
            productor (`dominio/calculo/variaciones.py`), no este constructor
            — mismo criterio que `ResultadoIndice.indice_replicado`: la
            invariante vive en la capa que construye el DataFrame, el modelo
            solo lo empaqueta.
        estado_calculo (str): `{ok, parcial}` — `parcial` cuando el periodo base
            o final se calculó con datos incompletos (`estado_calculo` del
            `ResultadoIndice` de origen, propagado; ver `_estado_derivado`).
    """

    def __init__(
        self,
        df_resultado: pd.DataFrame,
        manifiesto: ManifestDerivado,
        df_reporte: pd.DataFrame,
        df_diagnostico: pd.DataFrame,
        indices_parciales: pd.DataFrame | None = None,
    ) -> None:
        faltantes = _COLUMNAS_MINIMAS - set(df_resultado.columns)
        if faltantes:
            raise InvarianteViolado(
                f"ResultadoVariacion.df_resultado requiere columnas {sorted(faltantes)}"
            )
        clases = set(df_resultado["clase_variacion"].unique())
        if len(clases) != 1:
            raise InvarianteViolado(
                "ResultadoVariacion.df_resultado.clase_variacion debe ser homogénea"
            )
        clase = clases.pop()
        if clase not in _CLASES_VALIDAS:
            raise InvarianteViolado(
                f"clase_variacion '{clase}' no está en {sorted(_CLASES_VALIDAS)}"
            )
        if (clase == "desde") != (indices_parciales is not None):
            raise InvarianteViolado("indices_parciales is not None ⇔ clase_variacion == 'desde'")
        if manifiesto.clase != clase:
            raise InvarianteViolado(
                f"manifiesto.clase='{manifiesto.clase}' no coincide con "
                f"df_resultado.clase_variacion='{clase}'"
            )
        agregaciones = set(df_resultado["agregacion"].unique())
        if len(agregaciones) != 1:
            raise InvarianteViolado(
                "ResultadoVariacion.df_resultado['agregacion'] debe ser homogéneo"
            )
        agregacion_df = agregaciones.pop()
        if manifiesto.agregacion != agregacion_df:
            raise InvarianteViolado(
                f"manifiesto.agregacion='{manifiesto.agregacion}' no coincide con "
                f"df_resultado['agregacion']='{agregacion_df}'"
            )
        rubros = set(df_resultado["rubro"].unique())
        if len(rubros) != 1:
            raise InvarianteViolado("ResultadoVariacion.df_resultado['rubro'] debe ser homogéneo")
        rubro_df = rubros.pop()
        if manifiesto.rubro != rubro_df:
            raise InvarianteViolado(
                f"manifiesto.rubro='{manifiesto.rubro}' no coincide con "
                f"df_resultado['rubro']='{rubro_df}'"
            )
        estados_invalidos = set(df_resultado["estado_calculo"].unique()) - _ESTADOS_VALIDOS
        if estados_invalidos:
            raise InvarianteViolado(
                f"ResultadoVariacion.df_resultado.estado_calculo solo admite 'ok'/'parcial'; "
                f"recibió {sorted(estados_invalidos, key=repr)}"
            )
        super().__init__(df_resultado[["variacion_pp"]])
        self._df_resultado = df_resultado
        self._manifiesto = manifiesto
        self._df_reporte = df_reporte
        self._df_diagnostico = df_diagnostico
        self._indices_parciales = indices_parciales

    @property
    def manifiesto(self) -> ManifestDerivado:
        return self._manifiesto

    @property
    def resultado(self) -> Vista:
        return Vista(self._df_resultado, ["variacion_pp"])

    @property
    def reporte(self) -> pd.DataFrame:
        return self._df_reporte

    @property
    def diagnostico(self) -> pd.DataFrame:
        return self._df_diagnostico

    @property
    def indices_parciales(self) -> pd.DataFrame | None:
        return self._indices_parciales

    @property
    def resumen(self) -> pd.DataFrame:
        df = self._df_resultado
        estados = df["estado_calculo"].unique()
        estado = max(estados, key=lambda e: _ORDEN_SEVERIDAD[e])
        periodos = df.index.get_level_values("periodo")
        return pd.DataFrame(
            [
                {
                    "agregacion": self._manifiesto.agregacion,
                    "rubro": self._manifiesto.rubro,
                    "clase_variacion": self._manifiesto.clase,
                    "descripcion": self._manifiesto.descripcion,
                    "estado_calculo": estado,
                    "periodo_inicio": min(periodos),
                    "periodo_fin": max(periodos),
                    "fecha": self._manifiesto.fecha,
                }
            ]
        )

    def _repr_html_(self) -> str:
        return self.resumen._repr_html_()  # type: ignore[operator]
