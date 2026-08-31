from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import pandas as pd

from replica_inpp.dominio.errores import ErrorCalculo
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import RANGOS_CANASTAS, VersionCanasta

_COLUMNAS_DIAGNOSTICO = [
    "version",
    "agregacion",
    "rubro",
    "periodo",
    "generico",
    "nivel_faltante",
    "tipo_faltante",
    "detalle",
]


def _validar_serie_cubre_grupo(
    genericos_del_grupo: pd.Index,
    serie: SerieNormalizada,
    version: VersionCanasta,
    agregacion: str,
    rubro: str,
) -> None:
    """Rechaza una serie a la que le falte algún genérico del grupo que se va a calcular."""
    faltantes = sorted(str(g) for g in genericos_del_grupo.difference(serie.df.index))
    if not faltantes:
        return
    muestra = ", ".join(faltantes[:3])
    raise ErrorCalculo(
        f"la serie no tiene {len(faltantes)} de los {len(genericos_del_grupo)} genéricos "
        f"que '{agregacion}'/'{rubro}' necesitan de la canasta {version} (por ejemplo: "
        f"{muestra}). Suele significar que la canasta y la serie son de versiones distintas."
    )


def _rellenar_dato_serie_faltante(
    df_serie: pd.DataFrame,
    version: VersionCanasta,
    agregacion: str,
    rubro: str,
) -> tuple[pd.DataFrame, pd.DataFrame, set[object]]:
    """Rellena NaN vía bfill→ffill por fila; documenta cada relleno con su periodo fuente."""
    mascara_faltante = df_serie.isna()
    if not mascara_faltante.any(axis=None):
        return df_serie, pd.DataFrame(columns=_COLUMNAS_DIAGNOSTICO), set()

    df_serie_rellenada = df_serie.bfill(axis=1).ffill(axis=1).infer_objects(copy=False)
    mascara_rellenada = mascara_faltante & df_serie_rellenada.notna()
    periodos_rellenados: set[object] = set(
        df_serie_rellenada.columns[mascara_rellenada.any(axis=0)]
    )

    # Ubica el periodo fuente de cada relleno propagando la ETIQUETA de columna
    # (en vez del dato) con el mismo bfill→ffill: evita re-escanear columnas por
    # celda rellenada, igual que en replica-inpc-mx.
    etiquetas_columna = pd.DataFrame(
        np.tile(df_serie.columns.to_numpy(), (len(df_serie.index), 1)),
        index=df_serie.index,
        columns=df_serie.columns,
    ).where(df_serie.notna())
    periodo_fuente_adelante = etiquetas_columna.bfill(axis=1)
    periodo_fuente_atras = etiquetas_columna.ffill(axis=1)
    periodo_fuente = periodo_fuente_adelante.where(
        periodo_fuente_adelante.notna(), periodo_fuente_atras
    )

    celdas_rellenadas = cast("pd.Series[bool]", mascara_rellenada.stack())
    celdas_rellenadas = celdas_rellenadas[celdas_rellenadas]
    if celdas_rellenadas.empty:
        return df_serie_rellenada, pd.DataFrame(columns=_COLUMNAS_DIAGNOSTICO), periodos_rellenados

    fuentes = periodo_fuente.stack().reindex(celdas_rellenadas.index)
    diagnostico = pd.DataFrame(
        {
            "version": version,
            "agregacion": agregacion,
            "rubro": rubro,
            "periodo": celdas_rellenadas.index.get_level_values(1),
            "generico": celdas_rellenadas.index.get_level_values(0),
            "nivel_faltante": "periodo",
            "tipo_faltante": "rellenado",
            "detalle": "NaN sustituido con valor de " + fuentes.astype(str),
        },
        columns=_COLUMNAS_DIAGNOSTICO,
    )

    return df_serie_rellenada, diagnostico, periodos_rellenados


def _recortar_series_fecha(df_serie: pd.DataFrame, version: VersionCanasta) -> pd.DataFrame:
    """Recorta las columnas de periodo de la serie al rango vigente de la versión de canasta."""
    periodo_inicio, periodo_fin = RANGOS_CANASTAS[version]
    columnas_en_rango = [
        periodo
        for periodo in df_serie.columns
        if isinstance(periodo, PeriodoMensual)
        and periodo >= periodo_inicio
        and (periodo_fin is None or periodo <= periodo_fin)
    ]
    if not columnas_en_rango:
        raise ErrorCalculo(
            f"La serie no tiene ningún periodo dentro del rango vigente de la "
            f"canasta versión {version} ({periodo_inicio} - {periodo_fin or 'sin límite'}). "
            "Revisa que la serie y la canasta correspondan a la misma versión."
        )
    return df_serie[columnas_en_rango]


class CalculadorBase(ABC):
    """Contrato abstracto para estrategias de cálculo del dominio.

    Implementaciones: `LaspeyresDirecto` (Etapa 2, sin encadenar, canastas
    2012/2019) y `LaspeyresEncadenado` (Etapa 3, encadenamiento 2025).
    `api/indices.py::calcular_indice` despacha a una u otra según
    `canasta.version`.
    """

    @abstractmethod
    def calcular(
        self,
        canasta: CanastaINPP,
        serie: SerieNormalizada,
        agregacion: str,
        rubro: str | None = None,
        sin_petroleo: bool = False,
    ) -> ResultadoIndice:
        """Calcula `ResultadoIndice` para una canasta y serie dadas."""


def _laspeyres_por_grupo(
    numerador: pd.DataFrame,
    ponderador: pd.Series,
    cat_por_gen: pd.Series,
) -> pd.DataFrame:
    """Laspeyres por grupo: Σ(ponderador·numerador)/Σponderador, agrupado por `cat_por_gen`."""
    resultado = (
        numerador.multiply(ponderador, axis=0)
        .groupby(cat_por_gen)
        .sum()
        .divide(ponderador.groupby(cat_por_gen).sum(), axis=0)
    )
    valores = resultado.to_numpy(dtype=float)
    # numerador ya viene finito-o-NaN (SerieNormalizada lo garantiza), pero
    # ponderar puede desbordar a inf incluso con entradas finitas (ej. 1e308)
    if (~pd.isna(valores) & ~np.isfinite(valores)).any():
        raise ErrorCalculo(
            "El cálculo de Laspeyres por grupo produjo un valor no finito — "
            "posible desbordamiento al ponderar la serie."
        )
    return resultado


def _construir_diagnostico(
    df_serie: pd.DataFrame,
    version: VersionCanasta,
    agregacion: str,
    rubro: str,
) -> pd.DataFrame:
    """Lista (periodo, generico) faltantes con schema de diagnóstico.

    Para subgrupos, `df_serie` debe ser el del subgrupo. Una fila por celda NaN.
    """
    mascara_faltante = df_serie.isna()
    genericos_idx, periodos_idx = mascara_faltante.to_numpy().nonzero()
    if genericos_idx.size == 0:
        return pd.DataFrame(columns=_COLUMNAS_DIAGNOSTICO)

    genericos_faltantes = mascara_faltante.index[genericos_idx]
    periodos_faltantes = mascara_faltante.columns[periodos_idx]

    return pd.DataFrame(
        {
            "version": version,
            "agregacion": agregacion,
            "rubro": rubro,
            "periodo": periodos_faltantes,
            "generico": genericos_faltantes,
            "nivel_faltante": "periodo",
            "tipo_faltante": "indice",
            "detalle": "valor NaN en serie publicada",
        },
        columns=_COLUMNAS_DIAGNOSTICO,
    )
