from __future__ import annotations

import numpy as np
import pandas as pd

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import RecorteINPP

_COLUMNA_GENERICO = "generico"


class SerieNormalizada:
    """Representa una matriz de índices por código de genérico y periodo, de un recorte.

    Args:
        df: DataFrame en formato ancho con `codigo` como índice, columna
            `generico` (texto, decorativa) más columnas `PeriodoMensual` con
            valores numéricos finitos no negativos o `NaN`.
        recorte: Recorte de destino/etapa al que pertenece la serie. Real, no
            decorativo — confirmado que el valor numérico del índice elemental
            cambia según el recorte para una parte de los genéricos.

    Raises:
        InvarianteViolado: Si el índice contiene duplicados o cadenas vacías,
            si falta la columna `generico`, si no hay al menos una columna de
            periodo, si alguna columna de periodo no es `PeriodoMensual`, si
            hay columnas de periodo duplicadas, si el DataFrame contiene
            valores negativos, o si contiene valores no finitos (`inf`/`-inf`).

    Esquema del DataFrame:
        Índice (str): `codigo` — código de 3 dígitos del genérico. Válido solo
            dentro de la MISMA versión de canasta; un genérico puede
            fusionarse/desagregarse y cambiar de código entre versiones — no
            comparar `codigo` entre versiones sin pasar antes por el módulo de
            correspondencia (sin escribir todavía).
        Columna `generico` (str): nombre del genérico. Puramente decorativa —
            no se usa para cruzar ni calcular nada, no está garantizado que
            coincida entre fuentes para el mismo `codigo`.
        Columnas (PeriodoMensual): una columna por mes.
        Valores (float64/NaN): índice del genérico en cada periodo.

    Example:
        DataFrame interno:
        | codigo | generico | Ene 2019 | Feb 2019 | Mar 2019 |
        | :----- | :------- | -------: | -------: | -------: |
        | 001    | soya...  | 100.0    | 102.2    | 100.9    |
        | 002    | frijol   | 100.0    | 101.2    | 101.0    |

        `NaN` indica que no hubo índice disponible para un genérico en ese
        periodo.
    """

    def __init__(self, df: pd.DataFrame, recorte: RecorteINPP) -> None:
        if df.index.duplicated().any():
            raise InvarianteViolado("El índice del DataFrame no puede contener valores duplicados.")
        if (df.index == "").any():
            raise InvarianteViolado("El índice del DataFrame no puede contener cadenas vacías.")
        if _COLUMNA_GENERICO not in df.columns:
            raise InvarianteViolado(f"Falta la columna decorativa '{_COLUMNA_GENERICO}'.")

        columnas_periodo = [c for c in df.columns if c != _COLUMNA_GENERICO]
        if len(columnas_periodo) == 0:
            raise InvarianteViolado("El DataFrame debe tener al menos una columna de periodo.")
        if not all(isinstance(col, PeriodoMensual) for col in columnas_periodo):
            raise InvarianteViolado(
                f"Las columnas del DataFrame, salvo '{_COLUMNA_GENERICO}', deben ser del tipo "
                "PeriodoMensual."
            )
        if pd.Index(columnas_periodo).duplicated().any():
            raise InvarianteViolado(
                "Las columnas del DataFrame no pueden contener periodos duplicados."
            )

        valores = df[columnas_periodo]
        if (valores < 0).any().any():
            raise InvarianteViolado("Los valores del DataFrame no pueden ser negativos.")
        if not (valores.isna() | np.isfinite(valores.astype(float))).all().all():
            raise InvarianteViolado("Los valores del DataFrame deben ser finitos o NaN (no ±inf).")

        # Orden cronológico explícito, mismo motivo que en replica-inpc-mx: relleno
        # bfill/ffill futuro opera por posición física de columna. `generico` va
        # primero, no participa del orden cronológico.
        self._df = df[[_COLUMNA_GENERICO, *sorted(columnas_periodo)]]
        self._recorte: RecorteINPP = recorte

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def recorte(self) -> RecorteINPP:
        """Devuelve el recorte de destino/etapa al que pertenece la serie."""
        return self._recorte

    def _repr_html_(self) -> str:
        """Renderiza la serie como tabla HTML en entornos interactivos."""
        return self._df._repr_html_()  # type: ignore[operator]
