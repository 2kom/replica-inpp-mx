from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import pandas as pd

from replica_inpp.dominio.errores import InvarianteViolado


class Vista:
    """Envuelve un DataFrame con MultiIndex `(periodo, indice)` y expone formato largo o ancho bajo demanda.

    Args:
        df: DataFrame con MultiIndex de 2 niveles `(periodo, indice)`.
        columnas: columna(s) de `df` a exponer en `.ancho` — pivotea por `periodo`.
    """

    def __init__(self, df: pd.DataFrame, columnas: list[str]) -> None:
        self._df = df
        self._columnas = columnas

    @property
    def largo(self) -> pd.DataFrame:
        """DataFrame completo tal cual se almacenó, con metadata."""
        return self._df

    @property
    def ancho(self) -> pd.DataFrame:
        """Pivotea `columnas` por `periodo`: filas=`indice` si 1 columna, MultiIndex `(indice, metrica)` si N."""
        if len(self._columnas) == 1:
            return self._df[self._columnas[0]].unstack("periodo")
        sub = self._df[self._columnas].rename_axis(columns="metrica")
        return sub.stack(future_stack=True).unstack("periodo")  # type: ignore

    def _repr_html_(self) -> str:
        """Delega el render HTML a `.largo`."""
        return self._df._repr_html_()  # type: ignore[operator]


class Resultado(ABC):
    """Contrato base para resultados de cálculo (`ResultadoIndice`, `ResultadoVariacion`, ...).

    Valida la estructura mínima del `df` recibido; la subclase pasa solo la
    columna calculada, nunca el DataFrame completo.

    Args:
        df: DataFrame con MultiIndex `(periodo, indice)` y exactamente 1 columna
            — el valor calculado por la subclase.

    Raises:
        InvarianteViolado: `df` vacío, MultiIndex distinto de `(periodo, indice)`,
            más de 1 columna, o índice con combinaciones duplicadas.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise InvarianteViolado("Resultado.df no puede estar vacío")
        if (
            not isinstance(df.index, pd.MultiIndex)
            or df.index.nlevels != 2
            or list(df.index.names) != ["periodo", "indice"]
        ):
            raise InvarianteViolado(
                "Resultado.df requiere MultiIndex exacto con niveles ('periodo', 'indice')"
            )
        if df.shape[1] != 1:
            raise InvarianteViolado("Resultado.df debe contener exactamente una columna calculada")
        if df.index.duplicated().any():
            raise InvarianteViolado("Resultado.df no puede tener índices duplicados")
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        """DataFrame mínimo validado — solo la columna calculada, sin metadata adicional."""
        return self._df

    @property
    @abstractmethod
    def resultado(self) -> Vista: ...

    @property
    @abstractmethod
    def resumen(self) -> pd.DataFrame: ...

    @property
    @abstractmethod
    def reporte(self) -> pd.DataFrame: ...

    @property
    @abstractmethod
    def diagnostico(self) -> pd.DataFrame: ...

    @abstractmethod
    def _repr_html_(self) -> str: ...

    def pipe(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Llama `fn(self, *args, **kwargs)` — encadenamiento estilo pandas."""
        return fn(self, *args, **kwargs)
