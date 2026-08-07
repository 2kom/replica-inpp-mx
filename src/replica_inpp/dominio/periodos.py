"""Periodos del dominio del INPP.

El INPP se publica con periodicidad mensual, por lo que `PeriodoMensual` es el
único tipo de periodo del dominio. `periodo_desde_str` es el punto de entrada
público para parsear periodos (se re-exportará en la fachada del paquete,
igual que en replica-inpc-mx); su despacho por cantidad de palabras es el
punto de extensión previsto si alguna periodicidad adicional se incorporara.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass

import pandas as pd

from replica_inpp.dominio.errores import InvarianteViolado, PeriodoNoInterpretable

# Meses str -> int
_MESES: dict[str, int] = {
    "Ene": 1,
    "Feb": 2,
    "Mar": 3,
    "Abr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Ago": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dic": 12,
}

# Meses int -> str
_MESES_INV: dict[int, str] = {v: k for k, v in _MESES.items()}


def _ultimo_dia(año: int, mes: int) -> int:
    return calendar.monthrange(año, mes)[1]


def _normalizar_espacios(texto: str) -> str:
    return " ".join(texto.split())


def _validar_año_mes(año: int, mes: int) -> None:
    if not (1 <= mes <= 12):
        raise InvarianteViolado(f"mes debe estar entre 1 y 12, se recibio {mes}")
    if año <= 0:
        raise InvarianteViolado(f"año debe ser un entero positivo, se recibio {año}")


@dataclass(frozen=True, order=True)
class PeriodoMensual:
    """Representa un periodo mensual del dominio.

    Es inmutable, su orden natural es cronológico y se puede usar como
    clave hashable. El INPP se publica con periodicidad mensual, por lo que
    este es el único tipo de periodo del dominio.

    Args:
        año: Año calendario. Debe ser entero positivo.
        mes: Mes calendario. Debe estar entre 1 y 12.

    Raises:
        InvarianteViolado: Si `año` no es positivo o `mes` no está entre 1 y 12.
    """

    año: int
    mes: int

    def __post_init__(self) -> None:
        _validar_año_mes(self.año, self.mes)

    def __str__(self) -> str:
        return f"{_MESES_INV[self.mes]} {self.año}"

    def __repr__(self) -> str:
        return f"PeriodoMensual({self.año}, {self.mes})"

    @classmethod
    def desde_str(cls, periodo_str: str) -> PeriodoMensual:
        """Construye un `PeriodoMensual` desde su representación textual canónica.

        Args:
            periodo_str: Texto en formato `"Mes AAAA"`, por ejemplo `"Jul 2024"`.
                El mes es insensible a mayúsculas (`"jul"`, `"JUL"`, `"Jul"`
                son equivalentes) y se toleran espacios extra o al
                inicio/final.

        Raises:
            PeriodoNoInterpretable: Si el texto no corresponde a un periodo válido.
            InvarianteViolado: Si el texto es interpretable pero año o mes
                están fuera de rango.
        """
        try:
            mes_str, año_str = _normalizar_espacios(periodo_str).split(" ")
            mes = _MESES[mes_str.capitalize()]
            año = int(año_str)
            return cls(año, mes)
        except InvarianteViolado:
            raise
        except (KeyError, ValueError) as e:
            raise PeriodoNoInterpretable(
                f"Formato de periodo mensual inválido: '{periodo_str}'. Se esperaba formato 'Mes AAAA'"
            ) from e

    def to_timestamp(self) -> pd.Timestamp:
        """Convierte el periodo a `pd.Timestamp` usando el último día del mes."""
        return pd.Timestamp(year=self.año, month=self.mes, day=_ultimo_dia(self.año, self.mes))


def periodo_desde_str(texto: str) -> PeriodoMensual:
    """Construye un `PeriodoMensual` a partir de su texto canónico `"Mes AAAA"`.

    Insensible a mayúsculas y tolera espacios extra o al inicio/final.

    Raises:
        PeriodoNoInterpretable: Si el texto no corresponde al formato mensual.
        InvarianteViolado: Si el texto encaja en el formato pero algún
            componente está fuera de rango.
    """
    texto = _normalizar_espacios(texto)
    partes = texto.split(" ")
    if len(partes) == 2:
        return PeriodoMensual.desde_str(texto)
    raise PeriodoNoInterpretable(
        f"Formato de periodo no reconocido: '{texto}'. Se esperaba 'Mes AAAA' (mensual)."
    )
