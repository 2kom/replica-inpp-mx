import operator
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from replica_inpp.dominio.errores import InvarianteViolado, PeriodoNoInterpretable
from replica_inpp.dominio.periodos import PeriodoMensual, periodo_desde_str

# --- Validación de construcción ---


@pytest.mark.parametrize(
    "mes",
    [13, 0, -1],  # cota superior, cota inferior, negativo
)
def test_mes_invalido(mes):
    with pytest.raises(InvarianteViolado):
        PeriodoMensual(2024, mes)


@pytest.mark.parametrize("año", [-1, 0])
def test_año_invalido(año):
    with pytest.raises(InvarianteViolado):
        PeriodoMensual(año, 1)


def test_construccion_valida():
    p = PeriodoMensual(2024, 7)
    assert p.año == 2024
    assert p.mes == 7


# --- desde_str ---


@pytest.mark.parametrize(
    "texto",
    [
        "formato incorrecto x",  # conteo de palabras
        "Xyz 2024",  # mes fuera de catálogo
        "Ene abcd",  # año no numérico
        "2024",  # una sola palabra
        "Ene 2024 extra",  # tres palabras
        "",  # vacío
    ],
)
def test_desde_str_invalido(texto):
    with pytest.raises(PeriodoNoInterpretable):
        PeriodoMensual.desde_str(texto)


def test_desde_str_valido():
    p = PeriodoMensual.desde_str("Jul 2024")
    assert p.año == 2024
    assert p.mes == 7


@pytest.mark.parametrize("mes_str", ["dic", "DIC", "Dic", "dIC"])
def test_desde_str_insensible_a_mayusculas(mes_str):
    p = PeriodoMensual.desde_str(f"{mes_str} 2024")
    assert p.mes == 12


def test_desde_str_normaliza_espacios_extra():
    p = PeriodoMensual.desde_str("  Jul   2018 ")
    assert p == PeriodoMensual(2018, 7)


def test_desde_str_año_invalido_lanza_invariante_violado():
    # texto interpretable, valor fuera de rango: InvarianteViolado del __init__
    # sale sin envolver, misma excepcion que la ruta de construccion directa
    with pytest.raises(InvarianteViolado):
        PeriodoMensual.desde_str("Ene 0")


# --- Representación ---


def test_str():
    assert str(PeriodoMensual(2024, 1)) == "Ene 2024"
    assert str(PeriodoMensual(2024, 12)) == "Dic 2024"


def test_repr():
    assert repr(PeriodoMensual(2024, 7)) == "PeriodoMensual(2024, 7)"


# --- Igualdad, orden, hash ---


def test_eq():
    p1 = PeriodoMensual(2024, 7)
    p2 = PeriodoMensual(2024, 7)
    p3 = PeriodoMensual(2024, 8)
    assert p1 == p2
    assert p1 != p3
    assert p1 != "no es un periodo"


def test_orden():
    p1 = PeriodoMensual(2024, 1)
    p2 = PeriodoMensual(2024, 6)
    p3 = PeriodoMensual(2025, 1)
    assert p1 < p2 < p3
    assert sorted([p3, p1, p2]) == [p1, p2, p3]


@pytest.mark.parametrize(
    "op,esperado",
    [
        (operator.le, True),
        (operator.lt, True),
        (operator.gt, False),
        (operator.ge, False),
    ],
)
def test_operadores_derivados(op, esperado):
    # <=, <, >, >= son generados por dataclass(order=True) a partir de (año, mes)
    p1 = PeriodoMensual(2024, 1)
    p2 = PeriodoMensual(2024, 6)
    assert op(p1, p2) is esperado


def test_orden_contra_tipo_ajeno_lanza_typeerror():
    with pytest.raises(TypeError):
        PeriodoMensual(2024, 7) < 5  # type: ignore[operator]


def test_hash():
    p1 = PeriodoMensual(2024, 7)
    p2 = PeriodoMensual(2024, 7)
    assert hash(p1) == hash(p2)
    assert len({p1, p2}) == 1  # mismo elemento, no se duplica en el set


@pytest.mark.parametrize(
    "periodo,atributo",
    [
        (PeriodoMensual(2024, 7), "mes"),
        (PeriodoMensual(2024, 7), "año"),
    ],
)
def test_inmutable(periodo, atributo):
    # mutar rompería el hash de un periodo ya usado como clave en dict/set/MultiIndex
    with pytest.raises(FrozenInstanceError):
        setattr(periodo, atributo, 999)


# --- to_timestamp ---


def test_to_timestamp():
    assert PeriodoMensual(2024, 1).to_timestamp() == pd.Timestamp(2024, 1, 31)
    assert PeriodoMensual(2024, 2).to_timestamp() == pd.Timestamp(2024, 2, 29)  # bisiesto
    assert PeriodoMensual(2023, 2).to_timestamp() == pd.Timestamp(2023, 2, 28)
    assert PeriodoMensual(2024, 4).to_timestamp() == pd.Timestamp(2024, 4, 30)  # mes de 30 días
    assert PeriodoMensual(2024, 11).to_timestamp() == pd.Timestamp(2024, 11, 30)


@pytest.mark.parametrize("texto", ["Ene 2024", "  jul   2018 ", "DIC 2025"])
def test_round_trip_str(texto):
    # str(PeriodoMensual.desde_str(s)) devuelve el formato canónico; re-parsearlo
    # debe ser idéntico al periodo original (round-trip estable)
    p = PeriodoMensual.desde_str(texto)
    assert PeriodoMensual.desde_str(str(p)) == p


# --- periodo_desde_str ---


def test_periodo_desde_str_mensual():
    p = periodo_desde_str("Jul 2024")
    assert isinstance(p, PeriodoMensual)
    assert p.año == 2024
    assert p.mes == 7


@pytest.mark.parametrize(
    "texto,mes",
    [
        ("DIC 2024", 12),
        ("dic 2024", 12),
        ("  Jul   2024 ", 7),
    ],
)
def test_periodo_desde_str_insensible_a_mayusculas_y_espacios(texto, mes):
    p = periodo_desde_str(texto)
    assert isinstance(p, PeriodoMensual)
    assert p.mes == mes


@pytest.mark.parametrize(
    "texto",
    [
        "formato totalmente incorrecto aqui",
        "2024",  # una palabra
        "1Q Ene 2024",  # formato quincenal no aplica al INPP
        "",  # vacío
    ],
)
def test_periodo_desde_str_invalido(texto):
    with pytest.raises(PeriodoNoInterpretable):
        periodo_desde_str(texto)


def test_periodo_desde_str_invariante_violado_propaga_limpio():
    # el dispatcher no debe re-envolver InvarianteViolado en PeriodoNoInterpretable
    with pytest.raises(InvarianteViolado):
        periodo_desde_str("Ene 0")
