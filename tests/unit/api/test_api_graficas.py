from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pytest

import replica_inpp as rep
from replica_inpp.api import graficas
from replica_inpp.dominio.errores import PeriodoNoDisponible
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import ManifestCalculo, ManifestDerivado
from replica_inpp.infraestructura.graficacion import graficador

# --------------------------------------------------------------------------- helpers


def _manifiesto(
    version: int = 2019, agregacion: str = "SECTOR", rubro: str = "produccion_total"
) -> ManifestCalculo:
    return ManifestCalculo(
        version=version,  # type: ignore[arg-type]
        agregacion=agregacion,
        rubro=rubro,
        incluir_petroleo=True,
        calculador="LaspeyresDirecto",
        fecha=datetime(2024, 1, 1),
    )


def _resultado(
    periodos: list[Any], version: int = 2019, agregacion: str = "SECTOR"
) -> ResultadoIndice:
    filas = [
        {
            "periodo": p,
            "indice": "cat",
            "version": version,
            "agregacion": agregacion,
            "rubro": "produccion_total",
            "indice_replicado": 100.0,
            "estado_calculo": "ok",
            "motivo_error": None,
        }
        for p in periodos
    ]
    df = pd.DataFrame(filas)
    df.index = pd.MultiIndex.from_arrays(
        [df.pop("periodo"), df.pop("indice")], names=["periodo", "indice"]
    )
    reporte = df[[]].copy()
    diag = pd.DataFrame(
        columns=["periodo", "generico", "nivel_faltante", "tipo_faltante", "detalle"]
    )
    return ResultadoIndice(df, [_manifiesto(version, agregacion)], reporte, diag)


_P1 = PeriodoMensual(2018, 1)
_P2 = PeriodoMensual(2018, 2)
_P3 = PeriodoMensual(2018, 3)


def test_rep_graficar_expuesto_en_fachada_publica(mocker: Any) -> None:
    # A diferencia de los demás tests de este archivo (import directo de
    # `graficas`/`graficador`), este pasa por `replica_inpp.__init__` --
    # protege contra que se rompa el wiring de `__all__`/import ahí sin que
    # ningún otro test lo note.
    assert "graficar" in rep.__all__
    grafica_falsa = mocker.Mock()
    mocker.patch.object(graficador, "_construir_grafica_linea", return_value=grafica_falsa)
    r = _resultado([_P1])
    rep.graficar(r)
    grafica_falsa.draw.assert_called_once_with(show=True)


def _resultado_variacion(
    periodos: list[Any], agregacion: str = "SECTOR", clase: str = "periodica_mensual"
) -> ResultadoVariacion:
    filas = [
        {
            "periodo": p,
            "indice": "cat",
            "agregacion": agregacion,
            "rubro": "produccion_total",
            "clase_variacion": clase,
            "variacion_pp": 0.5,
            "estado_calculo": "ok",
        }
        for p in periodos
    ]
    df = pd.DataFrame(filas)
    df.index = pd.MultiIndex.from_arrays(
        [df.pop("periodo"), df.pop("indice")], names=["periodo", "indice"]
    )
    reporte = df[[]].copy()
    diag = pd.DataFrame(columns=["periodo", "indice", "estado_calculo", "motivo_error"])
    manifiesto = ManifestDerivado(
        versiones=[2019],  # type: ignore[arg-type]
        agregacion=agregacion,
        rubro="produccion_total",
        clase=clase,
        descripcion="",
        fecha=datetime(2024, 1, 1),
    )
    return ResultadoVariacion(df, manifiesto, reporte, diag)


# --------------------------------------------------------------------------- delegación
#
# Parametrizado por tipo (indice/variacion): ambos pasan por el mismo
# graficas.graficar() y delegan al mismo graficas._graficar -- es justamente
# lo que unifica la API, no dos funciones públicas separadas por tipo.


def _construir(tipo_resultado: str, periodos: list[Any], agregacion: str = "SECTOR") -> Any:
    if tipo_resultado == "indice":
        return _resultado(periodos, agregacion=agregacion)
    return _resultado_variacion(periodos, agregacion=agregacion)


@pytest.mark.parametrize("tipo_resultado", ["indice", "variacion"])
def test_graficar_sin_tramo_delega_sin_validar_periodos(mocker: Any, tipo_resultado: str) -> None:
    fn = mocker.patch.object(graficas, "_graficar")
    r = _construir(tipo_resultado, [_P1, _P2])
    graficas.graficar(r)
    fn.assert_called_once_with(r, None, None, None)


@pytest.mark.parametrize("tipo_resultado", ["indice", "variacion"])
def test_graficar_convierte_desde_y_hasta(mocker: Any, tipo_resultado: str) -> None:
    fn = mocker.patch.object(graficas, "_graficar")
    r = _construir(tipo_resultado, [_P1, _P2, _P3])
    graficas.graficar(r, desde="Ene 2018", hasta="Mar 2018")
    fn.assert_called_once_with(r, None, _P1, _P3)


@pytest.mark.parametrize("tipo_resultado", ["indice", "variacion"])
def test_graficar_pasa_comparacion(mocker: Any, tipo_resultado: str) -> None:
    fn = mocker.patch.object(graficas, "_graficar")
    r = _construir(tipo_resultado, [_P1])
    comparacion = _construir(tipo_resultado, [_P1], agregacion="INPP")
    graficas.graficar(r, comparacion=comparacion)
    fn.assert_called_once_with(r, comparacion, None, None)


# --------------------------------------------------------------------------- validación de tramo


@pytest.mark.parametrize("tipo_resultado", ["indice", "variacion"])
def test_graficar_desde_ausente_lanza_periodo_no_disponible(
    mocker: Any, tipo_resultado: str
) -> None:
    mocker.patch.object(graficas, "_graficar")
    r = _construir(tipo_resultado, [_P1, _P2])
    with pytest.raises(PeriodoNoDisponible):
        graficas.graficar(r, desde="Ene 2030")


@pytest.mark.parametrize("tipo_resultado", ["indice", "variacion"])
def test_graficar_hasta_ausente_lanza_periodo_no_disponible(
    mocker: Any, tipo_resultado: str
) -> None:
    mocker.patch.object(graficas, "_graficar")
    r = _construir(tipo_resultado, [_P1, _P2])
    with pytest.raises(PeriodoNoDisponible):
        graficas.graficar(r, hasta="Ene 2030")


@pytest.mark.parametrize("tipo_resultado", ["indice", "variacion"])
def test_graficar_desde_presente_solo_en_comparacion_no_lanza(
    mocker: Any, tipo_resultado: str
) -> None:
    # el periodo pedido puede venir de resultado O de comparacion -- la
    # union de ambos es lo que realmente termina en el panel.
    fn = mocker.patch.object(graficas, "_graficar")
    r = _construir(tipo_resultado, [_P1], agregacion="INPP")
    comparacion = _construir(tipo_resultado, [_P1, _P3])
    graficas.graficar(r, comparacion=comparacion, desde="Mar 2018")
    fn.assert_called_once_with(r, comparacion, _P3, None)


# --------------------------------------------------------------------------- _periodos_disponibles


def test_periodos_disponibles_sin_comparacion() -> None:
    r = _resultado([_P1, _P2])
    assert graficas._periodos_disponibles(r, None) == {_P1, _P2}


def test_periodos_disponibles_union_con_comparacion() -> None:
    r = _resultado([_P1])
    comparacion = _resultado([_P2], agregacion="INPP")
    assert graficas._periodos_disponibles(r, comparacion) == {_P1, _P2}
