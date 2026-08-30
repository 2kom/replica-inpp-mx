from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import replica_inpp as rep
from replica_inpp.api.indices import calcular_indice
from replica_inpp.dominio.errores import CanastaSinGenericos, InvarianteViolado, VersionNoCoincide
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.periodos import PeriodoMensual

_PERIODOS = [PeriodoMensual(2019, 7), PeriodoMensual(2019, 8)]
_GENERICOS = ["001", "070", "339", "460"]


def _canasta() -> CanastaINPP:
    df = pd.DataFrame(
        {
            "generico": ["soya", "petroleo", "acero", "transporte"],
            "codigo sector": ["11", "21", "31", "48"],
            "sector": ["11", "21", "31", "48"],
            "codigo subsector": ["111", "211", "311", "481"],
            "subsector": ["111", "211", "311", "481"],
            "codigo rama": ["1111", "2111", "3111", "4811"],
            "rama": ["1111", "2111", "3111", "4811"],
            "codigo subrama": ["11111", "21111", "31111", "48111"],
            "subrama": ["11111", "21111", "31111", "48111"],
            "codigo clase": ["111111", "211111", "311111", "481111"],
            "clase": ["111111", "211111", "311111", "481111"],
            "produccion total": [25.0, 25.0, 25.0, 25.0],
            "bienes intermedios": [50.0, 0.0, 50.0, 0.0],
            "bienes finales": [10.0, 10.0, 40.0, 40.0],
            "demanda interna total": [20.0, 20.0, 30.0, 30.0],
            "demanda interna consumo": [20.0, 20.0, 30.0, 30.0],
            "demanda interna capital": [20.0, 20.0, 30.0, 30.0],
            "exportaciones": [25.0, 25.0, 25.0, 25.0],
            "encadenamiento total": [None] * 4,
            "encadenamiento produccion nacional": [None] * 4,
            "encadenamiento exportacion": [None] * 4,
            "encadenamiento uso final": [None] * 4,
        },
        index=_GENERICOS,
    )
    return CanastaINPP(df, 2019)


def _serie() -> SerieNormalizada:
    df = pd.DataFrame(
        {
            "generico": ["soya", "petroleo", "acero", "transporte"],
            _PERIODOS[0]: [100.0, 100.0, 100.0, 100.0],
            _PERIODOS[1]: [102.0, 110.0, 105.0, 101.0],
        },
        index=_GENERICOS,
    )
    return SerieNormalizada(df, "produccion_total")


def test_calcular_indice_retorna_resultado_indice() -> None:
    r = calcular_indice(_canasta(), _serie(), "INPP", rubro="produccion_total")
    assert isinstance(r, ResultadoIndice)


def test_rep_calcular_indice_expuesto_en_fachada_publica() -> None:
    # a diferencia de los demás tests de este archivo (import directo del
    # módulo), este pasa por `replica_inpp.__init__` -- protege contra que se
    # rompa el wiring de `__all__`/import ahí sin que ningún test lo note.
    assert "calcular_indice" in rep.__all__
    r = rep.calcular_indice(_canasta(), _serie(), "INPP", rubro="produccion_total")
    assert isinstance(r, ResultadoIndice)


def test_calcular_indice_delega_en_laspeyres_directo() -> None:
    r = calcular_indice(_canasta(), _serie(), "INPP", rubro="produccion_total")
    assert r.manifiesto[0].calculador == "LaspeyresDirecto"


def test_calcular_indice_propaga_sin_petroleo() -> None:
    r = calcular_indice(_canasta(), _serie(), "INPP", rubro="produccion_total", sin_petroleo=True)
    assert r.manifiesto[0].sin_petroleo is True


def test_calcular_indice_agregacion_invalida_lanza_invariante_violado() -> None:
    with pytest.raises(InvarianteViolado):
        calcular_indice(_canasta(), _serie(), "agregacion_inventada", rubro="produccion_total")


# -- canasta y serie de versiones distintas (negociación 2026-08-27) --------
# CSV sintéticos vía tmp_path -- corre en CI, no depende de datos reales
# marcados requires_data. Reproduce con la fachada pública completa: cargar_canasta +
# cargar_serie + calcular_indice.


def test_calcular_indice_canasta_y_serie_de_version_distinta_falla(tmp_path: Path) -> None:
    columnas_peso = [
        "produccion total",
        "bienes intermedios",
        "bienes finales",
        "demanda interna total",
        "demanda interna consumo",
        "demanda interna capital",
        "exportaciones",
    ]
    canasta_df = pd.DataFrame(
        {
            "codigo": ["001"],
            "generico": ["soya"],
            "sector": ["11 agricultura"],
            "subsector": ["111 agricultura"],
            "rama": ["1111 cultivo"],
            "subrama": ["11111 cultivo de soya"],
            "clase": ["111110 cultivo de soya"],
            **{col: [100.0] for col in columnas_peso},
            "encadenamiento total": [None],
            "encadenamiento produccion nacional": [None],
            "encadenamiento exportacion": [None],
            "encadenamiento uso final": [None],
        }
    )
    ruta_canasta = tmp_path / "canasta_2019.csv"
    canasta_df.to_csv(ruta_canasta, index=False)

    # recorte "mercado_nacional" (prefijo "1"), único genérico "001", declara
    # Base Julio 2012=100 -- cargar_serie(..., 2012) no falla (base coincide),
    # la version SÍ queda en attrs["version"] para que calcular_indice la vea.
    encabezado = "Instituto Nacional de Estadística y Geografía\n\n\n\n\n"
    serie_df = pd.DataFrame(
        {
            "Título": [
                "Índice nacional de precios productor. Base Julio 2012=100 (SCIAN 2013), "
                "Índices de precios de genéricos para mercado nacional, 1001 Soya"
            ],
            "Cifra": ["Indices"],
            "Serie": ["11111"],
            "Ene 2019": ["100.00"],
        }
    )
    ruta_serie = tmp_path / "serie_2012.csv"
    ruta_serie.write_text(encabezado + serie_df.to_csv(index=False), encoding="utf-8")

    canasta = rep.cargar_canasta(str(ruta_canasta), 2019)
    serie = rep.cargar_serie(str(ruta_serie), 2012)
    assert serie.df.attrs["version"] == 2012  # precondición: la mezcla es real

    with pytest.raises(VersionNoCoincide, match="2019.*2012|2012.*2019"):
        rep.calcular_indice(canasta, serie, "INPP")


def test_calcular_indice_sin_metadata_de_version_no_valida() -> None:
    # SerieNormalizada construida a mano (sin pasar por cargar_serie) no trae
    # attrs["version"] -- no hay nada contra qué comparar, no debe fallar por
    # esto. Mismo comportamiento de siempre para los demás tests de este repo.
    r = calcular_indice(_canasta(), _serie(), "INPP", rubro="produccion_total")
    assert isinstance(r, ResultadoIndice)


# -- sin_petroleo deja el grupo vacío, vía la fachada pública (negociación 2026-08-27,
# segunda ronda) -- el equivalente de dominio/calculo/test_calculo_laspeyres_directo.py
# ::test_sin_petroleo_deja_grupo_vacio_lanza_canasta_sin_genericos pero atravesando
# rep.calcular_indice, para que una regresión en la fachada (que capture, transforme o
# deje de propagar CanastaSinGenericos) sí quede detectada.


# -- canasta 2025 rechazada: la fachada todavía no expone LaspeyresEncadenado
# (negociación 2026-08-30) -- calcular_indice calculaba en silencio con
# LaspeyresDirecto sobre canasta 2025, dando un número incorrecto (mezcla la
# serie 2025, en escala absoluta continua de 2019, con ponderadores 2025) sin
# avisar. Ver dominio/calculo/laspeyres_encadenado.py::LaspeyresEncadenado.


def test_calcular_indice_canasta_2025_lanza_invariante_violado() -> None:
    df = _canasta().df.copy()
    canasta_2025 = CanastaINPP(df, 2025)
    with pytest.raises(InvarianteViolado, match="LaspeyresEncadenado"):
        calcular_indice(canasta_2025, _serie(), "INPP", rubro="produccion_total")


def test_rep_calcular_indice_canasta_2025_lanza_invariante_violado() -> None:
    # A diferencia del test anterior (import directo, protege el módulo API en
    # aislamiento), este atraviesa `rep.calcular_indice` -- el defecto original
    # (negociación 2026-08-30, ronda 2) estaba justo ahí: la fachada podía volver
    # a apuntar a una implementación sin guardia y este test seguiría siendo el
    # único que lo notaría, igual que ya pasa con
    # `test_calcular_indice_sin_petroleo_deja_grupo_vacio_lanza_canasta_sin_genericos`
    # arriba.
    df = _canasta().df.copy()
    canasta_2025 = CanastaINPP(df, 2025)
    with pytest.raises(InvarianteViolado, match="LaspeyresEncadenado"):
        rep.calcular_indice(canasta_2025, _serie(), "INPP", rubro="produccion_total")


def test_calcular_indice_sin_petroleo_deja_grupo_vacio_lanza_canasta_sin_genericos() -> None:
    df = pd.DataFrame(
        {
            "generico": ["petroleo"],
            "codigo sector": ["21"],
            "sector": ["21"],
            "codigo subsector": ["211"],
            "subsector": ["211"],
            "codigo rama": ["2111"],
            "rama": ["2111"],
            "codigo subrama": ["21111"],
            "subrama": ["21111"],
            "codigo clase": ["211111"],
            "clase": ["211111"],
            "produccion total": [100.0],
            "bienes intermedios": [100.0],
            "bienes finales": [100.0],
            "demanda interna total": [100.0],
            "demanda interna consumo": [100.0],
            "demanda interna capital": [100.0],
            "exportaciones": [100.0],
            "encadenamiento total": [None],
            "encadenamiento produccion nacional": [None],
            "encadenamiento exportacion": [None],
            "encadenamiento uso final": [None],
        },
        index=["070"],
    )
    canasta_solo_petroleo = CanastaINPP(df, 2019)
    serie_df = pd.DataFrame(
        {"generico": ["petroleo"], _PERIODOS[0]: [100.0], _PERIODOS[1]: [110.0]}, index=["070"]
    )
    serie = SerieNormalizada(serie_df, "produccion_total")

    with pytest.raises(CanastaSinGenericos):
        rep.calcular_indice(
            canasta_solo_petroleo, serie, "INPP", rubro="produccion_total", sin_petroleo=True
        )
