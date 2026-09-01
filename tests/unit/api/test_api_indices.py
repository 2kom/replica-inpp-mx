from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import replica_inpp as rep
from replica_inpp.api.indices import calcular_indice
from replica_inpp.dominio.errores import CanastaSinGenericos, InvarianteViolado, VersionNoCoincide
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import VersionCanasta

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


def _serie(recorte: Any = "produccion_total") -> SerieNormalizada:
    df = pd.DataFrame(
        {
            "generico": ["soya", "petroleo", "acero", "transporte"],
            _PERIODOS[0]: [100.0, 100.0, 100.0, 100.0],
            _PERIODOS[1]: [102.0, 110.0, 105.0, 101.0],
        },
        index=_GENERICOS,
    )
    return SerieNormalizada(df, recorte)


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


def test_calcular_indice_propaga_incluir_petroleo() -> None:
    r = calcular_indice(
        _canasta(), _serie(), "INPP", rubro="produccion_total", incluir_petroleo=False
    )
    assert r.manifiesto[0].incluir_petroleo is False


def test_calcular_indice_incluir_petroleo_posicional_lanza_typeerror() -> None:
    # incluir_petroleo es keyword-only (negociación 2026-08-31): el 5º argumento
    # posicional que en el contrato viejo era `sin_petroleo` (con semántica
    # invertida) debe fallar explícito en vez de correr en silencio con el
    # significado contrario -- ver data/negociaciones/2026-08-31-*.md.
    with pytest.raises(TypeError):
        rep.calcular_indice(_canasta(), _serie(), "INPP", "produccion_total", True)


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


# -- incluir_petroleo=False deja el grupo vacío, vía la fachada pública (negociación
# 2026-08-27, segunda ronda) -- el equivalente de
# dominio/calculo/test_calculo_laspeyres_directo.py
# ::test_incluir_petroleo_false_deja_grupo_vacio_lanza_canasta_sin_genericos pero
# atravesando rep.calcular_indice, para que una regresión en la fachada (que capture,
# transforme o deje de propagar CanastaSinGenericos) sí quede detectada.


# -- canasta 2025 rechazada sin `referencia` (negociación 2026-08-30) --
# calcular_indice calculaba en silencio con LaspeyresDirecto sobre canasta
# 2025, dando un número incorrecto (mezcla la serie 2025, en escala absoluta
# continua de 2019, con ponderadores 2025) sin avisar. Ahora despacha a
# LaspeyresEncadenado para canasta.version=2025 (ver test de despacho más
# abajo), pero sigue exigiendo `referencia` explícita -- sin ella, rechaza.
# Ver dominio/calculo/laspeyres_encadenado.py::LaspeyresEncadenado.


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
    # `test_calcular_indice_incluir_petroleo_false_deja_grupo_vacio_lanza_canasta_sin_genericos`
    # arriba.
    df = _canasta().df.copy()
    canasta_2025 = CanastaINPP(df, 2025)
    with pytest.raises(InvarianteViolado, match="LaspeyresEncadenado"):
        rep.calcular_indice(canasta_2025, _serie(), "INPP", rubro="produccion_total")


def test_calcular_indice_canasta_2025_despacha_a_laspeyres_encadenado() -> None:
    canasta_anterior = _canasta()  # version 2019, encadenamiento vacío (como debe ser en 2019)

    traslape = PeriodoMensual(2025, 7)
    serie_anterior = SerieNormalizada(
        pd.DataFrame(
            {"generico": ["soya", "petroleo", "acero", "transporte"], traslape: [100.0] * 4},
            index=_GENERICOS,
        ),
        "produccion_total",
    )
    # `referencia`: el ResultadoIndice de la MISMA combinación (INPP/
    # produccion_total/incluir_petroleo=True) ya calculado con canasta 2019 --
    # típicamente sale de una llamada previa a `calcular_indice`.
    referencia = rep.calcular_indice(
        canasta_anterior, serie_anterior, "INPP", rubro="produccion_total"
    )

    df_2025 = _canasta().df.copy()
    columnas_encadenamiento = [
        "encadenamiento total",
        "encadenamiento produccion nacional",
        "encadenamiento exportacion",
        "encadenamiento uso final",
    ]
    df_2025[columnas_encadenamiento] = 1.0
    canasta_2025 = CanastaINPP(df_2025, 2025)

    serie_2025 = SerieNormalizada(
        pd.DataFrame(
            {"generico": ["soya", "petroleo", "acero", "transporte"], traslape: [100.0] * 4},
            index=_GENERICOS,
        ),
        "produccion_total",
    )

    r = rep.calcular_indice(
        canasta_2025,
        serie_2025,
        "INPP",
        rubro="produccion_total",
        referencia=referencia,
    )
    assert r.manifiesto[0].calculador == "LaspeyresEncadenado"
    assert r.manifiesto[0].version == 2025


def test_calcular_indice_canasta_2025_incluir_petroleo_false_via_fachada() -> None:
    # Recorrido completo vía `rep.calcular_indice` (no `LaspeyresEncadenado`
    # directo) con `incluir_petroleo=False` y pesos DISTINTOS entre canasta
    # anterior (25/25/25/25) y nueva (10/20/30/40) -- mismo escenario ya probado en
    # dominio/calculo/test_calculo_laspeyres_encadenado.py
    # ::test_incluir_petroleo_false_excluye_070_de_i_tramo_y_de_factor_h, ahora
    # atravesando la fachada completa (incluida la construcción de `referencia` con
    # incluir_petroleo=False vía `rep.calcular_indice`). Un mutante que reenvíe
    # siempre `incluir_petroleo=True` al despachar a `LaspeyresEncadenado` hace que
    # el filtro de manifiesto no encuentre a `referencia` (construida con
    # incluir_petroleo=False) y explote, o -- si además se mutara la construcción de
    # `referencia` -- que el valor numérico deje de coincidir con el oráculo.
    traslape = PeriodoMensual(2025, 7)
    periodos_nuevos = [traslape, PeriodoMensual(2025, 8), PeriodoMensual(2025, 9)]

    def _canasta_pesos(pesos: list[float], version: VersionCanasta) -> CanastaINPP:
        encadenamiento: list[float | None] = (
            [2.0, 1.5, 1.2, 1.1] if version == 2025 else [None, None, None, None]
        )
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
                "produccion total": pesos,
                "bienes intermedios": pesos,
                "bienes finales": pesos,
                "demanda interna total": pesos,
                "demanda interna consumo": pesos,
                "demanda interna capital": pesos,
                "exportaciones": pesos,
                "encadenamiento total": encadenamiento,
                "encadenamiento produccion nacional": encadenamiento,
                "encadenamiento exportacion": encadenamiento,
                "encadenamiento uso final": encadenamiento,
            },
            index=_GENERICOS,
        )
        return CanastaINPP(df, version)

    canasta_anterior = _canasta_pesos([25.0, 25.0, 25.0, 25.0], 2019)
    canasta_2025 = _canasta_pesos([10.0, 20.0, 30.0, 40.0], 2025)

    serie_anterior = SerieNormalizada(
        pd.DataFrame(
            {
                "generico": ["soya", "petroleo", "acero", "transporte"],
                traslape: [130.0, 90.0, 115.0, 105.0],
            },
            index=_GENERICOS,
        ),
        "produccion_total",
    )
    serie_2025 = SerieNormalizada(
        pd.DataFrame(
            {
                "generico": ["soya", "petroleo", "acero", "transporte"],
                periodos_nuevos[0]: [200.0, 150.0, 120.0, 110.0],
                periodos_nuevos[1]: [204.0, 165.0, 126.0, 111.1],
                periodos_nuevos[2]: [208.0, 180.0, 129.6, 113.3],
            },
            index=_GENERICOS,
        ),
        "produccion_total",
    )

    referencia = rep.calcular_indice(
        canasta_anterior,
        serie_anterior,
        "INPP",
        rubro="produccion_total",
        incluir_petroleo=False,
    )
    r = rep.calcular_indice(
        canasta_2025,
        serie_2025,
        "INPP",
        rubro="produccion_total",
        incluir_petroleo=False,
        referencia=referencia,
    )

    # oráculo: i_tramo sin petróleo (pesos NUEVOS 10/30/40, soya/acero/transporte,
    # renormalizados sobre 80) x factor_h sin petróleo (pesos VIEJOS iguales
    # 25/25/25 -> promedio simple de soya/acero/transporte) -- derivación
    # completa en el test de dominio referenciado arriba.
    esperado = [116.6666667, 119.7291667, 122.5]
    assert list(r.resultado.ancho.loc["INPP"]) == pytest.approx(esperado)
    assert r.manifiesto[0].incluir_petroleo is False


def test_calcular_indice_canasta_2025_referencia_combinacion_distinta_lanza_invariante_violado() -> (
    None
):
    # referencia calculada para "bienes_intermedios" (recorte mercado_nacional),
    # se pide "produccion_total" -- no coincide, debe rechazarse en vez de dar
    # un factor_h equivocado.
    referencia = rep.calcular_indice(
        _canasta(), _serie("mercado_nacional"), "INPP", rubro="bienes_intermedios"
    )

    traslape = PeriodoMensual(2025, 7)
    df_2025 = _canasta().df.copy()
    columnas_encadenamiento = [
        "encadenamiento total",
        "encadenamiento produccion nacional",
        "encadenamiento exportacion",
        "encadenamiento uso final",
    ]
    df_2025[columnas_encadenamiento] = 1.0
    canasta_2025 = CanastaINPP(df_2025, 2025)
    serie_2025 = SerieNormalizada(
        pd.DataFrame(
            {"generico": ["soya", "petroleo", "acero", "transporte"], traslape: [100.0] * 4},
            index=_GENERICOS,
        ),
        "produccion_total",
    )

    with pytest.raises(InvarianteViolado, match="rubro='produccion_total'"):
        rep.calcular_indice(
            canasta_2025, serie_2025, "INPP", rubro="produccion_total", referencia=referencia
        )


def test_calcular_indice_incluir_petroleo_false_deja_grupo_vacio_lanza_canasta_sin_genericos() -> (
    None
):
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
            canasta_solo_petroleo,
            serie,
            "INPP",
            rubro="produccion_total",
            incluir_petroleo=False,
        )
