from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import replica_inpp as rep
from replica_inpp.api.indices import calcular_indice, empalmar, rebasar
from replica_inpp.dominio.errores import (
    CanastaSinGenericos,
    InvarianteViolado,
    PeriodoNoInterpretable,
    VersionNoCoincide,
)
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import VersionCanasta

_PERIODOS = [PeriodoMensual(2019, 7), PeriodoMensual(2019, 8)]
_GENERICOS = ["001", "070", "339", "460"]

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "inputs"


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
        rep.calcular_indice(_canasta(), _serie(), "INPP", "produccion_total", True)  # type: ignore[call-arg]


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


# -- cruce real contra datos reales (canasta 2012, LaspeyresDirecto) ---------
# Toda la mecánica de despacho (VersionCanasta/RANGOS_CANASTAS/calcular_indice)
# ya trataba 2012 igual que 2019, pero ningún test corría la ruta completa
# (cargar_canasta + cargar_serie + calcular_indice) con archivos reales de esa
# vintage -- solo con datos sintéticos. Exactitud numérica contra el BIE queda
# para `data/comprobacion.py` (fuera del repo versionado); estos tests solo
# confirman que la mecánica no rompe con la forma real de los CSV 2012 y que
# el rango vigente (`RANGOS_CANASTAS[2012]` = jun2012-jul2019, 86 meses) sale
# completo y sin huecos.


@pytest.mark.requires_data
@pytest.mark.parametrize(
    "carpeta,archivo,rubro",
    [
        ("produccion_total", "s12_h_nm_nae.CSV", "produccion_total"),
        ("bienes_finales", "s12_h_nm.CSV", "bienes_finales"),
        ("mercado_exportacion", "s12_h_nm.CSV", "exportaciones"),
        ("mercado_nacional", "s12_h_nm.CSV", "demanda_interna_total"),
    ],
)
def test_calcular_indice_real_2012(carpeta: str, archivo: str, rubro: str) -> None:
    canasta = rep.cargar_canasta(str(DATA_DIR / "canasta" / "ponderadores_2012.csv"), 2012)
    serie = rep.cargar_serie(str(DATA_DIR / carpeta / archivo), 2012)

    r = rep.calcular_indice(canasta, serie, "INPP", rubro=rubro)

    assert r.manifiesto[0].version == 2012
    assert r.manifiesto[0].calculador == "LaspeyresDirecto"
    valores = r.resultado.ancho.loc["INPP"]
    assert len(valores) == 86  # jun2012-jul2019 inclusive, sin huecos
    assert bool(valores.notna().all())
    assert bool((valores > 0).all())
    assert bool(valores.iloc[0] == pytest.approx(100.0))  # jun2012 es el mes base


@pytest.mark.requires_data
def test_calcular_indice_real_2012_incluir_petroleo_false_excluye_070() -> None:
    canasta = rep.cargar_canasta(str(DATA_DIR / "canasta" / "ponderadores_2012.csv"), 2012)
    serie = rep.cargar_serie(str(DATA_DIR / "produccion_total" / "s12_h_nm_nae.CSV"), 2012)

    con_petroleo = rep.calcular_indice(
        canasta, serie, "INPP", rubro="produccion_total", incluir_petroleo=True
    )
    sin_petroleo = rep.calcular_indice(
        canasta, serie, "INPP", rubro="produccion_total", incluir_petroleo=False
    )

    valores_sin = sin_petroleo.resultado.ancho.loc["INPP"]
    assert bool(valores_sin.notna().all())
    assert bool(valores_sin.iloc[0] == pytest.approx(100.0))
    # excluir petróleo cambia la serie -- no es un passthrough disfrazado.
    assert not valores_sin.equals(con_petroleo.resultado.ancho.loc["INPP"])  # type: ignore[arg-type]


# -- rebasar: fachada pública -------------------------------------------------


def test_rebasar_esta_en_all_de_la_fachada() -> None:
    assert "rebasar" in rep.__all__


def test_rep_rebasar_es_la_misma_funcion_que_indices_rebasar() -> None:
    assert rep.rebasar is rebasar


def test_rebasar_parsea_periodo_texto() -> None:
    r = calcular_indice(_canasta(), _serie(), "INPP", rubro="produccion_total")
    rb = rebasar(r, "Ago 2019")
    assert rb.resultado.ancho.loc["INPP", _PERIODOS[1]] == pytest.approx(100.0)  # type: ignore[call-overload]


def test_rebasar_periodo_no_interpretable_propaga() -> None:
    r = calcular_indice(_canasta(), _serie(), "INPP", rubro="produccion_total")
    with pytest.raises(PeriodoNoInterpretable):
        rebasar(r, "no es un periodo")


@pytest.mark.requires_data
def test_rebasar_real_2012_a_referencia_2019() -> None:
    # El caso real que motiva `rebasar`: canasta 2012 sale en su propia base
    # (Jun2012=100, RANGOS_CANASTAS[2012]) -- para compararla/expresarla junto a
    # la vintage 2019 hace falta reexpresarla con Jul2019=100, el mes de
    # traslape entre ambas (RANGOS_CANASTAS[2019][0]), sin tocar `codigo` ni
    # requerir un módulo de correspondencia entre versiones.
    canasta = rep.cargar_canasta(str(DATA_DIR / "canasta" / "ponderadores_2012.csv"), 2012)
    serie = rep.cargar_serie(str(DATA_DIR / "produccion_total" / "s12_h_nm_nae.CSV"), 2012)
    r = rep.calcular_indice(canasta, serie, "INPP", rubro="produccion_total")

    rb = rep.rebasar(r, "Jul 2019")

    valores = rb.resultado.ancho.loc["INPP"]
    assert len(valores) == 86
    assert bool(valores.notna().all())
    assert valores.loc[PeriodoMensual(2019, 7)] == pytest.approx(100.0)  # type: ignore[call-overload]
    # jun2012 ya no vale 100 -- se reescaló toda la serie por un factor común.
    assert bool(valores.iloc[0] != pytest.approx(100.0))
    # manifiesto/version no cambian -- rebasar no recalcula ni reclasifica.
    assert rb.manifiesto == r.manifiesto


# -- empalmar: fachada pública -------------------------------------------------


def _tramo(agregacion: str = "INPP") -> tuple[ResultadoIndice, ResultadoIndice]:
    """2 tramos consecutivos (2019/2025) listos para empalmar sin rebasar --
    `LaspeyresEncadenado` ya deja 2025 en la escala de 2019, mismo mecanismo
    que usa `data/comprobacion.py::_continuo` a mano. `r2019` cubre hasta la
    frontera (traslape=Jul2025) para compartir exactamente ese periodo con
    `r2025` -- se reusa como `referencia` de `LaspeyresEncadenado`, igual que
    en `data/comprobacion.py`."""
    canasta_2019 = _canasta()
    traslape = PeriodoMensual(2025, 7)
    serie_2019 = SerieNormalizada(
        pd.DataFrame(
            {
                "generico": ["soya", "petroleo", "acero", "transporte"],
                _PERIODOS[0]: [100.0] * 4,
                _PERIODOS[1]: [102.0, 110.0, 105.0, 101.0],
                traslape: [130.0, 90.0, 115.0, 105.0],
            },
            index=_GENERICOS,
        ),
        "produccion_total",
    )
    r2019 = calcular_indice(canasta_2019, serie_2019, agregacion, rubro="produccion_total")
    referencia = r2019
    df_2025 = _canasta().df.copy()
    columnas_encadenamiento = [
        "encadenamiento total",
        "encadenamiento produccion nacional",
        "encadenamiento exportacion",
        "encadenamiento uso final",
    ]
    df_2025[columnas_encadenamiento] = 1.0
    canasta_2025 = CanastaINPP(df_2025, 2025)
    # 2 periodos (no solo la frontera): si el manifiesto de 2025 quedara respaldado
    # SOLO por la fila de la frontera, el dedup de `empalmar` ("keep=first", manda
    # el tramo anterior) la eliminaría entera y el manifiesto quedaría huérfano.
    serie_2025 = SerieNormalizada(
        pd.DataFrame(
            {
                "generico": ["soya", "petroleo", "acero", "transporte"],
                traslape: [100.0] * 4,
                PeriodoMensual(2025, 8): [102.0] * 4,
            },
            index=_GENERICOS,
        ),
        "produccion_total",
    )
    r2025 = calcular_indice(
        canasta_2025, serie_2025, agregacion, rubro="produccion_total", referencia=referencia
    )
    return r2019, r2025


def test_empalmar_esta_en_all_de_la_fachada() -> None:
    assert "empalmar" in rep.__all__


def test_rep_empalmar_es_la_misma_funcion_que_indices_empalmar() -> None:
    assert rep.empalmar is empalmar


def test_empalmar_delega_correctamente_con_sintetico() -> None:
    r2019, r2025 = _tramo()
    r = empalmar([r2019, r2025])
    assert isinstance(r, ResultadoIndice)
    assert {m.version for m in r.manifiesto} == {2019, 2025}


def test_empalmar_menos_de_2_propaga_invariante_violado() -> None:
    r2019, _ = _tramo()
    with pytest.raises(InvarianteViolado):
        rep.empalmar([r2019])


def test_empalmar_version_nombres_propaga_a_traves_de_la_fachada() -> None:
    # atraviesa rep.empalmar (no _empalmar directo) para proteger el wiring del
    # parámetro nuevo -- version_nombres=2012 no corresponde a ningún tramo de
    # _tramo() (2019/2025), debe rechazar mencionando el parámetro.
    r2019, r2025 = _tramo()
    with pytest.raises(InvarianteViolado, match="version_nombres"):
        rep.empalmar([r2019, r2025], version_nombres=2012)


@pytest.mark.requires_data
def test_empalmar_real_2012_rebasado_con_2019_da_serie_continua() -> None:
    # El flujo completo que motivó rebasar/empalmar: canasta 2012 (base propia
    # Jun2012=100) rebasada a Jul2019=100 -- la base de la vintage 2019 -- y
    # empalmada con el tramo 2019, para una sola serie continua jun2012-jul2025.
    canasta_2012 = rep.cargar_canasta(str(DATA_DIR / "canasta" / "ponderadores_2012.csv"), 2012)
    serie_2012 = rep.cargar_serie(str(DATA_DIR / "produccion_total" / "s12_h_nm_nae.CSV"), 2012)
    r2012 = rep.calcular_indice(canasta_2012, serie_2012, "INPP", rubro="produccion_total")
    r2012_rebasado = rep.rebasar(r2012, "Jul 2019")

    canasta_2019 = rep.cargar_canasta(str(DATA_DIR / "canasta" / "ponderadores_2019.csv"), 2019)
    serie_2019 = rep.cargar_serie(str(DATA_DIR / "produccion_total" / "s19_h_nm_nae.CSV"), 2019)
    r2019 = rep.calcular_indice(canasta_2019, serie_2019, "INPP", rubro="produccion_total")

    combinado = rep.empalmar([r2012_rebasado, r2019])

    valores = combinado.resultado.ancho.loc["INPP"]
    assert len(valores) == 158  # 86 (2012) + 73 (2019) - 1 (frontera compartida)
    assert bool(valores.notna().all())
    assert valores.loc[PeriodoMensual(2019, 7)] == pytest.approx(100.0)  # type: ignore[call-overload]
    assert valores.index.min() == PeriodoMensual(2012, 6)
    assert valores.index.max() == PeriodoMensual(2025, 7)
    assert combinado.periodo_referencia == PeriodoMensual(2019, 7)
    assert {m.version for m in combinado.manifiesto} == {2012, 2019}
