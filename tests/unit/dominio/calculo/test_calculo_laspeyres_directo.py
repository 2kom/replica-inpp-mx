from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from replica_inpp.dominio.calculo.laspeyres_directo import LaspeyresDirecto
from replica_inpp.dominio.errores import (
    CanastaSinGenericos,
    ErrorCalculo,
    InvarianteViolado,
    VersionNoCoincide,
)
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.periodos import PeriodoMensual

_PERIODOS = [PeriodoMensual(2019, 7), PeriodoMensual(2019, 8), PeriodoMensual(2019, 9)]
_GENERICOS = ["001", "070", "339", "460"]

# 4 genéricos, 4 sectores distintos: "11" (soya) y "31" (acero) son MERCANCIAS,
# "21" (petróleo, código CODIGO_PETROLEO_CRUDO) también MERCANCIAS, "48"
# (transporte) es SERVICIOS -- misma estructura que SECTORES_MERCANCIAS real.
# bienes intermedios: solo soya (50) y acero (50) participan -- petróleo y
# transporte pesan EXACTO 0, regresión del bug real (peso 0 ≠ NaN, pero
# tampoco debe exigirse en el grupo).


def _canasta(rama_combinada: bool = True) -> CanastaINPP:
    rama_txt = (
        [
            "1111 cultivo de cereales",
            "2111 extraccion de petroleo",
            "3111 fabricacion de acero",
            "4811 transporte",
        ]
        if rama_combinada
        else ["1111", "2111", "3111", "4811"]
    )
    df = pd.DataFrame(
        {
            "generico": ["soya", "petroleo", "acero", "transporte"],
            "codigo sector": ["11", "21", "31", "48"],
            "sector": ["11", "21", "31", "48"],
            "codigo subsector": ["111", "211", "311", "481"],
            "subsector": ["111", "211", "311", "481"],
            "codigo rama": ["1111", "2111", "3111", "4811"],
            "rama": rama_txt,
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
            _PERIODOS[2]: [104.0, 120.0, 108.0, 103.0],
        },
        index=_GENERICOS,
    )
    return SerieNormalizada(df, recorte)


# ---------- básicos ----------


def test_calcular_retorna_resultado_indice() -> None:
    r = LaspeyresDirecto().calcular(_canasta(), _serie(), "INPP", rubro="produccion_total")
    assert isinstance(r, ResultadoIndice)


def test_valores_inpp_general_correctos() -> None:
    r = LaspeyresDirecto().calcular(_canasta(), _serie(), "INPP", rubro="produccion_total")
    valores = list(r.resultado.ancho.loc["INPP"])
    # pesos iguales (25 c/u) -> promedio simple de los 4
    assert valores == pytest.approx([100.0, 104.5, 108.75])


def test_agregacion_invalida_lanza_invariante_violado() -> None:
    with pytest.raises(InvarianteViolado):
        LaspeyresDirecto().calcular(
            _canasta(), _serie(), "agregacion_inventada", rubro="produccion_total"
        )


def test_manifiesto_campos_correctos() -> None:
    r = LaspeyresDirecto().calcular(
        _canasta(), _serie(), "INPP", rubro="produccion_total", incluir_petroleo=False
    )
    m = r.manifiesto[0]
    assert m.version == 2019
    assert m.agregacion == "INPP"
    assert m.rubro == "produccion_total"
    assert m.incluir_petroleo is False
    assert m.calculador == "LaspeyresDirecto"


# ---------- normalización de agregación ----------


def test_agregacion_abreviatura_se_normaliza_a_nombre_canonico() -> None:
    r = LaspeyresDirecto().calcular(_canasta(), _serie(), "s", rubro="produccion_total")
    assert r.manifiesto[0].agregacion == "SECTOR"


def test_agregacion_inpp_no_se_toca_por_normalizacion() -> None:
    r = LaspeyresDirecto().calcular(_canasta(), _serie(), "inpp", rubro="produccion_total")
    assert r.manifiesto[0].agregacion == "INPP"


# ---------- rubro: inferencia y validación contra recorte ----------


def test_rubro_se_infiere_para_produccion_total() -> None:
    # produccion_total dejó de ser ambiguo (bienes_intermedios se movió a
    # mercado_nacional, 2026-08-30) -- ahora se infiere solo, igual que
    # bienes_finales/mercado_exportacion.
    r = LaspeyresDirecto().calcular(_canasta(), _serie("produccion_total"), "INPP")
    assert r.manifiesto[0].rubro == "produccion_total"


def test_rubro_invalido_para_recorte_lanza_invariante_violado() -> None:
    with pytest.raises(InvarianteViolado, match="no es válido para el recorte"):
        LaspeyresDirecto().calcular(
            _canasta(), _serie("mercado_nacional"), "INPP", rubro="exportaciones"
        )


def test_rubro_se_infiere_para_recorte_1a1() -> None:
    r = LaspeyresDirecto().calcular(_canasta(), _serie("bienes_finales"), "INPP")
    assert r.manifiesto[0].rubro == "bienes_finales"


# ---------- demanda_interna_* con recorte mercado_nacional (negociación 2026-08-27) ----------
#
# La nota b/ del xlsx de ponderadores dice que estas 3 columnas de peso se
# combinan con precios de MERCADO NACIONAL, no produccion_total como asumía
# el mapa antes -- ver RUBROS_POR_RECORTE en dominio/tipos.py. Confirmado
# contra el BIE real: con mercado_nacional coincide dentro del error de punto
# flotante; con produccion_total no.


@pytest.mark.parametrize(
    "rubro", ["demanda_interna_total", "demanda_interna_consumo", "demanda_interna_capital"]
)
def test_rubro_demanda_interna_valido_para_recorte_mercado_nacional(rubro: str) -> None:
    r = LaspeyresDirecto().calcular(_canasta(), _serie("mercado_nacional"), "INPP", rubro=rubro)
    assert r.manifiesto[0].rubro == rubro


def test_rubro_demanda_interna_invalido_para_recorte_produccion_total() -> None:
    with pytest.raises(InvarianteViolado, match="no es válido para el recorte"):
        LaspeyresDirecto().calcular(
            _canasta(), _serie("produccion_total"), "INPP", rubro="demanda_interna_total"
        )


def test_rubro_ambiguo_para_recorte_mercado_nacional_sin_indicar() -> None:
    with pytest.raises(InvarianteViolado, match="admite varios rubros"):
        LaspeyresDirecto().calcular(_canasta(), _serie("mercado_nacional"), "INPP")


# ---------- incluir_petroleo ----------


def test_incluir_petroleo_false_filtra_070_y_renormaliza_implicito() -> None:
    r = LaspeyresDirecto().calcular(
        _canasta(), _serie(), "INPP", rubro="produccion_total", incluir_petroleo=False
    )
    valores = list(r.resultado.ancho.loc["INPP"])
    # sin 070: soya+acero+transporte, pesos iguales -> promedio simple de los 3
    esperado = [100.0, (102.0 + 105.0 + 101.0) / 3, (104.0 + 108.0 + 103.0) / 3]
    assert valores == pytest.approx(esperado)


# ---------- MERCANCIAS_SERVICIOS ----------


def test_mercancias_servicios_agrupa_correcto() -> None:
    r = LaspeyresDirecto().calcular(
        _canasta(), _serie(), "MERCANCIAS_SERVICIOS", rubro="produccion_total"
    )
    ancho = r.resultado.ancho
    # Mercancías = sectores 11, 21, 31 (soya, petróleo, acero) -- ver SECTORES_MERCANCIAS
    esperado_mercancias = [100.0, (102.0 + 110.0 + 105.0) / 3, (104.0 + 120.0 + 108.0) / 3]
    esperado_servicios = [100.0, 101.0, 103.0]
    assert list(ancho.loc["MERCANCIAS"]) == pytest.approx(esperado_mercancias)
    assert list(ancho.loc["SERVICIOS"]) == pytest.approx(esperado_servicios)


# ---------- peso exactamente 0 (regresión del bug real 2019) ----------


def test_peso_exactamente_cero_excluye_generico_del_grupo() -> None:
    # bienes_intermedios: petróleo ("21") y transporte ("48") pesan 0 -- deben
    # quedar FUERA del grupo entero, no aparecer como fila NaN. rubro válido
    # para recorte mercado_nacional (movido de produccion_total, 2026-08-30).
    r = LaspeyresDirecto().calcular(
        _canasta(), _serie("mercado_nacional"), "SECTOR", rubro="bienes_intermedios"
    )
    assert set(r.resultado.ancho.index) == {"11", "31"}


def test_peso_cero_no_exige_cobertura_de_serie() -> None:
    # la serie NO trae a petróleo/transporte -- no debe fallar: su peso en
    # "bienes intermedios" es 0, no participan de este rubro.
    canasta = _canasta()
    serie_incompleta = SerieNormalizada(
        _serie("mercado_nacional").df.drop(index=["070", "460"]), "mercado_nacional"
    )
    r = LaspeyresDirecto().calcular(canasta, serie_incompleta, "INPP", rubro="bienes_intermedios")
    assert (r.resultado.largo["estado_calculo"] == "ok").all()


def test_serie_le_falta_generico_con_peso_no_cero_lanza_error_calculo() -> None:
    canasta = _canasta()
    serie_incompleta = SerieNormalizada(_serie().df.drop(index=["070"]), "produccion_total")
    with pytest.raises(ErrorCalculo, match="produccion_total"):
        LaspeyresDirecto().calcular(canasta, serie_incompleta, "INPP", rubro="produccion_total")


# ---------- columna "nombre" (best-effort, solo agregación SCIAN) ----------


def test_nombre_aparece_cuando_canasta_trae_texto_combinado() -> None:
    r = LaspeyresDirecto().calcular(
        _canasta(rama_combinada=True), _serie(), "RAMA", rubro="produccion_total"
    )
    ancho = r.resultado.ancho
    assert "nombre" in ancho.columns
    assert ancho.loc["1111", "nombre"] == "1111 cultivo de cereales"


def test_nombre_ausente_cuando_canasta_es_bare() -> None:
    r = LaspeyresDirecto().calcular(
        _canasta(rama_combinada=False), _serie(), "RAMA", rubro="produccion_total"
    )
    assert "nombre" not in r.resultado.ancho.columns


def test_nombre_ausente_para_agregacion_inpp() -> None:
    r = LaspeyresDirecto().calcular(_canasta(), _serie(), "INPP", rubro="produccion_total")
    assert "nombre" not in r.resultado.ancho.columns


def test_nombre_ausente_para_mercancias_servicios() -> None:
    r = LaspeyresDirecto().calcular(
        _canasta(), _serie(), "MERCANCIAS_SERVICIOS", rubro="produccion_total"
    )
    assert "nombre" not in r.resultado.ancho.columns


# ---------- faltantes / relleno ----------


def test_nan_parcial_produce_estado_rellenado() -> None:
    df = pd.DataFrame(
        {
            "generico": ["soya", "petroleo", "acero", "transporte"],
            _PERIODOS[0]: [100.0, 100.0, 100.0, 100.0],
            _PERIODOS[1]: [float("nan"), 110.0, 105.0, 101.0],
            _PERIODOS[2]: [104.0, 120.0, 108.0, 103.0],
        },
        index=_GENERICOS,
    )
    serie = SerieNormalizada(df, "produccion_total")
    r = LaspeyresDirecto().calcular(_canasta(), serie, "INPP", rubro="produccion_total")
    largo = r.resultado.largo
    estados = dict(zip(largo.index.get_level_values("periodo"), largo["estado_calculo"]))
    assert estados[_PERIODOS[0]] == "ok"
    assert estados[_PERIODOS[1]] == "rellenado"
    assert estados[_PERIODOS[2]] == "ok"


def test_nan_total_generico_produce_sin_datos() -> None:
    df = pd.DataFrame(
        {
            "generico": ["soya", "petroleo", "acero", "transporte"],
            _PERIODOS[0]: [float("nan"), 100.0, 100.0, 100.0],
            _PERIODOS[1]: [float("nan"), 110.0, 105.0, 101.0],
            _PERIODOS[2]: [float("nan"), 120.0, 108.0, 103.0],
        },
        index=_GENERICOS,
    )
    serie = SerieNormalizada(df, "produccion_total")
    r = LaspeyresDirecto().calcular(_canasta(), serie, "INPP", rubro="produccion_total")
    largo = r.resultado.largo
    assert (largo["estado_calculo"] == "sin_datos").all()
    assert largo["indice_replicado"].isna().all()


# ---------- recorte de fechas ----------


def test_periodos_fuera_de_rango_2019_se_recortan() -> None:
    periodos_con_extra = [
        PeriodoMensual(2019, 5),
        PeriodoMensual(2019, 6),
        *_PERIODOS,
    ]
    df = pd.DataFrame(
        {
            "generico": ["soya", "petroleo", "acero", "transporte"],
            periodos_con_extra[0]: [99.0, 99.0, 99.0, 99.0],
            periodos_con_extra[1]: [99.0, 99.0, 99.0, 99.0],
            periodos_con_extra[2]: [100.0, 100.0, 100.0, 100.0],
            periodos_con_extra[3]: [102.0, 110.0, 105.0, 101.0],
            periodos_con_extra[4]: [104.0, 120.0, 108.0, 103.0],
        },
        index=_GENERICOS,
    )
    serie_extra = SerieNormalizada(df, "produccion_total")
    r = LaspeyresDirecto().calcular(_canasta(), serie_extra, "INPP", rubro="produccion_total")
    periodos_resultado = r.resultado.largo.index.get_level_values("periodo").tolist()
    assert PeriodoMensual(2019, 6) not in periodos_resultado
    assert PeriodoMensual(2019, 7) in periodos_resultado


# ---------- versión de canasta vs. versión de la serie (negociación 2026-08-27) ----------


def test_version_serie_no_coincide_con_canasta_lanza_version_no_coincide() -> None:
    canasta = _canasta()
    serie = _serie()
    serie.df.attrs["version"] = 2012  # como haría cargar_serie(..., 2012)
    with pytest.raises(VersionNoCoincide):
        LaspeyresDirecto().calcular(canasta, serie, "INPP", rubro="produccion_total")


def test_version_serie_coincide_con_canasta_no_falla() -> None:
    canasta = _canasta()
    serie = _serie()
    serie.df.attrs["version"] = 2019  # misma versión que la canasta
    r = LaspeyresDirecto().calcular(canasta, serie, "INPP", rubro="produccion_total")
    assert isinstance(r, ResultadoIndice)


def test_version_serie_ausente_en_attrs_no_valida() -> None:
    # serie construida a mano, sin pasar por cargar_serie -- sin attrs["version"]
    # no hay nada contra qué comparar; mismo comportamiento que todos los demás
    # tests de este archivo (ninguno setea attrs["version"]).
    canasta = _canasta()
    serie = _serie()
    assert "version" not in serie.df.attrs
    r = LaspeyresDirecto().calcular(canasta, serie, "INPP", rubro="produccion_total")
    assert isinstance(r, ResultadoIndice)


# ---------- incluir_petroleo=False deja el grupo vacío (negociación 2026-08-27) ----------


def test_incluir_petroleo_false_deja_grupo_vacio_lanza_canasta_sin_genericos() -> None:
    # canasta mínima donde el ÚNICO genérico con peso es el 070 (petróleo) --
    # incluir_petroleo=False lo excluye y no queda nada con qué calcular.
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
        LaspeyresDirecto().calcular(
            canasta_solo_petroleo, serie, "INPP", rubro="produccion_total", incluir_petroleo=False
        )
