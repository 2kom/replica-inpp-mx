from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from replica_inpp.dominio.calculo.laspeyres_encadenado import LaspeyresEncadenado
from replica_inpp.dominio.errores import ErrorCalculo, InvarianteViolado
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.periodos import PeriodoMensual

_TRASLAPE = PeriodoMensual(2025, 7)
_PERIODOS_NUEVOS = [_TRASLAPE, PeriodoMensual(2025, 8), PeriodoMensual(2025, 9)]
_GENERICOS = ["001", "070", "339", "460"]

# 4 genéricos, 4 sectores distintos: "11" (soya), "21" (petróleo,
# CODIGO_PETROLEO_CRUDO), "31" (acero), "48" (transporte) -- mismos códigos que
# test_calculo_laspeyres_directo.py, para que quede claro que es el mismo
# universo, solo con canasta 2019 (anterior) y 2025 (nueva) por separado.
#
# Pesos DISTINTOS a propósito entre la canasta anterior (25/25/25/25, igual)
# y la nueva (10/20/30/40, distinto) -- esto es justo lo que expone la
# regresión real: `factor_h` debe salir de correr `LaspeyresDirecto` sobre la
# canasta ANTERIOR con SUS pesos, nunca promediar `f_j` con los pesos de la
# canasta nueva (ver docstring de `LaspeyresEncadenado`, bug real encontrado
# validando contra el BIE: daba un error multiplicativo constante ~0.87%).


def _canasta_anterior() -> CanastaINPP:
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
            "bienes intermedios": [25.0, 25.0, 25.0, 25.0],
            "bienes finales": [25.0, 25.0, 25.0, 25.0],
            "demanda interna total": [25.0, 25.0, 25.0, 25.0],
            "demanda interna consumo": [25.0, 25.0, 25.0, 25.0],
            "demanda interna capital": [25.0, 25.0, 25.0, 25.0],
            "exportaciones": [25.0, 25.0, 25.0, 25.0],
            "encadenamiento total": [None] * 4,
            "encadenamiento produccion nacional": [None] * 4,
            "encadenamiento exportacion": [None] * 4,
            "encadenamiento uso final": [None] * 4,
        },
        index=_GENERICOS,
    )
    return CanastaINPP(df, 2019)


def _canasta_nueva(
    f_j_total_vacio: bool = False, f_j_exportacion_transporte_nan: bool = False
) -> CanastaINPP:
    f_j: list[float | None] = [None] * 4 if f_j_total_vacio else [2.0, 1.5, 1.2, 1.1]
    f_j_exportacion = [2.0, 1.5, 1.2, None if f_j_exportacion_transporte_nan else 1.1]
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
            "produccion total": [10.0, 20.0, 30.0, 40.0],
            "bienes intermedios": [10.0, 20.0, 30.0, 40.0],
            "bienes finales": [10.0, 20.0, 30.0, 40.0],
            "demanda interna total": [10.0, 20.0, 30.0, 40.0],
            "demanda interna consumo": [10.0, 20.0, 30.0, 40.0],
            "demanda interna capital": [10.0, 20.0, 30.0, 40.0],
            "exportaciones": [10.0, 20.0, 30.0, 40.0],
            "encadenamiento total": f_j,
            "encadenamiento produccion nacional": f_j,
            "encadenamiento exportacion": f_j_exportacion,
            "encadenamiento uso final": f_j,
        },
        index=_GENERICOS,
    )
    return CanastaINPP(df, 2025)


def _serie_anterior(recorte: Any = "produccion_total") -> SerieNormalizada:
    # Valor del índice VIEJO en el traslape (jul-2025) -- promedio simple
    # (pesos iguales 25/25/25/25) da 110.0 exacto: (130+90+115+105)/4.
    df = pd.DataFrame(
        {
            "generico": ["soya", "petroleo", "acero", "transporte"],
            _TRASLAPE: [130.0, 90.0, 115.0, 105.0],
        },
        index=_GENERICOS,
    )
    return SerieNormalizada(df, recorte)


def _serie_nueva(recorte: Any = "produccion_total") -> SerieNormalizada:
    # Cruda en escala absoluta continua; entre f_j da escala local limpia:
    # soya 100/102/104, petroleo 100/110/120, acero 100/105/108,
    # transporte 100/101/103 (mismos números que test_calculo_laspeyres_directo).
    df = pd.DataFrame(
        {
            "generico": ["soya", "petroleo", "acero", "transporte"],
            _PERIODOS_NUEVOS[0]: [200.0, 150.0, 120.0, 110.0],
            _PERIODOS_NUEVOS[1]: [204.0, 165.0, 126.0, 111.1],
            _PERIODOS_NUEVOS[2]: [208.0, 180.0, 129.6, 113.3],
        },
        index=_GENERICOS,
    )
    return SerieNormalizada(df, recorte)


# ---------- básicos ----------


def test_calcular_retorna_resultado_indice() -> None:
    r = LaspeyresEncadenado(_canasta_anterior(), _serie_anterior()).calcular(
        _canasta_nueva(), _serie_nueva(), "INPP", rubro="produccion_total"
    )
    assert isinstance(r, ResultadoIndice)


def test_valores_inpp_usa_peso_de_canasta_anterior_para_factor_h() -> None:
    r = LaspeyresEncadenado(_canasta_anterior(), _serie_anterior()).calcular(
        _canasta_nueva(), _serie_nueva(), "INPP", rubro="produccion_total"
    )
    # i_tramo (pesos NUEVOS 10/20/30/40 sobre escala local): 100.0, 104.10, 108.00
    # factor_h = I_viejo(traslape)/100 con pesos VIEJOS (25/25/25/25 -> promedio
    # simple) = 110.0/100 = 1.10 -- NO 107.5/100=1.075, que es lo que daría
    # promediar f_j con el peso NUEVO (10*2.0+20*1.5+30*1.2+40*1.1)/100=1.33...
    # ya no cuadra ni de casualidad, la regresión real quedó peor que esto).
    esperado = [110.0, 114.51, 118.8]
    assert list(r.resultado.ancho.loc["INPP"]) == pytest.approx(esperado)


def test_manifiesto_campos_correctos() -> None:
    r = LaspeyresEncadenado(_canasta_anterior(), _serie_anterior()).calcular(
        _canasta_nueva(), _serie_nueva(), "INPP", rubro="produccion_total", sin_petroleo=True
    )
    m = r.manifiesto[0]
    assert m.version == 2025
    assert m.agregacion == "INPP"
    assert m.rubro == "produccion_total"
    assert m.sin_petroleo is True
    assert m.calculador == "LaspeyresEncadenado"


# ---------- validación de versión ----------


def test_canasta_anterior_version_incorrecta_lanza_invariante_violado() -> None:
    canasta_2025_como_anterior = _canasta_nueva()  # version=2025, no 2019
    with pytest.raises(InvarianteViolado, match="canasta_anterior.version=2019"):
        LaspeyresEncadenado(canasta_2025_como_anterior, _serie_anterior())


def test_canasta_nueva_version_incorrecta_lanza_invariante_violado() -> None:
    with pytest.raises(InvarianteViolado, match="canasta.version=2025"):
        LaspeyresEncadenado(_canasta_anterior(), _serie_anterior()).calcular(
            _canasta_anterior(), _serie_nueva(), "INPP", rubro="produccion_total"
        )


# ---------- factor de encadenamiento (f_j) ----------


def test_f_j_totalmente_vacio_lanza_error_calculo() -> None:
    canasta_sin_encadenamiento = _canasta_nueva(f_j_total_vacio=True)
    with pytest.raises(ErrorCalculo, match="sin --encadenamientos"):
        LaspeyresEncadenado(_canasta_anterior(), _serie_anterior()).calcular(
            canasta_sin_encadenamiento, _serie_nueva(), "INPP", rubro="produccion_total"
        )


def test_f_j_nan_parcial_en_un_generico_produce_sin_datos_solo_para_ese_grupo() -> None:
    # "encadenamiento exportacion" trae NaN para transporte (N/A real de INEGI,
    # CanastaINPP lo permite en las columnas parciales) -- no debe reventar,
    # solo dejar ese sector como sin_datos; el resto sigue en "ok". El motivo y
    # el diagnóstico deben apuntar al factor de encadenamiento, NO a la serie
    # (regresión: antes ambos decían "faltantes en serie"/"valor NaN en serie
    # publicada" aunque el precio de transporte sí estaba completo).
    canasta = _canasta_nueva(f_j_exportacion_transporte_nan=True)
    serie_exp = _serie_nueva(recorte="mercado_exportacion")
    r = LaspeyresEncadenado(
        _canasta_anterior(), _serie_anterior(recorte="mercado_exportacion")
    ).calcular(canasta, serie_exp, "SECTOR", rubro="exportaciones")
    largo = r.resultado.largo
    estados = dict(zip(largo.index.get_level_values("indice"), largo["estado_calculo"]))
    assert estados["48"] == "sin_datos"  # transporte
    assert estados["11"] == "ok"
    assert estados["21"] == "ok"
    assert estados["31"] == "ok"

    motivos = largo[largo.index.get_level_values("indice") == "48"]["motivo_error"]
    assert (motivos == "factor de encadenamiento ausente en 'encadenamiento exportacion'").all()

    diagnostico_transporte = r.diagnostico[r.diagnostico["generico"] == "460"]
    # len() explícito, no solo .all() -- un DataFrame vacío hace ambos .all() de
    # abajo verdaderos por vacuidad (regresión: un mutante que borrara las filas
    # de diagnóstico de f_j pasaba este test igual).
    assert len(diagnostico_transporte) == 3  # una fila por periodo (jul/ago/sep 2025)
    assert (diagnostico_transporte["tipo_faltante"] == "factor_encadenamiento").all()
    assert (
        diagnostico_transporte["detalle"]
        == "factor de encadenamiento ausente en 'encadenamiento exportacion'"
    ).all()


def test_traslape_faltante_en_serie_anterior_lanza_error_calculo() -> None:
    serie_sin_traslape = SerieNormalizada(
        pd.DataFrame(
            {
                "generico": ["soya", "petroleo", "acero", "transporte"],
                PeriodoMensual(2019, 7): [100.0] * 4,
            },
            index=_GENERICOS,
        ),
        "produccion_total",
    )
    with pytest.raises(ErrorCalculo, match="traslape"):
        LaspeyresEncadenado(_canasta_anterior(), serie_sin_traslape).calcular(
            _canasta_nueva(), _serie_nueva(), "INPP", rubro="produccion_total"
        )


def test_producto_final_no_finito_lanza_error_calculo() -> None:
    # i_tramo y factor_h son finitos por separado (~1e200 cada uno, muy por
    # debajo de sys.float_info.max) pero su producto desborda a inf -- ninguno
    # de los dos _laspeyres_por_grupo internos lo detecta por su cuenta.
    serie_anterior_enorme = SerieNormalizada(
        pd.DataFrame(
            {"generico": ["soya", "petroleo", "acero", "transporte"], _TRASLAPE: [1e200] * 4},
            index=_GENERICOS,
        ),
        "produccion_total",
    )
    serie_nueva_enorme = SerieNormalizada(
        pd.DataFrame(
            {
                "generico": ["soya", "petroleo", "acero", "transporte"],
                _PERIODOS_NUEVOS[0]: [1e200] * 4,
                _PERIODOS_NUEVOS[1]: [1e200] * 4,
                _PERIODOS_NUEVOS[2]: [1e200] * 4,
            },
            index=_GENERICOS,
        ),
        "produccion_total",
    )
    with pytest.raises(ErrorCalculo, match="no finito"):
        LaspeyresEncadenado(_canasta_anterior(), serie_anterior_enorme).calcular(
            _canasta_nueva(), serie_nueva_enorme, "INPP", rubro="produccion_total"
        )


def test_grupo_sin_factor_h_en_tramo_anterior_lanza_error_calculo() -> None:
    # transporte cambia de sector "48" (canasta anterior) a "99" (canasta
    # nueva, sin equivalente en el tramo viejo) -- simula una reclasificación
    # de agrupación entre versiones: SECTOR "99" no existe en `factor_h`.
    df_nueva = _canasta_nueva().df.copy()
    df_nueva.loc["460", ["codigo sector", "sector"]] = ["99", "99"]
    canasta_reclasificada = CanastaINPP(df_nueva, 2025)
    with pytest.raises(ErrorCalculo, match="reclasificación"):
        LaspeyresEncadenado(_canasta_anterior(), _serie_anterior()).calcular(
            canasta_reclasificada, _serie_nueva(), "SECTOR", rubro="produccion_total"
        )


# ---------- sin_petroleo (debe excluir 070 en ambos cálculos: i_tramo y factor_h) ----------


def test_sin_petroleo_excluye_070_de_i_tramo_y_de_factor_h() -> None:
    r_sin = LaspeyresEncadenado(_canasta_anterior(), _serie_anterior()).calcular(
        _canasta_nueva(), _serie_nueva(), "INPP", rubro="produccion_total", sin_petroleo=True
    )
    # i_tramo sin petróleo (pesos NUEVOS 10/30/40 sobre soya/acero/transporte,
    # renormalizados sobre 80): jul=100.0, ago=(10*102+30*105+40*101)/80=102.625,
    # sep=(10*104+30*108+40*103)/80=105.0
    # factor_h sin petróleo = I_viejo(traslape)/100 con pesos VIEJOS iguales
    # (25/25/25 -> promedio simple de soya/acero/transporte) = (130+115+105)/3
    # /100 = 116.6667/100 = 1.166667
    #
    # Un mutante que excluya 070 SOLO de i_tramo (deja factor_h calculado con
    # LaspeyresDirecto(sin_petroleo=False), petróleo incluido) sigue dando un
    # resultado != con_petroleo -- ese mutante pasaba el assert viejo
    # ("valores_con != valores_sin"). Este oráculo exacto sí lo detecta: con el
    # mutante, factor_h queda en 1.10 (ver test de arriba) y el resultado en
    # ago sería 102.625*1.10=112.8875, no 119.7291667.
    esperado = [116.6666667, 119.7291667, 122.5]
    assert list(r_sin.resultado.ancho.loc["INPP"]) == pytest.approx(esperado)
    assert r_sin.manifiesto[0].sin_petroleo is True
