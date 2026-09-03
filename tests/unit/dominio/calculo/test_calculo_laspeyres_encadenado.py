from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from replica_inpp.dominio.calculo.laspeyres_directo import LaspeyresDirecto
from replica_inpp.dominio.calculo.laspeyres_encadenado import LaspeyresEncadenado
from replica_inpp.dominio.errores import ErrorCalculo, InvarianteViolado
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import ManifestCalculo

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
# regresión real: `factor_h` debe salir de `referencia` (calculado con
# LaspeyresDirecto sobre la canasta ANTERIOR con SUS pesos), nunca de
# promediar `f_j` con los pesos de la canasta nueva (ver docstring de
# `LaspeyresEncadenado`, bug real encontrado validando contra el BIE).
#
# `LaspeyresEncadenado` recibe `referencia: ResultadoIndice` (no
# canasta/serie crudas) -- se arma con `LaspeyresDirecto` sobre la canasta/
# serie anterior, igual que haría `calcular_indice` en la práctica.


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


def _referencia(
    agregacion: str = "INPP",
    rubro: str = "produccion_total",
    incluir_petroleo: bool = True,
    recorte: Any = "produccion_total",
    canasta: CanastaINPP | None = None,
    serie: SerieNormalizada | None = None,
) -> ResultadoIndice:
    """Arma el `ResultadoIndice` de referencia (canasta.version=2019) que
    `LaspeyresEncadenado` espera -- misma combinación agregacion/rubro/
    incluir_petroleo que se le va a pedir a `calcular`, igual que haría
    `calcular_indice` en la práctica.
    """
    return LaspeyresDirecto().calcular(
        canasta if canasta is not None else _canasta_anterior(),
        serie if serie is not None else _serie_anterior(recorte),
        agregacion,
        rubro=rubro,
        incluir_petroleo=incluir_petroleo,
    )


# ---------- básicos ----------


def test_calcular_retorna_resultado_indice() -> None:
    r = LaspeyresEncadenado(_referencia()).calcular(
        _canasta_nueva(), _serie_nueva(), "INPP", rubro="produccion_total"
    )
    assert isinstance(r, ResultadoIndice)


def test_valores_inpp_usa_peso_de_canasta_anterior_para_factor_h() -> None:
    r = LaspeyresEncadenado(_referencia()).calcular(
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
    r = LaspeyresEncadenado(_referencia(incluir_petroleo=False)).calcular(
        _canasta_nueva(), _serie_nueva(), "INPP", rubro="produccion_total", incluir_petroleo=False
    )
    m = r.manifiesto[0]
    assert m.version == 2025
    assert m.agregacion == "INPP"
    assert m.rubro == "produccion_total"
    assert m.incluir_petroleo is False
    assert m.calculador == "LaspeyresEncadenado"


# ---------- validación de referencia ----------


def test_referencia_version_incorrecta_lanza_invariante_violado() -> None:
    # referencia calculada con canasta.version=2025 (no 2019) -- no sirve como
    # tramo anterior.
    referencia_2025 = LaspeyresDirecto().calcular(
        _canasta_nueva(), _serie_nueva(), "INPP", rubro="produccion_total"
    )
    with pytest.raises(InvarianteViolado, match="version=2019"):
        LaspeyresEncadenado(referencia_2025).calcular(
            _canasta_nueva(), _serie_nueva(), "INPP", rubro="produccion_total"
        )


def test_referencia_combinacion_distinta_lanza_invariante_violado() -> None:
    # referencia calculada para SECTOR, se pide INPP -- no coincide.
    referencia_sector = _referencia(agregacion="SECTOR")
    with pytest.raises(InvarianteViolado, match="agregacion='INPP'"):
        LaspeyresEncadenado(referencia_sector).calcular(
            _canasta_nueva(), _serie_nueva(), "INPP", rubro="produccion_total"
        )


def test_referencia_con_manifiesto_ajeno_en_traslape_lanza_error_calculo() -> None:
    # `referencia` compuesta por 2 manifiestos, construida 100% vía API pública
    # (ResultadoIndice, sin mutar _df_resultado/_manifiesto): el manifiesto
    # correcto (2019/INPP/produccion_total) NO tiene fila en el traslape
    # (jul-2025); un manifiesto ajeno (2012/INPP/bienes_finales) sí la tiene,
    # con valor 999 -- ningún índice duplicado, el constructor lo acepta. Antes
    # de este fix, `.ancho` apilaba ambas filas bajo la misma etiqueta "INPP" y
    # el filtro de traslape tomaba la fila ajena sin darse cuenta.
    manifiesto_propio = ManifestCalculo(
        version=2019,
        agregacion="INPP",
        rubro="produccion_total",
        incluir_petroleo=True,
        calculador="LaspeyresDirecto",
    )
    manifiesto_ajeno = ManifestCalculo(
        version=2012,
        agregacion="INPP",
        rubro="bienes_finales",
        incluir_petroleo=True,
        calculador="LaspeyresDirecto",
    )
    df = pd.DataFrame(
        {
            "version": [2019, 2012],
            "agregacion": ["INPP", "INPP"],
            "rubro": ["produccion_total", "bienes_finales"],
            "indice_replicado": [100.0, 999.0],
            "estado_calculo": ["ok", "ok"],
            "motivo_error": [None, None],
        },
        index=pd.MultiIndex.from_tuples(
            [(PeriodoMensual(2019, 7), "INPP"), (_TRASLAPE, "INPP")], names=["periodo", "indice"]
        ),
    )
    referencia_compuesta = ResultadoIndice(
        df,
        [manifiesto_propio, manifiesto_ajeno],
        pd.DataFrame(index=df.index),
        pd.DataFrame(
            columns=[
                "version",
                "agregacion",
                "rubro",
                "periodo",
                "generico",
                "nivel_faltante",
                "tipo_faltante",
                "detalle",
            ]
        ),
    )
    assert df.index.is_unique  # precondición: el escenario no depende de duplicados

    with pytest.raises(ErrorCalculo, match="traslape"):
        LaspeyresEncadenado(referencia_compuesta).calcular(
            _canasta_nueva(), _serie_nueva(), "INPP", rubro="produccion_total"
        )


def test_canasta_nueva_version_incorrecta_lanza_invariante_violado() -> None:
    with pytest.raises(InvarianteViolado, match="canasta.version=2025"):
        LaspeyresEncadenado(_referencia()).calcular(
            _canasta_anterior(), _serie_nueva(), "INPP", rubro="produccion_total"
        )


# ---------- factor de encadenamiento (f_j) ----------


def test_f_j_totalmente_vacio_lanza_error_calculo() -> None:
    canasta_sin_encadenamiento = _canasta_nueva(f_j_total_vacio=True)
    with pytest.raises(ErrorCalculo, match="sin --encadenamientos"):
        LaspeyresEncadenado(_referencia()).calcular(
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
    referencia = _referencia(
        agregacion="SECTOR", rubro="exportaciones", recorte="mercado_exportacion"
    )
    r = LaspeyresEncadenado(referencia).calcular(
        canasta, serie_exp, "SECTOR", rubro="exportaciones"
    )
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


def test_traslape_faltante_en_referencia_lanza_error_calculo() -> None:
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
    referencia_sin_traslape = _referencia(serie=serie_sin_traslape)
    with pytest.raises(ErrorCalculo, match="traslape"):
        LaspeyresEncadenado(referencia_sin_traslape).calcular(
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
    referencia_enorme = _referencia(serie=serie_anterior_enorme)
    with pytest.raises(ErrorCalculo, match="no finito"):
        LaspeyresEncadenado(referencia_enorme).calcular(
            _canasta_nueva(), serie_nueva_enorme, "INPP", rubro="produccion_total"
        )


def test_grupo_sin_factor_h_en_tramo_anterior_usa_respaldo_por_f_j() -> None:
    # transporte cambia de sector "48" (canasta anterior) a "99" (canasta
    # nueva, sin equivalente en el tramo viejo) -- simula una reclasificación
    # de agrupación entre versiones: SECTOR "99" no existe en `referencia`, así
    # que cae al respaldo (promedio ponderado de f_j, ver docstring). Único
    # genérico del grupo -> el respaldo es exactamente su propio f_j (1.1),
    # y de-encadenar con ese mismo f_j y volver a encadenar con él recupera la
    # serie cruda tal cual (110.0/111.1/113.3), sin importar el peso.
    df_nueva = _canasta_nueva().df.copy()
    df_nueva.loc["460", ["codigo sector", "sector"]] = ["99", "99"]
    canasta_reclasificada = CanastaINPP(df_nueva, 2025)
    referencia_sector = _referencia(agregacion="SECTOR")
    r = LaspeyresEncadenado(referencia_sector).calcular(
        canasta_reclasificada, _serie_nueva(), "SECTOR", rubro="produccion_total"
    )
    ancho = r.resultado.ancho
    assert list(ancho.loc["99"]) == pytest.approx([110.0, 111.1, 113.3])


def test_grupo_sin_factor_h_ni_respaldo_por_f_j_lanza_error_calculo() -> None:
    # mismo escenario de reclasificación, pero además transporte -- único
    # genérico del grupo "99" -- no tiene f_j válido en 'encadenamiento
    # exportacion' (columna que sí admite NaN parcial, a diferencia de
    # 'encadenamiento total'/'produccion nacional', ver
    # CanastaINPP._COLUMNAS_ENCADENAMIENTO_PARCIAL): ni referencia ni respaldo,
    # no queda forma de calcular factor_h para el grupo "99".
    df_nueva = _canasta_nueva(f_j_exportacion_transporte_nan=True).df.copy()
    df_nueva.loc["460", ["codigo sector", "sector"]] = ["99", "99"]
    canasta_reclasificada = CanastaINPP(df_nueva, 2025)
    referencia_sector = _referencia(
        agregacion="SECTOR", rubro="exportaciones", recorte="mercado_exportacion"
    )
    with pytest.raises(ErrorCalculo, match="No hay factor_h"):
        LaspeyresEncadenado(referencia_sector).calcular(
            canasta_reclasificada,
            _serie_nueva(recorte="mercado_exportacion"),
            "SECTOR",
            rubro="exportaciones",
        )


def test_grupo_sin_factor_h_respaldo_con_2_genericos_promedia_ponderado() -> None:
    # acero (339, peso 30, f_j 1.2) y transporte (460, peso 40, f_j 1.1) pasan
    # los dos a sector "99" -- grupo nuevo de 2 genéricos, no 1: a diferencia
    # de test_..._usa_respaldo_por_f_j (grupo de 1, el respaldo se reduce a
    # "es igual a su propio f_j" y no prueba la fórmula), acá el respaldo
    # tiene que ser el promedio ponderado real:
    # factor_h = (30*1.2 + 40*1.1)/(30+40) = 80/70 = 1.142857...
    # i_tramo (peso 2025, de-encadenado): jul=(30*100+40*100)/70=100.0,
    # ago=(30*105+40*101)/70=102.714286, sep=(30*108+40*103)/70=105.142857
    # resultado = i_tramo * factor_h
    df_nueva = _canasta_nueva().df.copy()
    df_nueva.loc[["339", "460"], ["codigo sector", "sector"]] = [["99", "99"], ["99", "99"]]
    canasta_reclasificada = CanastaINPP(df_nueva, 2025)
    referencia_sector = _referencia(agregacion="SECTOR")
    r = LaspeyresEncadenado(referencia_sector).calcular(
        canasta_reclasificada, _serie_nueva(), "SECTOR", rubro="produccion_total"
    )
    ancho = r.resultado.ancho
    assert list(ancho.loc["99"]) == pytest.approx(
        [114.28571428571428, 117.3877551020408, 120.16326530612244]
    )


def test_grupo_sin_factor_h_respaldo_con_f_j_parcial_no_promedia_solo_lo_valido() -> None:
    # mismo grupo de 2 genéricos (acero+transporte -> sector "99"), pero
    # transporte sin f_j válido en 'encadenamiento exportacion' -- el respaldo
    # debe excluir el GRUPO entero (NaN), no promediar solo acero (que sí
    # tiene f_j válido). Si un cambio futuro promediara solo lo válido, este
    # grupo no lanzaría ErrorCalculo -- lanzaría un resultado calculado (mal)
    # con el peso completo del grupo pero el numerador de un solo genérico.
    df_nueva = _canasta_nueva(f_j_exportacion_transporte_nan=True).df.copy()
    df_nueva.loc[["339", "460"], ["codigo sector", "sector"]] = [["99", "99"], ["99", "99"]]
    canasta_reclasificada = CanastaINPP(df_nueva, 2025)
    referencia_sector = _referencia(
        agregacion="SECTOR", rubro="exportaciones", recorte="mercado_exportacion"
    )
    with pytest.raises(ErrorCalculo, match="No hay factor_h") as exc_info:
        LaspeyresEncadenado(referencia_sector).calcular(
            canasta_reclasificada,
            _serie_nueva(recorte="mercado_exportacion"),
            "SECTOR",
            rubro="exportaciones",
        )
    # el mensaje debe decir "no todos" tienen factor -- acero (339) SÍ tiene
    # f_j válido, solo transporte (460) no. "ningún genérico" sería falso acá
    # (regresión: mensaje viejo, hallazgo de negociación 2026-09-02).
    assert "no todos" in str(exc_info.value)
    assert "ningún" not in str(exc_info.value)


def test_referencia_con_indice_replicado_nan_en_traslape_lanza_error_calculo() -> None:
    # sector 48 (transporte) SÍ existía en 2019 (no reclasificado, a diferencia
    # de los tests de respaldo de arriba) -- pero su referencia quedó sin dato
    # en el traslape (estado_calculo="sin_datos", ej. la serie 2019 no cubría
    # ese periodo para ese genérico). No es una categoría nueva: no debe caer
    # al respaldo de f_j -- sería sustituir en silencio una referencia
    # inválida por un número calculado con peso 2025, cuando el grupo sí tenía
    # ponderador de julio 2019 real que debía usarse (negociación 2026-09-02).
    manifiesto = ManifestCalculo(
        version=2019,
        agregacion="SECTOR",
        rubro="produccion_total",
        incluir_petroleo=True,
        calculador="LaspeyresDirecto",
    )
    idx = pd.MultiIndex.from_tuples(
        [(_TRASLAPE, "11"), (_TRASLAPE, "21"), (_TRASLAPE, "31"), (_TRASLAPE, "48")],
        names=["periodo", "indice"],
    )
    df_largo = pd.DataFrame(
        {
            "version": 2019,
            "agregacion": "SECTOR",
            "rubro": "produccion_total",
            "indice_replicado": [130.0, 90.0, 115.0, float("nan")],
            "estado_calculo": ["ok", "ok", "ok", "sin_datos"],
            "motivo_error": [None, None, None, "faltantes en serie"],
        },
        index=idx,
    )
    referencia_sector = ResultadoIndice(
        df_largo,
        [manifiesto],
        pd.DataFrame(index=idx),
        pd.DataFrame(
            columns=[
                "version",
                "agregacion",
                "rubro",
                "periodo",
                "generico",
                "nivel_faltante",
                "tipo_faltante",
                "detalle",
            ]
        ),
    )
    with pytest.raises(ErrorCalculo, match="referencia inválida"):
        LaspeyresEncadenado(referencia_sector).calcular(
            _canasta_nueva(), _serie_nueva(), "SECTOR", rubro="produccion_total"
        )


def test_grupo_sin_factor_h_respaldo_con_f_j_extremo_no_desborda() -> None:
    # f_j=1e307 en acero y transporte (grupo nuevo "99") -- multiplicar el
    # peso CRUDO (30/40) por f_j desbordaría a inf antes de dividir
    # (30*1e307=3e308 > float64 max ≈1.7977e308) aunque el resultado
    # matemático real sea perfectamente finito. Con el mismo f_j en ambos
    # genéricos, el respaldo se cancela con el de-encadenado (serie/f_j*f_j)
    # y el resultado queda igual al promedio ponderado de la serie CRUDA, sin
    # importar el valor de f_j -- por eso no desborda con el fix (peso
    # normalizado antes de multiplicar) y antes sí (negociación 2026-09-02).
    f_j_extremo = 1e307
    df_nueva = _canasta_nueva().df.copy()
    df_nueva.loc[["339", "460"], ["codigo sector", "sector"]] = [["99", "99"], ["99", "99"]]
    columnas_f_j = [
        "encadenamiento total",
        "encadenamiento produccion nacional",
        "encadenamiento exportacion",
        "encadenamiento uso final",
    ]
    df_nueva.loc[["339", "460"], columnas_f_j] = f_j_extremo
    canasta_reclasificada = CanastaINPP(df_nueva, 2025)
    referencia_sector = _referencia(agregacion="SECTOR")
    r = LaspeyresEncadenado(referencia_sector).calcular(
        canasta_reclasificada, _serie_nueva(), "SECTOR", rubro="produccion_total"
    )
    assert list(r.resultado.ancho.loc["99"]) == pytest.approx(
        [114.28571428571429, 117.48571428571428, 120.28571428571429]
    )


# ---------- incluir_petroleo=False (debe excluir 070 en ambos cálculos: i_tramo y factor_h) ----------


def test_incluir_petroleo_false_excluye_070_de_i_tramo_y_de_factor_h() -> None:
    r_sin = LaspeyresEncadenado(_referencia(incluir_petroleo=False)).calcular(
        _canasta_nueva(), _serie_nueva(), "INPP", rubro="produccion_total", incluir_petroleo=False
    )
    # i_tramo sin petróleo (pesos NUEVOS 10/30/40 sobre soya/acero/transporte,
    # renormalizados sobre 80): jul=100.0, ago=(10*102+30*105+40*101)/80=102.625,
    # sep=(10*104+30*108+40*103)/80=105.0
    # factor_h sin petróleo = I_viejo(traslape)/100 con pesos VIEJOS iguales
    # (25/25/25 -> promedio simple de soya/acero/transporte) = (130+115+105)/3
    # /100 = 116.6667/100 = 1.166667
    #
    # Un mutante que excluya 070 SOLO de i_tramo (deja factor_h calculado con
    # una referencia con incluir_petroleo=True, petróleo incluido) sigue dando un
    # resultado != con_petroleo -- ese mutante pasaba el assert viejo
    # ("valores_con != valores_sin"). Este oráculo exacto sí lo detecta: con el
    # mutante, factor_h queda en 1.10 (ver test de arriba) y el resultado en
    # ago sería 102.625*1.10=112.8875, no 119.7291667.
    esperado = [116.6666667, 119.7291667, 122.5]
    assert list(r_sin.resultado.ancho.loc["INPP"]) == pytest.approx(esperado)
    assert r_sin.manifiesto[0].incluir_petroleo is False
