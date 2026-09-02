from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import replica_inpp as rep
from replica_inpp.api import variaciones
from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import ManifestCalculo, ManifestDerivado

_PERIODOS = [PeriodoMensual(2019, 7), PeriodoMensual(2019, 8), PeriodoMensual(2019, 9)]
_GENERICOS = ["001", "070"]


def _canasta() -> CanastaINPP:
    df = pd.DataFrame(
        {
            "generico": ["soya", "petroleo"],
            "codigo sector": ["11", "21"],
            "sector": ["11", "21"],
            "codigo subsector": ["111", "211"],
            "subsector": ["111", "211"],
            "codigo rama": ["1111", "2111"],
            "rama": ["1111", "2111"],
            "codigo subrama": ["11111", "21111"],
            "subrama": ["11111", "21111"],
            "codigo clase": ["111111", "211111"],
            "clase": ["111111", "211111"],
            "produccion total": [50.0, 50.0],
            "bienes intermedios": [50.0, 50.0],
            "bienes finales": [50.0, 50.0],
            "demanda interna total": [50.0, 50.0],
            "demanda interna consumo": [50.0, 50.0],
            "demanda interna capital": [50.0, 50.0],
            "exportaciones": [50.0, 50.0],
            "encadenamiento total": [None] * 2,
            "encadenamiento produccion nacional": [None] * 2,
            "encadenamiento exportacion": [None] * 2,
            "encadenamiento uso final": [None] * 2,
        },
        index=_GENERICOS,
    )
    return CanastaINPP(df, 2019)


def _serie() -> SerieNormalizada:
    df = pd.DataFrame(
        {
            "generico": ["soya", "petroleo"],
            _PERIODOS[0]: [100.0, 100.0],
            _PERIODOS[1]: [101.0, 103.0],
            _PERIODOS[2]: [102.0, 106.0],
        },
        index=_GENERICOS,
    )
    return SerieNormalizada(df, "produccion_total")


def test_rep_expone_variaciones_en_fachada_publica() -> None:
    # a diferencia de los demás tests de este archivo (import directo del
    # módulo + mocks), este pasa por `replica_inpp.__init__` -- protege
    # contra que se rompa el wiring de `__all__`/import ahí sin que ningún
    # otro test lo note. Negociado en /negociar-hallazgos (hallazgo #5,
    # 2026-09-01).
    for nombre in (
        "variacion_periodica",
        "variacion_acumulada_anual",
        "variacion_desde",
        "inflacion_en",
        "inflacion_acumulada",
        "inflacion_promedio",
        "inflacion_maxima",
        "inflacion_minima",
        "ResultadoVariacion",
        "ManifestDerivado",
    ):
        assert nombre in rep.__all__

    indice = rep.calcular_indice(_canasta(), _serie(), "INPP", rubro="produccion_total")
    v = rep.variacion_periodica(indice, "mensual")
    assert isinstance(v, ResultadoVariacion)
    assert isinstance(v.manifiesto, ManifestDerivado)

    # variacion_acumulada_anual requiere diciembre del año anterior, fuera del
    # rango chico de este fixture -- ya cubierta en
    # tests/unit/dominio/calculo/test_calculo_variaciones.py; este test solo
    # protege el wiring de `__all__`, no re-prueba la lógica.
    vd = rep.variacion_desde(indice, "jul 2019", "sep 2019")
    assert isinstance(vd, ResultadoVariacion)

    en: Any = rep.inflacion_en(v, "ago 2019")
    assert "variacion_pp" in en.columns

    acumulada = rep.inflacion_acumulada(v, "ago 2019", "sep 2019", indice="INPP")
    assert isinstance(acumulada, float)

    promedio = rep.inflacion_promedio(v, indice="INPP", metodo="simple")
    assert isinstance(promedio, float)

    periodo, idx, valor = rep.inflacion_maxima(v)
    assert isinstance(periodo, str)
    assert isinstance(idx, str)
    assert isinstance(valor, float)

    periodo_min, idx_min, valor_min = rep.inflacion_minima(v)
    assert isinstance(periodo_min, str)
    assert isinstance(idx_min, str)
    assert isinstance(valor_min, float)


def test_rep_inflacion_promedio_tcac_reenvia_rango_y_falla_por_extremo_ausente() -> None:
    # Cierra el hueco señalado en /negociar-hallazgos (ronda 4, 2026-09-01):
    # rep.inflacion_promedio no tenía ningún test con rango explícito +
    # metodo="tcac" a través de la fachada -- una regresión en la conversión
    # de periodos o el reenvío de desde/hasta podría pasar desapercibida.
    # "OTRO" tiene May completo; "INPP" no (sin_datos), para que May exista
    # globalmente y el extremo ausente sea específico de "INPP".
    periodos = [PeriodoMensual(2024, m) for m in range(1, 6)]
    valores_inpp = [100.0 + 5 * i for i in range(5)]
    valores_inpp[4] = float("nan")  # Mayo sin dato para INPP
    valores_otro = [50.0 + 3 * i for i in range(5)]
    rows = []
    for p, v in zip(periodos, valores_inpp):
        rows.append(
            {
                "periodo": p,
                "indice": "INPP",
                "version": 2019,
                "agregacion": "INPP",
                "rubro": "produccion_total",
                "indice_replicado": v,
                "estado_calculo": "ok" if v == v else "sin_datos",
            }
        )
    for p, v in zip(periodos, valores_otro):
        rows.append(
            {
                "periodo": p,
                "indice": "OTRO",
                "version": 2019,
                "agregacion": "INPP",
                "rubro": "produccion_total",
                "indice_replicado": v,
                "estado_calculo": "ok",
            }
        )
    df = pd.DataFrame(rows).set_index(["periodo", "indice"])
    manifiesto = [ManifestCalculo(2019, "INPP", "produccion_total", True, "LaspeyresDirecto")]
    indice = ResultadoIndice(df, manifiesto, pd.DataFrame(), pd.DataFrame())
    v = rep.variacion_periodica(indice, "mensual")

    # match exacto (no solo "hasta") -- negociado en /negociar-hallazgos (ronda
    # 5, 2026-09-01): un match genérico también matchea el mensaje de un
    # mutante distinto ("'desde' (May 2024) no puede ser posterior a 'hasta'
    # (Feb 2024)"), que también contiene la palabra "hasta".
    with pytest.raises(InvarianteViolado, match=r"falta el extremo 'hasta' \(May 2024\)"):
        rep.inflacion_promedio(v, "feb 2024", "may 2024", indice="INPP", metodo="tcac")


# -- series: conversión de frontera --------------------------------------------


def test_variacion_periodica_delega(mocker) -> None:
    fn = mocker.patch.object(variaciones, "_variacion_periodica", return_value="rv")
    assert variaciones.variacion_periodica("idx", "mensual") == "rv"  # type: ignore[arg-type]
    fn.assert_called_once_with("idx", "mensual")


def test_variacion_desde_convierte_desde_y_hasta(mocker) -> None:
    fn = mocker.patch.object(variaciones, "_variacion_desde", return_value="rv")

    variaciones.variacion_desde("idx", "ene 2015", "DIC 2024", incluir_parciales=False)  # type: ignore[arg-type]

    fn.assert_called_once_with("idx", PeriodoMensual(2015, 1), PeriodoMensual(2024, 12), False)


def test_variacion_desde_hasta_none_pasa_none(mocker) -> None:
    fn = mocker.patch.object(variaciones, "_variacion_desde", return_value="rv")
    variaciones.variacion_desde("idx", "jul 2019")  # type: ignore[arg-type]
    fn.assert_called_once_with("idx", PeriodoMensual(2019, 7), None, True)


# -- análisis: Periodo -> str en las tuplas ------------------------------------


def test_inflacion_maxima_devuelve_periodo_como_str(mocker) -> None:
    mocker.patch.object(
        variaciones._consulta,
        "inflacion_maxima",
        return_value=(PeriodoMensual(2024, 12), "INPP", 1.5),
    )
    periodo, indice, valor = variaciones.inflacion_maxima("rv")  # type: ignore[arg-type]
    assert (periodo, indice, valor) == ("Dic 2024", "INPP", 1.5)
    assert isinstance(periodo, str)


def test_inflacion_minima_devuelve_periodo_como_str(mocker) -> None:
    mocker.patch.object(
        variaciones._consulta,
        "inflacion_minima",
        return_value=(PeriodoMensual(2020, 4), "SECTOR", -0.3),
    )
    periodo, indice, valor = variaciones.inflacion_minima("rv")  # type: ignore[arg-type]
    assert periodo == "Abr 2020"
    assert (indice, valor) == ("SECTOR", -0.3)


def test_inflacion_en_parsea_periodo(mocker) -> None:
    fn = mocker.patch.object(variaciones._consulta, "inflacion_en", return_value="df")
    assert variaciones.inflacion_en("rv", "dic 2024") == "df"  # type: ignore[arg-type]
    fn.assert_called_once_with("rv", PeriodoMensual(2024, 12))


def test_inflacion_acumulada_convierte_rango(mocker) -> None:
    fn = mocker.patch.object(variaciones._consulta, "inflacion_acumulada", return_value=2.0)
    variaciones.inflacion_acumulada("rv", "ene 2015", "dic 2024", indice="INPP")  # type: ignore[arg-type]
    fn.assert_called_once_with(
        "rv", PeriodoMensual(2015, 1), PeriodoMensual(2024, 12), indice="INPP"
    )


def test_inflacion_promedio_convierte_rango_y_reenvia_metodo(mocker) -> None:
    # metodo="simple" (NO el default "tcac") + rango explícito -- negociado en
    # /negociar-hallazgos (ronda 5, 2026-09-01): el test end-to-end de más
    # arriba usaba "tcac", que coincide con el default y no distingue si la
    # fachada realmente reenvía `metodo` o lo ignora.
    fn = mocker.patch.object(variaciones._consulta, "inflacion_promedio", return_value=3.5)
    variaciones.inflacion_promedio("rv", "ene 2015", "dic 2024", indice="INPP", metodo="simple")  # type: ignore[arg-type]
    fn.assert_called_once_with(
        "rv",
        PeriodoMensual(2015, 1),
        PeriodoMensual(2024, 12),
        indice="INPP",
        metodo="simple",
    )
