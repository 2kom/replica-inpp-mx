from __future__ import annotations

import pandas as pd
import pytest

from replica_inpp.dominio.calculo.variaciones import (
    variacion_acumulada_anual,
    variacion_desde,
    variacion_periodica,
)
from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import ManifestCalculo

# Tabla de lags esperada, independiente de LAG_MENSUAL (deliberadamente NO importada
# de _temporal.py: si el productor cambia un lag por error, este oráculo no debe
# moverse con él.
_LAGS_MENSUAL_ESPERADOS = {
    "mensual": 1,
    "bimestral": 2,
    "trimestral": 3,
    "cuatrimestral": 4,
    "semestral": 6,
    "anual": 12,
}

# -- helpers -------------------------------------------------------------------


def _indice(
    data: dict[str, list[tuple[object, float | None]]],
    *,
    agregacion: str = "INPP",
    rubro: str = "produccion_total",
    version: int = 2019,
    estados: dict[tuple[object, str], str] | None = None,
    reporte: pd.DataFrame | None = None,
) -> ResultadoIndice:
    rows = []
    for indice, pares in data.items():
        for periodo, valor in pares:
            est = "ok" if valor is not None else "sin_datos"
            if estados and (periodo, indice) in estados:
                est = estados[(periodo, indice)]
            rows.append(
                {
                    "periodo": periodo,
                    "indice": indice,
                    "version": version,
                    "agregacion": agregacion,
                    "rubro": rubro,
                    "indice_replicado": float("nan") if valor is None else float(valor),
                    "estado_calculo": est,
                }
            )
    df = pd.DataFrame(rows).set_index(["periodo", "indice"])
    manifiesto = [
        ManifestCalculo(
            version,  # type: ignore[arg-type]
            agregacion,
            rubro,
            incluir_petroleo=True,
            calculador="LaspeyresDirecto",
        )
    ]
    return ResultadoIndice(
        df,
        manifiesto,
        reporte if reporte is not None else pd.DataFrame(),
        pd.DataFrame(),
    )


_M1 = PeriodoMensual(2019, 7)
_M2 = PeriodoMensual(2019, 8)
_M3 = PeriodoMensual(2019, 9)
_M4 = PeriodoMensual(2019, 10)


def _indice_mensual() -> ResultadoIndice:
    return _indice({"INPP": [(_M1, 100.0), (_M2, 103.0), (_M3, 106.0), (_M4, 109.0)]})


# -- variacion_periodica -------------------------------------------------------


def test_periodica_retorna_resultado_variacion() -> None:
    r = variacion_periodica(_indice_mensual(), "mensual")
    assert isinstance(r, ResultadoVariacion)


def test_periodica_clase_embebe_frecuencia() -> None:
    r = variacion_periodica(_indice_mensual(), "mensual")
    assert (r.resultado.largo["clase_variacion"] == "periodica_mensual").all()
    assert r.manifiesto.clase == "periodica_mensual"


def test_periodica_valores_en_pp() -> None:
    r = variacion_periodica(_indice_mensual(), "mensual")
    assert r.df["variacion_pp"].tolist() == pytest.approx(
        [3.0, 106 / 103 * 100 - 100, 109 / 106 * 100 - 100]
    )


def test_periodica_primer_periodo_sin_base_ausente() -> None:
    r = variacion_periodica(_indice_mensual(), "mensual")
    assert _M1 not in r.df.index.get_level_values("periodo")
    assert len(r.df) == 3


def test_periodica_frecuencia_invalida_falla() -> None:
    with pytest.raises(InvarianteViolado):
        variacion_periodica(_indice_mensual(), "decenal")  # type: ignore[arg-type]


def test_periodica_sin_filas_computables_falla() -> None:
    solo_uno = _indice({"INPP": [(_M1, 100.0)]})
    with pytest.raises(InvarianteViolado):
        variacion_periodica(solo_uno, "mensual")


def test_periodica_estado_parcial_propagado() -> None:
    indice = _indice(
        {"INPP": [(_M1, 100.0), (_M2, 103.0)]},
        estados={(_M2, "INPP"): "parcial"},
    )
    r = variacion_periodica(indice, "mensual")
    assert r.resultado.largo.loc[(_M2, "INPP"), "estado_calculo"] == "parcial"  # type: ignore[index]


def test_periodica_fuente_sin_datos_ausente_y_en_reporte() -> None:
    indice = _indice({"INPP": [(_M1, 100.0), (_M2, 103.0), (_M3, None), (_M4, 109.0)]})
    r = variacion_periodica(indice, "mensual")
    assert (_M3, "INPP") not in r.df.index
    assert (_M3, "INPP") in r.reporte.index
    assert r.reporte.loc[(_M3, "INPP"), "estado_calculo"] == "sin_datos"  # type: ignore[index]


def test_periodica_base_cero_falla() -> None:
    indice = _indice({"INPP": [(_M1, 0.0), (_M2, 103.0)]})
    with pytest.raises(InvarianteViolado):
        variacion_periodica(indice, "mensual")


def test_periodica_base_no_finita_falla() -> None:
    indice = _indice({"INPP": [(_M1, float("inf")), (_M2, 103.0)]})
    with pytest.raises(InvarianteViolado):
        variacion_periodica(indice, "mensual")


def test_periodica_overflow_en_variacion_falla() -> None:
    # Ambos extremos finitos; el cociente 1e308/1e-308 desborda a inf.
    indice = _indice({"INPP": [(_M1, 1e-308), (_M2, 1e308)]})
    with pytest.raises(InvarianteViolado):
        variacion_periodica(indice, "mensual")


def test_periodica_numerador_cero_acepta_menos_cien() -> None:
    indice = _indice({"INPP": [(_M1, 100.0), (_M2, 0.0)]})
    r = variacion_periodica(indice, "mensual")
    assert r.df.loc[(_M2, "INPP"), "variacion_pp"] == pytest.approx(-100.0)  # type: ignore[index]


def test_periodica_manifiesto_y_diagnostico() -> None:
    indice = _indice({"INPP": [(_M1, 100.0), (_M2, 103.0), (_M3, None), (_M4, 109.0)]})
    r = variacion_periodica(indice, "mensual")
    assert r.manifiesto.versiones == [2019]
    assert len(r.diagnostico) == 3
    assert r.indices_parciales is None


def _avanzar_meses(base: PeriodoMensual, pasos: int) -> PeriodoMensual:
    """Avanza `pasos` meses desde `base`, mes por mes.

    Implementación propia (loop, no aritmética de ordinal), deliberadamente
    independiente de `restar_meses` de producción: si fixture y lookup
    comparten la misma función, un bug de desplazamiento en esa función se
    cancela entre ambos lados y el test queda ciego a él.
    """
    año, mes = base.año, base.mes
    for _ in range(pasos):
        mes, año = (1, año + 1) if mes == 12 else (mes + 1, año)
    return PeriodoMensual(año, mes)


def _serie_mensual_n(n: int, base: PeriodoMensual = _M1) -> ResultadoIndice:
    periodos = [_avanzar_meses(base, i) for i in range(n)]
    valores = [100.0 + 3.0 * i for i in range(n)]
    return _indice({"INPP": list(zip(periodos, valores))})


@pytest.mark.parametrize(("frecuencia", "lag"), sorted(_LAGS_MENSUAL_ESPERADOS.items()))
def test_periodica_todas_las_frecuencias(frecuencia: str, lag: int) -> None:
    n = lag + 2
    r = variacion_periodica(_serie_mensual_n(n), frecuencia)  # type: ignore[arg-type]
    assert (r.resultado.largo["clase_variacion"] == f"periodica_{frecuencia}").all()
    ultimo = _avanzar_meses(_M1, n - 1)
    val_t = 100.0 + 3.0 * (n - 1)
    val_base = 100.0 + 3.0 * (n - 1 - lag)
    esperado = (val_t / val_base - 1.0) * 100.0
    assert r.df.loc[(ultimo, "INPP"), "variacion_pp"] == pytest.approx(esperado)  # type: ignore[index]


# -- variacion_acumulada_anual -------------------------------------------------


def test_acumulada_base_diciembre_anio_anterior() -> None:
    indice = _indice(
        {"INPP": [(PeriodoMensual(2018, 12), 100.0), (PeriodoMensual(2019, 12), 110.0)]}
    )
    r = variacion_acumulada_anual(indice)
    assert (r.resultado.largo["clase_variacion"] == "acumulada_anual").all()
    assert r.df.loc[(PeriodoMensual(2019, 12), "INPP"), "variacion_pp"] == pytest.approx(  # type: ignore[index]
        10.0
    )
    assert len(r.df) == 1


def test_acumulada_periodo_ordinario_enero() -> None:
    indice = _indice(
        {
            "INPP": [
                (PeriodoMensual(2019, 12), 100.0),
                (PeriodoMensual(2020, 1), 101.0),
                (PeriodoMensual(2020, 2), 102.0),
            ]
        }
    )
    r = variacion_acumulada_anual(indice)
    assert r.df.loc[(PeriodoMensual(2020, 1), "INPP"), "variacion_pp"] == pytest.approx(  # type: ignore[index]
        1.0
    )
    assert r.df.loc[(PeriodoMensual(2020, 2), "INPP"), "variacion_pp"] == pytest.approx(  # type: ignore[index]
        2.0
    )


# -- variacion_desde -----------------------------------------------------------


def _indice_dos() -> ResultadoIndice:
    return _indice({"A": [(_M1, 100.0), (_M2, 110.0)], "B": [(_M1, 100.0), (_M2, 90.0)]})


def test_desde_una_fila_por_indice() -> None:
    r = variacion_desde(_indice_dos(), _M1, _M2)
    assert len(r.df) == 2
    assert set(r.df.index.get_level_values("indice")) == {"A", "B"}


def test_desde_valores_correctos() -> None:
    r = variacion_desde(_indice_dos(), _M1, _M2)
    assert r.df.loc[(_M2, "A"), "variacion_pp"] == pytest.approx(10.0)  # type: ignore[index]
    assert r.df.loc[(_M2, "B"), "variacion_pp"] == pytest.approx(-10.0)  # type: ignore[index]


def test_desde_indices_parciales_vacio_si_exacto() -> None:
    r = variacion_desde(_indice_dos(), _M1, _M2)
    assert r.indices_parciales is not None
    assert r.indices_parciales.empty
    assert list(r.indices_parciales.columns) == ["periodo_desde_real", "periodo_hasta_real"]


def test_desde_incluir_parciales_ajusta_periodo() -> None:
    indice = _indice(
        {
            "A": [(_M1, 100.0), (_M2, 95.0), (_M3, 110.0)],
            "B": [(_M1, None), (_M2, 95.0), (_M3, 90.0)],
        }
    )
    r = variacion_desde(indice, _M1, _M3, incluir_parciales=True)
    assert len(r.df) == 2
    assert r.indices_parciales.loc["B", "periodo_desde_real"] == _M2  # type: ignore[union-attr]
    assert r.df.loc[(_M3, "B"), "variacion_pp"] == pytest.approx(  # type: ignore[index]
        90 / 95 * 100 - 100
    )


def test_desde_sin_parciales_excluye_indice() -> None:
    indice = _indice(
        {
            "A": [(_M1, 100.0), (_M3, 110.0)],
            "B": [(_M1, None), (_M3, 90.0)],
        }
    )
    r = variacion_desde(indice, _M1, _M3, incluir_parciales=False)
    assert set(r.df.index.get_level_values("indice")) == {"A"}
    assert len(r.diagnostico) == 1


def test_desde_sin_parciales_excluye_indice_con_estado_parcial() -> None:
    indice = _indice(
        {"A": [(_M1, 100.0), (_M2, 110.0)], "B": [(_M1, 100.0), (_M2, 90.0)]},
        estados={(_M2, "A"): "parcial"},
    )
    r_con = variacion_desde(indice, _M1, _M2, incluir_parciales=True)
    assert set(r_con.df.index.get_level_values("indice")) == {"A", "B"}
    r_sin = variacion_desde(indice, _M1, _M2, incluir_parciales=False)
    assert set(r_sin.df.index.get_level_values("indice")) == {"B"}


def test_desde_hasta_anterior_a_desde_falla() -> None:
    with pytest.raises(InvarianteViolado):
        variacion_desde(_indice_dos(), _M2, _M1)


def test_desde_periodo_inexistente_falla() -> None:
    with pytest.raises(InvarianteViolado):
        variacion_desde(_indice_dos(), PeriodoMensual(2099, 1), _M2)


def test_desde_base_cero_falla() -> None:
    indice = _indice({"A": [(_M1, 0.0), (_M2, 110.0)]})
    with pytest.raises(InvarianteViolado):
        variacion_desde(indice, _M1, _M2)


def test_desde_extremo_no_finito_falla() -> None:
    indice = _indice({"A": [(_M1, 100.0), (_M2, float("inf"))]})
    with pytest.raises(InvarianteViolado):
        variacion_desde(indice, _M1, _M2)


def test_desde_overflow_en_variacion_falla() -> None:
    # Ambos extremos finitos; el cociente 1e308/1e-308 desborda a inf.
    indice = _indice({"A": [(_M1, 1e-308), (_M2, 1e308)]})
    with pytest.raises(InvarianteViolado):
        variacion_desde(indice, _M1, _M2)


def test_desde_numerador_cero_acepta_menos_cien() -> None:
    indice = _indice({"A": [(_M1, 100.0), (_M2, 0.0)]})
    r = variacion_desde(indice, _M1, _M2)
    assert r.df.loc[(_M2, "A"), "variacion_pp"] == pytest.approx(-100.0)  # type: ignore[index]


def test_desde_hasta_none_usa_ultimo_periodo() -> None:
    r = variacion_desde(_indice_dos(), _M1)
    assert set(r.df.index.get_level_values("periodo")) == {_M2}


def test_desde_incluir_parciales_ajusta_periodo_hasta() -> None:
    indice = _indice(
        {
            "A": [(_M1, 100.0), (_M2, 105.0), (_M3, 110.0)],
            "B": [(_M1, 100.0), (_M2, 105.0), (_M3, None)],
        }
    )
    r = variacion_desde(indice, _M1, _M3, incluir_parciales=True)
    assert r.indices_parciales.loc["B", "periodo_hasta_real"] == _M2  # type: ignore[union-attr]
    assert r.df.loc[(_M2, "B"), "variacion_pp"] == pytest.approx(5.0)  # type: ignore[index]


def test_desde_reporte_usa_periodos_efectivos_no_los_solicitados() -> None:
    # "B" cae por ambos lados (falta en _M1/_M4 -> desde_real=_M2, hasta_real=_M3).
    # version/cobertura llevan un valor centinela en los periodos NOMINALES
    # (_M1/_M4, version=2012) distinto de los reales (_M2=2019, _M3=2025) para
    # que, si _construir_fila_reporte alguna vez vuelve a leer el periodo
    # pedido en vez del resuelto, el assert lo note. 2012/2019/2025 son
    # VersionCanasta válidas (no un centinela fuera de rango como 9999).
    filas = [
        {
            "periodo": _M1,
            "indice": "B",
            "version": 2012,
            "agregacion": "INPP",
            "rubro": "produccion_total",
            "indice_replicado": float("nan"),
            "estado_calculo": "sin_datos",
        },
        {
            "periodo": _M2,
            "indice": "B",
            "version": 2019,
            "agregacion": "INPP",
            "rubro": "produccion_total",
            "indice_replicado": 100.0,
            "estado_calculo": "ok",
        },
        {
            "periodo": _M3,
            "indice": "B",
            "version": 2025,
            "agregacion": "INPP",
            "rubro": "produccion_total",
            "indice_replicado": 120.0,
            "estado_calculo": "ok",
        },
        {
            "periodo": _M4,
            "indice": "B",
            "version": 2012,
            "agregacion": "INPP",
            "rubro": "produccion_total",
            "indice_replicado": float("nan"),
            "estado_calculo": "sin_datos",
        },
    ]
    df = pd.DataFrame(filas).set_index(["periodo", "indice"])
    reporte_fuente = pd.DataFrame(
        {"cobertura_genericos_pct": [999.0, 40.0, 80.0, 999.0]},
        index=pd.MultiIndex.from_tuples(
            [(_M1, "B"), (_M2, "B"), (_M3, "B"), (_M4, "B")], names=["periodo", "indice"]
        ),
    )
    manifiesto = [
        ManifestCalculo(2012, "INPP", "produccion_total", True, "LaspeyresDirecto"),
        ManifestCalculo(2019, "INPP", "produccion_total", True, "LaspeyresDirecto"),
        ManifestCalculo(2025, "INPP", "produccion_total", True, "LaspeyresDirecto"),
    ]
    indice = ResultadoIndice(df, manifiesto, reporte_fuente, pd.DataFrame())

    r = variacion_desde(indice, _M1, _M4, incluir_parciales=True)

    assert r.indices_parciales.loc["B", "periodo_desde_real"] == _M2  # type: ignore[union-attr]
    assert r.indices_parciales.loc["B", "periodo_hasta_real"] == _M3  # type: ignore[union-attr]

    fila_reporte: pd.Series = r.reporte.loc[(_M3, "B")]  # type: ignore[index,assignment]
    assert fila_reporte["periodo_lag"] == _M2
    assert fila_reporte["indice_t"] == pytest.approx(120.0)
    assert fila_reporte["indice_lag"] == pytest.approx(100.0)
    assert fila_reporte["version_t"] == 2025
    assert fila_reporte["version_lag"] == 2019
    assert fila_reporte["cobertura_pct_t"] == pytest.approx(80.0)
    assert fila_reporte["cobertura_pct_lag"] == pytest.approx(40.0)


def test_desde_ningun_indice_computable_falla() -> None:
    indice = _indice({"A": [(_M1, None), (_M2, None)]})
    with pytest.raises(InvarianteViolado):
        variacion_desde(indice, _M1, _M2, incluir_parciales=False)


# -- cobertura -----------------------------------------------------------------


# -- combinación única (agregacion, rubro) -------------------------------------


def _indice_heterogeneo() -> ResultadoIndice:
    # SECTOR/exportaciones v2025 mezclado con INPP/produccion_total v2019 en un
    # mismo ResultadoIndice -- contractual vía la API pública (2 manifiestos,
    # cada uno con al menos una fila que lo respalda). Negociado en
    # /negociar-hallazgos (hallazgo #3, 2026-09-01).
    rows = [
        {
            "periodo": _M1,
            "indice": "X",
            "version": 2019,
            "agregacion": "INPP",
            "rubro": "produccion_total",
            "indice_replicado": 100.0,
            "estado_calculo": "ok",
        },
        {
            "periodo": _M2,
            "indice": "X",
            "version": 2025,
            "agregacion": "SECTOR",
            "rubro": "exportaciones",
            "indice_replicado": 200.0,
            "estado_calculo": "ok",
        },
    ]
    df = pd.DataFrame(rows).set_index(["periodo", "indice"])
    manifiesto = [
        ManifestCalculo(2019, "INPP", "produccion_total", True, "LaspeyresDirecto"),
        ManifestCalculo(2025, "SECTOR", "exportaciones", True, "LaspeyresDirecto"),
    ]
    return ResultadoIndice(df, manifiesto, pd.DataFrame(), pd.DataFrame())


def test_periodica_combinacion_heterogenea_falla() -> None:
    with pytest.raises(InvarianteViolado, match="agregacion, rubro"):
        variacion_periodica(_indice_heterogeneo(), "mensual")


def test_desde_combinacion_heterogenea_falla() -> None:
    with pytest.raises(InvarianteViolado, match="agregacion, rubro"):
        variacion_desde(_indice_heterogeneo(), _M1, _M2)


def test_reporte_propaga_cobertura_del_fuente() -> None:
    reporte = pd.DataFrame(
        {"cobertura_genericos_pct": [88.0, 90.0]},
        index=pd.MultiIndex.from_tuples(
            [(_M1, "INPP"), (_M2, "INPP")], names=["periodo", "indice"]
        ),
    )
    indice = _indice({"INPP": [(_M1, 100.0), (_M2, 103.0)]}, reporte=reporte)
    r = variacion_periodica(indice, "mensual")
    assert r.reporte.loc[(_M2, "INPP"), "cobertura_pct_t"] == pytest.approx(  # type: ignore[index]
        90.0
    )
    assert r.reporte.loc[(_M2, "INPP"), "cobertura_pct_lag"] == pytest.approx(  # type: ignore[index]
        88.0
    )


# -- en_indefinido="marcar" -----------------------------------------------------
# Caso real que motiva esto: un `indice` puntual (ej. subsector '517' del INPP,
# mercado_exportación) cae permanentemente a 0 por un cambio estructural real
# (reforma de telecomunicaciones 2015) -- con "error" (default) eso revienta
# TODA la corrida aunque el resto de los índices calculen bien.


def test_periodica_en_indefinido_marcar_excluye_base_cero_y_conserva_los_demas() -> None:
    indice = _indice({"A": [(_M1, 100.0), (_M2, 103.0)], "B": [(_M1, 0.0), (_M2, 103.0)]})
    r = variacion_periodica(indice, "mensual", en_indefinido="marcar")
    assert (_M2, "A") in r.df.index
    assert (_M2, "B") not in r.df.index
    assert r.df.loc[(_M2, "A"), "variacion_pp"] == pytest.approx(3.0)  # type: ignore[index]


def test_periodica_en_indefinido_marcar_diagnostico_marca_indefinido_con_motivo() -> None:
    indice = _indice({"A": [(_M1, 100.0), (_M2, 103.0)], "B": [(_M1, 0.0), (_M2, 103.0)]})
    r = variacion_periodica(indice, "mensual", en_indefinido="marcar")
    fila = r.diagnostico[(r.diagnostico["periodo"] == _M2) & (r.diagnostico["indice"] == "B")]
    assert len(fila) == 1
    assert fila["estado_calculo"].iloc[0] == "indefinido"
    assert "base=0" in fila["motivo_error"].iloc[0]


def test_periodica_en_indefinido_marcar_overflow_tambien_se_excluye() -> None:
    # mismo escenario que test_periodica_overflow_en_variacion_falla, sin raise.
    indice = _indice({"A": [(_M1, 100.0), (_M2, 103.0)], "B": [(_M1, 1e-308), (_M2, 1e308)]})
    r = variacion_periodica(indice, "mensual", en_indefinido="marcar")
    assert (_M2, "A") in r.df.index
    assert (_M2, "B") not in r.df.index
    fila = r.diagnostico[(r.diagnostico["periodo"] == _M2) & (r.diagnostico["indice"] == "B")]
    assert fila["estado_calculo"].iloc[0] == "indefinido"


def test_periodica_en_indefinido_default_error_revienta_aunque_haya_indice_valido() -> None:
    indice = _indice({"A": [(_M1, 100.0), (_M2, 103.0)], "B": [(_M1, 0.0), (_M2, 103.0)]})
    with pytest.raises(InvarianteViolado):
        variacion_periodica(indice, "mensual")


def test_acumulada_en_indefinido_marcar_excluye_base_cero_y_conserva_los_demas() -> None:
    indice = _indice(
        {
            "A": [(PeriodoMensual(2018, 12), 100.0), (PeriodoMensual(2019, 12), 110.0)],
            "B": [(PeriodoMensual(2018, 12), 0.0), (PeriodoMensual(2019, 12), 110.0)],
        }
    )
    r = variacion_acumulada_anual(indice, en_indefinido="marcar")
    assert (PeriodoMensual(2019, 12), "A") in r.df.index
    assert (PeriodoMensual(2019, 12), "B") not in r.df.index


def test_acumulada_en_indefinido_default_error_revienta() -> None:
    indice = _indice(
        {
            "A": [(PeriodoMensual(2018, 12), 100.0), (PeriodoMensual(2019, 12), 110.0)],
            "B": [(PeriodoMensual(2018, 12), 0.0), (PeriodoMensual(2019, 12), 110.0)],
        }
    )
    with pytest.raises(InvarianteViolado):
        variacion_acumulada_anual(indice)


def test_desde_en_indefinido_marcar_excluye_indice_base_cero_y_conserva_los_demas() -> None:
    indice = _indice({"A": [(_M1, 100.0), (_M2, 110.0)], "C": [(_M1, 0.0), (_M2, 110.0)]})
    r = variacion_desde(indice, _M1, _M2, en_indefinido="marcar")
    assert set(r.df.index.get_level_values("indice")) == {"A"}


def test_desde_en_indefinido_marcar_diagnostico_marca_indefinido_con_motivo() -> None:
    indice = _indice({"A": [(_M1, 100.0), (_M2, 110.0)], "C": [(_M1, 0.0), (_M2, 110.0)]})
    r = variacion_desde(indice, _M1, _M2, en_indefinido="marcar")
    fila = r.diagnostico[r.diagnostico["indice"] == "C"]
    assert len(fila) == 1
    assert fila["estado_calculo"].iloc[0] == "indefinido"
    assert "base=0" in fila["motivo_error"].iloc[0]


def test_desde_en_indefinido_marcar_overflow_tambien_se_excluye() -> None:
    indice = _indice({"A": [(_M1, 100.0), (_M2, 110.0)], "C": [(_M1, 1e-308), (_M2, 1e308)]})
    r = variacion_desde(indice, _M1, _M2, en_indefinido="marcar")
    assert set(r.df.index.get_level_values("indice")) == {"A"}


def test_desde_en_indefinido_default_error_revienta_aunque_haya_indice_valido() -> None:
    indice = _indice({"A": [(_M1, 100.0), (_M2, 110.0)], "C": [(_M1, 0.0), (_M2, 110.0)]})
    with pytest.raises(InvarianteViolado):
        variacion_desde(indice, _M1, _M2)


def test_periodica_en_indefinido_valor_invalido_falla() -> None:
    # negociado 2026-09-03: sin esto, cualquier typo ("marcar_", "Marcar", "si")
    # caía en la rama implícita "no es 'error'" y excluía filas en silencio
    # (repro real: 'B' desaparecía sin excepción, resultado "válido" con solo 'A').
    indice = _indice({"A": [(_M1, 100.0), (_M2, 103.0)], "B": [(_M1, 0.0), (_M2, 103.0)]})
    with pytest.raises(InvarianteViolado):
        variacion_periodica(indice, "mensual", en_indefinido="cualquier-cosa")  # type: ignore[arg-type]


def test_periodica_en_indefinido_valor_invalido_falla_aunque_no_haya_fila_indefinida() -> None:
    # la validación es incondicional -- no solo dentro del `if invalido.any()`,
    # si no un typo pasaría desapercibido en cualquier corrida sin filas malas.
    with pytest.raises(InvarianteViolado):
        variacion_periodica(_indice_mensual(), "mensual", en_indefinido="cualquier-cosa")  # type: ignore[arg-type]


def test_acumulada_en_indefinido_valor_invalido_falla() -> None:
    indice = _indice(
        {"INPP": [(PeriodoMensual(2018, 12), 100.0), (PeriodoMensual(2019, 12), 110.0)]}
    )
    with pytest.raises(InvarianteViolado):
        variacion_acumulada_anual(indice, en_indefinido="cualquier-cosa")  # type: ignore[arg-type]


def test_desde_en_indefinido_valor_invalido_falla() -> None:
    with pytest.raises(InvarianteViolado):
        variacion_desde(_indice_dos(), _M1, _M2, en_indefinido="cualquier-cosa")  # type: ignore[arg-type]
