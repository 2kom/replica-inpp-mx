from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.base import Vista
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import ManifestDerivado


def _manifiesto(
    agregacion: str = "INPP",
    rubro: str = "produccion_total",
    clase: str = "periodica_mensual",
    descripcion: str = "",
) -> ManifestDerivado:
    return ManifestDerivado(
        versiones=[2019],
        agregacion=agregacion,
        rubro=rubro,
        clase=clase,
        descripcion=descripcion,
        fecha=datetime(2024, 1, 1),
    )


def _df_var(
    estados: list[str] | None = None,
    clase: str = "periodica_mensual",
    agregacion: str = "INPP",
    rubro: str = "produccion_total",
) -> pd.DataFrame:
    estados = estados or ["ok", "ok"]
    n = len(estados)
    periodos = [PeriodoMensual(2019, 6 + m) for m in range(1, n + 1)]
    idx = pd.MultiIndex.from_tuples([(p, "INPP") for p in periodos], names=["periodo", "indice"])
    return pd.DataFrame(
        {
            "agregacion": [agregacion] * n,
            "rubro": [rubro] * n,
            "clase_variacion": [clase] * n,
            "variacion_pp": [0.5 + i * 0.1 for i in range(n)],
            "estado_calculo": estados,
            "version_t": [2019] * n,
        },
        index=idx,
    )


def _rep_vacio() -> pd.DataFrame:
    return pd.DataFrame({"estado_calculo": []})


def _diag_vacio() -> pd.DataFrame:
    return pd.DataFrame({"id_corrida": []})


# ---------- Construcción ----------


def test_construccion_valida_periodica() -> None:
    r = ResultadoVariacion(_df_var(), _manifiesto(), _rep_vacio(), _diag_vacio())
    assert r.df.shape == (2, 1)


def test_construccion_valida_desde() -> None:
    df = _df_var(clase="desde")
    ip = pd.DataFrame({"periodo_desde_real": []})
    r = ResultadoVariacion(
        df, _manifiesto(clase="desde"), _rep_vacio(), _diag_vacio(), indices_parciales=ip
    )
    assert r.indices_parciales is ip


# ---------- Invariantes ----------


@pytest.mark.parametrize(
    "falta", ["agregacion", "rubro", "clase_variacion", "variacion_pp", "estado_calculo"]
)
def test_df_falta_columna_minima_falla(falta: str) -> None:
    df = _df_var().drop(columns=[falta])
    with pytest.raises(InvarianteViolado):
        ResultadoVariacion(df, _manifiesto(), _rep_vacio(), _diag_vacio())


def test_clase_variacion_heterogenea_falla() -> None:
    df = _df_var()
    df.loc[df.index[1], "clase_variacion"] = "periodica_anual"
    with pytest.raises(InvarianteViolado, match="homogénea"):
        ResultadoVariacion(df, _manifiesto(), _rep_vacio(), _diag_vacio())


def test_clase_variacion_fuera_catalogo_falla() -> None:
    df = _df_var(clase="inventada")
    with pytest.raises(InvarianteViolado):
        ResultadoVariacion(df, _manifiesto(clase="inventada"), _rep_vacio(), _diag_vacio())


def test_desde_sin_indices_parciales_falla() -> None:
    df = _df_var(clase="desde")
    with pytest.raises(InvarianteViolado):
        ResultadoVariacion(df, _manifiesto(clase="desde"), _rep_vacio(), _diag_vacio())


def test_periodica_con_indices_parciales_falla() -> None:
    df = _df_var(clase="periodica_mensual")
    ip = pd.DataFrame({"x": []})
    with pytest.raises(InvarianteViolado):
        ResultadoVariacion(
            df,
            _manifiesto(clase="periodica_mensual"),
            _rep_vacio(),
            _diag_vacio(),
            indices_parciales=ip,
        )


def test_manifiesto_clase_mismatch_falla() -> None:
    df = _df_var(clase="periodica_mensual")
    with pytest.raises(InvarianteViolado):
        ResultadoVariacion(df, _manifiesto(clase="periodica_anual"), _rep_vacio(), _diag_vacio())


def test_df_agregacion_heterogenea_falla() -> None:
    df = _df_var()
    df.loc[df.index[1], "agregacion"] = "SECTOR"
    with pytest.raises(InvarianteViolado, match="homogéneo"):
        ResultadoVariacion(df, _manifiesto(), _rep_vacio(), _diag_vacio())


def test_manifiesto_agregacion_mismatch_falla() -> None:
    df = _df_var(agregacion="INPP")
    with pytest.raises(InvarianteViolado):
        ResultadoVariacion(df, _manifiesto(agregacion="SECTOR"), _rep_vacio(), _diag_vacio())


def test_df_rubro_heterogeneo_falla() -> None:
    df = _df_var()
    df.loc[df.index[1], "rubro"] = "bienes_finales"
    with pytest.raises(InvarianteViolado, match="homogéneo"):
        ResultadoVariacion(df, _manifiesto(), _rep_vacio(), _diag_vacio())


def test_manifiesto_rubro_mismatch_falla() -> None:
    df = _df_var(rubro="produccion_total")
    with pytest.raises(InvarianteViolado):
        ResultadoVariacion(df, _manifiesto(rubro="bienes_finales"), _rep_vacio(), _diag_vacio())


@pytest.mark.parametrize("estado_invalido", ["sin_datos", "fallida"])
def test_estado_calculo_invalido_falla(estado_invalido: str) -> None:
    df = _df_var(estados=["ok", estado_invalido])
    with pytest.raises(InvarianteViolado):
        ResultadoVariacion(df, _manifiesto(), _rep_vacio(), _diag_vacio())


def test_estados_invalidos_heterogeneos_no_filtra_typeerror() -> None:
    # sorted() sobre tipos no comparables (str vs int) debe seguir dando
    # InvarianteViolado, no un TypeError crudo de Python -- regresión de
    # negociación (evaluador: estado_calculo=["malo", 1]).
    df = _df_var(estados=["malo", 1])  # type: ignore[list-item]
    with pytest.raises(InvarianteViolado):
        ResultadoVariacion(df, _manifiesto(), _rep_vacio(), _diag_vacio())


# ---------- Properties ----------


def test_df_minimal_una_columna() -> None:
    r = ResultadoVariacion(_df_var(), _manifiesto(), _rep_vacio(), _diag_vacio())
    assert list(r.df.columns) == ["variacion_pp"]


def test_resultado_retorna_vista_con_largo_extendido() -> None:
    r = ResultadoVariacion(_df_var(), _manifiesto(), _rep_vacio(), _diag_vacio())
    vista = r.resultado
    assert isinstance(vista, Vista)
    for col in (
        "agregacion",
        "rubro",
        "clase_variacion",
        "variacion_pp",
        "estado_calculo",
        "version_t",
    ):
        assert col in vista.largo.columns


def test_reporte_y_diagnostico_propagados() -> None:
    rep = pd.DataFrame({"x": [1]})
    diag = pd.DataFrame({"y": [2]})
    r = ResultadoVariacion(_df_var(), _manifiesto(), rep, diag)
    assert r.reporte is rep
    assert r.diagnostico is diag


def test_indices_parciales_propagado_con_clase_desde() -> None:
    ip = pd.DataFrame({"periodo_desde_real": [PeriodoMensual(2019, 7)]})
    r = ResultadoVariacion(
        _df_var(clase="desde"),
        _manifiesto(clase="desde"),
        _rep_vacio(),
        _diag_vacio(),
        indices_parciales=ip,
    )
    assert r.indices_parciales is ip


def test_resumen_una_fila_con_cols_esperadas() -> None:
    r = ResultadoVariacion(_df_var(), _manifiesto(), _rep_vacio(), _diag_vacio())
    res = r.resumen
    assert res.shape == (1, 8)
    assert list(res.columns) == [
        "agregacion",
        "rubro",
        "clase_variacion",
        "descripcion",
        "estado_calculo",
        "periodo_inicio",
        "periodo_fin",
        "fecha",
    ]


def test_resumen_valores_concretos() -> None:
    r = ResultadoVariacion(
        _df_var(), _manifiesto(descripcion="jul→ago 2019"), _rep_vacio(), _diag_vacio()
    )
    fila: pd.Series = r.resumen.loc[0]  # type: ignore[assignment]
    assert fila["agregacion"] == "INPP"
    assert fila["rubro"] == "produccion_total"
    assert fila["clase_variacion"] == "periodica_mensual"
    assert fila["descripcion"] == "jul→ago 2019"
    assert fila["estado_calculo"] == "ok"
    assert fila["periodo_inicio"] == PeriodoMensual(2019, 7)
    assert fila["periodo_fin"] == PeriodoMensual(2019, 8)
    assert fila["fecha"] == datetime(2024, 1, 1)


def test_manifiesto_propagado() -> None:
    m = _manifiesto(descripcion="x")
    r = ResultadoVariacion(_df_var(), m, _rep_vacio(), _diag_vacio())
    assert r.manifiesto is m


def test_resumen_mezcla_ok_parcial_devuelve_parcial() -> None:
    r = ResultadoVariacion(
        _df_var(estados=["ok", "parcial"]), _manifiesto(), _rep_vacio(), _diag_vacio()
    )
    assert r.resumen.loc[0, "estado_calculo"] == "parcial"


def test_repr_html_devuelve_string() -> None:
    r = ResultadoVariacion(_df_var(), _manifiesto(), _rep_vacio(), _diag_vacio())
    assert isinstance(r._repr_html_(), str)
