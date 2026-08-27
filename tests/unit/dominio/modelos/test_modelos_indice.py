from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.base import Vista
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import ManifestCalculo


def _manifiesto(
    version: int = 2019,
    agregacion: str = "INPP",
    rubro: str = "produccion_total",
    sin_petroleo: bool = False,
) -> ManifestCalculo:
    return ManifestCalculo(
        version=version,  # type: ignore[arg-type]
        agregacion=agregacion,
        rubro=rubro,
        sin_petroleo=sin_petroleo,
        calculador="LaspeyresDirecto",
        ruta_canasta=Path("/tmp/c.csv"),
        ruta_series=Path("/tmp/s.csv"),
        fecha=datetime(2024, 1, 1),
    )


def _df_indice(
    estados: Sequence[str | float] | None = None,
    version: int = 2019,
    agregacion: str = "INPP",
    rubro: str = "produccion_total",
    año: int = 2019,
    mes_inicio: int = 7,
) -> pd.DataFrame:
    estados = estados or ["ok", "ok"]
    n = len(estados)
    periodos = [PeriodoMensual(año, mes_inicio + i) for i in range(n)]
    idx = pd.MultiIndex.from_tuples([(p, "INPP") for p in periodos], names=["periodo", "indice"])
    return pd.DataFrame(
        {
            "version": [version] * n,
            "agregacion": [agregacion] * n,
            "rubro": [rubro] * n,
            "indice_replicado": [100.0 + i for i in range(n)],
            "estado_calculo": estados,
            "motivo_error": [None] * n,
        },
        index=idx,
    )


def _reporte_vacio() -> pd.DataFrame:
    return pd.DataFrame({"version": [], "estado_calculo": []})


def _diagnostico_vacio() -> pd.DataFrame:
    return pd.DataFrame({"version": []})


def test_construccion_valida() -> None:
    r = ResultadoIndice(_df_indice(), [_manifiesto()], _reporte_vacio(), _diagnostico_vacio())
    assert r.df.shape == (2, 1)
    assert list(r.df.columns) == ["indice_replicado"]


def test_manifiesto_vacio_falla() -> None:
    with pytest.raises(InvarianteViolado):
        ResultadoIndice(_df_indice(), [], _reporte_vacio(), _diagnostico_vacio())


@pytest.mark.parametrize(
    "falta", ["version", "agregacion", "rubro", "indice_replicado", "estado_calculo"]
)
def test_df_falta_columna_minima_falla(falta: str) -> None:
    df = _df_indice().drop(columns=[falta])
    with pytest.raises(InvarianteViolado):
        ResultadoIndice(df, [_manifiesto()], _reporte_vacio(), _diagnostico_vacio())


def test_estado_calculo_invalido_falla() -> None:
    df = _df_indice(estados=["ok", "indefinido"])
    with pytest.raises(InvarianteViolado):
        ResultadoIndice(df, [_manifiesto()], _reporte_vacio(), _diagnostico_vacio())


def test_manifiesto_sin_filas_por_version_falla() -> None:
    df = _df_indice(version=2019)
    huerfano = _manifiesto(version=2025)
    with pytest.raises(InvarianteViolado):
        ResultadoIndice(df, [huerfano], _reporte_vacio(), _diagnostico_vacio())


def test_manifiesto_sin_filas_por_agregacion_falla() -> None:
    df = _df_indice(agregacion="INPP")
    huerfano = _manifiesto(agregacion="SECTOR")
    with pytest.raises(InvarianteViolado):
        ResultadoIndice(df, [huerfano], _reporte_vacio(), _diagnostico_vacio())


def test_manifiesto_sin_filas_por_rubro_falla() -> None:
    df = _df_indice(rubro="produccion_total")
    huerfano = _manifiesto(rubro="bienes_finales")
    with pytest.raises(InvarianteViolado):
        ResultadoIndice(df, [huerfano], _reporte_vacio(), _diagnostico_vacio())


def test_manifiesto_sin_petroleo_no_se_valida_contra_columna() -> None:
    # sin_petroleo no vive en df_resultado — un manifiesto con sin_petroleo=True
    # contra un df que no lo distingue de ningún modo no debe fallar por eso.
    df = _df_indice()
    m = _manifiesto(sin_petroleo=True)
    r = ResultadoIndice(df, [m], _reporte_vacio(), _diagnostico_vacio())
    assert r.manifiesto[0].sin_petroleo is True


def test_fila_huerfana_sin_manifiesto_falla() -> None:
    # df trae filas de 2019 Y 2025, pero el manifiesto solo declara 2019 —
    # las filas de 2025 quedarían huérfanas, sin manifiesto que las respalde.
    df19 = _df_indice(estados=["ok"], version=2019, año=2019, mes_inicio=7)
    df25 = _df_indice(estados=["ok"], version=2025, año=2025, mes_inicio=7)
    df = pd.concat([df19, df25])
    with pytest.raises(InvarianteViolado):
        ResultadoIndice(df, [_manifiesto(version=2019)], _reporte_vacio(), _diagnostico_vacio())


def test_manifiestos_duplicados_por_version_agregacion_rubro_falla() -> None:
    # 2 manifiestos con la misma (version, agregacion, rubro) pero distinto
    # sin_petroleo apuntarían a las mismas filas sin que nada los distinga.
    df = _df_indice()
    m1 = _manifiesto(sin_petroleo=False)
    m2 = _manifiesto(sin_petroleo=True)
    with pytest.raises(InvarianteViolado):
        ResultadoIndice(df, [m1, m2], _reporte_vacio(), _diagnostico_vacio())


def test_estados_invalidos_heterogeneos_no_lanza_typeerror() -> None:
    # estado_calculo con un string inválido y NaN mezclados no debe escapar
    # como TypeError (str vs float no son comparables con sorted() directo).
    df = _df_indice(estados=["malo", float("nan")])
    with pytest.raises(InvarianteViolado):
        ResultadoIndice(df, [_manifiesto()], _reporte_vacio(), _diagnostico_vacio())


def test_resultado_retorna_vista() -> None:
    r = ResultadoIndice(_df_indice(), [_manifiesto()], _reporte_vacio(), _diagnostico_vacio())
    vista = r.resultado
    assert isinstance(vista, Vista)
    assert "version" in vista.largo.columns
    assert "estado_calculo" in vista.largo.columns


def test_indice_incidencia_oculto_en_resultado_pero_presente_en_completo() -> None:
    df = _df_indice()
    df["indice_incidencia"] = [95.0, 105.0]
    r = ResultadoIndice(df, [_manifiesto()], _reporte_vacio(), _diagnostico_vacio())
    assert "indice_incidencia" not in r.resultado.largo.columns
    assert "indice_incidencia" in r._completo.columns
    assert list(r._completo["indice_incidencia"]) == pytest.approx([95.0, 105.0])


def test_indice_incidencia_ausente_no_falla() -> None:
    # no está en _COLUMNAS_MINIMAS — construir sin ella debe funcionar.
    df = _df_indice()
    r = ResultadoIndice(df, [_manifiesto()], _reporte_vacio(), _diagnostico_vacio())
    assert "indice_incidencia" not in r.resultado.largo.columns


def test_reporte_y_diagnostico_propagados() -> None:
    rep = pd.DataFrame({"x": [1]})
    diag = pd.DataFrame({"y": [2]})
    r = ResultadoIndice(_df_indice(), [_manifiesto()], rep, diag)
    assert r.reporte is rep
    assert r.diagnostico is diag


def test_resumen_una_fila_por_manifiesto() -> None:
    m1 = _manifiesto(version=2019)
    m2 = _manifiesto(version=2025)
    df1 = _df_indice(version=2019, año=2019, mes_inicio=7)
    df2 = _df_indice(version=2025, año=2025, mes_inicio=7)
    df = pd.concat([df1, df2])
    r = ResultadoIndice(df, [m1, m2], _reporte_vacio(), _diagnostico_vacio())
    res = r.resumen
    assert list(res.index) == [
        "2019:INPP:produccion_total:con_petroleo",
        "2025:INPP:produccion_total:con_petroleo",
    ]
    assert list(res.columns) == [
        "estado_calculo",
        "periodo_inicio",
        "periodo_fin",
        "fecha",
    ]
    assert list(res["fecha"]) == [datetime(2024, 1, 1), datetime(2024, 1, 1)]
    clave_2019 = "2019:INPP:produccion_total:con_petroleo"
    clave_2025 = "2025:INPP:produccion_total:con_petroleo"
    assert res.loc[clave_2019, "periodo_inicio"] == PeriodoMensual(2019, 7)
    assert res.loc[clave_2019, "periodo_fin"] == PeriodoMensual(2019, 8)
    assert res.loc[clave_2025, "periodo_inicio"] == PeriodoMensual(2025, 7)
    assert res.loc[clave_2025, "periodo_fin"] == PeriodoMensual(2025, 8)


def test_resumen_clave_marca_sin_petroleo() -> None:
    r = ResultadoIndice(
        _df_indice(), [_manifiesto(sin_petroleo=True)], _reporte_vacio(), _diagnostico_vacio()
    )
    assert list(r.resumen.index) == ["2019:INPP:produccion_total:sin_petroleo"]


def test_resumen_peor_estado_segun_severidad() -> None:
    r = ResultadoIndice(
        _df_indice(estados=["ok", "parcial"]),
        [_manifiesto()],
        _reporte_vacio(),
        _diagnostico_vacio(),
    )
    clave = "2019:INPP:produccion_total:con_petroleo"
    assert r.resumen.loc[clave, "estado_calculo"] == "parcial"


def test_resumen_estado_fallida_mas_severo_que_sin_datos() -> None:
    r = ResultadoIndice(
        _df_indice(estados=["sin_datos", "fallida"]),
        [_manifiesto()],
        _reporte_vacio(),
        _diagnostico_vacio(),
    )
    clave = "2019:INPP:produccion_total:con_petroleo"
    assert r.resumen.loc[clave, "estado_calculo"] == "fallida"


@pytest.mark.parametrize(
    ("estados", "esperado"),
    [
        (["ok", "rellenado"], "rellenado"),
        (["rellenado", "parcial"], "parcial"),
        (["rellenado", "sin_datos"], "sin_datos"),
        (["parcial", "sin_datos"], "sin_datos"),
    ],
)
def test_resumen_severidad_cadena_completa_incluye_rellenado(
    estados: list[str], esperado: str
) -> None:
    r = ResultadoIndice(
        _df_indice(estados=estados), [_manifiesto()], _reporte_vacio(), _diagnostico_vacio()
    )
    clave = "2019:INPP:produccion_total:con_petroleo"
    assert r.resumen.loc[clave, "estado_calculo"] == esperado
