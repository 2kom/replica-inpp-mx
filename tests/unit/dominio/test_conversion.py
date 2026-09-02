from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from replica_inpp.dominio.conversion import empalmar, rebasar
from replica_inpp.dominio.errores import ErrorCalculo, InvarianteViolado
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import ManifestCalculo, VersionCanasta

_r1 = PeriodoMensual(2012, 6)
_r2 = PeriodoMensual(2019, 7)
_r3 = PeriodoMensual(2019, 8)


def _manifiesto(
    version: int = 2012, agregacion: str = "INPP", rubro: str = "produccion_total"
) -> ManifestCalculo:
    return ManifestCalculo(
        version=version,  # type: ignore[arg-type]
        agregacion=agregacion,
        rubro=rubro,
        incluir_petroleo=True,
        calculador="LaspeyresDirecto",
        ruta_canasta=Path("/tmp/c.csv"),
        ruta_series=Path("/tmp/s.csv"),
        fecha=datetime(2024, 1, 1),
    )


def _resultado(
    rows: list[tuple[Any, str, float | None, str, str | None]],
    version: int = 2012,
    agregacion: str = "INPP",
    rubro: str = "produccion_total",
    con_indice_incidencia: bool = False,
    periodo_referencia: PeriodoMensual | None = None,
    nombres: pd.Series | None = None,
    nombres_por_version: dict[VersionCanasta, pd.Series] | None = None,
) -> ResultadoIndice:
    """rows = (periodo, indice, valor, estado, motivo). Un solo manifiesto
    (version/agregacion/rubro) puede cubrir varios valores de `indice` -- igual
    que un `calcular_indice(..., agregacion="SECTOR")` real, que agrega varios
    sectores bajo una sola corrida."""
    filas = []
    for periodo, indice, valor, estado, motivo in rows:
        fila = {
            "periodo": periodo,
            "indice": indice,
            "version": version,
            "agregacion": agregacion,
            "rubro": rubro,
            "indice_replicado": valor,
            "estado_calculo": estado,
            "motivo_error": motivo,
        }
        if con_indice_incidencia:
            fila["indice_incidencia"] = valor
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.index = pd.MultiIndex.from_arrays(
        [df.pop("periodo"), df.pop("indice")], names=["periodo", "indice"]
    )
    reporte = pd.DataFrame(
        {"version": version, "estado_calculo": [estado for _, _, _, estado, _ in rows]},
        index=df.index,
    )
    diag = pd.DataFrame({"version": []})
    manifiesto = [_manifiesto(version=version, agregacion=agregacion, rubro=rubro)]
    return ResultadoIndice(
        df,
        manifiesto,
        reporte,
        diag,
        nombres=nombres,
        periodo_referencia=periodo_referencia,
        nombres_por_version=nombres_por_version,
    )


# --------------------------------------------------------------------------- básico


def test_rebasar_periodo_referencia_queda_en_valor_base() -> None:
    r = _resultado(
        [
            (_r1, "INPP", 100.0, "ok", None),
            (_r2, "INPP", 128.863769, "ok", None),
            (_r3, "INPP", 129.5, "ok", None),
        ]
    )
    rb = rebasar(r, _r2)
    assert rb.df.at[(_r2, "INPP"), "indice_replicado"] == pytest.approx(100.0)


def test_rebasar_proporcional() -> None:
    r = _resultado(
        [
            (_r1, "INPP", 100.0, "ok", None),
            (_r2, "INPP", 128.863769, "ok", None),
            (_r3, "INPP", 129.5, "ok", None),
        ]
    )
    rb = rebasar(r, _r2)
    assert rb.df.at[(_r1, "INPP"), "indice_replicado"] == pytest.approx(100.0 * 100.0 / 128.863769)
    assert rb.df.at[(_r3, "INPP"), "indice_replicado"] == pytest.approx(129.5 * 100.0 / 128.863769)


def test_rebasar_valor_base_distinto_de_100() -> None:
    r = _resultado([(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 128.0, "ok", None)])
    rb = rebasar(r, _r2, valor_base=200.0)
    assert rb.df.at[(_r2, "INPP"), "indice_replicado"] == pytest.approx(200.0)
    assert rb.df.at[(_r1, "INPP"), "indice_replicado"] == pytest.approx(100.0 * 200.0 / 128.0)


def test_rebasar_propaga_manifiesto() -> None:
    r = _resultado([(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 128.0, "ok", None)])
    rb = rebasar(r, _r2)
    assert rb.manifiesto == r.manifiesto


def test_rebasar_no_toca_indice_incidencia() -> None:
    r = _resultado(
        [(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 128.0, "ok", None)],
        con_indice_incidencia=True,
    )
    rb = rebasar(r, _r2)
    # indice_replicado sí se reescala, indice_incidencia queda intacta (columna
    # interna previa a cualquier rebase -- ver comentario en laspeyres_directo.py).
    assert rb._df_resultado.at[(_r1, "INPP"), "indice_incidencia"] == pytest.approx(100.0)
    assert rb._df_resultado.at[(_r2, "INPP"), "indice_incidencia"] == pytest.approx(128.0)


# --------------------------------------------------------------------------- varios `indice`


def test_rebasar_varios_indice_cada_uno_con_su_propio_factor() -> None:
    # Un solo manifiesto (SECTOR/produccion_total) con 2 sectores -- cada uno debe
    # reescalarse con SU PROPIO valor en periodo_referencia, no uno compartido.
    r = _resultado(
        [
            (_r1, "11", 100.0, "ok", None),
            (_r2, "11", 150.0, "ok", None),  # factor = 100/150
            (_r1, "21", 100.0, "ok", None),
            (_r2, "21", 200.0, "ok", None),  # factor = 100/200 (distinto)
        ],
        agregacion="SECTOR",
    )
    rb = rebasar(r, _r2)
    assert rb.df.at[(_r1, "11"), "indice_replicado"] == pytest.approx(100.0 * 100.0 / 150.0)
    assert rb.df.at[(_r1, "21"), "indice_replicado"] == pytest.approx(100.0 * 100.0 / 200.0)


def test_rebasar_indice_sin_referencia_emite_warning_y_no_rebasa() -> None:
    # "11" tiene dato en periodo_referencia, "21" no -- "21" queda sin rebasar.
    r = _resultado(
        [
            (_r1, "11", 100.0, "ok", None),
            (_r2, "11", 150.0, "ok", None),
            (_r1, "21", 50.0, "ok", None),
            (_r3, "21", 55.0, "ok", None),
        ],
        agregacion="SECTOR",
    )
    with pytest.warns(UserWarning, match="21"):
        rb = rebasar(r, _r2)
    assert rb.df.at[(_r2, "11"), "indice_replicado"] == pytest.approx(100.0)
    assert rb.df.at[(_r1, "21"), "indice_replicado"] == pytest.approx(50.0)
    assert rb.df.at[(_r3, "21"), "indice_replicado"] == pytest.approx(55.0)


# --------------------------------------------------------------------------- fallos


def test_rebasar_periodo_inexistente_falla() -> None:
    r = _resultado([(_r1, "INPP", 100.0, "ok", None), (_r3, "INPP", 129.5, "ok", None)])
    with pytest.raises(InvarianteViolado, match="ningún 'indice' tiene dato"):
        rebasar(r, _r2)


def test_rebasar_sin_datos_en_referencia_falla() -> None:
    r = _resultado(
        [
            (_r1, "INPP", 100.0, "ok", None),
            (_r2, "INPP", None, "sin_datos", "faltantes"),
            (_r3, "INPP", 129.5, "ok", None),
        ]
    )
    with pytest.raises(InvarianteViolado):
        rebasar(r, _r2)


def test_rebasar_referencia_fallida_falla() -> None:
    r = _resultado(
        [
            (_r1, "INPP", 100.0, "ok", None),
            (_r2, "INPP", None, "fallida", "error interno"),
            (_r3, "INPP", 129.5, "ok", None),
        ]
    )
    with pytest.raises(InvarianteViolado):
        rebasar(r, _r2)


def test_rebasar_referencia_parcial_o_rellenado_acepta() -> None:
    # "parcial"/"rellenado" traen valor real -- se aceptan como base, igual que "ok".
    r = _resultado(
        [
            (_r1, "INPP", 100.0, "ok", None),
            (_r2, "INPP", 128.0, "rellenado", None),
            (_r3, "INPP", 129.5, "ok", None),
        ]
    )
    rb = rebasar(r, _r2)
    assert rb.df.at[(_r2, "INPP"), "indice_replicado"] == pytest.approx(100.0)


def test_rebasar_nan_con_estado_ok_inconsistente_falla() -> None:
    # estado_calculo=ok pero indice_replicado=NaN -- inconsistente (no debería
    # ocurrir con datos reales, pero rebasar debe detectarlo si pasa).
    r = _resultado([(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", None, "ok", None)])
    with pytest.raises(InvarianteViolado, match="NaN"):
        rebasar(r, _r2)


def test_rebasar_cero_en_referencia_falla() -> None:
    r = _resultado(
        [
            (_r1, "INPP", 100.0, "ok", None),
            (_r2, "INPP", 0.0, "ok", None),
            (_r3, "INPP", 129.5, "ok", None),
        ]
    )
    with pytest.raises(InvarianteViolado, match="0"):
        rebasar(r, _r2)


@pytest.mark.parametrize(
    "valor_base", [float("nan"), float("inf"), float("-inf"), 0.0, -100.0], ids=str
)
def test_rebasar_valor_base_invalido_falla(valor_base: float) -> None:
    r = _resultado([(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 128.0, "ok", None)])
    with pytest.raises(InvarianteViolado, match="valor_base"):
        rebasar(r, _r2, valor_base=valor_base)


def test_rebasar_valor_base_finito_que_causa_overflow_lanza_errorcalculo() -> None:
    # valor_base=1.7e308 pasa la validación de "finito y positivo" -- el
    # desbordamiento ocurre recién al multiplicar contra otro periodo (200.0),
    # no contra el propio periodo_referencia.
    r = _resultado([(_r1, "INPP", 200.0, "ok", None), (_r2, "INPP", 100.0, "ok", None)])
    with pytest.raises(ErrorCalculo, match="overflow"):
        rebasar(r, _r2, valor_base=1.7e308)


def test_rebasar_valor_base_grande_pero_representable_no_lanza() -> None:
    # protege contra una guardia demasiado agresiva -- un valor_base gigante
    # pero cuyo resultado sigue siendo representable en float64 no debe fallar.
    r = _resultado([(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 100.0, "ok", None)])
    rb = rebasar(r, _r2, valor_base=1e300)
    assert rb.df.at[(_r1, "INPP"), "indice_replicado"] == pytest.approx(1e300)
    assert rb.df.at[(_r2, "INPP"), "indice_replicado"] == pytest.approx(1e300)


def test_rebasar_filas_sin_datos_y_fallida_fuera_de_referencia_preservadas() -> None:
    # Centinelas FINITOS (999.0/888.0, no NaN real) para distinguir "la máscara
    # excluyó la fila" de "ya era NaN y seguiría siéndolo con cualquier factor".
    r = _resultado(
        [
            (_r1, "INPP", 999.0, "sin_datos", "faltantes"),
            (_r2, "INPP", 128.0, "ok", None),
            (_r3, "INPP", 888.0, "fallida", "error interno"),
        ]
    )
    rb = rebasar(r, _r2)
    assert rb.df.at[(_r1, "INPP"), "indice_replicado"] == pytest.approx(999.0)
    assert rb.df.at[(_r3, "INPP"), "indice_replicado"] == pytest.approx(888.0)
    assert rb.df.at[(_r2, "INPP"), "indice_replicado"] == pytest.approx(100.0)


# --------------------------------------------------------------------------- periodo_referencia


def test_resultado_sin_rebasar_tiene_periodo_referencia_none() -> None:
    r = _resultado([(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 128.0, "ok", None)])
    assert r.periodo_referencia is None


def test_rebasar_setea_periodo_referencia() -> None:
    r = _resultado([(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 128.0, "ok", None)])
    rb = rebasar(r, _r2)
    assert rb.periodo_referencia == _r2


# =============================================================================
# empalmar
# =============================================================================


def test_empalmar_dos_tramos_concatena_y_frontera_la_manda_el_tramo_anterior() -> None:
    # tramo_2012 ya rebasado a Jul2019 (periodo_referencia=_r2) -- así su valor
    # ahí (100.0) es numéricamente coherente con el propio 100.0 de tramo_2019
    # en su base natural, mismo caso real que motivó todo esto.
    tramo_2012 = _resultado(
        [(_r1, "INPP", 77.6, "ok", None), (_r2, "INPP", 100.0, "ok", None)],
        version=2012,
        periodo_referencia=_r2,
    )
    # tramo_2019 vale 100.0000001 en la frontera -- dentro de tolerancia (1e-6)
    # de los 100.0 de tramo_2012, pero NO idéntico, para poder distinguir en el
    # assert que el resultado final es el del tramo ANTERIOR, no una casualidad.
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0000001, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    r = empalmar([tramo_2012, tramo_2019])
    # en la frontera, el tramo ANTERIOR manda -- el 100.0 exacto de 2012
    assert r.df.at[(_r2, "INPP"), "indice_replicado"] == pytest.approx(100.0, abs=1e-9)
    assert r.df.at[(_r1, "INPP"), "indice_replicado"] == pytest.approx(77.6)
    assert r.df.at[(_r3, "INPP"), "indice_replicado"] == pytest.approx(101.0)


def test_empalmar_orden_de_entrada_no_importa() -> None:
    # mismo caso de arriba, pero pasado en orden inverso -- `empalmar` ordena solo.
    tramo_2012 = _resultado(
        [(_r1, "INPP", 77.6, "ok", None), (_r2, "INPP", 100.0, "ok", None)],
        version=2012,
        periodo_referencia=_r2,
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0000001, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    r = empalmar([tramo_2019, tramo_2012])
    assert r.df.at[(_r2, "INPP"), "indice_replicado"] == pytest.approx(100.0, abs=1e-9)


def test_empalmar_varios_indice_cada_uno_respeta_su_propia_frontera() -> None:
    tramo_2012 = _resultado(
        [
            (_r1, "11", 100.0, "ok", None),
            (_r2, "11", 100.0, "ok", None),
            (_r1, "21", 100.0, "ok", None),
            (_r2, "21", 100.0, "ok", None),
        ],
        version=2012,
        agregacion="SECTOR",
    )
    tramo_2019 = _resultado(
        [
            (_r2, "11", 100.0000001, "ok", None),  # dentro de tolerancia, no idéntico
            (_r3, "11", 105.0, "ok", None),
            (_r2, "21", 99.9999999, "ok", None),  # idem, del otro lado
            (_r3, "21", 110.0, "ok", None),
        ],
        version=2019,
        agregacion="SECTOR",
    )
    r = empalmar([tramo_2012, tramo_2019])
    assert r.df.at[(_r2, "11"), "indice_replicado"] == pytest.approx(100.0, abs=1e-9)
    assert r.df.at[(_r2, "21"), "indice_replicado"] == pytest.approx(100.0, abs=1e-9)
    assert r.df.at[(_r3, "11"), "indice_replicado"] == pytest.approx(105.0)
    assert r.df.at[(_r3, "21"), "indice_replicado"] == pytest.approx(110.0)


def test_empalmar_concatena_manifiesto_sin_colapsar() -> None:
    tramo_2012 = _resultado(
        [(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 100.0, "ok", None)], version=2012
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    r = empalmar([tramo_2012, tramo_2019])
    assert len(r.manifiesto) == 2
    assert {m.version for m in r.manifiesto} == {2012, 2019}


def test_empalmar_periodo_referencia_resultado_es_del_ultimo_tramo_con_referencia() -> None:
    tramo_2012 = _resultado(
        [(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 100.0, "ok", None)],
        version=2012,
        periodo_referencia=_r2,
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    r = empalmar([tramo_2012, tramo_2019])
    assert r.periodo_referencia == _r2


def test_empalmar_sin_tramos_rebasados_periodo_referencia_es_none() -> None:
    tramo_2012 = _resultado(
        [(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 100.0, "ok", None)], version=2012
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    r = empalmar([tramo_2012, tramo_2019])
    assert r.periodo_referencia is None


# --------------------------------------------------------------------------- fallos de empalmar


def test_empalmar_menos_de_2_falla() -> None:
    tramo = _resultado([(_r1, "INPP", 100.0, "ok", None)], version=2012)
    with pytest.raises(InvarianteViolado, match="al menos 2"):
        empalmar([tramo])


def test_empalmar_combinaciones_distintas_falla() -> None:
    tramo_a = _resultado(
        [(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 100.0, "ok", None)],
        version=2012,
        rubro="produccion_total",
    )
    tramo_b = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)],
        version=2019,
        rubro="bienes_finales",
    )
    with pytest.raises(InvarianteViolado, match="agregacion, rubro, incluir_petroleo"):
        empalmar([tramo_a, tramo_b])


def test_empalmar_par_consecutivo_sin_periodo_compartido_falla() -> None:
    tramo_a = _resultado([(_r1, "INPP", 100.0, "ok", None)], version=2012)
    tramo_b = _resultado([(_r3, "INPP", 101.0, "ok", None)], version=2019)
    with pytest.raises(InvarianteViolado, match="no comparte ningún periodo"):
        empalmar([tramo_a, tramo_b])


def test_empalmar_par_consecutivo_comparte_mas_de_un_periodo_falla() -> None:
    tramo_a = _resultado(
        [
            (_r1, "INPP", 100.0, "ok", None),
            (_r2, "INPP", 100.0, "ok", None),
            (_r3, "INPP", 101.0, "ok", None),
        ],
        version=2012,
    )
    tramo_b = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    with pytest.raises(InvarianteViolado, match="topología PATH"):
        empalmar([tramo_a, tramo_b])


def test_empalmar_par_no_consecutivo_comparte_periodo_falla() -> None:
    # tramo1 trae de más el periodo final de tramo3 (p4) -- comparten un periodo
    # sin ser consecutivos (tramo2 queda en medio de los dos en el orden).
    p4 = PeriodoMensual(2025, 7)
    tramo1 = _resultado(
        [
            (_r1, "INPP", 100.0, "ok", None),
            (_r2, "INPP", 100.0, "ok", None),
            (p4, "INPP", 999.0, "ok", None),
        ],
        version=2012,
    )
    tramo2 = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    tramo3 = _resultado(
        [(_r3, "INPP", 101.0, "ok", None), (p4, "INPP", 130.0, "ok", None)], version=2025
    )
    with pytest.raises(InvarianteViolado, match="no-consecutivo"):
        empalmar([tramo1, tramo2, tramo3])


def test_empalmar_periodo_referencia_discontinuo_sin_forzar_falla() -> None:
    tramo_2012 = _resultado(
        [(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 100.0, "ok", None)],
        version=2012,
        periodo_referencia=_r1,  # rebasado a un periodo que NO es la frontera con 2019
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    with pytest.raises(InvarianteViolado, match="periodo_referencia"):
        empalmar([tramo_2012, tramo_2019])


def test_empalmar_periodo_referencia_discontinuo_con_forzar_advierte() -> None:
    tramo_2012 = _resultado(
        [(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 100.0, "ok", None)],
        version=2012,
        periodo_referencia=_r1,
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    with pytest.warns(UserWarning, match="periodo_referencia"):
        r = empalmar([tramo_2012, tramo_2019], forzar=True)
    assert isinstance(r, ResultadoIndice)


# --------------------------------------------------------------------------- escala numérica en la frontera


def test_empalmar_escala_dentro_de_tolerancia_acepta() -> None:
    # diferencia de 1e-7 en la frontera -- dentro de la tolerancia interna (1e-6).
    tramo_2012 = _resultado(
        [(_r1, "INPP", 77.6, "ok", None), (_r2, "INPP", 100.0, "ok", None)], version=2012
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0000001, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    r = empalmar([tramo_2012, tramo_2019])
    assert isinstance(r, ResultadoIndice)


def test_empalmar_escala_tolerancia_es_absoluta_no_relativa() -> None:
    # discrimina rel_tol=0.0 de un posible mutante rel_tol=1e-5: diferencia
    # absoluta 5e-4 (>1e-6) rechaza bajo tolerancia absoluta pura, pero
    # math.isclose(100, 100.0005, rel_tol=1e-5, abs_tol=1e-6) da True -- si
    # alguna vez se cuela una tolerancia relativa, este test lo detecta.
    tramo_2012 = _resultado(
        [(_r1, "INPP", 77.6, "ok", None), (_r2, "INPP", 100.0, "ok", None)], version=2012
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0005, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    with pytest.raises(InvarianteViolado, match="escala no es coherente"):
        empalmar([tramo_2012, tramo_2019])


def test_empalmar_escala_fuera_de_tolerancia_falla() -> None:
    # caso real reportado: rebasar 2012 a valor_base=200 y empalmarlo con 2019
    # (frontera=100) debía rechazarse -- antes se aceptaba en silencio.
    tramo_2012 = _resultado(
        [(_r1, "INPP", 155.2, "ok", None), (_r2, "INPP", 200.0, "ok", None)], version=2012
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    with pytest.raises(InvarianteViolado, match="escala no es coherente"):
        empalmar([tramo_2012, tramo_2019])


def test_empalmar_escala_fuera_de_tolerancia_con_forzar_advierte() -> None:
    tramo_2012 = _resultado(
        [(_r1, "INPP", 155.2, "ok", None), (_r2, "INPP", 200.0, "ok", None)], version=2012
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    with pytest.warns(UserWarning, match="escala no es coherente"):
        r = empalmar([tramo_2012, tramo_2019], forzar=True)
    assert isinstance(r, ResultadoIndice)


def test_empalmar_tres_tramos_escala_incompatible_en_uno_falla() -> None:
    # tramo1-tramo2 coherentes en su frontera; tramo2-tramo3 NO -- debe fallar
    # igual, sin importar en qué par de la cadena esté la incompatibilidad.
    p4 = PeriodoMensual(2025, 7)
    tramo1 = _resultado(
        [(_r1, "INPP", 77.6, "ok", None), (_r2, "INPP", 100.0, "ok", None)], version=2012
    )
    tramo2 = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    tramo3 = _resultado(
        [(_r3, "INPP", 500.0, "ok", None), (p4, "INPP", 130.0, "ok", None)], version=2025
    )
    with pytest.raises(InvarianteViolado, match="escala no es coherente"):
        empalmar([tramo1, tramo2, tramo3])


# --------------------------------------------------------------------------- nombres


def test_empalmar_combina_nombres_precedencia_del_tramo_mas_reciente() -> None:
    tramo_2012 = _resultado(
        [(_r1, "11", 100.0, "ok", None), (_r2, "11", 100.0, "ok", None)],
        version=2012,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "Agricultura (nombre viejo)"}),
    )
    tramo_2019 = _resultado(
        [(_r2, "11", 100.0, "ok", None), (_r3, "11", 101.0, "ok", None)],
        version=2019,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "Agricultura, cría y explotación de animales"}),
    )
    r = empalmar([tramo_2012, tramo_2019])
    ancho = r.resultado.ancho
    assert "nombre" in ancho.columns
    # el nombre vigente (tramo más reciente) es el único que se muestra para
    # "11" -- `nombre` es 1 valor por `indice`, no por (periodo, indice), así
    # que aplica igual a las filas que vinieron originalmente del tramo 2012.
    assert ancho.loc["11", "nombre"] == "Agricultura, cría y explotación de animales"


def test_empalmar_nombre_solo_en_tramo_viejo_se_preserva() -> None:
    # el tramo nuevo no trae nombre para "21" -- no debe perderse el del viejo.
    tramo_2012 = _resultado(
        [(_r1, "21", 100.0, "ok", None), (_r2, "21", 100.0, "ok", None)],
        version=2012,
        agregacion="SECTOR",
        nombres=pd.Series({"21": "Minería"}),
    )
    tramo_2019 = _resultado(
        [(_r2, "21", 100.0, "ok", None), (_r3, "21", 101.0, "ok", None)],
        version=2019,
        agregacion="SECTOR",
    )
    r = empalmar([tramo_2012, tramo_2019])
    assert r.resultado.ancho.loc["21", "nombre"] == "Minería"


def test_empalmar_sin_nombres_en_ningun_tramo_no_agrega_columna() -> None:
    tramo_2012 = _resultado(
        [(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 100.0, "ok", None)], version=2012
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    r = empalmar([tramo_2012, tramo_2019])
    assert "nombre" not in r.resultado.ancho.columns


# --------------------------------------------------------------------------- version_nombres


def test_empalmar_version_nombres_prioriza_tramo_elegido() -> None:
    # 3 tramos, cada uno con su propio nombre para "11" -- version_nombres=2012
    # debe ganar aunque 2012 sea el tramo más VIEJO de los tres.
    p4 = PeriodoMensual(2025, 7)
    tramo_2012 = _resultado(
        [(_r1, "11", 100.0, "ok", None), (_r2, "11", 100.0, "ok", None)],
        version=2012,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "Agricultura (2012)"}),
    )
    tramo_2019 = _resultado(
        [(_r2, "11", 100.0, "ok", None), (_r3, "11", 101.0, "ok", None)],
        version=2019,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "Agricultura (2019)"}),
    )
    tramo_2025 = _resultado(
        [(_r3, "11", 101.0, "ok", None), (p4, "11", 105.0, "ok", None)],
        version=2025,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "Agricultura (2025)"}),
    )
    r = empalmar([tramo_2012, tramo_2019, tramo_2025], version_nombres=2012)
    assert r.resultado.ancho.loc["11", "nombre"] == "Agricultura (2012)"


def test_empalmar_version_nombres_fallback_en_huecos() -> None:
    # el tramo elegido (2012) no nombra "21" -- sigue el fallback normal
    # (cascada cronológica) entre los demás: gana 2019, el más reciente que sí lo nombra.
    p4 = PeriodoMensual(2025, 7)
    tramo_2012 = _resultado(
        [(_r1, "21", 100.0, "ok", None), (_r2, "21", 100.0, "ok", None)],
        version=2012,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "Agricultura (2012)"}),  # no nombra "21"
    )
    tramo_2019 = _resultado(
        [(_r2, "21", 100.0, "ok", None), (_r3, "21", 101.0, "ok", None)],
        version=2019,
        agregacion="SECTOR",
        nombres=pd.Series({"21": "Minería (2019)"}),
    )
    tramo_2025 = _resultado(
        [(_r3, "21", 101.0, "ok", None), (p4, "21", 105.0, "ok", None)],
        version=2025,
        agregacion="SECTOR",
    )
    r = empalmar([tramo_2012, tramo_2019, tramo_2025], version_nombres=2012)
    assert r.resultado.ancho.loc["21", "nombre"] == "Minería (2019)"


def test_empalmar_version_nombres_inexistente_falla() -> None:
    tramo_2012 = _resultado(
        [(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 100.0, "ok", None)], version=2012
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)], version=2019
    )
    with pytest.raises(InvarianteViolado, match="version_nombres"):
        empalmar([tramo_2012, tramo_2019], version_nombres=2025)


def test_empalmar_version_nombres_none_mantiene_precedencia_del_mas_reciente() -> None:
    # default explícito -- mismo comportamiento que no pasar el argumento.
    tramo_2012 = _resultado(
        [(_r1, "11", 100.0, "ok", None), (_r2, "11", 100.0, "ok", None)],
        version=2012,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "viejo"}),
    )
    tramo_2019 = _resultado(
        [(_r2, "11", 100.0, "ok", None), (_r3, "11", 101.0, "ok", None)],
        version=2019,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "nuevo"}),
    )
    r = empalmar([tramo_2012, tramo_2019], version_nombres=None)
    assert r.resultado.ancho.loc["11", "nombre"] == "nuevo"


def test_empalmar_version_nombres_robusto_a_empalme_incremental() -> None:
    # hallazgo negociado 2026-09-01: empalmar 2012+2019 primero, y recién
    # después sumar 2025 con version_nombres=2012, debe dar el mismo resultado
    # que empalmar los 3 de una -- antes, el tramo ya empalmado (2012+2019)
    # aportaba su `nombre` ya mezclado (el de 2019, no el de 2012 puro).
    p4 = PeriodoMensual(2025, 7)
    tramo_2012 = _resultado(
        [(_r1, "11", 100.0, "ok", None), (_r2, "11", 100.0, "ok", None)],
        version=2012,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "nombre-2012"}),
    )
    tramo_2019 = _resultado(
        [(_r2, "11", 100.0, "ok", None), (_r3, "11", 101.0, "ok", None)],
        version=2019,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "nombre-2019"}),
    )
    tramo_2025 = _resultado(
        [(_r3, "11", 101.0, "ok", None), (p4, "11", 105.0, "ok", None)],
        version=2025,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "nombre-2025"}),
    )

    directo = empalmar([tramo_2012, tramo_2019, tramo_2025], version_nombres=2012)

    parcial = empalmar([tramo_2012, tramo_2019])  # sin version_nombres -- gana 2019 acá
    incremental = empalmar([parcial, tramo_2025], version_nombres=2012)

    esperado = "nombre-2012"
    assert directo.resultado.ancho.loc["11", "nombre"] == esperado
    assert incremental.resultado.ancho.loc["11", "nombre"] == esperado


def test_empalmar_nombres_por_version_se_fusiona_a_traves_de_empalmes() -> None:
    # el registro interno de nombres por versión sobrevive un empalme, para
    # que un empalme POSTERIOR pueda seguir resolviendo version_nombres.
    tramo_2012 = _resultado(
        [(_r1, "INPP", 100.0, "ok", None), (_r2, "INPP", 100.0, "ok", None)],
        version=2012,
        nombres=pd.Series({"INPP": "nombre-2012"}),
    )
    tramo_2019 = _resultado(
        [(_r2, "INPP", 100.0, "ok", None), (_r3, "INPP", 101.0, "ok", None)],
        version=2019,
        nombres=pd.Series({"INPP": "nombre-2019"}),
    )
    r = empalmar([tramo_2012, tramo_2019])
    assert set(r._nombres_por_version) == {2012, 2019}
    assert r._nombres_por_version[2012]["INPP"] == "nombre-2012"
    assert r._nombres_por_version[2019]["INPP"] == "nombre-2019"


def test_empalmar_cascada_default_no_depende_de_como_se_agruparon_los_empalmes() -> None:
    # hallazgo negociado 2026-09-01: sin version_nombres, la cascada default
    # usaba el `nombre` ya colapsado de cada tramo -- si el tramo interior se
    # había empalmado con version_nombres explícito, ese sesgo se filtraba al
    # resultado externo aunque el afuera NO pidiera version_nombres. tramo_2025
    # deliberadamente NO nombra "11" -- sin el sesgo, gana 2019 (más reciente
    # que sí lo nombra), no 2012.
    p4 = PeriodoMensual(2025, 7)
    tramo_2012 = _resultado(
        [(_r1, "11", 100.0, "ok", None), (_r2, "11", 100.0, "ok", None)],
        version=2012,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "nombre-2012"}),
    )
    tramo_2019 = _resultado(
        [(_r2, "11", 100.0, "ok", None), (_r3, "11", 101.0, "ok", None)],
        version=2019,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "nombre-2019"}),
    )
    tramo_2025 = _resultado(
        [(_r3, "11", 101.0, "ok", None), (p4, "11", 105.0, "ok", None)],
        version=2025,
        agregacion="SECTOR",
        # sin nombres -- no aporta nada para "11"
    )

    directo = empalmar([tramo_2012, tramo_2019, tramo_2025])

    interno = empalmar([tramo_2012, tramo_2019], version_nombres=2012)
    incremental = empalmar([interno, tramo_2025])

    esperado = "nombre-2019"
    assert directo.resultado.ancho.loc["11", "nombre"] == esperado
    assert incremental.resultado.ancho.loc["11", "nombre"] == esperado


def _compuesto_sin_registro(nombres: pd.Series) -> ResultadoIndice:
    """`ResultadoIndice` con 2 manifiestos (2012+2019), `nombres` plano, y
    `nombres_por_version=None` -- construcción pública legítima (ver
    `test_modelos_indice.py::test_nombres_por_version_con_varios_manifiestos_sin_explicito_queda_vacio`),
    su registro queda vacío por diseño."""
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "version": [2012],
                    "agregacion": ["SECTOR"],
                    "rubro": ["produccion_total"],
                    "indice_replicado": [100.0],
                    "estado_calculo": ["ok"],
                    "motivo_error": [None],
                },
                index=pd.MultiIndex.from_tuples([(_r1, "11")], names=["periodo", "indice"]),
            ),
            pd.DataFrame(
                {
                    "version": [2019],
                    "agregacion": ["SECTOR"],
                    "rubro": ["produccion_total"],
                    "indice_replicado": [100.0],
                    "estado_calculo": ["ok"],
                    "motivo_error": [None],
                },
                index=pd.MultiIndex.from_tuples([(_r2, "11")], names=["periodo", "indice"]),
            ),
        ]
    )
    reporte = pd.DataFrame(
        {"version": [2012, 2019], "estado_calculo": ["ok", "ok"]}, index=df.index
    )
    return ResultadoIndice(
        df,
        [
            _manifiesto(version=2012, agregacion="SECTOR"),
            _manifiesto(version=2019, agregacion="SECTOR"),
        ],
        reporte,
        pd.DataFrame({"version": []}),
        nombres=nombres,
    )


def test_empalmar_nombre_plano_de_tramo_sin_historial_sobrevive_como_huerfano() -> None:
    # hallazgo negociado 2026-09-01 (ronda 2): registro COMPLETAMENTE vacío --
    # sin información por versión, el nombre plano entero se trata como huérfano.
    compuesto = _compuesto_sin_registro(pd.Series({"11": "nombre-plano"}))
    assert compuesto._nombres_por_version == {}  # precondición

    posterior = _resultado(
        [(_r2, "11", 100.0, "ok", None), (PeriodoMensual(2025, 7), "11", 105.0, "ok", None)],
        version=2025,
        agregacion="SECTOR",
    )

    r = empalmar([compuesto, posterior])
    assert r.resultado.ancho.loc["11", "nombre"] == "nombre-plano"


def test_empalmar_indice_huerfano_y_trazable_en_el_mismo_tramo_gana_el_trazable() -> None:
    # hallazgo negociado 2026-09-01 (ronda 2): registro parcial (NO vacío) --
    # "11" es huérfano (no aparece en el registro del propio tramo), "21" es
    # trazable (sí aparece) -- ambos deben sobrevivir, y "21" debe conservar el
    # valor TRAZABLE si alguna vez difiere del plano (acá coinciden a propósito
    # para poder afirmar el valor sin ambigüedad).
    tramo = _resultado(
        [(_r2, "11", 100.0, "ok", None), (_r2, "21", 100.0, "ok", None)],
        version=2019,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "plano-sin-version", "21": "trazable-2019"}),
        nombres_por_version={2019: pd.Series({"21": "trazable-2019"})},
    )
    assert tramo._nombres_por_version != {}  # precondición: NO vacío

    posterior = _resultado(
        [
            (_r2, "11", 100.0, "ok", None),
            (_r2, "21", 100.0, "ok", None),
            (PeriodoMensual(2025, 7), "11", 105.0, "ok", None),
            (PeriodoMensual(2025, 7), "21", 105.0, "ok", None),
        ],
        version=2025,
        agregacion="SECTOR",
    )

    r = empalmar([tramo, posterior])
    ancho = r.resultado.ancho
    assert ancho.loc["11", "nombre"] == "plano-sin-version"
    assert ancho.loc["21", "nombre"] == "trazable-2019"


def test_empalmar_huerfano_y_trazable_colisionan_en_el_mismo_indice_gana_el_trazable() -> None:
    # hallazgo negociado 2026-09-01 (ronda 3): el test mixto de arriba usa
    # índices DISTINTOS para huérfano ("11") y trazable ("21") -- nunca prueba
    # una colisión real donde ambas capas nombran el MISMO índice. Un mutante
    # que aplique el registro por versión ANTES que los huérfanos pasaba los
    # tests existentes en silencio y devolvía el huérfano en vez del trazable
    # (verificado con un mutante real antes de esta negociación).
    compuesto = _compuesto_sin_registro(pd.Series({"11": "huerfano"}))
    assert compuesto._nombres_por_version == {}  # precondición: sin historial

    posterior = _resultado(
        [(_r2, "11", 100.0, "ok", None), (PeriodoMensual(2025, 7), "11", 105.0, "ok", None)],
        version=2025,
        agregacion="SECTOR",
        nombres=pd.Series({"11": "trazable-2025"}),
    )

    r = empalmar([compuesto, posterior])
    assert r.resultado.ancho.loc["11", "nombre"] == "trazable-2025"
