from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from replica_inpp.dominio.calculo.variaciones import variacion_periodica
from replica_inpp.dominio.consulta.variaciones import (
    inflacion_acumulada,
    inflacion_en,
    inflacion_maxima,
    inflacion_minima,
    inflacion_promedio,
)
from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import ManifestCalculo, ManifestDerivado

_M1 = PeriodoMensual(2024, 1)
_M2 = PeriodoMensual(2024, 2)
_M3 = PeriodoMensual(2024, 3)
_M9 = PeriodoMensual(2099, 9)

# -- helpers -------------------------------------------------------------------


def _rv(
    data: dict[str, list[tuple[PeriodoMensual, float]]],
    *,
    agregacion: str = "INPP",
    rubro: str = "produccion_total",
    clase: str = "periodica_mensual",
) -> ResultadoVariacion:
    rows = []
    for indice, pares in data.items():
        for periodo, valor in pares:
            rows.append(
                {
                    "periodo": periodo,
                    "indice": indice,
                    "agregacion": agregacion,
                    "rubro": rubro,
                    "clase_variacion": clase,
                    "variacion_pp": float(valor),
                    "estado_calculo": "ok",
                }
            )
    df = pd.DataFrame(rows).set_index(["periodo", "indice"])
    manifiesto = ManifestDerivado(
        versiones=[2019],
        agregacion=agregacion,
        rubro=rubro,
        clase=clase,
        descripcion="",
        fecha=datetime(2024, 1, 1),
    )
    return ResultadoVariacion(df, manifiesto, pd.DataFrame(), pd.DataFrame())


def _rv_multi() -> ResultadoVariacion:
    return _rv(
        {
            "INPP": [(_M1, 1.0), (_M2, 3.0), (_M3, 2.0)],
            "Alimentos": [(_M1, 5.0), (_M2, -1.0), (_M3, 4.0)],
        }
    )


# -- inflacion_en --------------------------------------------------------------


def test_en_devuelve_dataframe_con_categorias() -> None:
    df = inflacion_en(_rv_multi(), _M2)
    assert list(df.columns) == ["variacion_pp"]
    assert set(df.index) == {"INPP", "Alimentos"}
    assert df.loc["INPP", "variacion_pp"] == pytest.approx(3.0)


def test_en_periodo_inexistente_falla() -> None:
    with pytest.raises(InvarianteViolado):
        inflacion_en(_rv_multi(), _M9)


def test_en_no_muta_resultado() -> None:
    r = _rv_multi()
    df = inflacion_en(r, _M2)
    df.loc["INPP", "variacion_pp"] = 999.0
    assert r.df.loc[(_M2, "INPP"), "variacion_pp"] == pytest.approx(3.0)  # type: ignore[index]


# -- inflacion_acumulada -------------------------------------------------------


def test_acumulada_compone_el_rango() -> None:
    # 100*(1.01*1.03*1.02 - 1) = 6.1106000000000105, NO sum(1,3,2)=6.0 --
    # negociado en /negociar-hallazgos (hallazgo #1, 2026-09-01): componer,
    # no sumar, tasas periódicas.
    r = _rv({"INPP": [(_M1, 1.0), (_M2, 3.0), (_M3, 2.0)]})
    assert inflacion_acumulada(r, _M1, _M3, indice="INPP") == pytest.approx(6.1106000000000105)


def test_acumulada_hasta_none_usa_ultimo() -> None:
    # 100*(1.03*1.02 - 1) = 5.06
    r = _rv({"INPP": [(_M1, 1.0), (_M2, 3.0), (_M3, 2.0)]})
    assert inflacion_acumulada(r, _M2, indice="INPP") == pytest.approx(5.06)


def test_acumulada_requiere_periodica_mensual_falla() -> None:
    r = _rv({"INPP": [(_M1, 1.0), (_M2, 3.0)]}, clase="periodica_bimestral")
    with pytest.raises(InvarianteViolado, match="periodica_mensual"):
        inflacion_acumulada(r, _M1, _M2, indice="INPP")


def test_acumulada_overflow_falla() -> None:
    # Dos factores finitos cuyo producto desborda a inf (float64 tope ~1.8e308).
    r = _rv({"INPP": [(_M1, (1e160 - 1) * 100), (_M2, (1e160 - 1) * 100)]})
    with pytest.raises(InvarianteViolado):
        inflacion_acumulada(r, _M1, _M2, indice="INPP")


def test_acumulada_hueco_en_secuencia_falla() -> None:
    # Feb y May presentes en .df (Mar/Abr fueron sin_datos, ya excluidos por
    # variacion_periodica) -- negociado en /negociar-hallazgos (hallazgo A,
    # ronda 3, 2026-09-01): sin este chequeo, compondría Feb->May como si
    # fueran consecutivos.
    r = _rv({"INPP": [(_M2, 10.0), (PeriodoMensual(2024, 5), 10.0)]})
    with pytest.raises(InvarianteViolado, match="no consecutivos"):
        inflacion_acumulada(r, _M2, PeriodoMensual(2024, 5), indice="INPP")


def test_acumulada_tasa_menor_a_menos_cien_falla() -> None:
    r = _rv({"INPP": [(_M1, -150.0), (_M2, 5.0)]})
    with pytest.raises(InvarianteViolado, match="-100%"):
        inflacion_acumulada(r, _M1, _M2, indice="INPP")


def test_acumulada_tasa_exacta_menos_cien_permitida() -> None:
    r = _rv({"INPP": [(_M1, -100.0), (_M2, 5.0)]})
    assert inflacion_acumulada(r, _M1, _M2, indice="INPP") == pytest.approx(-100.0)


def test_acumulada_indice_inexistente_falla() -> None:
    with pytest.raises(InvarianteViolado):
        inflacion_acumulada(_rv_multi(), _M1, _M3, indice="Inexistente")


def test_acumulada_desde_posterior_a_hasta_falla() -> None:
    with pytest.raises(InvarianteViolado):
        inflacion_acumulada(_rv_multi(), _M3, _M1, indice="INPP")


def test_acumulada_rango_vacio_falla() -> None:
    # 'Alimentos' existe globalmente y _M2/_M3 existen globalmente,
    # pero 'Alimentos' no tiene filas dentro de [_M2, _M3].
    r = _rv(
        {
            "INPP": [(_M1, 1.0), (_M2, 3.0), (_M3, 2.0)],
            "Alimentos": [(_M1, 5.0)],
        }
    )
    with pytest.raises(InvarianteViolado):
        inflacion_acumulada(r, _M2, _M3, indice="Alimentos")


# -- inflacion_promedio --------------------------------------------------------


def test_promedio_simple_es_media_aritmetica() -> None:
    r = _rv({"INPP": [(_M1, 1.0), (_M2, 3.0), (_M3, 2.0)]})
    assert inflacion_promedio(r, indice="INPP", metodo="simple") == pytest.approx(2.0)


def test_promedio_tcac_formula_congelada() -> None:
    # Dos variaciones de 10 pp; factor = 1.1 * 1.1 = 1.21; ppy=12 (INPP siempre
    # mensual), n=2. tcac = (1.21 ** 6 - 1) * 100 = 213.8428376721
    r = _rv({"INPP": [(_M1, 10.0), (_M2, 10.0)]})
    assert inflacion_promedio(r, indice="INPP", metodo="tcac") == pytest.approx(213.8428376721)


def test_tcac_hueco_en_secuencia_falla() -> None:
    r = _rv({"INPP": [(_M2, 10.0), (PeriodoMensual(2024, 5), 10.0)]})
    with pytest.raises(InvarianteViolado, match="no consecutivos"):
        inflacion_promedio(r, indice="INPP", metodo="tcac")


def test_tcac_tasa_menor_a_menos_cien_falla() -> None:
    r = _rv({"INPP": [(_M1, -150.0), (_M2, 5.0)]})
    with pytest.raises(InvarianteViolado, match="-100%"):
        inflacion_promedio(r, indice="INPP", metodo="tcac")


def test_tcac_producto_desborda_falla() -> None:
    # Mismo caso que test_acumulada_overflow_falla, vía tcac.
    r = _rv({"INPP": [(_M1, (1e160 - 1) * 100), (_M2, (1e160 - 1) * 100)]})
    with pytest.raises(InvarianteViolado):
        inflacion_promedio(r, indice="INPP", metodo="tcac")


def test_tcac_potencia_desborda_con_producto_finito_falla() -> None:
    # factor=1e100 (finito, un solo periodo no desborda el producto), pero
    # factor**12 (la potencia de anualización) sí desborda -- negociado en
    # /negociar-hallazgos (hallazgo B, ronda 3, 2026-09-01): np.errstate NO
    # captura el overflow de `float ** int` nativo de Python, hace falta
    # `np.power`.
    r = _rv({"INPP": [(_M1, (1e100 - 1) * 100)]})
    with pytest.raises(InvarianteViolado):
        inflacion_promedio(r, indice="INPP", metodo="tcac")


def test_promedio_metodo_invalido_falla() -> None:
    r = _rv({"INPP": [(_M1, 1.0), (_M2, 3.0)]})
    with pytest.raises(InvarianteViolado):
        inflacion_promedio(r, indice="INPP", metodo="geometrico")  # type: ignore[arg-type]


def test_promedio_simple_admite_periodica_anual() -> None:
    # Negociado en /negociar-hallazgos (hallazgo #2, 2026-09-01): "simple" es
    # estadística descriptiva válida sobre cualquier periodica_*, no compone.
    r = _rv({"INPP": [(_M1, 10.0), (_M2, 10.0)]}, clase="periodica_anual")
    assert inflacion_promedio(r, indice="INPP", metodo="simple") == pytest.approx(10.0)


def test_promedio_tcac_rechaza_periodica_anual_falla() -> None:
    r = _rv({"INPP": [(_M1, 10.0), (_M2, 10.0)]}, clase="periodica_anual")
    with pytest.raises(InvarianteViolado, match="periodica_mensual"):
        inflacion_promedio(r, indice="INPP", metodo="tcac")


def test_promedio_simple_rechaza_acumulada_anual_falla() -> None:
    r = _rv({"INPP": [(_M1, 1.0), (_M2, 3.0)]}, clase="acumulada_anual")
    with pytest.raises(InvarianteViolado, match="periodica"):
        inflacion_promedio(r, indice="INPP", metodo="simple")


def test_promedio_simple_rechaza_desde_falla() -> None:
    df = pd.DataFrame(
        {
            "agregacion": ["INPP"],
            "rubro": ["produccion_total"],
            "clase_variacion": ["desde"],
            "variacion_pp": [5.0],
            "estado_calculo": ["ok"],
        },
        index=pd.MultiIndex.from_tuples([(_M1, "INPP")], names=["periodo", "indice"]),
    )
    manifiesto = ManifestDerivado(
        versiones=[2019],
        agregacion="INPP",
        rubro="produccion_total",
        clase="desde",
        descripcion="",
        fecha=datetime(2024, 1, 1),
    )
    r = ResultadoVariacion(
        df,
        manifiesto,
        pd.DataFrame(),
        pd.DataFrame(),
        indices_parciales=pd.DataFrame({"periodo_desde_real": []}),
    )
    with pytest.raises(InvarianteViolado, match="periodica"):
        inflacion_promedio(r, indice="INPP", metodo="simple")


# -- inflacion_maxima / inflacion_minima ---------------------------------------


def test_maxima_global() -> None:
    periodo, indice, valor = inflacion_maxima(_rv_multi())
    assert (periodo, indice, valor) == (_M1, "Alimentos", pytest.approx(5.0))


def test_minima_global() -> None:
    periodo, indice, valor = inflacion_minima(_rv_multi())
    assert (periodo, indice, valor) == (_M2, "Alimentos", pytest.approx(-1.0))


def test_maxima_con_indice() -> None:
    periodo, indice, valor = inflacion_maxima(_rv_multi(), indice="INPP")
    assert (periodo, indice, valor) == (_M2, "INPP", pytest.approx(3.0))


def test_maxima_con_rango() -> None:
    periodo, indice, valor = inflacion_maxima(_rv_multi(), desde=_M2, hasta=_M3)
    assert (periodo, indice, valor) == (_M3, "Alimentos", pytest.approx(4.0))


def test_maxima_indice_inexistente_falla() -> None:
    with pytest.raises(InvarianteViolado):
        inflacion_maxima(_rv_multi(), indice="Inexistente")


# -- extremos ausentes / secuencia desordenada (ronda 4, 2026-09-01) -----------


def _indice_extremo_ausente(mes_faltante_inpp: int, meses_totales: int = 5) -> ResultadoVariacion:
    """`ResultadoIndice` -> `variacion_periodica` con 2 índices: "INPP" le falta
    un mes (queda `sin_datos`, fuera del `.df`), "OTRO" tiene todos los meses
    -- para que ese mes exista GLOBALMENTE en `df` y `_comun._verificar_periodo`
    (que solo chequea existencia global, no por `indice`) no detecte nada raro.
    Ruta pública completa, tal como la reprodujo el evaluador.
    """
    periodos = [PeriodoMensual(2024, m) for m in range(1, meses_totales + 1)]
    valores_inpp = [100.0 + 5 * i for i in range(meses_totales)]
    valores_inpp[mes_faltante_inpp - 1] = float("nan")
    valores_otro = [50.0 + 3 * i for i in range(meses_totales)]
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
    return variacion_periodica(indice, "mensual")


def test_acumulada_extremo_desde_ausente_falla() -> None:
    # Feb sin dato para INPP (existe para OTRO) -- pedir explícito desde=Feb
    # debe fallar, no silenciarse a un rango más angosto.
    r = _indice_extremo_ausente(mes_faltante_inpp=2)
    with pytest.raises(InvarianteViolado, match="desde"):
        inflacion_acumulada(r, PeriodoMensual(2024, 2), PeriodoMensual(2024, 5), indice="INPP")


def test_tcac_extremo_desde_ausente_falla() -> None:
    r = _indice_extremo_ausente(mes_faltante_inpp=2)
    with pytest.raises(InvarianteViolado, match="desde"):
        inflacion_promedio(
            r, PeriodoMensual(2024, 2), PeriodoMensual(2024, 5), indice="INPP", metodo="tcac"
        )


def test_acumulada_extremo_hasta_ausente_falla() -> None:
    # May sin dato para INPP (existe para OTRO) -- pedir explícito hasta=May
    # debe fallar, no recortarse en silencio al último periodo disponible.
    r = _indice_extremo_ausente(mes_faltante_inpp=5)
    with pytest.raises(InvarianteViolado, match="hasta"):
        inflacion_acumulada(r, PeriodoMensual(2024, 2), PeriodoMensual(2024, 5), indice="INPP")


def test_tcac_extremo_hasta_ausente_falla() -> None:
    # Paridad con test_acumulada_extremo_hasta_ausente_falla -- negociado en
    # /negociar-hallazgos (ronda 4, 2026-09-01): faltaba el caso hasta ausente
    # para TCAC, solo existía para desde.
    r = _indice_extremo_ausente(mes_faltante_inpp=5)
    with pytest.raises(InvarianteViolado, match="hasta"):
        inflacion_promedio(
            r, PeriodoMensual(2024, 2), PeriodoMensual(2024, 5), indice="INPP", metodo="tcac"
        )


def _rv_desordenado() -> ResultadoVariacion:
    # Mar, Ene, Feb en orden FÍSICO (MultiIndex no exige cronológico) --
    # secuencia completa, sin huecos reales, solo desordenada.
    return _rv({"INPP": [(PeriodoMensual(2024, 3), 10.0), (_M1, 5.0), (_M2, 8.0)]})


def test_acumulada_secuencia_desordenada_no_es_hueco() -> None:
    # 100*(1.05*1.08*1.10 - 1) = 24.74, calculado sin importar el orden físico
    # de las filas en el ResultadoVariacion.
    r = _rv_desordenado()
    assert inflacion_acumulada(r, _M1, PeriodoMensual(2024, 3), indice="INPP") == pytest.approx(
        24.74
    )


def test_tcac_secuencia_desordenada_no_es_hueco() -> None:
    r = _rv_desordenado()
    # No debe lanzar InvarianteViolado por "no consecutivos" -- la secuencia
    # SÍ es consecutiva, solo está desordenada en el DataFrame.
    inflacion_promedio(r, _M1, PeriodoMensual(2024, 3), indice="INPP", metodo="tcac")
