from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import ManifestCalculo, ManifestDerivado
from replica_inpp.infraestructura.graficacion import _prepocesamiento as pp

# --------------------------------------------------------------------------- helpers


def _manifiesto(
    version: int = 2019,
    agregacion: str = "SECTOR",
    rubro: str = "produccion_total",
    incluir_petroleo: bool = True,
) -> ManifestCalculo:
    return ManifestCalculo(
        version=version,  # type: ignore[arg-type]
        agregacion=agregacion,
        rubro=rubro,
        incluir_petroleo=incluir_petroleo,
        calculador="LaspeyresDirecto",
        fecha=datetime(2024, 1, 1),
    )


def _resultado(
    filas: list[tuple[Any, str, float, str]],
    version: int = 2019,
    agregacion: str = "SECTOR",
    rubro: str = "produccion_total",
    incluir_petroleo: bool = True,
    periodo_referencia: Any = None,
) -> ResultadoIndice:
    """filas = list of (periodo, indice, valor, estado)."""
    registros = [
        {
            "periodo": p,
            "indice": i,
            "version": version,
            "agregacion": agregacion,
            "rubro": rubro,
            "indice_replicado": v,
            "estado_calculo": e,
            "motivo_error": None,
        }
        for p, i, v, e in filas
    ]
    df = pd.DataFrame(registros)
    df.index = pd.MultiIndex.from_arrays(
        [df.pop("periodo"), df.pop("indice")], names=["periodo", "indice"]
    )
    reporte = df[[]].copy()
    diag = pd.DataFrame(
        columns=["periodo", "generico", "nivel_faltante", "tipo_faltante", "detalle"]
    )
    return ResultadoIndice(
        df,
        [_manifiesto(version, agregacion, rubro, incluir_petroleo)],
        reporte,
        diag,
        periodo_referencia=periodo_referencia,
    )


def _resultado_heterogeneo(filas: list[tuple[Any, str, str, str, float]]) -> ResultadoIndice:
    """filas = list of (periodo, agregacion, rubro, indice, valor) -- una combinación (agregacion,rubro) por fila.

    A diferencia de `_resultado` (agregacion/rubro fijos), arma un manifiesto
    con una entrada `ManifestCalculo` por cada combinación `(agregacion,
    rubro)` distinta presente -- simula un `ResultadoIndice` heterogéneo,
    construido a mano fuera del flujo guiado (`calcular_indice`/`empalmar`
    nunca producen esto).
    """
    registros = [
        {
            "periodo": p,
            "indice": i,
            "version": 2019,
            "agregacion": a,
            "rubro": r,
            "indice_replicado": v,
            "estado_calculo": "ok",
            "motivo_error": None,
        }
        for p, a, r, i, v in filas
    ]
    df = pd.DataFrame(registros)
    df.index = pd.MultiIndex.from_arrays(
        [df.pop("periodo"), df.pop("indice")], names=["periodo", "indice"]
    )
    reporte = df[[]].copy()
    diag = pd.DataFrame(
        columns=["periodo", "generico", "nivel_faltante", "tipo_faltante", "detalle"]
    )
    combos = dict.fromkeys((a, r) for _, a, r, _, _ in filas)
    manifiestos = [_manifiesto(2019, a, r) for a, r in combos]
    return ResultadoIndice(df, manifiestos, reporte, diag)


def _manifiesto_variacion(
    agregacion: str = "SECTOR", rubro: str = "produccion_total", clase: str = "periodica_mensual"
) -> ManifestDerivado:
    return ManifestDerivado(
        versiones=[2019],  # type: ignore[arg-type]
        agregacion=agregacion,
        rubro=rubro,
        clase=clase,
        descripcion="",
        fecha=datetime(2024, 1, 1),
    )


def _resultado_variacion(
    filas: list[tuple[Any, str, float, str]],
    agregacion: str = "SECTOR",
    rubro: str = "produccion_total",
    clase: str = "periodica_mensual",
) -> ResultadoVariacion:
    """filas = list of (periodo, indice, variacion_pp, estado)."""
    registros = [
        {
            "periodo": p,
            "indice": i,
            "agregacion": agregacion,
            "rubro": rubro,
            "clase_variacion": clase,
            "variacion_pp": v,
            "estado_calculo": e,
        }
        for p, i, v, e in filas
    ]
    df = pd.DataFrame(registros)
    df.index = pd.MultiIndex.from_arrays(
        [df.pop("periodo"), df.pop("indice")], names=["periodo", "indice"]
    )
    reporte = df[[]].copy()
    diag = pd.DataFrame(columns=["periodo", "indice", "estado_calculo", "motivo_error"])
    indices_parciales = pd.DataFrame() if clase == "desde" else None
    return ResultadoVariacion(
        df, _manifiesto_variacion(agregacion, rubro, clase), reporte, diag, indices_parciales
    )


_P1 = PeriodoMensual(2018, 1)
_P2 = PeriodoMensual(2018, 2)
_P3 = PeriodoMensual(2018, 3)


def _datos_n_categorias(n: int) -> pd.DataFrame:
    filas = [(_P1, f"cat{i:02d}", float(i), "ok") for i in range(n)]
    r = _resultado(filas)
    return pp._aplanar_resultado(r)


def _meses(n: int) -> list[PeriodoMensual]:
    return [PeriodoMensual(2018 + i // 12, i % 12 + 1) for i in range(n)]


def _datos_serie(periodos: list[Any], indice: str = "011 Cría de bovinos") -> pd.DataFrame:
    r = _resultado([(p, indice, 100.0 + i, "ok") for i, p in enumerate(periodos)])
    return pp._aplanar_resultado(r)


# --------------------------------------------------------------------------- _titulo


def test_titulo_una_sola_combinacion_con_petroleo() -> None:
    r = _resultado([(_P1, "cat", 100.0, "ok")], agregacion="SECTOR", rubro="produccion_total")
    datos = pp._aplanar_resultado(r)
    # "produccion_total" -> "Produccion total": sin guion bajo, legible.
    assert pp._titulo(datos) == "SECTOR Produccion total con petróleo"


def test_titulo_sin_petroleo() -> None:
    r = _resultado([(_P1, "cat", 100.0, "ok")], incluir_petroleo=False)
    datos = pp._aplanar_resultado(r)
    assert pp._titulo(datos) == "SECTOR Produccion total sin petróleo"


def test_titulo_variacion_sin_segmento_de_petroleo() -> None:
    # ManifestDerivado no trae incluir_petroleo -- ese segmento se omite entero,
    # no se fuerza texto sin sentido.
    rv = _resultado_variacion([(_P1, "cat", 0.5, "ok")])
    datos = pp._aplanar_resultado(rv)
    assert pp._titulo(datos) == "SECTOR Produccion total"


def test_titulo_junta_combinaciones_de_resultado_y_comparacion() -> None:
    r = _resultado([(_P1, "cat", 100.0, "ok")], agregacion="SECTOR", rubro="produccion_total")
    comparacion = _resultado([(_P1, "cat2", 90.0, "ok")], agregacion="INPP", rubro="exportaciones")
    datos = pp._aplanar_resultado(r, comparacion)
    assert (
        pp._titulo(datos)
        == "SECTOR Produccion total con petróleo + INPP Exportaciones con petróleo"
    )


# --------------------------------------------------------------------------- _aplanar_resultado


def test_aplanar_sin_comparacion_linetype_solid() -> None:
    r = _resultado([(_P1, "cat", 100.0, "ok")])
    datos = pp._aplanar_resultado(r)
    assert len(datos) == 1
    assert datos["linetype"].tolist() == ["solid"]
    assert "periodo_ts" in datos.columns
    assert datos["incluir_petroleo"].tolist() == [True]


def test_aplanar_con_comparacion_concatena_y_marca_linetype() -> None:
    principal = _resultado([(_P1, "cat_a", 90.0, "ok")])
    comparacion = _resultado([(_P1, "cat_b", 100.0, "ok")])
    datos = pp._aplanar_resultado(principal, comparacion)
    assert len(datos) == 2
    linetypes = dict(zip(datos["indice"], datos["linetype"], strict=True))
    assert linetypes == {"cat_a": "solid", "cat_b": "dashed"}


def test_aplanar_variacion_incluir_petroleo_es_none() -> None:
    rv = _resultado_variacion([(_P1, "cat", 0.5, "ok")])
    datos = pp._aplanar_resultado(rv)
    assert datos["incluir_petroleo"].iloc[0] is None


def test_aplanar_swap_indice_por_rubro_cuando_agregacion_es_inpp() -> None:
    r = _resultado([(_P1, "INPP", 100.0, "ok")], agregacion="INPP", rubro="exportaciones")
    datos = pp._aplanar_resultado(r)
    assert datos["indice"].tolist() == ["exportaciones"]


def test_aplanar_no_swapea_indice_con_agregacion_scian() -> None:
    r = _resultado(
        [(_P1, "22 Generación de energía eléctrica", 100.0, "ok")],
        agregacion="SECTOR",
        rubro="produccion_total",
    )
    datos = pp._aplanar_resultado(r)
    assert datos["indice"].tolist() == ["22 Generación de energía eléctrica"]


def test_aplanar_desambigua_indice_repetido_entre_agregaciones_distintas_en_solid() -> None:
    # SECTOR/produccion_total "11" en enero y SUBSECTOR/exportaciones "11" en
    # febrero -- mediciones sin relación, mismo código, ambas en `resultado`
    # (linetype solid). Sin desambiguar, geom_line las conecta como una sola
    # serie continua (100 -> 200).
    r = _resultado_heterogeneo(
        [
            (_P1, "SECTOR", "produccion_total", "11", 100.0),
            (_P2, "SUBSECTOR", "exportaciones", "11", 200.0),
        ]
    )
    datos = pp._aplanar_resultado(r)
    assert datos["indice"].nunique() == 2


def test_aplanar_no_desambigua_mismo_indice_entre_solid_y_dashed() -> None:
    # El mismo índice en resultado (solid) y comparacion (dashed), aunque las
    # combinaciones agregacion/rubro DIFIERAN entre ellos, conserva su
    # identificador -- el propio linetype ya distingue cuál es cuál, y
    # desambiguar cruzando linetype rompería el color compartido que esa
    # comparación necesita para mostrarse.
    r = _resultado([(_P1, "11", 100.0, "ok")], agregacion="SECTOR", rubro="produccion_total")
    comparacion = _resultado(
        [(_P1, "11", 200.0, "ok")], agregacion="SUBSECTOR", rubro="exportaciones"
    )
    datos = pp._aplanar_resultado(r, comparacion)
    assert set(datos["indice"]) == {"11"}


# --------------------------------------------------------------------------- _recortar_tramo


def _datos_tres_periodos() -> pd.DataFrame:
    r = _resultado(
        [
            (_P1, "cat", 100.0, "ok"),
            (_P2, "cat", 101.0, "ok"),
            (_P3, "cat", 102.0, "ok"),
        ]
    )
    return pp._aplanar_resultado(r)


def test_recortar_tramo_sin_limites_no_cambia_nada() -> None:
    datos = _datos_tres_periodos()
    assert len(pp._recortar_tramo(datos, None, None)) == 3


def test_recortar_tramo_desde_excluye_anteriores() -> None:
    datos = _datos_tres_periodos()
    recortado = pp._recortar_tramo(datos, _P2, None)
    assert set(recortado["periodo"]) == {_P2, _P3}


def test_recortar_tramo_hasta_excluye_posteriores() -> None:
    datos = _datos_tres_periodos()
    recortado = pp._recortar_tramo(datos, None, _P2)
    assert set(recortado["periodo"]) == {_P1, _P2}


def test_recortar_tramo_desde_mayor_a_hasta_falla() -> None:
    datos = _datos_tres_periodos()
    with pytest.raises(InvarianteViolado):
        pp._recortar_tramo(datos, _P3, _P1)


def test_recortar_tramo_sin_datos_en_rango_falla() -> None:
    datos = _datos_tres_periodos()
    fuera = PeriodoMensual(2030, 1)
    with pytest.raises(InvarianteViolado):
        pp._recortar_tramo(datos, fuera, fuera)


# --------------------------------------------------------------------------- _descartar_grupos_no_graficables


def test_descartar_grupos_totalmente_no_finitos_falla() -> None:
    r = _resultado(
        [
            (_P1, "cat", float("nan"), "sin_datos"),
            (_P2, "cat", float("nan"), "sin_datos"),
            (_P3, "cat", float("nan"), "sin_datos"),
        ]
    )
    datos = pp._aplanar_resultado(r)
    with pytest.raises(InvarianteViolado):
        pp._descartar_grupos_no_graficables(datos, "indice_replicado")


def test_descartar_grupos_conserva_nan_interiores_de_grupo_con_algun_finito() -> None:
    # El hueco (Feb=NaN) no se filtra fila por fila -- solo se descartan
    # grupos SIN ningún valor finito. Ene/Mar finitos alcanzan para conservar
    # las 3 filas, NaN incluido.
    r = _resultado(
        [
            (_P1, "cat", 100.0, "ok"),
            (_P2, "cat", float("nan"), "sin_datos"),
            (_P3, "cat", 110.0, "ok"),
        ]
    )
    datos = pp._aplanar_resultado(r)
    graficable = pp._descartar_grupos_no_graficables(datos, "indice_replicado")
    assert len(graficable) == 3
    assert graficable["indice_replicado"].isna().sum() == 1


def test_descartar_grupos_quita_solo_la_categoria_sin_ningun_finito() -> None:
    # cat_b no tiene ningún valor real -- se descarta entera; cat_a conserva
    # sus 2 filas intactas.
    r = _resultado(
        [
            (_P1, "cat_a", 100.0, "ok"),
            (_P2, "cat_a", 105.0, "ok"),
            (_P1, "cat_b", float("nan"), "sin_datos"),
            (_P2, "cat_b", float("nan"), "sin_datos"),
        ]
    )
    datos = pp._aplanar_resultado(r)
    graficable = pp._descartar_grupos_no_graficables(datos, "indice_replicado")
    assert set(graficable["indice"]) == {"cat_a"}
    assert len(graficable) == 2


# --------------------------------------------------------------------------- _particionar_series


def test_particionar_series_bajo_capacidad_no_parte() -> None:
    datos = _datos_n_categorias(5)
    particiones = pp._particionar_series(datos, capacidad=8)
    assert len(particiones) == 1
    assert len(particiones[0]) == 5


def test_particionar_series_reparte_parejo() -> None:
    datos = _datos_n_categorias(13)
    particiones = pp._particionar_series(datos, capacidad=8)
    assert len(particiones) == 2
    assert sorted(len(p) for p in particiones) == [6, 7]


def test_particionar_series_comparacion_no_cuenta_y_se_repite_en_cada_particion() -> None:
    r = _resultado([(_P1, f"cat{i:02d}", float(i), "ok") for i in range(13)])
    comparacion = _resultado([(_P1, "INPP", 100.0, "ok")], agregacion="INPP")
    datos = pp._aplanar_resultado(r, comparacion)
    particiones = pp._particionar_series(datos, capacidad=8)
    assert len(particiones) == 2
    for parte in particiones:
        assert "exportaciones" not in set(parte["indice"])  # no se coló otro rubro
        assert set(parte[parte["linetype"] == "dashed"]["indice"]) == {"produccion_total"}


def _datos_con_comparacion_grande(
    n_propios: int, n_comparacion: int, propios_dentro_de_c: bool = False
) -> pd.DataFrame:
    """`resultado` con `n_propios` categorías + `comparacion` con `n_comparacion` categorías.

    Ambos objetos requieren al menos 1 fila (`ResultadoIndice` no admite un
    `df_resultado` vacío), así que `n_propios`/`n_comparacion` son >= 1.
    `propios_dentro_de_c=True` reusa los MISMOS nombres que `comparacion` (en
    vez de un prefijo disjunto) -- para el caso "todos los índices propios ya
    pertenecen a C".
    """
    prefijo_propio = "m" if propios_dentro_de_c else "s"
    r = _resultado(
        [(_P1, f"{prefijo_propio}{i:02d}", float(i), "ok") for i in range(n_propios)],
        agregacion="SECTOR",
    )
    comparacion = _resultado(
        [(_P1, f"m{i:02d}", float(i), "ok") for i in range(n_comparacion)],
        agregacion="MERCANCIAS_SERVICIOS",
    )
    return pp._aplanar_resultado(r, comparacion)


def test_particionar_series_invariante_nunique_indice_no_pasa_capacidad() -> None:
    # Propiedad general: cualquier combinación propios/comparación, cada panel
    # respeta el tope de colores -- antes, comparación grande la rompía.
    for n_propios, n_comparacion in [(13, 13), (1, 13), (13, 1), (5, 8), (8, 8), (3, 9)]:
        datos = _datos_con_comparacion_grande(n_propios, n_comparacion)
        for parte in pp._particionar_series(datos, capacidad=8):
            assert parte["indice"].nunique() <= 8


def test_particionar_series_comparacion_grande_reparte_la_union() -> None:
    # 13 propios + 13 de comparación (disjuntos): sin la corrección, un panel
    # llegaba a 20 series con colores repetidos.
    datos = _datos_con_comparacion_grande(13, 13)
    particiones = pp._particionar_series(datos, capacidad=8)
    for parte in particiones:
        assert parte["indice"].nunique() <= 8
    # la unión (13+13=26) se reparte completa entre las particiones, sin perder índices
    todos = set().union(*(set(p["indice"]) for p in particiones))
    assert todos == {f"s{i:02d}" for i in range(13)} | {f"m{i:02d}" for i in range(13)}


def test_particionar_series_comparacion_igual_a_capacidad_sin_sobrantes_no_parte() -> None:
    # |C| == 8 y ningún índice propio queda afuera de C (los propios reusan
    # los mismos nombres de comparación): cabe entero.
    datos = _datos_con_comparacion_grande(3, 8, propios_dentro_de_c=True)
    particiones = pp._particionar_series(datos, capacidad=8)
    assert len(particiones) == 1


def test_particionar_series_comparacion_igual_a_capacidad_con_sobrantes_parte() -> None:
    # |C| == 8 pero sobra 1 índice propio ajeno a C: no cabe repetir C entera
    # y agregar algo más sin pasar el tope -- cae a partición por unión.
    datos = _datos_con_comparacion_grande(1, 8)
    particiones = pp._particionar_series(datos, capacidad=8)
    assert len(particiones) > 1
    for parte in particiones:
        assert parte["indice"].nunique() <= 8


def test_particionar_series_mantiene_solida_y_punteada_del_mismo_indice_juntas() -> None:
    # Un índice que colisiona (mismo valor en solid y dashed) no debe quedar
    # partido entre dos imágenes distintas.
    r = _resultado([(_P1, f"s{i:02d}", float(i), "ok") for i in range(12)], agregacion="SECTOR")
    comparacion = _resultado(
        [(_P1, f"s{i:02d}", float(i) + 0.5, "ok") for i in range(9)], agregacion="SECTOR"
    )
    datos = pp._aplanar_resultado(r, comparacion)
    for parte in pp._particionar_series(datos, capacidad=8):
        for indice in pd.unique(parte["indice"]):
            indice_str = str(indice)
            linetypes_en_datos = set(datos.loc[datos["indice"] == indice_str, "linetype"])
            linetypes_en_parte = set(parte.loc[parte["indice"] == indice_str, "linetype"])
            assert linetypes_en_parte == linetypes_en_datos


# --------------------------------------------------------------------------- colores estables entre paneles


def test_indices_comparacion_repetidos_none_si_no_aplica_particion_fija() -> None:
    # Cabe entero (pocos propios): no hay nada que repetir entre paneles.
    datos = _datos_con_comparacion_grande(3, 1)
    assert pp._indices_comparacion_repetidos(datos, capacidad=8) is None


def test_indices_comparacion_repetidos_no_es_falsy_cuando_no_hay_comparacion() -> None:
    # Sin comparación pero con más propios que capacidad: la partición "fija"
    # sigue aplicando (con `fijos=[]`, nada que reservar) -- `[]` no debe
    # confundirse con "no aplica" por ser una lista vacía y falsy en Python.
    r = _resultado([(_P1, f"cat{i:02d}", float(i), "ok") for i in range(13)])
    datos = pp._aplanar_resultado(r)
    assert pp._indices_comparacion_repetidos(datos, capacidad=8) == []
    assert len(pp._particionar_series(datos, capacidad=8)) == 2


def test_reservar_colores_comparacion_asigna_en_orden_de_aparicion() -> None:
    reservados = pp._reservar_colores_comparacion(["m00", "m01", "m02"])
    assert reservados == {
        "m00": pp._PALETA_OTROS_TIPOS[0],
        "m01": pp._PALETA_OTROS_TIPOS[1],
        "m02": pp._PALETA_OTROS_TIPOS[2],
    }


def test_reservar_colores_comparacion_vacia_no_reserva_nada() -> None:
    assert pp._reservar_colores_comparacion([]) == {}


def _colores_por_panel(datos: pd.DataFrame, capacidad: int = 8) -> list[dict[str, str]]:
    reservados = pp._reservar_colores_comparacion(
        pp._indices_comparacion_repetidos(datos, capacidad) or []
    )
    resultado = []
    for parte in pp._particionar_series(datos, capacidad=capacidad):
        parte = parte.copy()
        parte["indice"] = pp._ordenar_series_dibujo(parte["indice"], parte["linetype"])
        colores, _ = pp._colores_y_etiquetas(list(pd.unique(parte["indice"])), reservados)
        resultado.append(colores)
    return resultado


def test_colores_comparacion_repetida_conserva_el_mismo_color_en_todos_los_paneles() -> None:
    # Repro exacta del hallazgo: 9 propios + 3 de comparación, capacidad 8 --
    # antes, m00/m01/m02 cambiaban de color entre el panel 0 y el panel 1.
    colores_por_panel = _colores_por_panel(_datos_con_comparacion_grande(9, 3))
    assert len(colores_por_panel) == 2
    for indice in ("m00", "m01", "m02"):
        colores_vistos = {colores[indice] for colores in colores_por_panel if indice in colores}
        assert len(colores_vistos) == 1


def test_colores_dentro_de_un_panel_no_tiene_duplicados_accidentales() -> None:
    for colores in _colores_por_panel(_datos_con_comparacion_grande(9, 3)):
        genericos = [c for indice, c in colores.items() if indice not in pp._RUBROS_CONOCIDOS]
        assert len(genericos) == len(set(genericos)), "colores duplicados dentro del panel"


# --------------------------------------------------------------------------- _ordenar_series_dibujo


def test_ordenar_series_dibujo_comparacion_al_final() -> None:
    valores = pd.Series(["cat_b", "cat_a", "ref"])
    linetype = pd.Series(["solid", "solid", "dashed"])
    categorico = pp._ordenar_series_dibujo(valores, linetype)
    assert list(categorico.categories) == ["cat_b", "cat_a", "ref"]


def test_ordenar_series_dibujo_orden_explicito_reemplaza_el_de_aparicion() -> None:
    valores = pd.Series(["c", "a", "b"])
    linetype = pd.Series(["solid", "solid", "solid"])
    resultado = pp._ordenar_series_dibujo(valores, linetype, ["b", "a", "c"])
    assert list(resultado.categories) == ["b", "a", "c"]


def test_ordenar_series_dibujo_orden_parcial_no_pierde_categorias() -> None:
    valores = pd.Series(["c", "a", "b"])
    linetype = pd.Series(["solid", "solid", "solid"])
    resultado = pp._ordenar_series_dibujo(valores, linetype, ["b"])
    assert list(resultado.categories) == ["b", "c", "a"]


def test_ordenar_series_dibujo_orden_explicito_no_adelanta_comparacion() -> None:
    valores = pd.Series(["a", "ref", "b"])
    linetype = pd.Series(["solid", "dashed", "solid"])
    resultado = pp._ordenar_series_dibujo(valores, linetype, ["ref", "b", "a"])
    assert list(resultado.categories)[-1] == "ref"


def test_ordenar_series_dibujo_sin_comparacion_mantiene_orden_aparicion() -> None:
    valores = pd.Series(["cat_b", "cat_a", "cat_b"])
    linetype = pd.Series(["solid", "solid", "solid"])
    categorico = pp._ordenar_series_dibujo(valores, linetype)
    assert list(categorico.categories) == ["cat_b", "cat_a"]


# --------------------------------------------------------------------------- _breaks_y_etiquetas_x


def test_breaks_x_incluye_siempre_primero_y_ultimo() -> None:
    datos = _datos_serie(_meses(24))
    breaks, _ = pp._breaks_y_etiquetas_x(datos)
    assert breaks[0] == datos["periodo_ts"].min()
    assert breaks[-1] == datos["periodo_ts"].max()


def test_breaks_x_pocos_periodos_devuelve_todos() -> None:
    datos = _datos_serie(_meses(5))
    breaks, etiquetas = pp._breaks_y_etiquetas_x(datos)
    assert len(breaks) == 5
    assert len(etiquetas) == 5


def test_breaks_x_un_solo_periodo() -> None:
    datos = _datos_serie(_meses(1))
    breaks, etiquetas = pp._breaks_y_etiquetas_x(datos)
    assert len(breaks) == 1 and len(etiquetas) == 1


def test_breaks_x_no_pasa_del_maximo_de_etiquetas() -> None:
    datos = _datos_serie(_meses(186))
    breaks, _ = pp._breaks_y_etiquetas_x(datos)
    assert len(breaks) <= pp._MAX_ETIQUETAS_EJE_X


def test_breaks_x_reparte_parejo_hasta_el_ultimo_periodo() -> None:
    periodos = _meses(186)
    datos = _datos_serie(periodos)
    breaks, _ = pp._breaks_y_etiquetas_x(datos)
    posicion = {p.to_timestamp(): i for i, p in enumerate(periodos)}
    saltos = [posicion[b] - posicion[a] for a, b in zip(breaks, breaks[1:], strict=False)]
    assert max(saltos) - min(saltos) <= 1


# --------------------------------------------------------------------------- _breaks_y / _etiqueta_y_indice


def test_breaks_y_incluye_minimo_y_maximo_reales() -> None:
    r = _resultado([(_P1, "cat", 80.0, "ok"), (_P2, "cat", 120.0, "ok")])
    datos = pp._aplanar_resultado(r)
    breaks = pp._breaks_y(datos, "indice_replicado", pp._VALOR_BASE)
    assert breaks[0] == 80.0
    assert breaks[-1] == 120.0


def test_breaks_y_incluye_100_si_esta_en_rango() -> None:
    r = _resultado([(_P1, "cat", 80.0, "ok"), (_P2, "cat", 120.0, "ok")])
    datos = pp._aplanar_resultado(r)
    breaks = pp._breaks_y(datos, "indice_replicado", pp._VALOR_BASE)
    assert 100.0 in breaks


def test_breaks_y_no_agrega_100_fuera_de_rango() -> None:
    r = _resultado([(_P1, "cat", 120.0, "ok"), (_P2, "cat", 150.0, "ok")])
    datos = pp._aplanar_resultado(r)
    breaks = pp._breaks_y(datos, "indice_replicado", pp._VALOR_BASE)
    assert 100.0 not in breaks


def test_etiqueta_y_sin_periodo_referencia() -> None:
    r = _resultado([(_P1, "cat", 100.0, "ok")])
    assert pp._etiqueta_y_indice(r) == "Indice"


def test_etiqueta_y_con_periodo_referencia() -> None:
    r = _resultado([(_P1, "cat", 100.0, "ok")], periodo_referencia=_P1)
    assert pp._etiqueta_y_indice(r) == f"Indice ({_P1} = 100)"


# --------------------------------------------------------------------------- _colores_y_etiquetas


def test_colores_rubro_siempre_negro() -> None:
    # Sin variedad por rubro: la única condición para ser negro es venir de
    # agregacion=="INPP" (indice ya sobrescrito con rubro), sea cual sea el
    # rubro o el orden en que aparezcan -- distinguir 2 rubros de INPP entre sí
    # depende de `linetype`, no de color (ver graficador._construir_grafica_linea).
    colores_a, _ = pp._colores_y_etiquetas(["exportaciones", "produccion_total"])
    colores_b, _ = pp._colores_y_etiquetas(["produccion_total", "exportaciones"])
    assert colores_a == {"exportaciones": "black", "produccion_total": "black"}
    assert colores_b == {"exportaciones": "black", "produccion_total": "black"}


def test_colores_no_rubro_toman_paleta_en_orden_aparicion() -> None:
    colores, _ = pp._colores_y_etiquetas(["cat_a", "cat_b"])
    assert colores["cat_a"] == pp._PALETA_OTROS_TIPOS[0]
    assert colores["cat_b"] == pp._PALETA_OTROS_TIPOS[1]


def test_colores_rubro_no_consume_turno_de_paleta() -> None:
    colores, _ = pp._colores_y_etiquetas(["cat_a", "produccion_total", "cat_b"])
    assert colores["cat_a"] == pp._PALETA_OTROS_TIPOS[0]
    assert colores["cat_b"] == pp._PALETA_OTROS_TIPOS[1]


def test_etiquetas_rubro_son_legibles() -> None:
    _, etiquetas = pp._colores_y_etiquetas(["demanda_interna_consumo"])
    assert etiquetas["demanda_interna_consumo"] == "Demanda interna consumo"


def test_etiquetas_nombre_corto_no_se_trunca() -> None:
    _, etiquetas = pp._colores_y_etiquetas(["cat_a"])
    assert etiquetas["cat_a"] == "cat_a"


def test_etiquetas_nombre_largo_se_trunca_con_puntos() -> None:
    nombre = "22 generacion transmision distribucion y comercializacion de energia"
    _, etiquetas = pp._colores_y_etiquetas([nombre])
    assert len(etiquetas[nombre]) == pp._MAX_CARACTERES_LEYENDA
    assert etiquetas[nombre].endswith("...")


def test_etiquetas_limite_exacto_no_se_trunca() -> None:
    nombre = "x" * pp._MAX_CARACTERES_LEYENDA
    _, etiquetas = pp._colores_y_etiquetas([nombre])
    assert etiquetas[nombre] == nombre


# --------------------------------------------------------------------------- _primero_y_ultimo_para_anotar


def test_primero_ultimo_none_si_hay_mas_de_una_serie() -> None:
    r = _resultado([(_P1, "cat_a", 100.0, "ok"), (_P1, "cat_b", 90.0, "ok")])
    datos = pp._aplanar_resultado(r)
    assert pp._primero_y_ultimo_para_anotar(datos, ["cat_a", "cat_b"], "indice_replicado") is None


def test_primero_ultimo_none_si_resultado_y_comparacion_comparten_indice() -> None:
    principal = _resultado([(_P1, "cat", 100.0, "ok"), (_P2, "cat", 105.0, "ok")])
    comparacion = _resultado([(_P1, "cat", 200.0, "ok"), (_P2, "cat", 210.0, "ok")])
    datos = pp._aplanar_resultado(principal, comparacion)
    assert pp._primero_y_ultimo_para_anotar(datos, ["cat"], "indice_replicado") is None


def test_primero_ultimo_devuelve_extremos_con_una_sola_serie() -> None:
    r = _resultado(
        [(_P1, "cat", 100.0, "ok"), (_P2, "cat", 105.0, "ok"), (_P3, "cat", 110.0, "ok")]
    )
    datos = pp._aplanar_resultado(r)
    resultado = pp._primero_y_ultimo_para_anotar(datos, ["cat"], "indice_replicado")
    assert resultado is not None
    primero, ultimo = resultado
    assert primero["indice_replicado"] == 100.0
    assert ultimo["indice_replicado"] == 110.0


# --------------------------------------------------------------------------- _breaks_desde_extremos


def test_breaks_extremos_siempre_incluye_minimo_y_maximo() -> None:
    breaks = pp._breaks_desde_extremos(-1.21, 8.77, 0.0)
    assert breaks[0] == -1.21
    assert breaks[-1] == 8.77


def test_breaks_extremos_descarta_automatico_pegado_al_minimo() -> None:
    breaks = pp._breaks_desde_extremos(-1.21, 8.77, 0.0)
    assert -1.0 not in breaks
    assert -1.21 in breaks


def test_breaks_extremos_rango_cero_devuelve_un_solo_break() -> None:
    assert pp._breaks_desde_extremos(5.0, 5.0, 0.0) == [5.0]


def test_breaks_extremos_descarta_base_pegada_a_un_extremo() -> None:
    breaks = pp._breaks_desde_extremos(-0.001, 10.0, 0.0)
    assert 0.0 not in breaks


# --------------------------------------------------------------------------- _datos_para_puntos


@pytest.mark.parametrize("cuantos", [12, 13])
def test_puntos_en_todas_las_filas_hasta_13_meses(cuantos: int) -> None:
    datos = _datos_serie(_meses(cuantos))
    resultado = pp._datos_para_puntos(datos, "indice_replicado")
    assert resultado is not None
    assert len(resultado) == len(datos)


def test_sin_puntos_con_14_meses() -> None:
    assert pp._datos_para_puntos(_datos_serie(_meses(14)), "indice_replicado") is None


def test_puntos_solo_en_la_categoria_de_una_sola_aparicion() -> None:
    periodos = _meses(24)
    largas = [(p, "cat_larga", 100.0, "ok") for p in periodos]
    r = _resultado([*largas, (periodos[10], "rara", 95.0, "ok")])
    datos = pp._aplanar_resultado(r)
    resultado = pp._datos_para_puntos(datos, "indice_replicado")
    assert resultado is not None
    assert resultado["indice"].tolist() == ["rara"]


def test_puntos_agrupan_por_indice_y_linetype_no_solo_indice() -> None:
    periodos = _meses(24)
    principal = _resultado([(p, "cat", 100.0, "ok") for p in periodos])
    comparacion = _resultado([(periodos[0], "cat", 200.0, "ok")])
    datos = pp._aplanar_resultado(principal, comparacion)
    resultado = pp._datos_para_puntos(datos, "indice_replicado")
    assert resultado is not None
    assert resultado["linetype"].tolist() == ["dashed"]


# --------------------------------------------------------------------------- _aplanar_resultado (variaciones)


def test_aplanar_variacion_sin_comparacion_linetype_solid() -> None:
    rv = _resultado_variacion([(_P1, "cat", 0.5, "ok")])
    datos = pp._aplanar_resultado(rv)
    assert len(datos) == 1
    assert datos["variacion_pp"].tolist() == [0.5]
    assert datos["linetype"].tolist() == ["solid"]
    assert "periodo_ts" in datos.columns


def test_aplanar_variacion_con_comparacion_marca_linetype() -> None:
    principal = _resultado_variacion([(_P1, "cat", 0.5, "ok")])
    comparacion = _resultado_variacion([(_P1, "cat2", 0.3, "ok")])
    datos = pp._aplanar_resultado(principal, comparacion)
    assert len(datos) == 2
    linetypes = dict(zip(datos["indice"], datos["linetype"], strict=True))
    assert linetypes == {"cat": "solid", "cat2": "dashed"}


def test_aplanar_variacion_swap_indice_por_rubro_cuando_agregacion_es_inpp() -> None:
    rv = _resultado_variacion([(_P1, "INPP", 0.5, "ok")], agregacion="INPP", rubro="exportaciones")
    datos = pp._aplanar_resultado(rv)
    assert datos["indice"].tolist() == ["exportaciones"]


# --------------------------------------------------------------------------- _breaks_y (variaciones)


def test_breaks_y_variacion_incluye_minimo_maximo_y_cero() -> None:
    datos = pd.DataFrame({"variacion_pp": [-2.0, 3.0]})
    breaks = pp._breaks_y(datos, "variacion_pp", pp._VALOR_BASE_VARIACION)
    assert breaks[0] == -2.0
    assert breaks[-1] == 3.0
    assert 0.0 in breaks


def test_breaks_y_variacion_no_agrega_cero_fuera_de_rango() -> None:
    datos = pd.DataFrame({"variacion_pp": [5.0, 10.0]})
    breaks = pp._breaks_y(datos, "variacion_pp", pp._VALOR_BASE_VARIACION)
    assert 0.0 not in breaks
