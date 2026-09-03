from __future__ import annotations

from datetime import datetime
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

from replica_inpp.dominio.errores import InvarianteViolado
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.variacion import ResultadoVariacion
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import ManifestCalculo, ManifestDerivado
from replica_inpp.infraestructura.graficacion import graficador

# Backend sin GUI: este es el único módulo de test que llama a `.draw()`, y
# plotnine construye la figura con `plt.figure()`, que instancia un manager del
# backend activo. En Linux headless el default ya es Agg; en otros entornos la
# detección automática puede elegir un backend con GUI que revienta sin
# display. Mismo guard que `replica-inpc-mx`.
matplotlib.use("Agg")

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
        df, [_manifiesto(version, agregacion, rubro, incluir_petroleo)], reporte, diag
    )


def _resultado_heterogeneo(filas: list[tuple[Any, str, str, str, float]]) -> ResultadoIndice:
    """filas = list of (periodo, agregacion, rubro, indice, valor) -- una combinación (agregacion,rubro) por fila."""
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


_P1 = PeriodoMensual(2018, 1)
_P2 = PeriodoMensual(2018, 2)
_P3 = PeriodoMensual(2018, 3)


def _geoms(grafica: Any) -> list[str]:
    return [type(layer.geom).__name__ for layer in grafica.layers]


def _n_categorias(n: int) -> list[tuple[Any, str, float, str]]:
    return [(_P1, f"cat{i:02d}", float(i), "ok") for i in range(n)]


def _meses(n: int) -> list[PeriodoMensual]:
    return [PeriodoMensual(2018 + i // 12, i % 12 + 1) for i in range(n)]


# --------------------------------------------------------------------------- _construir_grafica_linea


def test_construir_grafica_agrupa_por_indice() -> None:
    r = _resultado([(_P1, "cat_a", 90.0, "ok"), (_P1, "cat_b", 110.0, "ok")])
    datos = graficador._aplanar_resultado(r)
    grafica = graficador._construir_grafica_linea(datos, r)
    assert grafica.mapping["color"] == "indice"


def test_construir_grafica_incluye_geom_point_en_tramo_corto() -> None:
    r = _resultado([(_P1, "cat", 90.0, "ok"), (_P2, "cat", 110.0, "ok")])
    datos = graficador._aplanar_resultado(r)
    grafica = graficador._construir_grafica_linea(datos, r)
    assert "geom_point" in _geoms(grafica)


def test_construir_grafica_sin_geom_point_en_tramo_largo() -> None:
    r = _resultado([(p, "cat", 100.0 + i, "ok") for i, p in enumerate(_meses(24))])
    datos = graficador._aplanar_resultado(r)
    grafica = graficador._construir_grafica_linea(datos, r)
    assert "geom_point" not in _geoms(grafica)


def test_construir_grafica_renderiza_punto_de_serie_solitaria_en_tramo_largo() -> None:
    periodos = _meses(24)
    filas = [(p, "cat", 100.0 + i, "ok") for i, p in enumerate(periodos)]
    r = _resultado([*filas, (periodos[10], "rara", 95.0, "ok")])
    datos = graficador._aplanar_resultado(r)
    grafica = graficador._construir_grafica_linea(datos, r)
    assert "geom_point" in _geoms(grafica)
    figura = grafica.draw()
    try:
        assert len(figura.axes[0].collections) > 0, "el punto de la serie solitaria no se dibujó"
    finally:
        import matplotlib.pyplot as plt

        plt.close(figura)


def test_construir_grafica_renderiza_puntos_con_un_solo_dato_por_serie() -> None:
    r = _resultado([(_P1, "A", 90.0, "ok"), (_P1, "B", 110.0, "ok")])
    datos = graficador._aplanar_resultado(r)
    grafica = graficador._construir_grafica_linea(datos, r)
    figura = grafica.draw()
    try:
        ejes = figura.axes[0]
        assert len(ejes.collections) > 0, "geom_point no dibujó nada con 1 dato por serie"
    finally:
        import matplotlib.pyplot as plt

        plt.close(figura)


def test_construir_grafica_conserva_hueco_interior_en_geom_line() -> None:
    # Ene/Mar finitos, Feb sin_datos: el grupo sobrevive a
    # _descartar_grupos_no_graficables (algún valor finito), y el NaN interior
    # llega intacto a geom_line -- matplotlib corta el trazo solo ahí, sin que
    # el código tenga que reconstruir el hueco a mano.
    r = _resultado(
        [
            (_P1, "cat", 100.0, "ok"),
            (_P2, "cat", float("nan"), "sin_datos"),
            (_P3, "cat", 110.0, "ok"),
        ]
    )
    datos = graficador._descartar_grupos_no_graficables(
        graficador._aplanar_resultado(r), "indice_replicado"
    )
    grafica = graficador._construir_grafica_linea(datos, r)
    figura = grafica.draw()
    try:
        linea = next(
            line for line in figura.axes[0].lines if len(np.asarray(line.get_xydata())) == 3
        )
        y = np.asarray(linea.get_xydata())[:, 1]
        assert y[0] == pytest.approx(100.0)
        assert np.isnan(y[1])
        assert y[2] == pytest.approx(110.0)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figura)


def test_construir_grafica_serie_larga_con_unico_valor_finito_aparece_como_punto() -> None:
    # 23 sin_datos + 1 valor real: cuenta como "una observación" (finita), no
    # como 24 -- sin el fix, el conteo crudo de filas no la trataba como
    # solitaria y el único dato real quedaba sin punto ni línea, invisible.
    periodos = _meses(24)
    filas = [(p, "cat", float("nan"), "sin_datos") for p in periodos]
    filas[10] = (periodos[10], "cat", 95.0, "ok")
    r = _resultado(filas)
    datos = graficador._descartar_grupos_no_graficables(
        graficador._aplanar_resultado(r), "indice_replicado"
    )
    grafica = graficador._construir_grafica_linea(datos, r)
    assert "geom_point" in _geoms(grafica)
    figura = grafica.draw()
    try:
        assert len(figura.axes[0].collections) > 0, "el único valor finito no se dibujó"
    finally:
        import matplotlib.pyplot as plt

        plt.close(figura)


def test_construir_grafica_no_conecta_mediciones_heterogeneas_con_mismo_codigo() -> None:
    # SECTOR/produccion_total "11" en enero y SUBSECTOR/exportaciones "11" en
    # febrero -- mediciones sin relación. Cada una queda como su propia serie
    # de 1 punto (desambiguadas en _aplanar_resultado): geom_line no dibuja
    # nada con un solo punto por grupo, así que ninguna línea de >=2 puntos
    # debería aparecer conectándolas.
    r = _resultado_heterogeneo(
        [
            (_P1, "SECTOR", "produccion_total", "11", 100.0),
            (_P2, "SUBSECTOR", "exportaciones", "11", 200.0),
        ]
    )
    datos = graficador._descartar_grupos_no_graficables(
        graficador._aplanar_resultado(r), "indice_replicado"
    )
    assert datos["indice"].nunique() == 2
    grafica = graficador._construir_grafica_linea(datos, r)
    figura = grafica.draw()
    try:
        lineas_conectadas = [
            ln for ln in figura.axes[0].lines if len(np.asarray(ln.get_xydata())) >= 2
        ]
        assert not lineas_conectadas, "no debería conectar mediciones sin relación"
    finally:
        import matplotlib.pyplot as plt

        plt.close(figura)


def test_construir_grafica_anota_en_periodo_del_valor_finito_no_en_extremo_del_eje() -> None:
    import matplotlib.dates as mdates

    r = _resultado(
        [
            (_P1, "cat", float("nan"), "sin_datos"),
            (_P2, "cat", 100.0, "ok"),
            (_P3, "cat", 110.0, "ok"),
        ]
    )
    datos = graficador._descartar_grupos_no_graficables(
        graficador._aplanar_resultado(r), "indice_replicado"
    )
    grafica = graficador._construir_grafica_linea(datos, r)
    figura = grafica.draw()
    try:
        texto = next(t for t in figura.axes[0].texts if t.get_text() == "100.00")
        x_texto = texto.get_position()[0]
        x_periodo_real = mdates.date2num(
            datos.loc[datos["indice_replicado"] == 100.0, "periodo_ts"].iloc[0]
        )
        x_extremo_eje_ene_nan = mdates.date2num(datos["periodo_ts"].min())
        assert x_texto == pytest.approx(x_periodo_real)
        assert x_texto != pytest.approx(x_extremo_eje_ene_nan)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figura)


def test_construir_grafica_incluye_hline_si_100_en_rango() -> None:
    r = _resultado([(_P1, "cat", 90.0, "ok"), (_P2, "cat", 110.0, "ok")])
    datos = graficador._aplanar_resultado(r)
    grafica = graficador._construir_grafica_linea(datos, r)
    assert "geom_hline" in _geoms(grafica)


def test_construir_grafica_sin_hline_si_100_fuera_de_rango() -> None:
    r = _resultado([(_P1, "cat", 120.0, "ok"), (_P2, "cat", 150.0, "ok")])
    datos = graficador._aplanar_resultado(r)
    grafica = graficador._construir_grafica_linea(datos, r)
    assert "geom_hline" not in _geoms(grafica)


def test_construir_grafica_anota_texto_con_una_sola_serie() -> None:
    r = _resultado([(_P1, "cat", 100.0, "ok"), (_P2, "cat", 110.0, "ok")])
    datos = graficador._aplanar_resultado(r)
    grafica = graficador._construir_grafica_linea(datos, r)
    assert "geom_text" in _geoms(grafica)


def test_construir_grafica_no_anota_texto_con_varias_series() -> None:
    r = _resultado([(_P1, "cat_a", 90.0, "ok"), (_P1, "cat_b", 110.0, "ok")])
    datos = graficador._aplanar_resultado(r)
    grafica = graficador._construir_grafica_linea(datos, r)
    assert "geom_text" not in _geoms(grafica)


def test_construir_grafica_titulo_junta_combinaciones_de_resultado_y_comparacion() -> None:
    r = _resultado([(_P1, "cat", 90.0, "ok")], agregacion="SECTOR", rubro="produccion_total")
    comparacion = _resultado([(_P1, "INPP", 100.0, "ok")], agregacion="INPP", rubro="exportaciones")
    datos = graficador._aplanar_resultado(r, comparacion)
    grafica = graficador._construir_grafica_linea(datos, r)
    assert grafica.labels.title == (
        "SECTOR Produccion total con petróleo + INPP Exportaciones con petróleo"
    )


def test_construir_grafica_agregacion_inpp_siempre_negro() -> None:
    # Sin variedad por rubro: cualquier línea de agregacion=="INPP" es negra,
    # sea cual sea el rubro -- distinguir "produccion_total" de "exportaciones"
    # acá depende únicamente de la leyenda de linetype (Resultado/Comparación),
    # no de color.
    r = _resultado([(_P1, "INPP", 100.0, "ok")], agregacion="INPP", rubro="produccion_total")
    comparacion = _resultado([(_P1, "INPP", 95.0, "ok")], agregacion="INPP", rubro="exportaciones")
    datos = graficador._aplanar_resultado(r, comparacion)
    assert set(datos["indice"]) == {"produccion_total", "exportaciones"}
    colores, _ = graficador._colores_y_etiquetas(list(pd.unique(datos["indice"])))
    assert colores == {"produccion_total": "black", "exportaciones": "black"}


def _escala_linetype(grafica: Any) -> Any:
    return next(e for e in grafica.scales if "linetype" in e.aesthetics)


def test_construir_grafica_leyenda_linetype_cuando_indice_colisiona() -> None:
    # Mismo rubro con/sin petróleo: mismo `indice` en resultado y comparacion
    # (1 sola serie) -- sin la leyenda de linetype, no habría forma de saber
    # cuál línea es cuál.
    r = _resultado([(_P1, "INPP", 100.0, "ok")], agregacion="INPP", rubro="produccion_total")
    comparacion = _resultado(
        [(_P1, "INPP", 95.0, "ok")], agregacion="INPP", rubro="produccion_total"
    )
    datos = graficador._aplanar_resultado(r, comparacion)
    grafica = graficador._construir_grafica_linea(datos, r)
    theme_dict = grafica.theme.themeables
    assert theme_dict["legend_position"].properties["value"] == "bottom"
    assert _escala_linetype(grafica).guide == "legend"


def test_construir_grafica_leyenda_linetype_describe_la_combinacion_no_un_rol_generico() -> None:
    # La etiqueta tiene que decir QUÉ distingue a cada línea (ej. con/sin
    # petróleo) -- un rol genérico ("Resultado"/"Comparación") no dice nada
    # cuando lo único que cambia es eso.
    r = _resultado([(_P1, "INPP", 100.0, "ok")], agregacion="INPP", rubro="produccion_total")
    comparacion = _resultado(
        [(_P1, "INPP", 95.0, "ok")],
        agregacion="INPP",
        rubro="produccion_total",
        incluir_petroleo=False,
    )
    datos = graficador._aplanar_resultado(r, comparacion)
    grafica = graficador._construir_grafica_linea(datos, r)
    etiquetas = _escala_linetype(grafica).labels
    assert etiquetas == {
        "solid": "INPP Produccion total con petróleo",
        "dashed": "INPP Produccion total sin petróleo",
    }


def test_construir_grafica_sin_leyenda_linetype_cuando_color_ya_distingue() -> None:
    # SECTOR desglosado (muchas categorías, cada una su color) + INPP como
    # comparación (su propio color negro, con etiqueta propia): no colisionan
    # en `indice`, así que un rol genérico "Resultado"/"Comparación" no
    # aportaría nada -- el color y su leyenda ya lo dicen todo.
    r = _resultado(
        [(_P1, "11 Agricultura", 90.0, "ok"), (_P1, "21 Minería", 110.0, "ok")],
        agregacion="SECTOR",
    )
    comparacion = _resultado(
        [(_P1, "INPP", 100.0, "ok")], agregacion="INPP", rubro="produccion_total"
    )
    datos = graficador._aplanar_resultado(r, comparacion)
    grafica = graficador._construir_grafica_linea(datos, r)
    assert _escala_linetype(grafica).guide is None


# --------------------------------------------------------------------------- graficar (índices)


def test_graficar_indice_una_sola_imagen_bajo_capacidad(mocker: Any) -> None:
    r = _resultado(_n_categorias(5))
    grafica_falsa = mocker.Mock()
    mocker.patch.object(graficador, "_construir_grafica_linea", return_value=grafica_falsa)
    graficador.graficar(r)
    grafica_falsa.draw.assert_called_once_with(show=True)


def test_graficar_indice_particiona_en_varias_imagenes(mocker: Any) -> None:
    r = _resultado(_n_categorias(13))
    grafica_falsa = mocker.Mock()
    mocker.patch.object(graficador, "_construir_grafica_linea", return_value=grafica_falsa)
    graficador.graficar(r)
    assert grafica_falsa.draw.call_count == 2
    grafica_falsa.draw.assert_called_with(show=True)


def test_graficar_indice_reserva_colores_estables_entre_paneles(mocker: Any) -> None:
    # Protege la integración real (graficador.graficar -> _graficar_indice)
    # contra una regresión que sustituya `colores_reservados` por `{}` en el
    # wiring -- los tests que reconstruyen manualmente
    # _reservar_colores_comparacion/_particionar_series/_colores_y_etiquetas
    # (test_prepocesamiento.py) no pasan por acá, así que no la detectarían.
    r = _resultado([(_P1, f"s{i:02d}", float(i), "ok") for i in range(9)], agregacion="SECTOR")
    comparacion = _resultado(
        [(_P1, f"m{i:02d}", float(i), "ok") for i in range(3)], agregacion="MERCANCIAS_SERVICIOS"
    )
    construir = mocker.patch.object(
        graficador, "_construir_grafica_linea", return_value=mocker.Mock()
    )
    graficador.graficar(r, comparacion=comparacion)
    assert construir.call_count == 2
    mapas = [llamada.kwargs["colores_reservados"] for llamada in construir.call_args_list]
    assert mapas[0] == mapas[1]
    # No basta con "igual y no vacío": ese par pasa igual con una reserva
    # PARCIAL (ej. solo m00) -- el mismo dict incompleto se pasa idéntico a
    # ambos paneles por construcción, sin que eso implique que m01/m02 no
    # sigan cambiando de color (se calculan local, fuera de lo reservado).
    assert set(mapas[0]) == {"m00", "m01", "m02"}
    assert len(set(mapas[0].values())) == 3


def test_graficar_indice_recorta_tramo_antes_de_graficar(mocker: Any) -> None:
    r = _resultado(
        [(_P1, "cat", 100.0, "ok"), (_P2, "cat", 105.0, "ok"), (_P3, "cat", 110.0, "ok")]
    )
    construir = mocker.patch.object(
        graficador, "_construir_grafica_linea", return_value=mocker.Mock()
    )
    graficador.graficar(r, desde=_P2)
    datos_recibidos = construir.call_args[0][0]
    assert set(datos_recibidos["periodo"]) == {_P2, _P3}


def test_graficar_dibuja_cada_grafica_para_indice(mocker: Any) -> None:
    r = _resultado([(_P1, "cat", 100.0, "ok")])
    grafica_falsa = mocker.Mock()
    construir = mocker.patch.object(
        graficador, "_construir_grafica_linea", return_value=grafica_falsa
    )
    graficador.graficar(r)
    grafica_falsa.draw.assert_called_once_with(show=True)
    construir.assert_called_once_with(mocker.ANY, r, colores_reservados={})


def test_graficar_indice_sin_ningun_valor_finito_lanza_invariante_violado() -> None:
    r = _resultado(
        [
            (_P1, "cat", float("nan"), "sin_datos"),
            (_P2, "cat", float("nan"), "sin_datos"),
            (_P3, "cat", float("nan"), "sin_datos"),
        ]
    )
    with pytest.raises(InvarianteViolado):
        graficador.graficar(r)


def test_graficar_tipo_invalido_no_lanza(mocker: Any, capsys: Any) -> None:
    construir = mocker.patch.object(graficador, "_construir_grafica_linea")
    graficador.graficar("no es un ResultadoIndice ni ResultadoVariacion")  # type: ignore[arg-type]
    assert "Error" in capsys.readouterr().out
    construir.assert_not_called()


def test_graficar_comparacion_de_otro_tipo_no_lanza(mocker: Any, capsys: Any) -> None:
    construir = mocker.patch.object(graficador, "_construir_grafica_linea")
    r = _resultado([(_P1, "cat", 100.0, "ok")])
    rv = _resultado_variacion([(_P1, "cat", 0.5, "ok")])
    graficador.graficar(r, comparacion=rv)
    assert "Error" in capsys.readouterr().out
    construir.assert_not_called()


# --------------------------------------------------------------------------- helpers variaciones


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


def _n_categorias_variacion(n: int) -> list[tuple[Any, str, float, str]]:
    return [(_P1, f"cat{i:02d}", float(i) / 100.0, "ok") for i in range(n)]


def _construir_grafica_variacion(datos: pd.DataFrame, resultado: ResultadoVariacion) -> Any:
    """Helper de test: `_construir_grafica_linea` con los kwargs que usa `_graficar_variacion`."""
    return graficador._construir_grafica_linea(
        datos,
        resultado,
        columna_valor="variacion_pp",
        valor_base=0.0,
        etiqueta_y="Variación (pp)",
    )


# --------------------------------------------------------------------------- _construir_grafica_linea (variaciones)


def test_construir_grafica_variacion_mapea_y_a_variacion_pp() -> None:
    rv = _resultado_variacion([(_P1, "cat", 0.5, "ok")])
    datos = graficador._aplanar_resultado(rv)
    grafica = _construir_grafica_variacion(datos, rv)
    assert grafica.mapping["y"] == "variacion_pp"


def test_construir_grafica_variacion_incluye_hline_si_0_en_rango() -> None:
    rv = _resultado_variacion([(_P1, "cat", -1.0, "ok"), (_P2, "cat", 1.0, "ok")])
    datos = graficador._aplanar_resultado(rv)
    grafica = _construir_grafica_variacion(datos, rv)
    assert "geom_hline" in _geoms(grafica)


def test_construir_grafica_variacion_sin_hline_si_0_fuera_de_rango() -> None:
    rv = _resultado_variacion([(_P1, "cat", 5.0, "ok"), (_P2, "cat", 8.0, "ok")])
    datos = graficador._aplanar_resultado(rv)
    grafica = _construir_grafica_variacion(datos, rv)
    assert "geom_hline" not in _geoms(grafica)


def test_construir_grafica_variacion_etiqueta_y() -> None:
    rv = _resultado_variacion([(_P1, "cat", 0.5, "ok")])
    datos = graficador._aplanar_resultado(rv)
    grafica = _construir_grafica_variacion(datos, rv)
    assert grafica.labels.y == "Variación (pp)"


def test_construir_grafica_variacion_titulo_sin_petroleo() -> None:
    principal = _resultado_variacion([(_P1, "cat", 0.5, "ok")], agregacion="SECTOR")
    datos = graficador._aplanar_resultado(principal)
    grafica = _construir_grafica_variacion(datos, principal)
    assert grafica.labels.title == "SECTOR Produccion total"


# --------------------------------------------------------------------------- graficar (variaciones)


def test_graficar_variacion_sin_ningun_valor_finito_lanza_invariante_violado() -> None:
    rv = _resultado_variacion([(_P1, "cat", float("nan"), "ok"), (_P2, "cat", float("nan"), "ok")])
    with pytest.raises(InvarianteViolado):
        graficador.graficar(rv)


def test_graficar_variacion_una_sola_imagen_bajo_capacidad(mocker: Any) -> None:
    rv = _resultado_variacion(_n_categorias_variacion(5))
    grafica_falsa = mocker.Mock()
    mocker.patch.object(graficador, "_construir_grafica_linea", return_value=grafica_falsa)
    graficador.graficar(rv)
    grafica_falsa.draw.assert_called_once_with(show=True)


def test_graficar_variacion_particiona_en_varias_imagenes(mocker: Any) -> None:
    rv = _resultado_variacion(_n_categorias_variacion(13))
    grafica_falsa = mocker.Mock()
    mocker.patch.object(graficador, "_construir_grafica_linea", return_value=grafica_falsa)
    graficador.graficar(rv)
    assert grafica_falsa.draw.call_count == 2
    grafica_falsa.draw.assert_called_with(show=True)


def test_graficar_variacion_reserva_colores_estables_entre_paneles(mocker: Any) -> None:
    # Contraparte de variaciones del test de índices -- mismo wiring
    # (_graficar_variacion), misma protección de integración.
    rv = _resultado_variacion(
        [(_P1, f"s{i:02d}", float(i), "ok") for i in range(9)], agregacion="SECTOR"
    )
    comparacion = _resultado_variacion(
        [(_P1, f"m{i:02d}", float(i), "ok") for i in range(3)], agregacion="MERCANCIAS_SERVICIOS"
    )
    construir = mocker.patch.object(
        graficador, "_construir_grafica_linea", return_value=mocker.Mock()
    )
    graficador.graficar(rv, comparacion=comparacion)
    assert construir.call_count == 2
    mapas = [llamada.kwargs["colores_reservados"] for llamada in construir.call_args_list]
    assert mapas[0] == mapas[1]
    assert set(mapas[0]) == {"m00", "m01", "m02"}
    assert len(set(mapas[0].values())) == 3


def test_graficar_variacion_recorta_tramo_antes_de_graficar(mocker: Any) -> None:
    rv = _resultado_variacion(
        [(_P1, "cat", 0.1, "ok"), (_P2, "cat", 0.2, "ok"), (_P3, "cat", 0.3, "ok")]
    )
    construir = mocker.patch.object(
        graficador, "_construir_grafica_linea", return_value=mocker.Mock()
    )
    graficador.graficar(rv, desde=_P2)
    datos_recibidos = construir.call_args[0][0]
    assert set(datos_recibidos["periodo"]) == {_P2, _P3}


def test_graficar_dibuja_cada_grafica_para_variacion(mocker: Any) -> None:
    rv = _resultado_variacion([(_P1, "cat", 0.5, "ok")])
    grafica_falsa = mocker.Mock()
    construir = mocker.patch.object(
        graficador, "_construir_grafica_linea", return_value=grafica_falsa
    )
    graficador.graficar(rv)
    grafica_falsa.draw.assert_called_once_with(show=True)
    construir.assert_called_once_with(
        mocker.ANY,
        rv,
        columna_valor="variacion_pp",
        valor_base=0.0,
        etiqueta_y="Variación (pp)",
        colores_reservados={},
    )


def test_graficar_variacion_comparacion_con_otra_clase_no_lanza(mocker: Any, capsys: Any) -> None:
    construir = mocker.patch.object(graficador, "_construir_grafica_linea")
    rv = _resultado_variacion([(_P1, "cat", 0.5, "ok")], clase="periodica_mensual")
    comparacion = _resultado_variacion([(_P1, "cat", 0.3, "ok")], clase="periodica_trimestral")
    graficador.graficar(rv, comparacion=comparacion)
    assert "Error" in capsys.readouterr().out
    construir.assert_not_called()


def test_graficar_variacion_comparacion_misma_clase_dibuja(mocker: Any) -> None:
    rv = _resultado_variacion([(_P1, "cat", 0.5, "ok")], clase="periodica_mensual")
    comparacion = _resultado_variacion([(_P1, "cat2", 0.3, "ok")], clase="periodica_mensual")
    grafica_falsa = mocker.Mock()
    mocker.patch.object(graficador, "_construir_grafica_linea", return_value=grafica_falsa)
    graficador.graficar(rv, comparacion=comparacion)
    grafica_falsa.draw.assert_called_once_with(show=True)


# --------------------------------------------------------------------------- regresiones


def test_construir_grafica_linea_no_emite_warning_con_series_solitarias() -> None:
    import warnings

    r = _resultado([(_P1, "A", 90.0, "ok"), (_P1, "B", 110.0, "ok")])
    datos = graficador._aplanar_resultado(r)
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        figura = graficador._construir_grafica_linea(datos, r).draw()
        import matplotlib.pyplot as plt

        plt.close(figura)
    assert not [w for w in capturados if "only one observation" in str(w.message)]


def test_graficar_con_nan_terminal_no_emite_warning_bajo_dash_w_error() -> None:
    # Protege el flujo público (graficador.graficar) contra una regresión que
    # reintroduzca na_rm=False: bajo -W error, un PlotnineWarning abortaría la
    # llamada -- un NaN terminal (ej. Ene sin_datos) es un caso real, no
    # sintético (LaspeyresDirecto lo produce cuando el histórico empieza sin
    # cobertura). Filtra solo `PlotnineWarning` como error (no TODO warning,
    # que -W error a secas haría) -- `graficar()` llama `.draw(show=True)`,
    # y bajo el backend Agg sin display eso emite su propio `UserWarning`
    # ("FigureCanvasAgg is non-interactive") sin relación con este hallazgo.
    import warnings

    from plotnine.exceptions import PlotnineWarning

    r = _resultado(
        [
            (_P1, "cat", float("nan"), "sin_datos"),
            (_P2, "cat", 100.0, "ok"),
            (_P3, "cat", 110.0, "ok"),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("error", category=PlotnineWarning)
        graficador.graficar(r)  # no debe lanzar
