from __future__ import annotations

from datetime import datetime
from pathlib import Path

from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import (
    AGREGACION_A_COLUMNA,
    AGREGACION_A_NOMBRE,
    AGREGACIONES_ESPECIALES,
    AGREGACIONES_VALIDAS,
    CODIGO_PETROLEO_CRUDO,
    COLUMNA_ENCADENAMIENTO_POR_RECORTE,
    RANGOS_CANASTAS,
    RUBRO_A_COLUMNA_PESO,
    RUBROS_POR_RECORTE,
    RUBROS_VALIDOS,
    SECTORES_MERCANCIAS,
    TIPO_INPP,
    ManifestCalculo,
)

# -- TIPO_INPP --


def test_tipo_inpp_es_inpp_mayuscula() -> None:
    assert TIPO_INPP == "INPP"


# -- AGREGACION_A_COLUMNA / AGREGACIONES_VALIDAS --


def test_agregacion_a_columna_contenido_exacto() -> None:
    assert AGREGACION_A_COLUMNA == {
        "SECTOR": "codigo sector",
        "S": "codigo sector",
        "SUBSECTOR": "codigo subsector",
        "SB": "codigo subsector",
        "RAMA": "codigo rama",
        "R": "codigo rama",
        "SUBRAMA": "codigo subrama",
        "SR": "codigo subrama",
        "CLASE": "codigo clase",
        "C": "codigo clase",
    }


# (negociación 2026-08-28, H10): sin estos 2 tests, `AGREGACION_A_NOMBRE` no
# tenía ninguna cobertura directa -- solo se ejercitaba indirecto vía
# `test_agregacion_abreviatura_se_normaliza_a_nombre_canonico` en
# test_calculo_laspeyres_directo.py, que no protege sus claves ni su contenido.


def test_agregacion_a_nombre_contenido_exacto() -> None:
    assert AGREGACION_A_NOMBRE == {
        "SECTOR": "SECTOR",
        "S": "SECTOR",
        "SUBSECTOR": "SUBSECTOR",
        "SB": "SUBSECTOR",
        "RAMA": "RAMA",
        "R": "RAMA",
        "SUBRAMA": "SUBRAMA",
        "SR": "SUBRAMA",
        "CLASE": "CLASE",
        "C": "CLASE",
    }


def test_agregacion_a_nombre_mismas_claves_que_columna() -> None:
    assert set(AGREGACION_A_NOMBRE) == set(AGREGACION_A_COLUMNA)


def test_agregaciones_especiales_contenido_exacto() -> None:
    assert AGREGACIONES_ESPECIALES == {TIPO_INPP, "MERCANCIAS_SERVICIOS"}


def test_agregaciones_validas_es_union_de_columna_y_especiales() -> None:
    assert AGREGACIONES_VALIDAS == set(AGREGACION_A_COLUMNA) | AGREGACIONES_ESPECIALES


# -- RUBRO_A_COLUMNA_PESO / RUBROS_VALIDOS --


def test_rubro_a_columna_peso_contenido_exacto() -> None:
    assert RUBRO_A_COLUMNA_PESO == {
        "produccion_total": "produccion total",
        "bienes_intermedios": "bienes intermedios",
        "bienes_finales": "bienes finales",
        "demanda_interna_total": "demanda interna total",
        "demanda_interna_consumo": "demanda interna consumo",
        "demanda_interna_capital": "demanda interna capital",
        "exportaciones": "exportaciones",
    }


def test_rubros_validos_coincide_con_llaves_del_mapeo() -> None:
    assert RUBROS_VALIDOS == set(RUBRO_A_COLUMNA_PESO)


# -- RUBROS_POR_RECORTE --


def test_rubros_por_recorte_cubre_los_4_recortes() -> None:
    assert set(RUBROS_POR_RECORTE) == {
        "produccion_total",
        "mercado_nacional",
        "bienes_finales",
        "mercado_exportacion",
    }


def test_rubros_por_recorte_mercado_nacional_es_ambiguo() -> None:
    # produccion_total ya no es ambiguo (bienes_intermedios se movió a
    # mercado_nacional, 2026-08-30) -- ver dominio/tipos.py::RUBROS_POR_RECORTE.
    assert RUBROS_POR_RECORTE["produccion_total"] == {"produccion_total"}
    assert RUBROS_POR_RECORTE["mercado_nacional"] == {
        "demanda_interna_total",
        "demanda_interna_consumo",
        "demanda_interna_capital",
        "bienes_intermedios",
    }
    assert RUBROS_POR_RECORTE["bienes_finales"] == {"bienes_finales"}
    assert RUBROS_POR_RECORTE["mercado_exportacion"] == {"exportaciones"}


def test_rubros_por_recorte_todos_los_rubros_son_validos() -> None:
    todos = set().union(*RUBROS_POR_RECORTE.values())
    assert todos <= RUBROS_VALIDOS


# -- COLUMNA_ENCADENAMIENTO_POR_RECORTE --


def test_columna_encadenamiento_por_recorte_contenido_exacto() -> None:
    assert COLUMNA_ENCADENAMIENTO_POR_RECORTE == {
        "produccion_total": "encadenamiento total",
        "mercado_nacional": "encadenamiento produccion nacional",
        "mercado_exportacion": "encadenamiento exportacion",
        "bienes_finales": "encadenamiento uso final",
    }


def test_columna_encadenamiento_por_recorte_cubre_los_4_recortes() -> None:
    assert set(COLUMNA_ENCADENAMIENTO_POR_RECORTE) == set(RUBROS_POR_RECORTE)


# -- SECTORES_MERCANCIAS --


def test_sectores_mercancias_contenido_exacto() -> None:
    # "31-33" (combinado) NUNCA aparece en `codigo sector` real -- la canasta ya
    # trae 31/32/33 resueltos por separado (`resolver_sector_agrupado`). Ver
    # comentario de `SECTORES_MERCANCIAS` en tipos.py: con "31-33" el grupo
    # "Mercancías" perdía TODO el peso de manufacturas (20.98 en vez de 66.46585,
    # bug encontrado comparando contra INEGI con datos reales de 2019).
    assert SECTORES_MERCANCIAS == {"11", "21", "22", "23", "31", "32", "33"}


# -- CODIGO_PETROLEO_CRUDO --


def test_codigo_petroleo_crudo() -> None:
    assert CODIGO_PETROLEO_CRUDO == "070"


# -- RANGOS_CANASTAS --


def test_rangos_canastas_cubre_las_3_versiones() -> None:
    assert set(RANGOS_CANASTAS) == {2012, 2019, 2025}


def test_rangos_canastas_limites_exactos() -> None:
    assert RANGOS_CANASTAS[2012] == (PeriodoMensual(2012, 6), PeriodoMensual(2019, 7))
    assert RANGOS_CANASTAS[2019] == (PeriodoMensual(2019, 7), PeriodoMensual(2025, 7))
    assert RANGOS_CANASTAS[2025] == (PeriodoMensual(2025, 7), None)


def test_rangos_canastas_juntas_son_continuas() -> None:
    assert RANGOS_CANASTAS[2012][1] == RANGOS_CANASTAS[2019][0]
    assert RANGOS_CANASTAS[2019][1] == RANGOS_CANASTAS[2025][0]


def test_rangos_canastas_ultima_version_sin_fin() -> None:
    assert RANGOS_CANASTAS[2025][1] is None


# -- ManifestCalculo --


def test_manifest_calculo_construccion_valida() -> None:
    m = ManifestCalculo(
        version=2019,
        agregacion="INPP",
        rubro="produccion_total",
        sin_petroleo=False,
        calculador="LaspeyresDirecto",
        ruta_canasta=Path("/tmp/c.csv"),
        ruta_series=Path("/tmp/s.csv"),
        fecha=datetime(2019, 7, 1),
    )
    assert m.version == 2019
    assert m.agregacion == "INPP"
    assert m.rubro == "produccion_total"
    assert m.sin_petroleo is False


def test_manifest_calculo_rutas_y_fecha_por_defecto() -> None:
    antes = datetime.now()
    m = ManifestCalculo(
        version=2019,
        agregacion="INPP",
        rubro="produccion_total",
        sin_petroleo=False,
        calculador="LaspeyresDirecto",
    )
    despues = datetime.now()
    assert m.ruta_canasta is None
    assert m.ruta_series is None
    assert antes <= m.fecha <= despues
