from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from canasta_inpp.esquema import (
    COL_ENCADENAMIENTO_EXPORTACION,
    COL_ENCADENAMIENTO_PRODUCCION_NACIONAL,
    COL_ENCADENAMIENTO_TOTAL,
    COL_ENCADENAMIENTO_USO_FINAL,
    COLUMNAS_BASE,
    HOJA_ENCADENAMIENTO,
    LAYOUTS_XLSX,
    LayoutXlsx,
    VersionCanastaScian,
)

# openpyxl vive en el extra "ponderadores", no en "dev" -- try/except (no
# `pytest.importorskip`) para que la falta de la librería solo salte
# `TestContraXlsxReales` (más abajo), sin tocar los tests puros de config de
# arriba, que no la necesitan.
try:
    import openpyxl
except ImportError:
    openpyxl = None  # type: ignore[assignment]

_VERSIONES: tuple[VersionCanastaScian, ...] = (2012, 2019, 2025)

# snapshot congelado de LAYOUTS_XLSX, verificado fila por fila contra los xlsx
# reales en esta sesión -- protege contra edición accidental (ej. swap de
# col_peso_demanda_consumo/col_peso_demanda_capital) que un chequeo de solo
# tipo/cantidad no detectaría.
_LAYOUTS_ESPERADOS: dict[VersionCanastaScian, LayoutXlsx] = {
    2012: LayoutXlsx(
        hoja_produccion_total="Producción Total",
        hoja_bienes_intermedios="Bienes Intermedios ",
        hoja_bienes_finales="Bienes Finales",
        hoja_demanda_interna="Demanda Interna",
        hoja_exportaciones="Exportaciones",
        col_s=1,
        col_sb=2,
        col_r=3,
        col_sr=4,
        col_c=5,
        col_g=6,
        col_actividad=7,
        col_peso_simple=10,
        col_peso_demanda_total=10,
        col_peso_demanda_consumo=11,
        col_peso_demanda_capital=12,
    ),
    2019: LayoutXlsx(
        hoja_produccion_total="ProduccionTotal",
        hoja_bienes_intermedios="BienesIntermedios",
        hoja_bienes_finales="BienesFinales",
        hoja_demanda_interna="DemandaInterna",
        hoja_exportaciones="Exportaciones",
        col_s=1,
        col_sb=2,
        col_r=3,
        col_sr=4,
        col_c=5,
        col_g=6,
        col_actividad=7,
        col_peso_simple=8,
        col_peso_demanda_total=8,
        col_peso_demanda_consumo=9,
        col_peso_demanda_capital=10,
    ),
    2025: LayoutXlsx(
        hoja_produccion_total="ProducciónTotal",
        hoja_bienes_intermedios="BienesIntermedios",
        hoja_bienes_finales="BienesFinales",
        hoja_demanda_interna="DemandaInterna",
        hoja_exportaciones="Exportaciones",
        col_s=1,
        col_sb=2,
        col_r=3,
        col_sr=4,
        col_c=5,
        col_g=6,
        col_actividad=7,
        col_peso_simple=8,
        col_peso_demanda_total=8,
        col_peso_demanda_consumo=9,
        col_peso_demanda_capital=10,
    ),
}


# -- COLUMNAS_BASE -------------------------------------------------------


def test_columnas_base_orden_y_cantidad() -> None:
    assert COLUMNAS_BASE == (
        "generico",
        "codigo",
        "sector",
        "subsector",
        "rama",
        "subrama",
        "clase",
        "produccion_total",
        "bienes_intermedios",
        "bienes_finales",
        "demanda_interna_total",
        "demanda_interna_consumo",
        "demanda_interna_capital",
        "exportaciones",
        "encadenamiento_total",
        "encadenamiento_produccion_nacional",
        "encadenamiento_exportacion",
        "encadenamiento_uso_final",
    )


def test_columnas_base_sin_columna_version() -> None:
    # la versión va como parámetro de carga, no como columna repetida --
    # mismo criterio que replica-inpc-mx
    assert "version" not in COLUMNAS_BASE


# -- LayoutXlsx ------------------------------------------------------------


def test_layouts_xlsx_cubre_exactamente_las_3_versiones() -> None:
    assert set(LAYOUTS_XLSX) == set(_VERSIONES)


def test_layout_xlsx_bloquea_reasignar_un_campo() -> None:
    # setattr(...) en vez de `LAYOUTS_XLSX[2019].col_g = 99`: la asignación
    # directa a un campo de un dataclass frozen es un error ESTÁTICO real
    # (mypy: "Property col_g ... is read-only [misc]"), no solo algo que
    # falla en runtime -- setattr conserva el mismo chequeo en runtime sin
    # introducir ese error de tipo.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(LAYOUTS_XLSX[2019], "col_g", 99)


@pytest.mark.parametrize("version", _VERSIONES)
def test_layout_xlsx_coincide_con_snapshot_esperado(version: VersionCanastaScian) -> None:
    assert LAYOUTS_XLSX[version] == _LAYOUTS_ESPERADOS[version]


@pytest.mark.parametrize("version", _VERSIONES)
def test_columnas_s_a_actividad_son_1_a_7_consecutivas(version: VersionCanastaScian) -> None:
    # S/SB/R/SR/C/G/ACTIVIDAD ECONÓMICA cae en la misma posición en las 3
    # versiones -- confirmado con los xlsx reales, no asumido por similitud
    layout = LAYOUTS_XLSX[version]
    assert [
        layout.col_s,
        layout.col_sb,
        layout.col_r,
        layout.col_sr,
        layout.col_c,
        layout.col_g,
        layout.col_actividad,
    ] == list(range(1, 8))


@pytest.mark.parametrize("version", _VERSIONES)
def test_columnas_demanda_interna_son_consecutivas_al_peso_simple(
    version: VersionCanastaScian,
) -> None:
    # DEMANDA INTERNA trae TOTAL/CONSUMO/FORMACIÓN DE CAPITAL en 3 columnas
    # seguidas, arrancando en col_peso_simple -- confirmado en 2012/2019/2025
    layout = LAYOUTS_XLSX[version]
    assert layout.col_peso_demanda_total == layout.col_peso_simple
    assert layout.col_peso_demanda_consumo == layout.col_peso_simple + 1
    assert layout.col_peso_demanda_capital == layout.col_peso_simple + 2


def test_col_peso_simple_distinto_en_2012_vs_2019_2025() -> None:
    # el hallazgo concreto que motivó tener LayoutXlsx por versión en vez de
    # una posición fija: 2012 parte el header en 2 filas con 2 columnas de
    # separación, 2019/2025 no
    assert LAYOUTS_XLSX[2012].col_peso_simple == 10
    assert LAYOUTS_XLSX[2019].col_peso_simple == 8
    assert LAYOUTS_XLSX[2025].col_peso_simple == 8


# -- archivo de encadenamiento (solo 2025) --------------------------------


def test_hoja_y_columnas_de_encadenamiento() -> None:
    assert HOJA_ENCADENAMIENTO == "FACTOR DE ENCADENAMIENTO"
    assert COL_ENCADENAMIENTO_TOTAL == 8
    assert COL_ENCADENAMIENTO_PRODUCCION_NACIONAL == 10
    assert COL_ENCADENAMIENTO_EXPORTACION == 12
    assert COL_ENCADENAMIENTO_USO_FINAL == 14


def test_columnas_de_encadenamiento_son_pares_consecutivos() -> None:
    # cada columna de factor va seguida de una columna vacía en el xlsx
    # real (confirmado) -- de ahí el salto de 2 en 2 en vez de 1 en 1
    columnas = [
        COL_ENCADENAMIENTO_TOTAL,
        COL_ENCADENAMIENTO_PRODUCCION_NACIONAL,
        COL_ENCADENAMIENTO_EXPORTACION,
        COL_ENCADENAMIENTO_USO_FINAL,
    ]
    assert columnas == list(range(8, 15, 2))


# -- 2003 queda fuera de este módulo, a propósito ---------------------------


def test_2003_no_esta_en_layouts_xlsx() -> None:
    # 2003 usa P/GD/DIV/R/SG/G-03 (previo a SCIAN), no un layout más de este
    # módulo -- VersionCanastaScian ya lo excluye por tipo, esto confirma que
    # tampoco se cuela en tiempo de ejecución (ej. por un typo al ampliar el
    # dict a mano). Diferido a v1.1 -- requiere PDF además de xlsx.
    assert 2003 not in LAYOUTS_XLSX


# ============================================================================
# -- LAYOUTS_XLSX contra los xlsx reales de INEGI ---------------------------
# ============================================================================
#
# Todo lo de arriba compara esquema.py contra SÍ MISMO (un snapshot
# congelado) -- protege contra edición accidental, pero no contra que el
# valor haya estado mal desde el principio, ni contra que INEGI cambie el
# xlsx real. Lo de acá abajo abre los xlsx reales y confirma que las
# posiciones de LAYOUTS_XLSX describen la realidad del archivo fuente, no
# solo la memoria congelada de esquema.py.
#
# requires_data: los xlsx viven en docs/requerimientos/ (gitignoreado, no
# se descargan con el checkout). openpyxl vive en el extra "ponderadores".
# Ninguna de las 2 ausencias debe romper los tests de arriba -- por eso el
# skip está a nivel de clase (pytestmark), no de módulo.

_REPO_ROOT = Path(__file__).resolve().parents[3]

_RUTAS_PONDERADORES: dict[VersionCanastaScian, Path] = {
    2012: _REPO_ROOT / "data/tests/xlsx/2012/ponderadores_inpp_inegi_2012.xlsx",
    2019: _REPO_ROOT
    / "data/tests/xlsx/2019"
    / "COU_2017_Estructura_de_ponderaciones_PR_Julio_2019_2_Agosto_2019.xlsx",
    2025: _REPO_ROOT / "data/tests/xlsx/2025/ponderadores_inpp_2025.xlsx",
}

# el archivo de encadenamiento solo existe para 2025 -- ver esquema.py y
# docs/requerimientos/explicacion_encadenamiento.md
_RUTA_ENCADENAMIENTO = _REPO_ROOT / "data/tests/xlsx/2025/factor_de_encadenamiento_ti.xlsx"

_RUTAS_TODAS = [*_RUTAS_PONDERADORES.values(), _RUTA_ENCADENAMIENTO]
_FALTANTES = [str(r) for r in _RUTAS_TODAS if not r.exists()]

_MOTIVO_SKIP: str | None = None
if openpyxl is None:
    _MOTIVO_SKIP = "openpyxl no instalado (extra 'ponderadores' de pyproject.toml)"
elif _FALTANTES:
    _MOTIVO_SKIP = f"faltan xlsx reales (data/tests gitignoreado): {_FALTANTES}"

_HOJAS = (
    "hoja_produccion_total",
    "hoja_bienes_intermedios",
    "hoja_bienes_finales",
    "hoja_demanda_interna",
    "hoja_exportaciones",
)


def _es_codigo_generico(valor: object) -> bool:
    """True si `valor` es un código de genérico real (dígitos), no la etiqueta del header.

    2012 parte el header en 2 filas -- la segunda trae solo el texto literal
    'G-11' en la posición de col_g, sin nada más en la fila (confirmado con
    el xlsx real). `col_g is not None` sola confunde esa fila con un
    genérico; hace falta filtrar que sea numérico/dígitos.
    """
    if isinstance(valor, int):
        return True
    if isinstance(valor, str):
        return valor.strip().isdigit()
    return False


def _codigos_y_pesos(ruta: Path, hoja: str, layout: LayoutXlsx, cols_peso: list[int]):
    """Filas de genérico real (col_g es un código, no la etiqueta del header): (codigo, [pesos...])."""
    # solo se llama con TestContraXlsxReales corriendo de verdad (pytestmark
    # ya garantiza openpyxl != None en runtime); el assert es para que
    # Pylance/mypy lo sepan también, no una comprobación nueva
    assert openpyxl is not None
    wb = openpyxl.load_workbook(ruta, data_only=True)
    ws = wb[hoja]
    filas = []
    for row in ws.iter_rows(values_only=True):
        if len(row) <= layout.col_actividad:
            continue
        g = row[layout.col_g]
        if not _es_codigo_generico(g):
            continue
        pesos = [row[c] for c in cols_peso]
        if any(not isinstance(p, (int, float)) for p in pesos):
            continue  # no debería pasar tras el filtro de arriba, pero por si acaso
        filas.append((g, pesos))
    return filas


def _buscar_encabezado(ruta: Path, hoja: str, col: int, max_filas: int = 20) -> str:
    """Último texto no vacío en `col` antes de los datos -- ahí vive el sub-encabezado real.

    No asume un número de fila fijo: 2012 parte el header en más filas que
    2019/2025 (confirmado con los xlsx reales), así que contar filas a mano
    sería frágil. Tampoco alcanza con el PRIMER string: la columna del peso
    "simple"/total suele tener 2 niveles de header -- un título general
    (ej. "DEMANDA INTERNAb/", "FACTOR DE ENCADENAMIENTO") seguido del
    sub-encabezado real (ej. "TOTAL", "GENÉRICOS DE PRODUCCIÓN TOTAL") --
    confirmado con los xlsx reales de 2012/2019/2025. Las filas de genérico
    no tienen texto en estas columnas (son numéricas o "N/A"), así que el
    ÚLTIMO string antes de que empiecen los datos es el sub-encabezado.
    """
    assert openpyxl is not None  # ver nota en _codigos_y_pesos
    wb = openpyxl.load_workbook(ruta, data_only=True)
    ws = wb[hoja]
    encontrado: str | None = None
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i >= max_filas:
            break
        if len(row) <= col:
            continue
        valor = row[col]
        if isinstance(valor, str) and valor.strip():
            encontrado = valor.strip().upper()
    if encontrado is None:
        raise AssertionError(f"no se encontró encabezado de texto en {hoja!r} columna {col}")
    return encontrado


class TestContraXlsxReales:
    """Ver el bloque de comentario de arriba -- protege contra que esquema.py
    describa mal el xlsx real, no solo contra que se edite a sí mismo."""

    pytestmark = [
        pytest.mark.requires_data,
        pytest.mark.skipif(_MOTIVO_SKIP is not None, reason=_MOTIVO_SKIP or ""),
    ]

    @pytest.mark.parametrize("version", _VERSIONES)
    def test_las_5_hojas_existen_con_el_nombre_esperado(self, version: VersionCanastaScian) -> None:
        assert openpyxl is not None  # ver nota en _codigos_y_pesos
        layout = LAYOUTS_XLSX[version]
        wb = openpyxl.load_workbook(_RUTAS_PONDERADORES[version], data_only=True, read_only=True)
        for campo in _HOJAS:
            nombre_hoja = getattr(layout, campo)
            assert nombre_hoja in wb.sheetnames, f"{version}: falta hoja {campo!r}={nombre_hoja!r}"

    @pytest.mark.parametrize("version", _VERSIONES)
    def test_codigo_g_es_unico_en_produccion_total(self, version: VersionCanastaScian) -> None:
        layout = LAYOUTS_XLSX[version]
        filas = _codigos_y_pesos(
            _RUTAS_PONDERADORES[version],
            layout.hoja_produccion_total,
            layout,
            [layout.col_peso_simple],
        )
        codigos = [g for g, _ in filas]
        assert len(codigos) > 0
        assert len(codigos) == len(set(codigos))

    @pytest.mark.parametrize(
        ("version", "campo_hoja"),
        [
            (v, campo)
            for v in _VERSIONES
            for campo in (
                "hoja_produccion_total",
                "hoja_bienes_intermedios",
                "hoja_bienes_finales",
                "hoja_exportaciones",
            )
        ],
    )
    def test_suma_de_peso_simple_ronda_100(
        self, version: VersionCanastaScian, campo_hoja: str
    ) -> None:
        # verificación independiente de _LAYOUTS_ESPERADOS: si col_peso_simple
        # apuntara a la columna equivocada (ej. por un xlsx reemplazado con
        # otra estructura), la suma de una columna que no es de ponderador
        # casi seguro NO daría ~100.
        layout = LAYOUTS_XLSX[version]
        hoja = getattr(layout, campo_hoja)
        filas = _codigos_y_pesos(
            _RUTAS_PONDERADORES[version], hoja, layout, [layout.col_peso_simple]
        )
        suma = sum(peso for _, (peso,) in filas)
        assert suma == pytest.approx(100.0, abs=0.01), f"{version}/{hoja}: suma={suma}"

    @pytest.mark.parametrize("version", _VERSIONES)
    def test_suma_de_demanda_interna_ronda_100_en_las_3_columnas(
        self, version: VersionCanastaScian
    ) -> None:
        layout = LAYOUTS_XLSX[version]
        cols = [
            layout.col_peso_demanda_total,
            layout.col_peso_demanda_consumo,
            layout.col_peso_demanda_capital,
        ]
        filas = _codigos_y_pesos(
            _RUTAS_PONDERADORES[version], layout.hoja_demanda_interna, layout, cols
        )
        sumas = [sum(pesos[i] for _, pesos in filas) for i in range(3)]
        for suma in sumas:
            assert suma == pytest.approx(100.0, abs=0.01), f"{version}: sumas={sumas}"

    @pytest.mark.parametrize("version", _VERSIONES)
    def test_col_actividad_trae_texto_no_vacio_en_produccion_total(
        self, version: VersionCanastaScian
    ) -> None:
        assert openpyxl is not None  # ver nota en _codigos_y_pesos
        layout = LAYOUTS_XLSX[version]
        wb = openpyxl.load_workbook(_RUTAS_PONDERADORES[version], data_only=True, read_only=True)
        ws = wb[layout.hoja_produccion_total]
        vistos = 0
        for row in ws.iter_rows(values_only=True):
            if len(row) <= layout.col_actividad:
                continue
            if not _es_codigo_generico(row[layout.col_g]):
                continue
            # variable local (no `row[layout.col_actividad]` repetido): Pylance
            # no narrowea el tipo de una expresión de subíndice a través del
            # `and`, solo el de un nombre simple
            valor_actividad = row[layout.col_actividad]
            assert isinstance(valor_actividad, str) and valor_actividad.strip()
            vistos += 1
        assert vistos > 0

    # -- demanda interna: texto de encabezado, no solo suma ------------------
    # (la suma sola no distingue consumo de capital -- ambas columnas suman
    # ~100 por separado, un swap entre las dos no lo detecta ningún test de
    # suma de arriba)

    @pytest.mark.parametrize("version", _VERSIONES)
    def test_encabezados_de_demanda_interna_distinguen_consumo_de_capital(
        self, version: VersionCanastaScian
    ) -> None:
        layout = LAYOUTS_XLSX[version]
        ruta = _RUTAS_PONDERADORES[version]
        hoja = layout.hoja_demanda_interna

        encabezado_total = _buscar_encabezado(ruta, hoja, layout.col_peso_demanda_total)
        encabezado_consumo = _buscar_encabezado(ruta, hoja, layout.col_peso_demanda_consumo)
        encabezado_capital = _buscar_encabezado(ruta, hoja, layout.col_peso_demanda_capital)

        assert "TOTAL" in encabezado_total
        assert "CONSUMO" in encabezado_consumo
        assert "CAPITAL" in encabezado_capital
        # ninguno debe repetirse -- si hubiera un swap consumo<->capital, el
        # encabezado leído en la posición de consumo diría "CAPITAL", no
        # "CONSUMO"
        assert len({encabezado_total, encabezado_consumo, encabezado_capital}) == 3

    # -- archivo de encadenamiento (solo 2025) -------------------------------

    def test_hoja_de_encadenamiento_existe(self) -> None:
        assert openpyxl is not None  # ver nota en _codigos_y_pesos
        wb = openpyxl.load_workbook(_RUTA_ENCADENAMIENTO, data_only=True, read_only=True)
        assert HOJA_ENCADENAMIENTO in wb.sheetnames

    @pytest.mark.parametrize(
        ("col", "fragmento_esperado"),
        [
            (COL_ENCADENAMIENTO_TOTAL, "PRODUCCIÓN TOTAL"),
            (COL_ENCADENAMIENTO_PRODUCCION_NACIONAL, "PRODUCCIÓN NACIONAL"),
            (COL_ENCADENAMIENTO_EXPORTACION, "EXPORTACIÓN"),
            (COL_ENCADENAMIENTO_USO_FINAL, "USO FINAL"),
        ],
    )
    def test_encabezados_de_encadenamiento_coinciden_con_su_columna(
        self, col: int, fragmento_esperado: str
    ) -> None:
        encabezado = _buscar_encabezado(
            _RUTA_ENCADENAMIENTO, HOJA_ENCADENAMIENTO, col, max_filas=15
        )
        assert fragmento_esperado in encabezado

    def test_encadenamiento_valores_son_numericos_o_na(self) -> None:
        assert openpyxl is not None  # ver nota en _codigos_y_pesos
        wb = openpyxl.load_workbook(_RUTA_ENCADENAMIENTO, data_only=True)
        ws = wb[HOJA_ENCADENAMIENTO]
        cols = [
            COL_ENCADENAMIENTO_TOTAL,
            COL_ENCADENAMIENTO_PRODUCCION_NACIONAL,
            COL_ENCADENAMIENTO_EXPORTACION,
            COL_ENCADENAMIENTO_USO_FINAL,
        ]
        vistos = 0
        for row in ws.iter_rows(values_only=True):
            if len(row) <= COL_ENCADENAMIENTO_USO_FINAL:
                continue
            g = row[6]  # col_g -- mismo layout que ponderadores 2025
            if not _es_codigo_generico(g):
                continue
            for c in cols:
                valor = row[c]
                assert isinstance(valor, (int, float)) or valor == "N/A", (
                    f"genérico {g}, columna {c}: valor inesperado {valor!r}"
                )
            vistos += 1
        assert vistos > 0

    def test_encadenamiento_total_cubre_el_universo_completo_sin_na(self) -> None:
        # confirmado en esta sesión: la columna "Producción Total" no tiene
        # ningún "N/A" -- cubre el universo completo de genéricos, a
        # diferencia de exportación/uso final que sí son subconjuntos
        assert openpyxl is not None  # ver nota en _codigos_y_pesos
        wb = openpyxl.load_workbook(_RUTA_ENCADENAMIENTO, data_only=True)
        ws = wb[HOJA_ENCADENAMIENTO]
        vistos = 0
        for row in ws.iter_rows(values_only=True):
            if len(row) <= COL_ENCADENAMIENTO_TOTAL:
                continue
            g = row[6]
            if not _es_codigo_generico(g):
                continue
            assert isinstance(row[COL_ENCADENAMIENTO_TOTAL], (int, float))
            vistos += 1
        assert vistos > 0
