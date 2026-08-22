from __future__ import annotations

from pathlib import Path

import pytest
from canasta_inpp.esquema import (
    COL_ENCADENAMIENTO_EXPORTACION,
    COL_ENCADENAMIENTO_PRODUCCION_NACIONAL,
    COL_ENCADENAMIENTO_TOTAL,
    COL_ENCADENAMIENTO_USO_FINAL,
    HOJA_ENCADENAMIENTO,
    LAYOUTS_XLSX,
    LayoutXlsx,
    VersionCanastaScian,
)

# openpyxl vive en el extra "ponderadores", no en "dev" -- sin esto, un
# checkout que solo instaló `.[dev]` rompe la RECOLECCIÓN de todo pytest
# (ImportError a nivel de módulo), no solo estos tests.
openpyxl = pytest.importorskip("openpyxl")

pytestmark = pytest.mark.requires_data

_REPO_ROOT = Path(__file__).resolve().parents[3]

_RUTAS_PONDERADORES: dict[VersionCanastaScian, Path] = {
    2012: _REPO_ROOT / "docs/requerimientos/xlsx/2012/ponderadores_inpp_inegi_2012.xlsx",
    2019: _REPO_ROOT
    / "docs/requerimientos/xlsx/2019"
    / "COU_2017_Estructura_de_ponderaciones_PR_Julio_2019_2_Agosto_2019.xlsx",
    2025: _REPO_ROOT / "docs/requerimientos/xlsx/2025/ponderadores_inpp_2025.xlsx",
}

# el archivo de encadenamiento solo existe para 2025 -- ver esquema.py y
# docs/requerimientos/explicacion_encadenamiento.md
_RUTA_ENCADENAMIENTO = _REPO_ROOT / "docs/requerimientos/xlsx/2025/factor_de_encadenamiento_ti.xlsx"

# `requires_data` solo ETIQUETA -- no omite nada por sí sola (hace falta
# correr pytest con -m "not requires_data" para excluirla, documentado en
# CLAUDE.md, pero nada lo fuerza: no hay CI en este repo). Sin este guardia,
# un checkout limpio sin los xlsx locales (docs/requerimientos está
# gitignoreado) rompe con FileNotFoundError crudo en vez de saltarse limpio.
_RUTAS_TODAS = [*_RUTAS_PONDERADORES.values(), _RUTA_ENCADENAMIENTO]
_FALTANTES = [str(r) for r in _RUTAS_TODAS if not r.exists()]
if _FALTANTES:
    pytest.skip(
        f"faltan xlsx reales para requires_data (docs/requerimientos gitignoreado): {_FALTANTES}",
        allow_module_level=True,
    )

_VERSIONES: tuple[VersionCanastaScian, ...] = (2012, 2019, 2025)
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


@pytest.mark.parametrize("version", _VERSIONES)
def test_las_5_hojas_existen_con_el_nombre_esperado(version: VersionCanastaScian) -> None:
    layout = LAYOUTS_XLSX[version]
    wb = openpyxl.load_workbook(_RUTAS_PONDERADORES[version], data_only=True, read_only=True)
    for campo in _HOJAS:
        nombre_hoja = getattr(layout, campo)
        assert nombre_hoja in wb.sheetnames, f"{version}: falta hoja {campo!r}={nombre_hoja!r}"


@pytest.mark.parametrize("version", _VERSIONES)
def test_codigo_g_es_unico_en_produccion_total(version: VersionCanastaScian) -> None:
    layout = LAYOUTS_XLSX[version]
    filas = _codigos_y_pesos(
        _RUTAS_PONDERADORES[version], layout.hoja_produccion_total, layout, [layout.col_peso_simple]
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
def test_suma_de_peso_simple_ronda_100(version: VersionCanastaScian, campo_hoja: str) -> None:
    # verificación independiente de los valores hardcodeados en
    # test_esquema.py: si col_peso_simple apuntara a la columna equivocada
    # (ej. por un xlsx reemplazado con otra estructura), la suma de una
    # columna que no es de ponderador casi seguro NO daría ~100.
    layout = LAYOUTS_XLSX[version]
    hoja = getattr(layout, campo_hoja)
    filas = _codigos_y_pesos(_RUTAS_PONDERADORES[version], hoja, layout, [layout.col_peso_simple])
    suma = sum(peso for _, (peso,) in filas)
    assert suma == pytest.approx(100.0, abs=0.01), f"{version}/{hoja}: suma={suma}"


@pytest.mark.parametrize("version", _VERSIONES)
def test_suma_de_demanda_interna_ronda_100_en_las_3_columnas(version: VersionCanastaScian) -> None:
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
    version: VersionCanastaScian,
) -> None:
    layout = LAYOUTS_XLSX[version]
    wb = openpyxl.load_workbook(_RUTAS_PONDERADORES[version], data_only=True, read_only=True)
    ws = wb[layout.hoja_produccion_total]
    vistos = 0
    for row in ws.iter_rows(values_only=True):
        if len(row) <= layout.col_actividad:
            continue
        if not _es_codigo_generico(row[layout.col_g]):
            continue
        assert isinstance(row[layout.col_actividad], str) and row[layout.col_actividad].strip()
        vistos += 1
    assert vistos > 0


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


# -- demanda interna: texto de encabezado, no solo suma ----------------------
# (la suma sola no distingue consumo de capital -- ambas columnas suman ~100
# por separado, un swap entre las dos no lo detecta ningún test de suma)


@pytest.mark.parametrize("version", _VERSIONES)
def test_encabezados_de_demanda_interna_distinguen_consumo_de_capital(
    version: VersionCanastaScian,
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
    # encabezado leído en la posición de consumo diría "CAPITAL", no "CONSUMO"
    assert len({encabezado_total, encabezado_consumo, encabezado_capital}) == 3


# -- archivo de encadenamiento (solo 2025) -----------------------------------


def test_hoja_de_encadenamiento_existe() -> None:
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
    col: int, fragmento_esperado: str
) -> None:
    encabezado = _buscar_encabezado(_RUTA_ENCADENAMIENTO, HOJA_ENCADENAMIENTO, col, max_filas=15)
    assert fragmento_esperado in encabezado


def test_encadenamiento_valores_son_numericos_o_na() -> None:
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


def test_encadenamiento_total_cubre_el_universo_completo_sin_na() -> None:
    # confirmado en esta sesión: la columna "Producción Total" no tiene
    # ningún "N/A" -- cubre el universo completo de genéricos, a diferencia
    # de exportación/uso final que sí son subconjuntos
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
