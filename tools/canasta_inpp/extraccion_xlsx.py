from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import openpyxl
import pandas as pd
from openpyxl.worksheet.worksheet import Worksheet

from canasta_inpp.esquema import (
    COL_ENCADENAMIENTO_EXPORTACION,
    COL_ENCADENAMIENTO_GENERICO,
    COL_ENCADENAMIENTO_PRODUCCION_NACIONAL,
    COL_ENCADENAMIENTO_TOTAL,
    COL_ENCADENAMIENTO_USO_FINAL,
    HOJA_ENCADENAMIENTO,
    LAYOUTS_CANASTA,
    LAYOUTS_XLSX,
    LayoutCanasta,
    LayoutXlsx,
    VersionCanastaScian,
)

_COLUMNAS_PONDERADORES: tuple[str, ...] = (
    "generico",
    "codigo",
    "sector",
    "subsector",
    "rama",
    "subrama",
    "clase",
    "produccion total",
    "bienes intermedios",
    "bienes finales",
    "demanda interna total",
    "demanda interna consumo",
    "demanda interna capital",
    "exportaciones",
)

_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _es_codigo_generico(valor: object) -> bool:
    """True si `valor` es un código de genérico real (dígitos), no la etiqueta del header."""
    if isinstance(valor, int):
        return True
    if isinstance(valor, str):
        return valor.strip().isdigit()
    return False


def _nombre_archivo_hoja(zf: zipfile.ZipFile, nombre_hoja: str) -> str:
    """Resuelve un nombre de hoja al `sheetN.xml` correspondiente (Target relativo o absoluto)."""
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rid = next(
        s.get(f"{{{_NS_REL}}}id")
        for s in workbook.iter(f"{{{_NS_MAIN}}}sheet")
        if s.get("name") == nombre_hoja
    )
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    destino = next(
        r.get("Target") for r in rels.iter(f"{{{_NS_PKG_REL}}}Relationship") if r.get("Id") == rid
    )
    assert destino is not None, f"relación {rid!r} sin atributo Target"
    if destino.startswith("/"):
        return destino.lstrip("/")
    return f"xl/{destino}"


def _valores_crudos(ruta: Path, nombre_hoja: str) -> dict[str, str]:
    """Lee el texto crudo (sin parsear a float) de las celdas numéricas de una hoja."""
    with zipfile.ZipFile(ruta) as zf:
        xml = zf.read(_nombre_archivo_hoja(zf, nombre_hoja))

    crudos: dict[str, str] = {}
    for celda in ET.fromstring(xml).iter(f"{{{_NS_MAIN}}}c"):
        tipo = celda.get("t")
        if tipo is not None and tipo != "n":
            continue
        ref = celda.get("r")
        valor = celda.find(f"{{{_NS_MAIN}}}v")
        if ref is not None and valor is not None and valor.text is not None:
            crudos[ref] = valor.text
    return crudos


def _leer_hoja_peso(
    ruta: Path,
    hoja: str,
    layout: LayoutXlsx,
    columnas_peso: dict[int, str],
    *,
    incluir_jerarquia: bool = False,
) -> pd.DataFrame:
    """Lee una hoja de ponderadores, filas de genérico real, indexadas por `codigo`."""
    wb = openpyxl.load_workbook(ruta, data_only=True)
    ws: Worksheet = wb[hoja]
    crudos = _valores_crudos(ruta, hoja)

    filas: list[dict[str, object]] = []
    for row in ws.iter_rows():
        if len(row) <= layout.col_actividad:
            continue
        codigo_crudo = row[layout.col_g].value
        if not _es_codigo_generico(codigo_crudo):
            continue

        fila: dict[str, object] = {"codigo": str(codigo_crudo).strip().zfill(3)}
        for col, nombre in columnas_peso.items():
            celda = row[col]
            valor_crudo = crudos.get(celda.coordinate)
            fila[nombre] = valor_crudo if valor_crudo is not None else celda.value
        if incluir_jerarquia:
            fila["generico"] = row[layout.col_actividad].value
            fila["sector"] = str(row[layout.col_s].value)
            fila["subsector"] = str(row[layout.col_sb].value)
            fila["rama"] = str(row[layout.col_r].value)
            fila["subrama"] = str(row[layout.col_sr].value)
            fila["clase"] = str(row[layout.col_c].value)
        filas.append(fila)

    df = pd.DataFrame(filas).set_index("codigo")
    if not df.index.is_unique:
        duplicados = sorted(df.index[df.index.duplicated()].unique())
        raise ValueError(f"'{hoja}' trae código(s) de genérico duplicado(s): {duplicados}")
    return df


def extraer_ponderadores(ruta: Path, version: VersionCanastaScian) -> pd.DataFrame:
    """Une las 5 hojas del xlsx de ponderadores en una sola tabla, una fila por genérico.

    Args:
        ruta: Ruta al xlsx de ponderadores (layout definido en `esquema.LAYOUTS_XLSX`).
        version: Versión de canasta -- determina el layout de columnas/hojas a leer.

    Returns:
        DataFrame con columnas `generico`, `codigo`, `sector`, `subsector`, `rama`,
        `subrama`, `clase` (código bare, sin nombre -- el nombre completo lo agrega
        `extraer_canasta`), y las 7 columnas de peso (`produccion total`,
        `bienes intermedios`, `bienes finales`, `demanda interna total/consumo/
        capital`, `exportaciones`) como texto crudo del xlsx (precisión exacta,
        sin castear a float).

    Raises:
        ValueError: Si alguna hoja trae código de genérico duplicado, o si el
            conjunto de códigos de una hoja no coincide con el de la hoja ancla
            (producción total).
    """
    layout = LAYOUTS_XLSX[version]

    ancla = _leer_hoja_peso(
        ruta,
        layout.hoja_produccion_total,
        layout,
        {layout.col_peso_simple: "produccion total"},
        incluir_jerarquia=True,
    )
    bienes_intermedios = _leer_hoja_peso(
        ruta, layout.hoja_bienes_intermedios, layout, {layout.col_peso_simple: "bienes intermedios"}
    )
    bienes_finales = _leer_hoja_peso(
        ruta, layout.hoja_bienes_finales, layout, {layout.col_peso_simple: "bienes finales"}
    )
    demanda_interna = _leer_hoja_peso(
        ruta,
        layout.hoja_demanda_interna,
        layout,
        {
            layout.col_peso_demanda_total: "demanda interna total",
            layout.col_peso_demanda_consumo: "demanda interna consumo",
            layout.col_peso_demanda_capital: "demanda interna capital",
        },
    )
    exportaciones = _leer_hoja_peso(
        ruta, layout.hoja_exportaciones, layout, {layout.col_peso_simple: "exportaciones"}
    )

    codigos_ancla = set(ancla.index)
    partes = (
        (bienes_intermedios, layout.hoja_bienes_intermedios),
        (bienes_finales, layout.hoja_bienes_finales),
        (demanda_interna, layout.hoja_demanda_interna),
        (exportaciones, layout.hoja_exportaciones),
    )
    for parte, hoja in partes:
        codigos_parte = set(parte.index)
        faltantes = codigos_ancla - codigos_parte
        sobrantes = codigos_parte - codigos_ancla
        if faltantes or sobrantes:
            raise ValueError(
                f"'{hoja}' no coincide con el universo de genéricos de "
                f"'{layout.hoja_produccion_total}' -- faltan {sorted(faltantes)}, "
                f"sobran {sorted(sobrantes)}"
            )

    resultado = ancla
    for parte, _hoja in partes:
        resultado = resultado.join(parte, how="left", validate="one_to_one")

    return resultado.reset_index()[list(_COLUMNAS_PONDERADORES)]


_NIVELES_JERARQUIA: tuple[str, ...] = ("sector", "subsector", "rama", "subrama", "clase")

_COLUMNAS_CANASTA: tuple[str, ...] = ("generico", "codigo", *_NIVELES_JERARQUIA)

# cota defensiva contra el "rango fantasma" que reporta openpyxl en algunos
# xlsx de INEGI (dimensions declara >1M filas aunque los datos reales
# terminan mucho antes) -- corta tras N filas en blanco consecutivas.
_MAX_FILAS_VACIAS_CONSECUTIVAS = 50

# nota al pie estándar de INEGI, cae en la posición de una columna de
# jerarquía -- sin filtrarla el state machine la confunde con un nivel válido.
_PATRON_NOTA_PIE = re.compile(r"^[a-z]/\s")


def _es_fila_nota_pie(fila: tuple[object, ...]) -> bool:
    """True si `fila` es una nota al pie (ver `_PATRON_NOTA_PIE`), no un dato real."""
    return any(isinstance(valor, str) and _PATRON_NOTA_PIE.match(valor.strip()) for valor in fila)


def _texto_nivel(layout: LayoutCanasta, nivel: str, fila: tuple[object, ...]) -> str | None:
    """Texto "código nombre" combinado de un nivel de jerarquía en `fila`, o `None` si vacío."""
    col_codigo = getattr(layout, f"col_{nivel}")
    valor_codigo = fila[col_codigo] if col_codigo < len(fila) else None
    if valor_codigo is None:
        return None

    col_nombre = getattr(layout, f"col_{nivel}_nombre")
    if col_nombre is None:
        return str(valor_codigo).strip()

    valor_nombre = fila[col_nombre] if col_nombre < len(fila) else None
    if valor_nombre is None:
        return str(valor_codigo).strip()
    return f"{valor_codigo} {str(valor_nombre).strip()}"


def _es_fila_generico(layout: LayoutCanasta, fila: tuple[object, ...]) -> bool:
    """True si `fila` es una fila de genérico (no de nivel de jerarquía).

    2012: el código de genérico comparte columna con Clase -- se distingue por
    tipo (`int` en fila de genérico, `str` en fila de Clase).
    """
    valor = fila[layout.col_codigo_generico]
    if layout.codigo_generico_en_columna_clase:
        return isinstance(valor, int)
    return valor is not None and str(valor).strip().isdigit()


def extraer_canasta(ruta: Path, version: VersionCanastaScian) -> pd.DataFrame:
    """Lee el xlsx de árbol SCIAN (canasta) y arma una fila por genérico con su jerarquía.

    Columnas devueltas: `generico`, `codigo`, `sector`, `subsector`, `rama`,
    `subrama`, `clase` -- `sector`..`clase` traen código+nombre combinado.
    Lanza `ValueError` si el xlsx trae código de genérico duplicado.
    """
    layout = LAYOUTS_CANASTA[version]
    wb = openpyxl.load_workbook(ruta, data_only=True)
    ws: Worksheet = wb[layout.hoja]

    max_col = max(
        layout.col_codigo_generico,
        layout.col_nombre_generico,
        layout.col_sector,
        layout.col_subsector,
        layout.col_rama,
        layout.col_subrama,
        layout.col_clase,
    )

    estado: dict[str, str] = dict.fromkeys(_NIVELES_JERARQUIA, "")
    filas: list[dict[str, object]] = []
    vio_datos = False
    filas_vacias_consecutivas = 0

    for fila in ws.iter_rows(min_row=layout.fila_datos_inicio, values_only=True):
        if len(fila) <= max_col or all(valor is None for valor in fila):
            if vio_datos:
                filas_vacias_consecutivas += 1
                if filas_vacias_consecutivas > _MAX_FILAS_VACIAS_CONSECUTIVAS:
                    break
            continue
        filas_vacias_consecutivas = 0
        vio_datos = True

        if _es_fila_nota_pie(fila):
            continue

        if _es_fila_generico(layout, fila):
            nombre_generico = fila[layout.col_nombre_generico]
            if nombre_generico is None:
                continue
            filas.append(
                {
                    "generico": str(nombre_generico).strip(),
                    "codigo": str(fila[layout.col_codigo_generico]).strip().zfill(3),
                    **estado,
                }
            )
            continue

        for nivel in _NIVELES_JERARQUIA:
            texto = _texto_nivel(layout, nivel, fila)
            if texto is not None:
                estado[nivel] = texto
                break

    df = pd.DataFrame(filas, columns=list(_COLUMNAS_CANASTA))
    if not df["codigo"].is_unique:
        duplicados = sorted(df.loc[df["codigo"].duplicated(), "codigo"].unique())
        raise ValueError(
            f"'{layout.hoja}' ({ruta.name}) trae código(s) de genérico duplicado(s): {duplicados}"
        )
    return df


_COLUMNAS_ENCADENAMIENTO_FACTOR: dict[int, str] = {
    COL_ENCADENAMIENTO_TOTAL: "encadenamiento total",
    COL_ENCADENAMIENTO_PRODUCCION_NACIONAL: "encadenamiento produccion nacional",
    COL_ENCADENAMIENTO_EXPORTACION: "encadenamiento exportacion",
    COL_ENCADENAMIENTO_USO_FINAL: "encadenamiento uso final",
}


def extraer_encadenamiento(ruta: Path) -> pd.DataFrame:
    """Lee el xlsx de factor de encadenamiento (solo 2025) -- una fila por genérico.

    Columnas: `codigo`, `encadenamiento total`, `encadenamiento produccion nacional`,
    `encadenamiento exportacion`, `encadenamiento uso final` (texto crudo). `"N/A"` se
    convierte a `NaN` real. Lanza `ValueError` si trae código duplicado.
    """
    wb = openpyxl.load_workbook(ruta, data_only=True)
    ws: Worksheet = wb[HOJA_ENCADENAMIENTO]
    crudos = _valores_crudos(ruta, HOJA_ENCADENAMIENTO)

    filas: list[dict[str, object]] = []
    for row in ws.iter_rows():
        if len(row) <= COL_ENCADENAMIENTO_USO_FINAL:
            continue
        codigo_crudo = row[COL_ENCADENAMIENTO_GENERICO].value
        if not _es_codigo_generico(codigo_crudo):
            continue

        fila: dict[str, object] = {"codigo": str(codigo_crudo).strip().zfill(3)}
        for col, nombre in _COLUMNAS_ENCADENAMIENTO_FACTOR.items():
            celda = row[col]
            if celda.value == "N/A":
                fila[nombre] = float("nan")
                continue
            valor_crudo = crudos.get(celda.coordinate)
            fila[nombre] = valor_crudo if valor_crudo is not None else celda.value
        filas.append(fila)

    df = pd.DataFrame(filas, columns=["codigo", *_COLUMNAS_ENCADENAMIENTO_FACTOR.values()])
    if not df["codigo"].is_unique:
        duplicados = sorted(df.loc[df["codigo"].duplicated(), "codigo"].unique())
        raise ValueError(
            f"'{HOJA_ENCADENAMIENTO}' ({ruta.name}) trae código(s) de genérico "
            f"duplicado(s): {duplicados}"
        )
    return df
