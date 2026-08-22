from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import openpyxl
import pandas as pd
from openpyxl.worksheet.worksheet import Worksheet

from canasta_inpp.esquema import LAYOUTS_XLSX, LayoutXlsx, VersionCanastaScian

_COLUMNAS_PONDERADORES: tuple[str, ...] = (
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
)

_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _es_codigo_generico(valor: object) -> bool:
    """True si `valor` es un código de genérico real (dígitos), no la etiqueta del header.

    2012 parte el header en 2 filas -- la segunda trae solo el texto literal
    'G-11' en la posición de col_g, sin nada más en la fila (confirmado con
    el xlsx real). `col_g is not None` sola confunde esa fila con un
    genérico; hace falta filtrar que sea numérico/dígitos. Mismo filtro que
    tests/unit/tools/test_esquema.py -- acá es la implementación real, ahí
    solo se reusa para verificar contra el xlsx.
    """
    if isinstance(valor, int):
        return True
    if isinstance(valor, str):
        return valor.strip().isdigit()
    return False


def _nombre_archivo_hoja(zf: zipfile.ZipFile, nombre_hoja: str) -> str:
    """Resuelve un nombre de hoja (ej. "ProduccionTotal") al `sheetN.xml` correspondiente.

    El `Target` de la relación puede venir relativo a `xl/` (`"worksheets/
    sheet3.xml"`, así lo escriben los xlsx reales de INEGI/Excel) o absoluto
    desde la raíz del paquete (`"/xl/worksheets/sheet1.xml"`, así lo escribe
    `openpyxl` al guardar -- confirmado armando un xlsx de prueba con
    `openpyxl.Workbook()`). Ambas formas son válidas en el estándar OOXML;
    hay que resolver las 2, no asumir una sola.
    """
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
    """Lee el texto crudo (sin parsear a float) de las celdas numéricas de una hoja.

    `openpyxl` parsea a `float` y no siempre preserva la representación
    exacta del XML (notación científica se vuelve decimal, etc.) -- se usa
    solo para las columnas de peso, donde perder un decimal sí importa (el
    resto -- texto, código, jerarquía -- usa el valor ya parseado, ahí la
    precisión no importa). Portado de
    `replica-inpc-mx/tools/canasta_inpc/extraccion_xlsx.py::_valores_crudos`.
    """
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
    """Lee una hoja de ponderadores, filas de genérico real, indexadas por `codigo`.

    `columnas_peso` mapea posición de columna (en la fila cruda de openpyxl)
    a nombre final de columna -- una hoja de peso simple pasa un solo par
    (produccion_total/bienes_intermedios/bienes_finales/exportaciones); la
    hoja de demanda interna pasa 3 (total/consumo/capital). El peso se
    guarda como el texto crudo del XML (`_valores_crudos`), no el `float`
    de `openpyxl` -- mismo criterio que `guardar_csv` de replica-inpc-mx:
    todos los decimales tal cual vienen en el xlsx, sin redondear.

    `incluir_jerarquia` solo hace falta en la hoja ancla (producción total):
    las 5 hojas repiten los mismos códigos S/SB/R/SR/C/generico para el
    mismo genérico -- confirmado con los xlsx reales de 2012/2019/2025 --
    así que no hace falta releerlos de cada hoja.
    """
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

    Columnas devueltas: `generico`, `codigo`, `sector`, `subsector`, `rama`,
    `subrama`, `clase`, `produccion_total`, `bienes_intermedios`,
    `bienes_finales`, `demanda_interna_total`, `demanda_interna_consumo`,
    `demanda_interna_capital`, `exportaciones` -- subconjunto de
    `esquema.COLUMNAS_BASE` (falta `encadenamiento_*`, que sale de
    `--encadenamientos` y solo aplica a 2025; y `sector`/`subsector`/etc.
    acá son código bare, sin nombre -- el nombre completo lo agrega
    `--canasta`, todavía sin implementar).

    Las columnas de peso vienen como `str` (texto crudo del xlsx, precisión
    exacta) -- para operar numéricamente hace falta castear (`.astype(float)`
    o `pd.to_numeric`), igual que hace `dominio/calculo` en replica-inpc-mx
    con `canasta.df["ponderador"]`.

    `sector` no arranca en "producción total" nomás por convención -- esa
    hoja es la única que nunca tiene ceros (universo completo de genéricos,
    confirmado con los 3 xlsx reales), así que sirve de ancla para el resto.
    """
    layout = LAYOUTS_XLSX[version]

    ancla = _leer_hoja_peso(
        ruta,
        layout.hoja_produccion_total,
        layout,
        {layout.col_peso_simple: "produccion_total"},
        incluir_jerarquia=True,
    )
    bienes_intermedios = _leer_hoja_peso(
        ruta, layout.hoja_bienes_intermedios, layout, {layout.col_peso_simple: "bienes_intermedios"}
    )
    bienes_finales = _leer_hoja_peso(
        ruta, layout.hoja_bienes_finales, layout, {layout.col_peso_simple: "bienes_finales"}
    )
    demanda_interna = _leer_hoja_peso(
        ruta,
        layout.hoja_demanda_interna,
        layout,
        {
            layout.col_peso_demanda_total: "demanda_interna_total",
            layout.col_peso_demanda_consumo: "demanda_interna_consumo",
            layout.col_peso_demanda_capital: "demanda_interna_capital",
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
