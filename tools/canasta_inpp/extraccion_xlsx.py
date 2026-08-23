from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import openpyxl
import pandas as pd
from openpyxl.worksheet.worksheet import Worksheet

from canasta_inpp.esquema import (
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


_NIVELES_JERARQUIA: tuple[str, ...] = ("sector", "subsector", "rama", "subrama", "clase")

_COLUMNAS_CANASTA: tuple[str, ...] = ("generico", "codigo", *_NIVELES_JERARQUIA)

# Cota defensiva contra el "rango fantasma" que reporta openpyxl en algunos
# xlsx de INEGI -- `data/tests/xlsx/2012/canasta.xlsx` declara
# `ws.dimensions == "A2:I1048565"` aunque los datos reales terminan en la
# fila 1448 (confirmado). Sin cortar, `iter_rows` recorrería >1M filas
# vacías. Se corta tras N filas en blanco consecutivas, no en una fila fija,
# para no depender de cuántas filas reales tenga cada versión.
_MAX_FILAS_VACIAS_CONSECUTIVAS = 50

# Nota al pie estándar de INEGI (ej. "a/   El número asignado al producto
# genérico corresponde al definido en el Cambio Año Base Julio 2019=100.0
# ...; excepto para aquellos productos genéricos de nueva creación donde se
# asigna el consecutivo siguiente."). Confirmada en 2019 y 2025 (2012 no la
# trae). Cae en la posición de una columna de jerarquía -- subsector en
# 2019, sector en 2025 -- así que sin filtrarla el state machine la
# confundiría con un nombre de nivel válido.
_PATRON_NOTA_PIE = re.compile(r"^[a-z]/\s")


def _es_fila_nota_pie(fila: tuple[object, ...]) -> bool:
    """True si `fila` es una nota al pie (ver `_PATRON_NOTA_PIE`), no un dato real."""
    return any(isinstance(valor, str) and _PATRON_NOTA_PIE.match(valor.strip()) for valor in fila)


def _texto_nivel(layout: LayoutCanasta, nivel: str, fila: tuple[object, ...]) -> str | None:
    """Texto "código nombre" combinado de un nivel de jerarquía en `fila`, o `None` si vacío.

    En 2012/2019 el código y el nombre ya vienen pegados en una sola celda
    de texto (`col_<nivel>_nombre` es `None`) -- se usa tal cual. En 2025
    vienen en columnas separadas y hay que unirlos, con el mismo formato
    `"{codigo} {nombre}"` que ya trae el texto combinado de 2012/2019
    (confirmado carácter por carácter contra el xlsx real de 2012, ej.
    `"11 Agricultura, cría y explotación de animales, aprovechamiento
    forestal, pesca y caza"`).
    """
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

    En 2012 el código de genérico comparte columna con Clase -- se
    distingue por tipo: `int` puro en la fila de genérico, `str` (texto de
    clase, ej. `"111110 Cultivo de soya"`) en la fila de Clase. En
    2019/2025 el código de genérico tiene columna propia, sin ambigüedad de
    tipo -- basta con que la celda traiga dígitos.
    """
    valor = fila[layout.col_codigo_generico]
    if layout.codigo_generico_en_columna_clase:
        return isinstance(valor, int)
    return valor is not None and str(valor).strip().isdigit()


def extraer_canasta(ruta: Path, version: VersionCanastaScian) -> pd.DataFrame:
    """Lee el xlsx de árbol SCIAN (canasta) y arma una fila por genérico con su jerarquía.

    A diferencia de `extraer_ponderadores` (donde cada fila de genérico ya
    trae los 5 códigos S/SB/R/SR/C completos), acá cada fila trae **un solo
    nivel jerárquico a la vez** -- Sector solo en su fila, Subsector solo en
    la suya, etc. -- así que hace falta un state machine: se arrastra el
    último valor visto de cada nivel (`estado`) hasta toparse con una fila
    de genérico, momento en el que se emite una fila con el genérico +
    el estado vigente de los 5 niveles.

    Columnas devueltas: `generico`, `codigo`, `sector`, `subsector`, `rama`,
    `subrama`, `clase` -- `sector`..`clase` traen código+nombre combinado en
    un solo texto (ej. `"11 Agricultura, cría y explotación de animales,
    aprovechamiento forestal, pesca y caza"`), decisión ya tomada de máxima
    densidad de información en el CSV (igual que en `replica-inpc-mx`). El
    código de genérico se guarda como texto de 3 dígitos (`str(...).zfill(3)`),
    mismo formato que usa `extraer_ponderadores` -- necesario para cruzar
    ambas tablas por código de genérico (NO es código SCIAN: ese vive en
    `sector`/`subsector`/`rama`/`subrama`/`clase`; el código de genérico es
    un identificador aparte, 001-570). Compartir formato no garantiza
    igualdad de valores entre los dos catálogos fuente -- confirmado que en
    2019 difieren para un genérico puntual (113 en este xlsx, 114 en el de
    ponderadores), ver CLAUDE.md § tools/canasta_inpp.

    Lanza `ValueError` si el xlsx trae código de genérico duplicado (mismo
    contrato que `extraer_ponderadores`/`_leer_hoja_peso`).
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
