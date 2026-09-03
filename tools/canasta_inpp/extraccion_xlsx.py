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


def _mapa_hojas(ruta: Path) -> dict[str, str]:
    """Nombre de hoja -> ruta interna `sheetN.xml` dentro del xlsx.

    Sin caché propia a propósito: cachear por `ruta` a nivel de módulo mezclaba
    contenido viejo y nuevo si el archivo se reescribía entre dos llamadas sobre
    la misma ruta (bug real, encontrado y revertido en esta misma sesión --
    `extraer_ponderadores` es quien decide si vale la pena reusar el resultado,
    llamando esta función una sola vez y pasándolo a sus 5 lecturas).
    """
    with zipfile.ZipFile(ruta) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))

        destinos_por_id: dict[str, str] = {}
        for r in rels.iter(f"{{{_NS_PKG_REL}}}Relationship"):
            rid, destino = r.get("Id"), r.get("Target")
            assert rid is not None and destino is not None, f"relación sin Id/Target: {r.attrib}"
            destinos_por_id[rid] = destino

        mapa: dict[str, str] = {}
        for s in workbook.iter(f"{{{_NS_MAIN}}}sheet"):
            nombre, rid = s.get("name"), s.get(f"{{{_NS_REL}}}id")
            assert nombre is not None and rid is not None, f"hoja sin name/r:id: {s.attrib}"
            destino = destinos_por_id[rid]
            mapa[nombre] = destino.lstrip("/") if destino.startswith("/") else f"xl/{destino}"
    return mapa


def _valores_crudos(
    ruta: Path, nombre_hoja: str, mapa_hojas: dict[str, str] | None = None
) -> dict[str, str]:
    """Lee el texto crudo (sin parsear a float) de las celdas numéricas de una hoja.

    `mapa_hojas`, si se pasa, evita recalcularlo -- lo usa `_leer_hoja_peso`
    cuando `extraer_ponderadores` ya lo resolvió una vez para sus 5 hojas. Sin
    pasarlo, se recalcula fresco acá mismo (sin caché ni estado que sobreviva a
    esta llamada).
    """
    nombre_archivo = (mapa_hojas if mapa_hojas is not None else _mapa_hojas(ruta))[nombre_hoja]
    with zipfile.ZipFile(ruta) as zf:
        xml = zf.read(nombre_archivo)

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
    wb: openpyxl.Workbook | None = None,
    mapa_hojas: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Lee una hoja de ponderadores, filas de genérico real, indexadas por `codigo`.

    `wb`/`mapa_hojas` son opcionales -- `extraer_ponderadores` los carga una sola
    vez y los reusa en sus 5 llamadas (evita reabrir/reparsear el mismo xlsx 5
    veces). Un llamador directo (tests) los deja en `None` y esta función carga
    los suyos, sin caché ni estado que sobreviva a la llamada.
    """
    if wb is None:
        wb = openpyxl.load_workbook(ruta, data_only=True)
    ws: Worksheet = wb[hoja]
    crudos = _valores_crudos(ruta, hoja, mapa_hojas)

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
    # cargados una sola vez acá y reusados en las 5 lecturas de abajo -- evita
    # reabrir/reparsear el mismo xlsx 5 veces. Viven solo durante esta llamada
    # (parámetros locales, no caché de módulo): no hay riesgo de mezclar
    # contenido viejo/nuevo si `ruta` cambia entre corridas distintas.
    wb = openpyxl.load_workbook(ruta, data_only=True)
    mapa_hojas = _mapa_hojas(ruta)

    ancla = _leer_hoja_peso(
        ruta,
        layout.hoja_produccion_total,
        layout,
        {layout.col_peso_simple: "produccion total"},
        incluir_jerarquia=True,
        wb=wb,
        mapa_hojas=mapa_hojas,
    )
    bienes_intermedios = _leer_hoja_peso(
        ruta,
        layout.hoja_bienes_intermedios,
        layout,
        {layout.col_peso_simple: "bienes intermedios"},
        wb=wb,
        mapa_hojas=mapa_hojas,
    )
    bienes_finales = _leer_hoja_peso(
        ruta,
        layout.hoja_bienes_finales,
        layout,
        {layout.col_peso_simple: "bienes finales"},
        wb=wb,
        mapa_hojas=mapa_hojas,
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
        wb=wb,
        mapa_hojas=mapa_hojas,
    )
    exportaciones = _leer_hoja_peso(
        ruta,
        layout.hoja_exportaciones,
        layout,
        {layout.col_peso_simple: "exportaciones"},
        wb=wb,
        mapa_hojas=mapa_hojas,
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

# SCIAN: cada nivel tiene un largo de código fijo (2/3/4/5/6 dígitos). Se usa
# para decidir a QUÉ nivel pertenece un texto encontrado, en vez de confiar en
# la columna donde cayó -- ver `_nivel_por_digitos`.
_NIVEL_POR_LARGO_CODIGO: dict[int, str] = {
    2: "sector",
    3: "subsector",
    4: "rama",
    5: "subrama",
    6: "clase",
}

# Filas de nivel que faltan por completo en el xlsx de canasta de una versión
# (no es error de columna: el xlsx salta directo al nivel siguiente sin
# escribir esta fila) -- nombre sacado a mano del Sistema de Clasificación
# Industrial de América del Norte (SCIAN), confirmado contra
# `Tabla_de_Correspondencia_SCIAN_2013_SCIAN_2007.xlsx`. 2019: códigos {337,
# 338} (rama 3279) y {373} (subrama 33299). 2012: código {471} (rama 4883,
# fila 1125) y {517} (subrama 56133, fila 1298). Encontrados escaneando los 3
# xlsx completos por continuidad de código, ver `investigacion_recorte_rubro.md`.
# Si aparece un código nuevo no listado acá, `extraer_canasta` rechaza en vez
# de dejar el nombre viejo.
_NOMBRES_NIVEL_FALTANTE: dict[VersionCanastaScian, dict[str, str]] = {
    2012: {
        "4883": "Servicios relacionados con el transporte por agua",
        "56133": "Suministro de personal permanente",
    },
    2019: {
        "3279": "Fabricación de otros productos a base de minerales no metálicos",
        "33299": "Fabricación de otros productos metálicos",
    },
    2025: {},
}

# Código de nivel mal tecleado en el xlsx de canasta (el nombre sí es
# correcto, solo el dígito líder está mal) -- distinto del caso de arriba
# (fila ausente): acá la fila SÍ existe, con el nombre real, pero el código no
# coincide con el de sus hijos. 2012: subrama del código {471} viene como
# "48832" pero sus hijos (clase `488330`, y el propio `codigo` en
# ponderadores_2012.csv) confirman "48833" -- mezcla de numeración SCIAN
# 2007/2013 en la misma celda, confirmado contra la tabla de correspondencia
# (fila 1129: "48832" SÍ es código válido, pero de la OTRA numeración, con
# clase "488320" -- no la que trae esta fila, "488330").
_CORRECCION_CODIGO: dict[VersionCanastaScian, dict[str, str]] = {
    2012: {"48832": "48833"},
    2019: {},
    2025: {},
}


def _corregir_codigo(texto: str, correcciones: dict[str, str]) -> str:
    """Reemplaza el código líder de `texto` si está en `correcciones` (ver `_CORRECCION_CODIGO`).

    Preserva el nombre tal cual viene del xlsx -- solo el código está mal, no
    hace falta (ni conviene) reinventar el nombre.
    """
    codigo, separador, resto = texto.partition(" ")
    codigo_correcto = correcciones.get(codigo)
    if codigo_correcto is None:
        return texto
    return f"{codigo_correcto}{separador}{resto}"


def _nivel_por_digitos(texto: str) -> str | None:
    """Nivel real de `texto` ("código nombre") según el largo de su código, o `None`.

    `None` cuando el código no es puramente numérico (ej. sector agrupado
    `"31-33 Industrias manufactureras"` de 2012) o su largo no es de ningún
    nivel SCIAN -- en ese caso el llamador debe usar la columna como respaldo.
    """
    codigo = texto.partition(" ")[0]
    if not codigo.isdigit():
        return None
    return _NIVEL_POR_LARGO_CODIGO.get(len(codigo))


def _completar_jerarquia_faltante(
    estado: dict[str, str], nombres_faltantes: dict[str, str]
) -> dict[str, str]:
    """Reconstruye niveles cuya fila no existe en el xlsx (ver `_NOMBRES_NIVEL_FALTANTE`).

    Detecta el hueco comparando el código de cada nivel contra el prefijo del
    nivel más profundo inmediato (ej. rama debe ser prefijo de 4 dígitos del
    código de subrama) -- si no coincide, ese nivel se saltó en el xlsx.
    Ignora sector agrupado (código no numérico, ej. `"31-33"`), lo resuelve
    `resolver_sector_agrupado` más adelante en el pipeline.

    Raises:
        ValueError: el nivel saltado no está en `nombres_faltantes` -- nombre
            desconocido, no se puede reconstruir en silencio.
    """
    estado = dict(estado)
    for nivel_padre, nivel_hijo in zip(_NIVELES_JERARQUIA, _NIVELES_JERARQUIA[1:], strict=False):
        codigo_padre = estado[nivel_padre].partition(" ")[0]
        codigo_hijo = estado[nivel_hijo].partition(" ")[0]
        if not (codigo_padre.isdigit() and codigo_hijo.isdigit()):
            continue
        prefijo_esperado = codigo_hijo[: len(codigo_padre)]
        if codigo_padre == prefijo_esperado:
            continue
        nombre = nombres_faltantes.get(prefijo_esperado)
        if nombre is None:
            raise ValueError(
                f"'{nivel_padre}' con código '{prefijo_esperado}' no aparece como fila propia "
                f"en el xlsx (implícito por '{nivel_hijo}'='{estado[nivel_hijo]}') y no está en "
                "la tabla de nombres conocidos -- agregar su nombre real (SCIAN) a "
                "_NOMBRES_NIVEL_FALTANTE_2019 antes de continuar."
            )
        estado[nivel_padre] = f"{prefijo_esperado} {nombre}"
    return estado


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
    nombres_faltantes = _NOMBRES_NIVEL_FALTANTE[version]
    correccion_codigo = _CORRECCION_CODIGO[version]
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
                    **_completar_jerarquia_faltante(estado, nombres_faltantes),
                }
            )
            continue

        for nivel_columna in _NIVELES_JERARQUIA:
            texto = _texto_nivel(layout, nivel_columna, fila)
            if texto is not None:
                texto = _corregir_codigo(texto, correccion_codigo)
                # `nivel_columna` es solo dónde cayó el texto -- el nivel real
                # sale del largo de su código (ver `_nivel_por_digitos`), no de
                # la columna: INEGI a veces mete el texto de un nivel más
                # profundo en la columna del padre cuando este tiene un solo
                # hijo (ej. subsector 511/rama 5111 en 2019, ver
                # investigacion_recorte_rubro.md) y confiar en la columna deja
                # ese nivel con el texto equivocado y el siguiente con basura
                # vieja de una rama distinta.
                nivel_real = _nivel_por_digitos(texto) or nivel_columna
                estado[nivel_real] = texto
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
