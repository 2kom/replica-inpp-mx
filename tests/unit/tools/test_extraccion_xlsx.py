from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import openpyxl
import pandas as pd
import pytest
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
    VersionCanastaScian,
)
from canasta_inpp.extraccion_xlsx import (
    _COLUMNAS_CANASTA,
    _COLUMNAS_ENCADENAMIENTO_FACTOR,
    _COLUMNAS_PONDERADORES,
    _NS_MAIN,
    _NS_PKG_REL,
    _NS_REL,
    _es_codigo_generico,
    _es_fila_generico,
    _es_fila_nota_pie,
    _leer_hoja_peso,
    _texto_nivel,
    _valores_crudos,
    extraer_canasta,
    extraer_encadenamiento,
    extraer_ponderadores,
)

# -- _es_codigo_generico ------------------------------------------------


@pytest.mark.parametrize(
    ("valor", "esperado"),
    [
        (1, True),
        (123, True),
        ("001", True),
        ("123", True),
        (" 045 ", True),
        ("G-11", False),  # etiqueta del header partido de 2012
        ("Gc/", False),  # etiqueta del header de 2019/2025
        (None, False),
        ("", False),
        ("Total", False),  # fila de agregado
        (1.5, False),  # nunca viene como float en el xlsx real
    ],
)
def test_es_codigo_generico(valor: object, esperado: bool) -> None:
    assert _es_codigo_generico(valor) is esperado


# -- fixtures: xlsx mínimos armados a mano -------------------------------

_LAYOUT_2019 = LAYOUTS_XLSX[2019]


def _fila_generico(codigo: str, nombre: str, peso: float, *pesos_extra: float) -> tuple:
    """Fila de genérico completa (col A vacía + S/SB/R/SR/C/G/ACTIVIDAD/peso...)."""
    return (None, 11, 111, 1111, 11111, 111110, codigo, nombre, peso, *pesos_extra)


def _fila_agregado(peso: float) -> tuple:
    """Fila de agregado (solo Sector, sin G) -- debe quedar filtrada, nunca contarse como genérico."""
    return (None, 11, None, None, None, None, None, "Agricultura", peso)


def _armar_xlsx_ponderadores_2019(tmp_path: Path) -> Path:
    """xlsx con las 5 hojas de 2019, 3 genéricos, un peso DISTINTO por hoja y por
    genérico (10*i/20*i/30*i/40*i/41*i/42*i/50*i) -- para que un bug de join
    (ej. leer el peso de la hoja equivocada) se note en el valor, no solo en
    que "hay un número ahí".
    """
    wb = openpyxl.Workbook()
    hoja_por_defecto = wb.active
    assert hoja_por_defecto is not None
    wb.remove(hoja_por_defecto)

    generico = {"001": "Soya y otras oleaginosas", "002": "Frijol", "003": "Garbanzo"}

    ws = wb.create_sheet(_LAYOUT_2019.hoja_produccion_total)
    ws.append(_fila_agregado(999.0))
    for i, (codigo, nombre) in enumerate(generico.items(), start=1):
        ws.append(_fila_generico(codigo, nombre, 10.0 * i))

    ws = wb.create_sheet(_LAYOUT_2019.hoja_bienes_intermedios)
    for i, codigo in enumerate(generico, start=1):
        ws.append(_fila_generico(codigo, generico[codigo], 20.0 * i))

    ws = wb.create_sheet(_LAYOUT_2019.hoja_bienes_finales)
    for i, codigo in enumerate(generico, start=1):
        ws.append(_fila_generico(codigo, generico[codigo], 30.0 * i))

    ws = wb.create_sheet(_LAYOUT_2019.hoja_demanda_interna)
    for i, codigo in enumerate(generico, start=1):
        ws.append(_fila_generico(codigo, generico[codigo], 40.0 * i, 41.0 * i, 42.0 * i))

    ws = wb.create_sheet(_LAYOUT_2019.hoja_exportaciones)
    for i, codigo in enumerate(generico, start=1):
        ws.append(_fila_generico(codigo, generico[codigo], 50.0 * i))

    ruta = tmp_path / "ponderadores_2019.xlsx"
    wb.save(ruta)
    return ruta


# -- extraer_ponderadores: wiring completo -------------------------------


def test_extraer_ponderadores_forma_y_orden_de_columnas(tmp_path: Path) -> None:
    ruta = _armar_xlsx_ponderadores_2019(tmp_path)
    df = extraer_ponderadores(ruta, 2019)

    assert len(df) == 3  # la fila de agregado NO debe colarse
    assert list(df.columns) == list(_COLUMNAS_PONDERADORES)
    assert list(df["codigo"]) == ["001", "002", "003"]
    assert list(df["generico"]) == ["Soya y otras oleaginosas", "Frijol", "Garbanzo"]


def test_extraer_ponderadores_jerarquia_sale_de_la_hoja_ancla(tmp_path: Path) -> None:
    ruta = _armar_xlsx_ponderadores_2019(tmp_path)
    df = extraer_ponderadores(ruta, 2019).set_index("codigo")

    # `.at` (celda única) en vez de `.loc[codigo]["col"]" (fila completa,
    # cuyo tipo es "Series | DataFrame" para el checker -- no puede probar
    # que "codigo" es único, aunque acá sí lo es por construcción)
    assert df.at["001", "sector"] == "11"
    assert df.at["001", "subsector"] == "111"
    assert df.at["001", "rama"] == "1111"
    assert df.at["001", "subrama"] == "11111"
    assert df.at["001", "clase"] == "111110"


def test_extraer_ponderadores_cada_columna_de_peso_viene_de_su_propia_hoja(
    tmp_path: Path,
) -> None:
    # el chequeo central del wiring: cada hoja aporta un peso DISTINTO -- si
    # el join estuviera mal (ej. todas las columnas leyendo produccion_total)
    # este test lo detecta, a diferencia de "hay un número parecido a 100"
    ruta = _armar_xlsx_ponderadores_2019(tmp_path)
    df = extraer_ponderadores(ruta, 2019).set_index("codigo")

    for i, codigo in enumerate(["001", "002", "003"], start=1):
        assert float(str(df.at[codigo, "produccion total"])) == pytest.approx(10.0 * i)
        assert float(str(df.at[codigo, "bienes intermedios"])) == pytest.approx(20.0 * i)
        assert float(str(df.at[codigo, "bienes finales"])) == pytest.approx(30.0 * i)
        assert float(str(df.at[codigo, "demanda interna total"])) == pytest.approx(40.0 * i)
        assert float(str(df.at[codigo, "demanda interna consumo"])) == pytest.approx(41.0 * i)
        assert float(str(df.at[codigo, "demanda interna capital"])) == pytest.approx(42.0 * i)
        assert float(str(df.at[codigo, "exportaciones"])) == pytest.approx(50.0 * i)


def test_extraer_ponderadores_peso_es_texto_no_float(tmp_path: Path) -> None:
    # el peso se guarda como texto crudo (precisión exacta), no como el
    # float ya parseado por openpyxl -- ver _valores_crudos
    ruta = _armar_xlsx_ponderadores_2019(tmp_path)
    df = extraer_ponderadores(ruta, 2019)
    for columna in (
        "produccion total",
        "bienes intermedios",
        "bienes finales",
        "demanda interna total",
        "demanda interna consumo",
        "demanda interna capital",
        "exportaciones",
    ):
        assert df[columna].dtype == object
        assert all(isinstance(v, str) for v in df[columna])


# -- _leer_hoja_peso: la posición de columna depende del layout, no está hardcodeada --


def test_leer_hoja_peso_usa_la_columna_del_layout_no_una_fija(tmp_path: Path) -> None:
    # 2012 tiene el peso en la columna 10, no en la 8 (confirmado con el
    # xlsx real, ver esquema.py) -- si `_leer_hoja_peso` tuviera la posición
    # hardcodeada en vez de leerla de `layout`, este test lo detecta
    layout_2012 = LAYOUTS_XLSX[2012]
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = layout_2012.hoja_produccion_total
    fila: list[object] = [None] * 11
    fila[1], fila[2], fila[3], fila[4], fila[5] = 11, 111, 1111, 11111, 111110
    fila[6] = "001"
    fila[7] = "Soya y otras oleaginosas"
    fila[10] = 12.5  # col_peso_simple de 2012, no la 8
    ws.append(fila)
    ruta = tmp_path / "ponderadores_2012.xlsx"
    wb.save(ruta)

    df = _leer_hoja_peso(
        ruta,
        layout_2012.hoja_produccion_total,
        layout_2012,
        {layout_2012.col_peso_simple: "produccion total"},
        incluir_jerarquia=True,
    )
    assert float(str(df.at["001", "produccion total"])) == pytest.approx(12.5)


# -- _valores_crudos: solo celdas numéricas, mapeadas por coordenada -----


def test_valores_crudos_solo_incluye_celdas_numericas(tmp_path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Hoja1"
    ws["A1"] = "un texto"  # no debe aparecer en crudos
    ws["B1"] = 3.14159265358979
    ruta = tmp_path / "prueba.xlsx"
    wb.save(ruta)

    crudos = _valores_crudos(ruta, "Hoja1")

    assert "A1" not in crudos
    assert "B1" in crudos
    assert float(crudos["B1"]) == pytest.approx(3.14159265358979)


def _forzar_valor_crudo(ruta: Path, hoja: str, celda: str, texto_crudo: str) -> None:
    """Reescribe el `<v>` de una celda ya guardada con un texto crudo elegido a mano.

    `openpyxl.Workbook.save` nunca escribiría algo como
    `"5.9196304989416844E-3"` -- normaliza el número al guardar. Para probar
    que `extraer_ponderadores` preserva el texto crudo EXACTO (no
    `str(valor_ya_parseado_por_openpyxl)`), hay que editar el XML ya
    guardado directo, sin volver a pasar por `openpyxl`.
    """
    with zipfile.ZipFile(ruta) as zf:
        contenidos = {nombre: zf.read(nombre) for nombre in zf.namelist()}

    workbook = ET.fromstring(contenidos["xl/workbook.xml"])
    rid = next(
        s.get(f"{{{_NS_REL}}}id")
        for s in workbook.iter(f"{{{_NS_MAIN}}}sheet")
        if s.get("name") == hoja
    )
    rels = ET.fromstring(contenidos["xl/_rels/workbook.xml.rels"])
    destino = next(
        r.get("Target") for r in rels.iter(f"{{{_NS_PKG_REL}}}Relationship") if r.get("Id") == rid
    )
    assert destino is not None
    nombre_hoja_xml = destino.lstrip("/") if destino.startswith("/") else f"xl/{destino}"

    hoja_xml = ET.fromstring(contenidos[nombre_hoja_xml])
    celda_el = next(c for c in hoja_xml.iter(f"{{{_NS_MAIN}}}c") if c.get("r") == celda)
    valor_el = celda_el.find(f"{{{_NS_MAIN}}}v")
    assert valor_el is not None, f"celda {celda!r} no tiene <v>"
    valor_el.text = texto_crudo
    contenidos[nombre_hoja_xml] = ET.tostring(hoja_xml, encoding="UTF-8", xml_declaration=True)

    with zipfile.ZipFile(ruta, "w", zipfile.ZIP_DEFLATED) as zf:
        for nombre, contenido in contenidos.items():
            zf.writestr(nombre, contenido)


def test_extraer_ponderadores_preserva_el_texto_crudo_exacto(tmp_path: Path) -> None:
    # protege lo que test_extraer_ponderadores_peso_es_texto_no_float NO
    # protegía: que el texto sea el CRUDO del XML, no `str(valor_parseado)`
    # -- ambos son `str`, pero solo uno preserva notación científica u otra
    # representación exacta que openpyxl normalizaría al parsear a float.
    ruta = _armar_xlsx_ponderadores_2019(tmp_path)
    layout = LAYOUTS_XLSX[2019]
    # fila 1 = agregado, fila 2 = genérico "001" -> col_peso_simple=8 = columna I
    _forzar_valor_crudo(ruta, layout.hoja_produccion_total, "I2", "5.9196304989416844E-3")

    df = extraer_ponderadores(ruta, 2019).set_index("codigo")
    assert df.at["001", "produccion total"] == "5.9196304989416844E-3"


# -- extraer_ponderadores: catálogos inconsistentes entre hojas -------------


def test_leer_hoja_peso_rechaza_codigos_duplicados(tmp_path: Path) -> None:
    layout = LAYOUTS_XLSX[2019]
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = layout.hoja_produccion_total
    ws.append(_fila_generico("001", "Soya", 10.0))
    ws.append(_fila_generico("001", "Soya (dup)", 15.0))
    ruta = tmp_path / "duplicado.xlsx"
    wb.save(ruta)

    with pytest.raises(ValueError, match="duplicado"):
        _leer_hoja_peso(
            ruta,
            layout.hoja_produccion_total,
            layout,
            {layout.col_peso_simple: "produccion total"},
            incluir_jerarquia=True,
        )


def test_extraer_ponderadores_rechaza_generico_faltante_en_una_hoja(tmp_path: Path) -> None:
    layout = LAYOUTS_XLSX[2019]
    generico = {"001": "Soya y otras oleaginosas", "002": "Frijol", "003": "Garbanzo"}

    wb = openpyxl.Workbook()
    hoja_por_defecto = wb.active
    assert hoja_por_defecto is not None
    wb.remove(hoja_por_defecto)

    ws = wb.create_sheet(layout.hoja_produccion_total)
    for i, (codigo, nombre) in enumerate(generico.items(), start=1):
        ws.append(_fila_generico(codigo, nombre, 10.0 * i))

    ws = wb.create_sheet(layout.hoja_bienes_intermedios)
    for i, codigo in enumerate(generico, start=1):
        ws.append(_fila_generico(codigo, generico[codigo], 20.0 * i))

    ws = wb.create_sheet(layout.hoja_bienes_finales)
    for i, codigo in enumerate(["001", "002"], start=1):  # falta "003" a propósito
        ws.append(_fila_generico(codigo, generico[codigo], 30.0 * i))

    ws = wb.create_sheet(layout.hoja_demanda_interna)
    for i, codigo in enumerate(generico, start=1):
        ws.append(_fila_generico(codigo, generico[codigo], 40.0 * i, 41.0 * i, 42.0 * i))

    ws = wb.create_sheet(layout.hoja_exportaciones)
    for i, codigo in enumerate(generico, start=1):
        ws.append(_fila_generico(codigo, generico[codigo], 50.0 * i))

    ruta = tmp_path / "faltante.xlsx"
    wb.save(ruta)

    with pytest.raises(ValueError, match="BienesFinales"):
        extraer_ponderadores(ruta, 2019)


def test_extraer_ponderadores_rechaza_generico_sobrante_en_una_hoja(tmp_path: Path) -> None:
    # complementa el test de arriba (faltante) -- un código que sobra (no
    # está en la ancla) es tan inconsistente como uno que falta, y sin este
    # test un `join(..., how="left")` lo descarta en silencio sin avisar
    layout = LAYOUTS_XLSX[2019]
    generico = {"001": "Soya y otras oleaginosas", "002": "Frijol", "003": "Garbanzo"}

    wb = openpyxl.Workbook()
    hoja_por_defecto = wb.active
    assert hoja_por_defecto is not None
    wb.remove(hoja_por_defecto)

    ws = wb.create_sheet(layout.hoja_produccion_total)
    for i, (codigo, nombre) in enumerate(generico.items(), start=1):
        ws.append(_fila_generico(codigo, nombre, 10.0 * i))

    ws = wb.create_sheet(layout.hoja_bienes_intermedios)
    for i, codigo in enumerate(generico, start=1):
        ws.append(_fila_generico(codigo, generico[codigo], 20.0 * i))

    ws = wb.create_sheet(layout.hoja_bienes_finales)
    for i, codigo in enumerate(generico, start=1):
        ws.append(_fila_generico(codigo, generico[codigo], 30.0 * i))
    ws.append(_fila_generico("004", "Sobrante", 999.0))  # no está en la ancla, a propósito

    ws = wb.create_sheet(layout.hoja_demanda_interna)
    for i, codigo in enumerate(generico, start=1):
        ws.append(_fila_generico(codigo, generico[codigo], 40.0 * i, 41.0 * i, 42.0 * i))

    ws = wb.create_sheet(layout.hoja_exportaciones)
    for i, codigo in enumerate(generico, start=1):
        ws.append(_fila_generico(codigo, generico[codigo], 50.0 * i))

    ruta = tmp_path / "sobrante.xlsx"
    wb.save(ruta)

    with pytest.raises(ValueError, match=r"BienesFinales") as exc_info:
        extraer_ponderadores(ruta, 2019)
    assert "sobran ['004']" in str(exc_info.value)


# =============================================================================
# -- extraer_canasta ----------------------------------------------------------
# =============================================================================
#
# A diferencia de las hojas de ponderadores (un genérico = una fila con los 5
# códigos S/SB/R/SR/C ya completos), el xlsx de canasta trae UN nivel por
# fila -- hay que arrastrar el nivel vigente hasta topar con una fila de
# genérico. Los 3 layouts reales tienen 3 "formas" distintas (ver
# esquema.py::LayoutCanasta): 2019 (texto ya combinado, código de genérico en
# columna propia), 2012 (igual, pero el código de genérico comparte columna
# con Clase, se distingue por tipo) y 2025 (código+nombre en columnas
# separadas por nivel, incluido el genérico).


def _ancho_layout_canasta(layout: LayoutCanasta) -> int:
    campos = (
        layout.col_sector,
        layout.col_sector_nombre,
        layout.col_subsector,
        layout.col_subsector_nombre,
        layout.col_rama,
        layout.col_rama_nombre,
        layout.col_subrama,
        layout.col_subrama_nombre,
        layout.col_clase,
        layout.col_clase_nombre,
        layout.col_codigo_generico,
        layout.col_nombre_generico,
    )
    return max(c for c in campos if c is not None) + 1


def _fila_canasta(
    layout: LayoutCanasta, valores: dict[int, str | int]
) -> tuple[str | int | None, ...]:
    """Fila de xlsx de canasta con columnas puestas por índice (col->valor), resto `None`.

    Tipo acotado a `str | int | None` (no `object`) -- son los únicos tipos
    de valor de celda que usan estos tests, y es lo que `Worksheet.append`
    espera según sus stubs (`object` genérico no matchea `_CellGetValue`).
    """
    fila: list[str | int | None] = [None] * _ancho_layout_canasta(layout)
    for col, valor in valores.items():
        fila[col] = valor
    return tuple(fila)


def _col(col: int | None) -> int:
    """Angosta `int | None` a `int` para las claves de `_fila_canasta`.

    Los campos `col_<nivel>_nombre` de `LayoutCanasta` son `int | None` (solo
    2025 los usa) -- acá siempre valen en los tests de 2025, el `assert` es
    para el checker, no una comprobación nueva.
    """
    assert col is not None
    return col


def _armar_xlsx_canasta(
    tmp_path: Path,
    version: VersionCanastaScian,
    filas_datos: list[tuple[str | int | None, ...]],
) -> Path:
    """xlsx sintético con la hoja y la `fila_datos_inicio` REALES de `version`.

    El relleno de filas vacías antes de `fila_datos_inicio` prueba que
    `extraer_canasta` usa el `min_row` del layout, no un valor fijo -- mismo
    criterio que `test_leer_hoja_peso_usa_la_columna_del_layout_no_una_fija`
    para el peso de ponderadores.
    """
    layout = LAYOUTS_CANASTA[version]
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = layout.hoja
    for _ in range(layout.fila_datos_inicio - 1):
        ws.append(())
    for fila in filas_datos:
        ws.append(fila)
    ruta = tmp_path / f"canasta_{version}.xlsx"
    wb.save(ruta)
    return ruta


# -- _texto_nivel ---------------------------------------------------------


def test_texto_nivel_texto_ya_combinado_se_usa_tal_cual() -> None:
    # 2012/2019: col_sector_nombre es None -- el código y el nombre ya vienen
    # pegados en una sola celda de texto, se usa tal cual
    layout = LAYOUTS_CANASTA[2019]
    fila = _fila_canasta(layout, {layout.col_sector: "11 Agricultura, cría y explotación..."})
    assert _texto_nivel(layout, "sector", fila) == "11 Agricultura, cría y explotación..."


def test_texto_nivel_columnas_separadas_se_unen_con_un_espacio() -> None:
    # 2025: código y nombre en columnas separadas, hay que unirlos
    layout = LAYOUTS_CANASTA[2025]
    fila = _fila_canasta(
        layout, {layout.col_sector: 11, _col(layout.col_sector_nombre): " Agricultura"}
    )
    assert _texto_nivel(layout, "sector", fila) == "11 Agricultura"


def test_texto_nivel_columnas_separadas_sin_espacio_inicial_en_el_nombre() -> None:
    # confirmado en el xlsx real de 2025: no todos los niveles traen el
    # espacio inicial en la celda de nombre (ej. Rama sí, Sector no) --
    # _texto_nivel no debe depender de eso
    layout = LAYOUTS_CANASTA[2025]
    fila = _fila_canasta(
        layout, {layout.col_rama: 1111, _col(layout.col_rama_nombre): "Cultivo de soya"}
    )
    assert _texto_nivel(layout, "rama", fila) == "1111 Cultivo de soya"


def test_texto_nivel_celda_vacia_devuelve_none() -> None:
    layout = LAYOUTS_CANASTA[2019]
    fila = _fila_canasta(layout, {})
    assert _texto_nivel(layout, "sector", fila) is None


# -- _es_fila_generico ------------------------------------------------------


def test_es_fila_generico_2019_por_digito_en_columna_propia() -> None:
    layout = LAYOUTS_CANASTA[2019]
    assert _es_fila_generico(layout, _fila_canasta(layout, {layout.col_codigo_generico: "001"}))
    assert not _es_fila_generico(
        layout, _fila_canasta(layout, {layout.col_clase: "111110 Clase A"})
    )


def test_es_fila_generico_2012_por_tipo_no_por_posicion() -> None:
    # el hallazgo concreto de 2012: el código de genérico comparte columna
    # con Clase -- `int` puro es genérico, `str` es texto de nivel
    layout = LAYOUTS_CANASTA[2012]
    assert layout.codigo_generico_en_columna_clase
    assert _es_fila_generico(layout, _fila_canasta(layout, {layout.col_codigo_generico: 1}))
    assert not _es_fila_generico(
        layout, _fila_canasta(layout, {layout.col_clase: "111110 Clase A"})
    )


# -- _es_fila_nota_pie --------------------------------------------------------


@pytest.mark.parametrize(
    ("fila", "esperado"),
    [
        ((None, "a/   El número asignado al producto genérico...", None), True),
        ((None, "b/ otra nota al pie", None), True),
        ((None, "11 Agricultura, cría y explotación de animales...", None), False),
        ((None, "Servicio doméstico y otros servicios para el hogar", None), False),
        ((None, None, None), False),
    ],
)
def test_es_fila_nota_pie(fila: tuple[object, ...], esperado: bool) -> None:
    assert _es_fila_nota_pie(fila) is esperado


# -- extraer_canasta: wiring completo, una "forma" de layout por test -------


def test_extraer_canasta_2019_arrastra_jerarquia_hasta_el_generico(tmp_path: Path) -> None:
    version: VersionCanastaScian = 2019
    layout = LAYOUTS_CANASTA[version]
    filas = [
        _fila_canasta(layout, {layout.col_sector: "11 Sector A"}),
        _fila_canasta(layout, {layout.col_subsector: "111 Subsector A"}),
        _fila_canasta(layout, {layout.col_rama: "1111 Rama A"}),
        _fila_canasta(layout, {layout.col_subrama: "11111 Subrama A"}),
        _fila_canasta(layout, {layout.col_clase: "111110 Clase A"}),
        _fila_canasta(
            layout, {layout.col_codigo_generico: "1", layout.col_nombre_generico: "Genérico A"}
        ),
        # segundo genérico bajo la misma rama -- solo cambia subrama/clase,
        # prueba que sector/subsector/rama NO se resetean entre niveles que
        # no cambiaron
        _fila_canasta(layout, {layout.col_subrama: "11112 Subrama B"}),
        _fila_canasta(layout, {layout.col_clase: "111120 Clase B"}),
        _fila_canasta(
            layout, {layout.col_codigo_generico: "2", layout.col_nombre_generico: "Genérico B"}
        ),
    ]
    ruta = _armar_xlsx_canasta(tmp_path, version, filas)

    df = extraer_canasta(ruta, version)

    assert list(df.columns) == list(_COLUMNAS_CANASTA)
    df = df.set_index("codigo")
    assert list(df.index) == ["001", "002"]
    assert df.at["001", "generico"] == "Genérico A"
    assert df.at["001", "sector"] == "11 Sector A"
    assert df.at["001", "subsector"] == "111 Subsector A"
    assert df.at["001", "rama"] == "1111 Rama A"
    assert df.at["001", "subrama"] == "11111 Subrama A"
    assert df.at["001", "clase"] == "111110 Clase A"
    assert df.at["002", "sector"] == "11 Sector A"
    assert df.at["002", "subsector"] == "111 Subsector A"
    assert df.at["002", "rama"] == "1111 Rama A"
    assert df.at["002", "subrama"] == "11112 Subrama B"
    assert df.at["002", "clase"] == "111120 Clase B"


def test_extraer_canasta_ignora_filas_vacias_intermedias(tmp_path: Path) -> None:
    # el xlsx real trae filas de separación visual entre bloques -- no deben
    # resetear ni alterar el estado arrastrado
    version: VersionCanastaScian = 2019
    layout = LAYOUTS_CANASTA[version]
    filas = [
        _fila_canasta(layout, {layout.col_sector: "11 Sector A"}),
        _fila_canasta(layout, {}),
        _fila_canasta(layout, {layout.col_subsector: "111 Subsector A"}),
        _fila_canasta(layout, {}),
        _fila_canasta(layout, {layout.col_rama: "1111 Rama A"}),
        _fila_canasta(layout, {layout.col_subrama: "11111 Subrama A"}),
        _fila_canasta(layout, {layout.col_clase: "111110 Clase A"}),
        _fila_canasta(
            layout, {layout.col_codigo_generico: "1", layout.col_nombre_generico: "Genérico A"}
        ),
    ]
    ruta = _armar_xlsx_canasta(tmp_path, version, filas)

    df = extraer_canasta(ruta, version)

    assert len(df) == 1
    assert df.iloc[0]["subsector"] == "111 Subsector A"


def test_extraer_canasta_filtra_nota_al_pie_sin_contaminar_estado(tmp_path: Path) -> None:
    version: VersionCanastaScian = 2019
    layout = LAYOUTS_CANASTA[version]
    nota = (
        "a/   El número asignado al producto genérico corresponde al definido "
        "en el Cambio Año Base Julio 2019=100.0 a efecto de facilitar la "
        "correspondencia en las series; excepto para aquellos productos "
        "genéricos de nueva creación donde se asigna el consecutivo siguiente."
    )
    filas = [
        _fila_canasta(layout, {layout.col_sector: "11 Sector A"}),
        _fila_canasta(layout, {layout.col_subsector: "111 Subsector A"}),
        _fila_canasta(layout, {layout.col_rama: "1111 Rama A"}),
        _fila_canasta(layout, {layout.col_subrama: "11111 Subrama A"}),
        _fila_canasta(layout, {layout.col_clase: "111110 Clase A"}),
        _fila_canasta(
            layout, {layout.col_codigo_generico: "1", layout.col_nombre_generico: "Genérico A"}
        ),
        # cae en la posición de "subsector", igual que en el xlsx real de 2019
        _fila_canasta(layout, {layout.col_subsector: nota}),
    ]
    ruta = _armar_xlsx_canasta(tmp_path, version, filas)

    df = extraer_canasta(ruta, version)

    assert len(df) == 1  # la nota no debe generar una fila propia
    assert df.iloc[0]["subsector"] == "111 Subsector A"  # ni pisar el estado vigente


def test_extraer_canasta_rechaza_codigos_duplicados(tmp_path: Path) -> None:
    # mismo contrato que extraer_ponderadores/_leer_hoja_peso -- un xlsx mal
    # formado o corregido a mano no debe colarse en silencio con genéricos
    # duplicados (multiplicaría filas/ponderadores en un cruce posterior)
    version: VersionCanastaScian = 2019
    layout = LAYOUTS_CANASTA[version]
    filas = [
        _fila_canasta(layout, {layout.col_sector: "11 Sector A"}),
        _fila_canasta(
            layout, {layout.col_codigo_generico: "1", layout.col_nombre_generico: "Genérico A"}
        ),
        _fila_canasta(
            layout,
            {layout.col_codigo_generico: "1", layout.col_nombre_generico: "Genérico A (dup)"},
        ),
    ]
    ruta = _armar_xlsx_canasta(tmp_path, version, filas)

    with pytest.raises(ValueError, match="duplicado") as exc_info:
        extraer_canasta(ruta, version)
    # no alcanza con "algún ValueError que diga duplicado" -- tiene que
    # traer el código concreto, si no no sirve para diagnosticar el xlsx
    assert "['001']" in str(exc_info.value)


def test_extraer_canasta_2012_codigo_generico_comparte_columna_con_clase(tmp_path: Path) -> None:
    version: VersionCanastaScian = 2012
    layout = LAYOUTS_CANASTA[version]
    filas = [
        _fila_canasta(layout, {layout.col_sector: "11 Sector A"}),
        _fila_canasta(layout, {layout.col_subsector: "111 Subsector A"}),
        _fila_canasta(layout, {layout.col_rama: "1111 Rama A"}),
        _fila_canasta(layout, {layout.col_subrama: "11111 Subrama A"}),
        _fila_canasta(layout, {layout.col_clase: "111110 Clase A"}),  # texto -> nivel, NO genérico
        _fila_canasta(
            layout, {layout.col_codigo_generico: 1, layout.col_nombre_generico: "Genérico A"}
        ),  # int -> genérico
    ]
    ruta = _armar_xlsx_canasta(tmp_path, version, filas)

    df = extraer_canasta(ruta, version)

    assert len(df) == 1
    assert df.iloc[0]["codigo"] == "001"
    assert df.iloc[0]["generico"] == "Genérico A"
    assert df.iloc[0]["clase"] == "111110 Clase A"  # quedó el texto, no el int del genérico


def test_extraer_canasta_2025_une_codigo_y_nombre_por_nivel(tmp_path: Path) -> None:
    version: VersionCanastaScian = 2025
    layout = LAYOUTS_CANASTA[version]
    filas = [
        _fila_canasta(layout, {layout.col_sector: 11, _col(layout.col_sector_nombre): " Sector A"}),
        _fila_canasta(
            layout, {layout.col_subsector: 111, _col(layout.col_subsector_nombre): " Subsector A"}
        ),
        _fila_canasta(layout, {layout.col_rama: 1111, _col(layout.col_rama_nombre): "Rama A"}),
        _fila_canasta(
            layout, {layout.col_subrama: 11111, _col(layout.col_subrama_nombre): "Subrama A"}
        ),
        _fila_canasta(layout, {layout.col_clase: 111110, _col(layout.col_clase_nombre): "Clase A"}),
        _fila_canasta(
            layout, {layout.col_codigo_generico: "001", layout.col_nombre_generico: "Genérico A"}
        ),
    ]
    ruta = _armar_xlsx_canasta(tmp_path, version, filas)

    df = extraer_canasta(ruta, version)

    assert len(df) == 1
    fila = df.iloc[0]
    assert fila["sector"] == "11 Sector A"
    assert fila["subsector"] == "111 Subsector A"
    assert fila["rama"] == "1111 Rama A"
    assert fila["subrama"] == "11111 Subrama A"
    assert fila["clase"] == "111110 Clase A"
    assert fila["codigo"] == "001"
    assert fila["generico"] == "Genérico A"


# ============================================================================
# -- extraer_canasta contra los xlsx reales de INEGI ------------------------
# ============================================================================
#
# Lo de arriba prueba la lógica del state machine de forma aislada. Lo de acá
# abajo confirma que, además, produce lo esperado sobre los 3 xlsx reales --
# en particular el cruce de códigos contra `extraer_ponderadores` (misma
# canasta, mismo universo de genéricos, deberían compartir join key).
#
# requires_data: los xlsx viven en data/tests/ (gitignoreado). Mismo
# criterio de skip a nivel de clase que TestContraXlsxReales en
# test_esquema.py.

_REPO_ROOT = Path(__file__).resolve().parents[3]

_RUTAS_CANASTA: dict[VersionCanastaScian, Path] = {
    2012: _REPO_ROOT / "data/tests/2012/canasta.xlsx",
    2019: _REPO_ROOT / "data/tests/2019/Canasta_de_Genericos_CAB_INPP_2019.xlsx",
    2025: _REPO_ROOT / "data/tests/2025/canasta_inpp_2025.xlsx",
}

_RUTAS_PONDERADORES_REALES: dict[VersionCanastaScian, Path] = {
    2012: _REPO_ROOT / "data/tests/2012/ponderadores_inpp_inegi_2012.xlsx",
    2019: _REPO_ROOT
    / "data/tests/2019"
    / "COU_2017_Estructura_de_ponderaciones_PR_Julio_2019_2_Agosto_2019.xlsx",
    2025: _REPO_ROOT / "data/tests/2025/ponderadores_inpp_2025.xlsx",
}

_RUTAS_TODAS_CANASTA = [*_RUTAS_CANASTA.values(), *_RUTAS_PONDERADORES_REALES.values()]
_FALTANTES_CANASTA = [str(r) for r in _RUTAS_TODAS_CANASTA if not r.exists()]
_MOTIVO_SKIP_CANASTA = (
    f"faltan xlsx reales (data/tests gitignoreado): {_FALTANTES_CANASTA}"
    if _FALTANTES_CANASTA
    else None
)

# genéricos esperados por versión -- confirmado en la sesión que armó
# extraer_canasta, mismo conteo que extraer_ponderadores (567/560/570)
_GENERICOS_ESPERADOS: dict[VersionCanastaScian, int] = {2012: 567, 2019: 560, 2025: 570}

# discrepancia real de fuente (no bug de extracción, confirmada carácter por
# carácter contra las celdas crudas de ambos xlsx del INEGI): "Chocolate en
# tableta y en polvo" trae código 113 en el xlsx de canasta de 2019 pero 114
# en el de ponderadores. Se deja explícita acá para que, si INEGI corrige el
# archivo y la discrepancia desaparece, el test de abajo lo note (en vez de
# quedar "de más" silenciosamente).
_DIFERENCIAS_CODIGO_CONOCIDAS: dict[VersionCanastaScian, tuple[set[str], set[str]]] = {
    2012: (set(), set()),
    2019: ({"113"}, {"114"}),  # (solo en canasta, solo en ponderadores)
    2025: (set(), set()),
}


class TestExtraerCanastaContraXlsxReales:
    pytestmark = [
        pytest.mark.requires_data,
        pytest.mark.skipif(_MOTIVO_SKIP_CANASTA is not None, reason=_MOTIVO_SKIP_CANASTA or ""),
    ]

    @pytest.mark.parametrize("version", [2012, 2019, 2025])
    def test_cantidad_de_generico_coincide_con_extraer_ponderadores(
        self, version: VersionCanastaScian
    ) -> None:
        df = extraer_canasta(_RUTAS_CANASTA[version], version)
        assert len(df) == _GENERICOS_ESPERADOS[version]
        assert df["codigo"].is_unique

    @pytest.mark.parametrize("version", [2012, 2019, 2025])
    def test_codigos_calzan_con_extraer_ponderadores_salvo_diferencia_conocida(
        self, version: VersionCanastaScian
    ) -> None:
        df_canasta = extraer_canasta(_RUTAS_CANASTA[version], version)
        df_ponderadores = extraer_ponderadores(_RUTAS_PONDERADORES_REALES[version], version)

        solo_canasta_esperado, solo_ponderadores_esperado = _DIFERENCIAS_CODIGO_CONOCIDAS[version]
        assert set(df_canasta["codigo"]) - set(df_ponderadores["codigo"]) == solo_canasta_esperado
        assert (
            set(df_ponderadores["codigo"]) - set(df_canasta["codigo"]) == solo_ponderadores_esperado
        )

    @pytest.mark.parametrize("version", [2012, 2019, 2025])
    def test_ninguna_columna_trae_texto_de_nota_al_pie(self, version: VersionCanastaScian) -> None:
        df = extraer_canasta(_RUTAS_CANASTA[version], version)
        for columna in ("sector", "subsector", "rama", "subrama", "clase", "generico"):
            assert not df[columna].str.strip().str.match(r"^[a-z]/\s").any(), (
                f"{version}/{columna}: quedó una nota al pie sin filtrar"
            )

    @pytest.mark.parametrize("version", [2012, 2019, 2025])
    def test_soya_trae_la_jerarquia_completa_esperada(self, version: VersionCanastaScian) -> None:
        # spot check contra un genérico conocido -- mismo código (001) y
        # misma jerarquía en las 3 versiones (confirmado con los xlsx reales)
        df = extraer_canasta(_RUTAS_CANASTA[version], version).set_index("codigo")
        assert df.at["001", "generico"] == "Soya y otras oleaginosas"
        assert str(df.at["001", "sector"]).startswith("11 Agricultura")
        assert df.at["001", "subsector"] == "111 Agricultura"
        assert str(df.at["001", "rama"]).startswith("1111 Cultivo de semillas")
        assert str(df.at["001", "subrama"]).startswith("11111 Cultivo de soya")
        assert df.at["001", "clase"] == "111110 Cultivo de soya"


# =============================================================================
# -- extraer_encadenamiento (solo 2025) ---------------------------------------
# =============================================================================
#
# Más simple que extraer_ponderadores/extraer_canasta: una sola hoja, sin
# state machine -- cada fila de genérico ya trae los 4 factores completos.
# Mismo layout S/SB/R/SR/C/G/ACTIVIDAD que LAYOUTS_XLSX[2025] en las
# columnas 1-7, factor en 4 columnas separadas por una columna vacía
# (8/10/12/14).

_ANCHO_ENCADENAMIENTO = COL_ENCADENAMIENTO_USO_FINAL + 1


_ValorCeldaEncadenamiento = str | int | float | None


def _fila_encadenamiento(
    codigo: str,
    actividad: str,
    total: _ValorCeldaEncadenamiento,
    produccion_nacional: _ValorCeldaEncadenamiento,
    exportacion: _ValorCeldaEncadenamiento,
    uso_final: _ValorCeldaEncadenamiento,
) -> tuple[_ValorCeldaEncadenamiento, ...]:
    """Fila de genérico completa del xlsx de encadenamiento (col A vacía + S/SB/R/SR/C/G/ACTIVIDAD/factores)."""
    fila: list[_ValorCeldaEncadenamiento] = [None] * _ANCHO_ENCADENAMIENTO
    fila[1], fila[2], fila[3], fila[4], fila[5] = 11, 111, 1111, 11111, 111110
    fila[COL_ENCADENAMIENTO_GENERICO] = codigo
    fila[7] = actividad
    fila[COL_ENCADENAMIENTO_TOTAL] = total
    fila[COL_ENCADENAMIENTO_PRODUCCION_NACIONAL] = produccion_nacional
    fila[COL_ENCADENAMIENTO_EXPORTACION] = exportacion
    fila[COL_ENCADENAMIENTO_USO_FINAL] = uso_final
    return tuple(fila)


def _fila_agregado_encadenamiento(
    sector: int, actividad: str, total: _ValorCeldaEncadenamiento
) -> tuple[_ValorCeldaEncadenamiento, ...]:
    """Fila de agregado (solo Sector + ACTIVIDAD + total, sin G) -- debe quedar filtrada."""
    fila: list[_ValorCeldaEncadenamiento] = [None] * _ANCHO_ENCADENAMIENTO
    fila[1] = sector
    fila[7] = actividad
    fila[COL_ENCADENAMIENTO_TOTAL] = total
    return tuple(fila)


def _armar_xlsx_encadenamiento(
    tmp_path: Path, filas: list[tuple[_ValorCeldaEncadenamiento, ...]]
) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = HOJA_ENCADENAMIENTO
    for fila in filas:
        ws.append(fila)
    ruta = tmp_path / "encadenamiento.xlsx"
    wb.save(ruta)
    return ruta


def test_extraer_encadenamiento_forma_y_orden_de_columnas(tmp_path: Path) -> None:
    filas = [
        _fila_agregado_encadenamiento(11, "Agricultura", 1.5),
        _fila_encadenamiento("1", "Soya", 1.1, 1.2, 1.3, 1.4),
        _fila_encadenamiento("2", "Frijol", 2.1, 2.2, 2.3, 2.4),
    ]
    ruta = _armar_xlsx_encadenamiento(tmp_path, filas)

    df = extraer_encadenamiento(ruta)

    assert len(df) == 2  # la fila de agregado NO debe colarse
    assert list(df.columns) == ["codigo", *_COLUMNAS_ENCADENAMIENTO_FACTOR.values()]
    assert list(df["codigo"]) == ["001", "002"]


def test_extraer_encadenamiento_convierte_na_a_nan_real(tmp_path: Path) -> None:
    filas = [
        _fila_encadenamiento("1", "Soya", 1.1, 1.2, "N/A", "N/A"),
    ]
    ruta = _armar_xlsx_encadenamiento(tmp_path, filas)

    df = extraer_encadenamiento(ruta).set_index("codigo")

    assert df.at["001", "encadenamiento exportacion"] != "N/A"
    assert df.at["001", "encadenamiento uso final"] != "N/A"
    assert pd.isna(df.at["001", "encadenamiento exportacion"])
    assert pd.isna(df.at["001", "encadenamiento uso final"])
    # las columnas SIN "N/A" no deben verse afectadas por la conversión
    assert not pd.isna(df.at["001", "encadenamiento total"])
    assert not pd.isna(df.at["001", "encadenamiento produccion nacional"])


def test_extraer_encadenamiento_preserva_el_texto_crudo_exacto_en_las_4_columnas(
    tmp_path: Path,
) -> None:
    # mismo criterio que test_extraer_ponderadores_preserva_el_texto_crudo_exacto
    # -- el factor se multiplica en el cálculo del índice, perder un decimal
    # ahí sí importa. Las 4 columnas, no solo total: un valor único
    # repetido en las 4 (o comprobado con pytest.approx) no detecta ni un
    # swap exportación<->uso_final ni que 3 de las 4 columnas se hayan
    # quedado con el valor ya parseado por openpyxl en vez del crudo --
    # confirmado reproduciendo ambos mutantes antes de este fix (ver
    # data/negociaciones/2026-08-22-extraer-encadenamiento.md).
    filas = [_fila_encadenamiento("1", "Soya", 1.1, 1.2, 1.3, 1.4)]
    ruta = _armar_xlsx_encadenamiento(tmp_path, filas)
    # fila 1 = genérico "001" -> total=col I, produccion_nacional=col K,
    # exportacion=col M, uso_final=col O (ver _fila_encadenamiento)
    _forzar_valor_crudo(ruta, HOJA_ENCADENAMIENTO, "I1", "1.0738593778483521")
    _forzar_valor_crudo(ruta, HOJA_ENCADENAMIENTO, "K1", "2.1487187556967042")
    _forzar_valor_crudo(ruta, HOJA_ENCADENAMIENTO, "M1", "3.2230781335450563")
    _forzar_valor_crudo(ruta, HOJA_ENCADENAMIENTO, "O1", "4.2974375113934084")

    df = extraer_encadenamiento(ruta).set_index("codigo")
    assert df.at["001", "encadenamiento total"] == "1.0738593778483521"
    assert df.at["001", "encadenamiento produccion nacional"] == "2.1487187556967042"
    assert df.at["001", "encadenamiento exportacion"] == "3.2230781335450563"
    assert df.at["001", "encadenamiento uso final"] == "4.2974375113934084"


def test_extraer_encadenamiento_rechaza_codigos_duplicados(tmp_path: Path) -> None:
    filas = [
        _fila_encadenamiento("1", "Soya", 1.1, 1.2, 1.3, 1.4),
        _fila_encadenamiento("1", "Soya (dup)", 9.1, 9.2, 9.3, 9.4),
    ]
    ruta = _armar_xlsx_encadenamiento(tmp_path, filas)

    with pytest.raises(ValueError, match="duplicado") as exc_info:
        extraer_encadenamiento(ruta)
    assert "['001']" in str(exc_info.value)


# ============================================================================
# -- extraer_encadenamiento contra el xlsx real de INEGI (solo 2025) --------
# ============================================================================

_RUTA_ENCADENAMIENTO_REAL = _REPO_ROOT / "data/tests/2025/factor_de_encadenamiento_ti.xlsx"
_MOTIVO_SKIP_ENCADENAMIENTO = (
    f"falta xlsx real (data/tests gitignoreado): {_RUTA_ENCADENAMIENTO_REAL}"
    if not _RUTA_ENCADENAMIENTO_REAL.exists()
    else None
)


class TestExtraerEncadenamientoContraXlsxReal:
    pytestmark = [
        pytest.mark.requires_data,
        pytest.mark.skipif(
            _MOTIVO_SKIP_ENCADENAMIENTO is not None, reason=_MOTIVO_SKIP_ENCADENAMIENTO or ""
        ),
    ]

    def test_cantidad_de_generico_y_codigo_unico(self) -> None:
        df = extraer_encadenamiento(_RUTA_ENCADENAMIENTO_REAL)
        assert len(df) == 570  # mismo conteo que extraer_ponderadores/extraer_canasta en 2025
        assert df["codigo"].is_unique

    def test_encadenamiento_total_y_produccion_nacional_nunca_son_nan(self) -> None:
        # confirmado contra el xlsx real: total y producción_nacional cubren
        # el universo completo de genéricos -- a diferencia de exportación y
        # uso_final, que sí traen "N/A" para genéricos sin esa cobertura
        df = extraer_encadenamiento(_RUTA_ENCADENAMIENTO_REAL)
        assert not df["encadenamiento total"].isna().any()
        assert not df["encadenamiento produccion nacional"].isna().any()

    def test_exportacion_y_uso_final_si_traen_nan(self) -> None:
        df = extraer_encadenamiento(_RUTA_ENCADENAMIENTO_REAL)
        for columna in ("encadenamiento exportacion", "encadenamiento uso final"):
            assert df[columna].isna().any(), f"{columna}: se esperaba al menos un NaN"

    def test_soya_trae_los_4_factores_esperados(self) -> None:
        # Soya (001) sirve para spot-check básico, pero NO para proteger
        # contra un swap exportación<->uso_final: las 3 columnas
        # produccion_nacional/exportacion/uso_final valen exactamente lo
        # mismo para este genérico en el xlsx real -- confirmado, un swap
        # ahí es invisible. Ver test de abajo (código 065) para eso.
        df = extraer_encadenamiento(_RUTA_ENCADENAMIENTO_REAL).set_index("codigo")
        assert df.at["001", "encadenamiento total"] == "1.0738593778483521"
        assert df.at["001", "encadenamiento produccion nacional"] == "1.0738593778483525"
        assert df.at["001", "encadenamiento exportacion"] == "1.0738593778483525"
        assert df.at["001", "encadenamiento uso final"] == "1.0738593778483525"

    def test_generico_con_4_valores_distintos_no_se_confunde_entre_columnas(self) -> None:
        # código 065: las 4 columnas traen valores DISTINTOS entre sí en el
        # xlsx real -- a diferencia de Soya (arriba), acá un swap
        # exportación<->uso_final, o perder la precisión cruda en alguna de
        # las 3 columnas no-total, sí cambia el resultado y el test lo nota
        df = extraer_encadenamiento(_RUTA_ENCADENAMIENTO_REAL).set_index("codigo")
        assert df.at["065", "encadenamiento total"] == "1.4165194322661865"
        assert df.at["065", "encadenamiento produccion nacional"] == "1.4141735579391901"
        assert df.at["065", "encadenamiento exportacion"] == "1.516797214010164"
        assert df.at["065", "encadenamiento uso final"] == "1.4598847030020605"
