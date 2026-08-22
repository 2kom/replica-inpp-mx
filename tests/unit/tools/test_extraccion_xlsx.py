from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import openpyxl
import pytest
from canasta_inpp.esquema import LAYOUTS_XLSX
from canasta_inpp.extraccion_xlsx import (
    _COLUMNAS_PONDERADORES,
    _NS_MAIN,
    _NS_PKG_REL,
    _NS_REL,
    _es_codigo_generico,
    _leer_hoja_peso,
    _valores_crudos,
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
        assert float(str(df.at[codigo, "produccion_total"])) == pytest.approx(10.0 * i)
        assert float(str(df.at[codigo, "bienes_intermedios"])) == pytest.approx(20.0 * i)
        assert float(str(df.at[codigo, "bienes_finales"])) == pytest.approx(30.0 * i)
        assert float(str(df.at[codigo, "demanda_interna_total"])) == pytest.approx(40.0 * i)
        assert float(str(df.at[codigo, "demanda_interna_consumo"])) == pytest.approx(41.0 * i)
        assert float(str(df.at[codigo, "demanda_interna_capital"])) == pytest.approx(42.0 * i)
        assert float(str(df.at[codigo, "exportaciones"])) == pytest.approx(50.0 * i)


def test_extraer_ponderadores_peso_es_texto_no_float(tmp_path: Path) -> None:
    # el peso se guarda como texto crudo (precisión exacta), no como el
    # float ya parseado por openpyxl -- ver _valores_crudos
    ruta = _armar_xlsx_ponderadores_2019(tmp_path)
    df = extraer_ponderadores(ruta, 2019)
    for columna in (
        "produccion_total",
        "bienes_intermedios",
        "bienes_finales",
        "demanda_interna_total",
        "demanda_interna_consumo",
        "demanda_interna_capital",
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
        {layout_2012.col_peso_simple: "produccion_total"},
        incluir_jerarquia=True,
    )
    assert float(str(df.at["001", "produccion_total"])) == pytest.approx(12.5)


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
    assert df.at["001", "produccion_total"] == "5.9196304989416844E-3"


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
            {layout.col_peso_simple: "produccion_total"},
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
