from __future__ import annotations

import pandas as pd
import pytest
from canasta_inpp.utilidades import (
    normalizar_columnas_con_codigo,
    normalizar_columnas_texto,
    normalizar_texto,
    normalizar_texto_con_codigo,
    resolver_sector_agrupado,
)

# -- normalizar_texto -------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("Árbol", "arbol"),
        ("ÁÉÍÓÚÜ", "aeiouu"),
        ("señor", "señor"),
        ("NIÑO", "niño"),
        ("hola, mundo!", "hola mundo"),
        ("hola   mundo", "hola mundo"),
        ("  hola  ", "hola"),
        ("11 Agricultura, cría y explotación", "11 agricultura cria y explotacion"),
    ],
)
def test_normalizar_texto(texto: str, esperado: str) -> None:
    assert normalizar_texto(texto) == esperado


# -- normalizar_columnas_texto -----------------------------------------------


def test_normalizar_columnas_texto_normaliza_solo_columnas_dadas() -> None:
    df = pd.DataFrame(
        {
            "generico": ["Soya y Otras Oleaginosas"],
            "sector": ["11 Agricultura, cría y explotación..."],
            "codigo": ["001"],
        }
    )
    resultado = normalizar_columnas_texto(df, ["generico", "sector"])
    assert resultado.loc[0, "generico"] == "soya y otras oleaginosas"
    assert resultado.loc[0, "sector"] == "11 agricultura cria y explotacion"
    assert resultado.loc[0, "codigo"] == "001"  # intacta, no pedida


def test_normalizar_columnas_texto_no_muta_el_df_original() -> None:
    df = pd.DataFrame({"generico": ["Frijol"]})
    normalizar_columnas_texto(df, ["generico"])
    assert df.loc[0, "generico"] == "Frijol"


def test_normalizar_columnas_texto_multiples_filas() -> None:
    df = pd.DataFrame({"generico": ["Soya", "Frijol", ""]})
    resultado = normalizar_columnas_texto(df, ["generico"])
    assert list(resultado["generico"]) == ["soya", "frijol", ""]


# -- normalizar_texto_con_codigo ---------------------------------------------
# Regresión: normalizar_texto (texto completo, sin distinguir código de
# nombre) destruye rangos SCIAN con guion -- "31-33 Industrias
# manufactureras" colapsa a "3133 industrias manufactureras", forma
# indistinguible de una rama real de 4 dígitos. Ver negociación
# data/negociaciones/2026-08-22-utilidades-normalizacion-scian.md.


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        # rangos SCIAN oficiales (agrupación de sectores, no texto libre) --
        # el guion es parte del código, se preserva intacto
        ("31-33 Industrias manufactureras", "31-33 industrias manufactureras"),
        (
            "48-49 Transportes, correos y almacenamiento",
            "48-49 transportes correos y almacenamiento",
        ),
        # código simple, mismo resultado que normalizar_texto (sin guion que proteger)
        ("11 Agricultura, cría y explotación", "11 agricultura cria y explotacion"),
        # guion DENTRO del nombre (no en el código) -- sigue la regla general,
        # se pierde igual que en normalizar_texto (no es parte del código)
        ("31-33 Industria químico-farmacéutica", "31-33 industria quimicofarmaceutica"),
        # solo código, sin nombre -- se devuelve tal cual
        ("11", "11"),
        ("", ""),
    ],
)
def test_normalizar_texto_con_codigo(texto: str, esperado: str) -> None:
    assert normalizar_texto_con_codigo(texto) == esperado


# -- normalizar_columnas_con_codigo ------------------------------------------


def test_normalizar_columnas_con_codigo_preserva_rango_scian() -> None:
    df = pd.DataFrame(
        {
            "generico": ["Hilados de fibras artificiales y/o sintéticas"],
            "sector": ["31-33 Industrias manufactureras"],
            "codigo": ["123"],
        }
    )
    resultado = normalizar_columnas_con_codigo(df, ["sector"])
    assert resultado.loc[0, "sector"] == "31-33 industrias manufactureras"
    assert (
        resultado.loc[0, "generico"] == "Hilados de fibras artificiales y/o sintéticas"
    )  # intacta, no pedida
    assert resultado.loc[0, "codigo"] == "123"  # intacta, no pedida


def test_normalizar_columnas_con_codigo_no_muta_el_df_original() -> None:
    df = pd.DataFrame({"sector": ["31-33 Industrias manufactureras"]})
    normalizar_columnas_con_codigo(df, ["sector"])
    assert df.loc[0, "sector"] == "31-33 Industrias manufactureras"


def test_normalizar_columnas_con_codigo_multiples_columnas_y_filas() -> None:
    df = pd.DataFrame(
        {
            "sector": ["31-33 Industrias manufactureras", "11 Agricultura"],
            "subsector": ["48-49 Transportes", "111 Agricultura"],
        }
    )
    resultado = normalizar_columnas_con_codigo(df, ["sector", "subsector"])
    assert list(resultado["sector"]) == ["31-33 industrias manufactureras", "11 agricultura"]
    assert list(resultado["subsector"]) == ["48-49 transportes", "111 agricultura"]


# -- resolver_sector_agrupado -------------------------------------------------
# Regresión: sector "31-33"/"48-49" es un rango SCIAN (agrupa varios códigos de
# sector bajo un mismo nombre) -- dominio/calculo necesita el código concreto
# (31/32/33) para agrupar, no el rango. subsector siempre trae el código
# concreto en sus primeros 2 dígitos.


def test_resolver_sector_agrupado_con_nombre_combinado() -> None:
    df = pd.DataFrame(
        {
            "sector": ["31-33 Industrias manufactureras"],
            "subsector": ["311 Industria alimentaria"],
        }
    )
    resultado = resolver_sector_agrupado(df)
    assert resultado.loc[0, "sector"] == "31 Industrias manufactureras"


def test_resolver_sector_agrupado_bare_sin_nombre() -> None:
    # solo --ponderadores: sector/subsector son código bare, sin nombre pegado
    df = pd.DataFrame({"sector": ["48-49"], "subsector": ["492"]})
    resultado = resolver_sector_agrupado(df)
    assert resultado.loc[0, "sector"] == "49"


def test_resolver_sector_agrupado_no_toca_sector_sin_rango() -> None:
    df = pd.DataFrame(
        {
            "sector": ["11 Agricultura, cría y explotación"],
            "subsector": ["111 Agricultura"],
        }
    )
    resultado = resolver_sector_agrupado(df)
    assert resultado.loc[0, "sector"] == "11 Agricultura, cría y explotación"


def test_resolver_sector_agrupado_multiples_filas_mismo_rango_distinto_subsector() -> None:
    df = pd.DataFrame(
        {
            "sector": ["31-33 Industrias manufactureras", "31-33 Industrias manufactureras"],
            "subsector": ["311 Industria alimentaria", "331 Industrias metálicas básicas"],
        }
    )
    resultado = resolver_sector_agrupado(df)
    assert list(resultado["sector"]) == [
        "31 Industrias manufactureras",
        "33 Industrias manufactureras",
    ]


def test_resolver_sector_agrupado_no_muta_el_df_original() -> None:
    df = pd.DataFrame(
        {"sector": ["31-33 Industrias manufactureras"], "subsector": ["311 Industria alimentaria"]}
    )
    resolver_sector_agrupado(df)
    assert df.loc[0, "sector"] == "31-33 Industrias manufactureras"


# -- resolver_sector_agrupado: validación de jerarquía inconsistente ---------
# Regresión: sin validar, una jerarquía inconsistente producía una
# clasificación falsa en silencio en vez de fallar. Ver negociación
# data/negociaciones/2026-08-22-utilidades-normalizacion-scian.md § Seguimiento 2.


def test_resolver_sector_agrupado_lanza_si_subsector_fuera_del_rango() -> None:
    # subsector 481 (sector real 48) no pertenece al rango 31-33 declarado por sector
    df = pd.DataFrame(
        {
            "sector": ["31-33 Industrias manufactureras"],
            "subsector": ["481 Transporte aéreo"],
            "codigo": ["099"],
        }
    )
    with pytest.raises(ValueError, match="no pertenece al rango"):
        resolver_sector_agrupado(df)


def test_resolver_sector_agrupado_lanza_si_subsector_vacio() -> None:
    df = pd.DataFrame({"sector": ["48-49 Transportes"], "subsector": [""], "codigo": ["099"]})
    with pytest.raises(ValueError, match="no trae un código de 3 dígitos"):
        resolver_sector_agrupado(df)


def test_resolver_sector_agrupado_lanza_si_subsector_no_numerico() -> None:
    df = pd.DataFrame(
        {"sector": ["31-33 Industrias manufactureras"], "subsector": ["xx dato"], "codigo": ["099"]}
    )
    with pytest.raises(ValueError, match="no trae un código de 3 dígitos"):
        resolver_sector_agrupado(df)


def test_resolver_sector_agrupado_lanza_si_subsector_corto() -> None:
    # ortogonal a "vacío"/"no numérico": numérico y no vacío, pero longitud != 3 --
    # sin el chequeo de longitud, "31".isdigit() es True y el bug pasaría desapercibido
    df = pd.DataFrame(
        {"sector": ["31-33 Industrias manufactureras"], "subsector": ["31"], "codigo": ["099"]}
    )
    with pytest.raises(ValueError, match="no trae un código de 3 dígitos"):
        resolver_sector_agrupado(df)


def test_resolver_sector_agrupado_lanza_si_subsector_3_caracteres_no_numerico() -> None:
    # ortogonal a "vacío"/"no numérico": longitud exacta de 3, pero no numérico --
    # sin el chequeo de isdigit(), pasaría el largo y colaría un código no numérico
    df = pd.DataFrame(
        {
            "sector": ["31-33 Industrias manufactureras"],
            "subsector": ["31x dato"],
            "codigo": ["099"],
        }
    )
    with pytest.raises(ValueError, match="no trae un código de 3 dígitos"):
        resolver_sector_agrupado(df)


def test_resolver_sector_agrupado_mensaje_de_error_identifica_codigo_de_fila() -> None:
    df = pd.DataFrame(
        {
            "sector": ["31-33 Industrias manufactureras"],
            "subsector": ["481 Transporte aéreo"],
            "codigo": ["099"],
        }
    )
    with pytest.raises(ValueError, match="código 099"):
        resolver_sector_agrupado(df)


def test_resolver_sector_agrupado_mensaje_de_error_usa_indice_de_fila_sin_columna_codigo() -> None:
    # sin columna "codigo" en el df (ej. solo --ponderadores sin cruce) -- cae a índice de fila
    df = pd.DataFrame({"sector": ["31-33 Industrias manufactureras"], "subsector": ["481"]})
    with pytest.raises(ValueError, match="fila 0"):
        resolver_sector_agrupado(df)
