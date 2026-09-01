from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from replica_inpp.dominio.errores import (
    ArchivoCorrupto,
    ArchivoNoEncontrado,
    ArchivoVacio,
    OrientacionNoDetectable,
    SerieVacia,
)
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.infraestructura.csv.lector_series_csv import LectorSeriesCsv

DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "inputs"

_ENCABEZADO_BIE = "Instituto Nacional de Estadística y Geografía\n\n\n\n\n"

"""
Serie plana sintética (mercado_nacional, prefijo 1) -- reproduce el formato real
confirmado contra los CSV reales del BIE: ", <1 dígito recorte><3 dígitos código>
<nombre>" al final del Título.
"""
df_plano = pd.DataFrame(
    {
        "Título": [
            "Índices de precios de genéricos para mercado nacional, 1001 Soya",
            "Índices de precios de genéricos para mercado nacional, 1002 Frijol",
            "Índices de precios de genéricos para mercado nacional, 1003 Garbanzo",
        ],
        "Cifra": ["Indices"] * 3,
        "Serie": ["11111", "22222", "33333"],
        "Ene 2019": ["100.00", "100.00", "100.00"],
        "Feb 2019": ["101.00", "102.00", "103.00"],
    }
)

"""
Jerarquía sintética estilo `ae` de produccion_total -- reproduce la cadena real
confirmada contra el xlsx fuente: código de 6 dígitos (Clase) seguido del código
de 3 dígitos (genérico), SIN prefijo de recorte. Incluye el caso real del
huérfano `622 Hospitales` -- misma posición estructural que un genérico válido
(después de un Clase real), pero excluido a mano.
"""


def _hijo(padre: str, sufijo: str) -> str:
    return f"{padre}, {sufijo}"


_sector_primario = "INPP, Producción total, según actividad económica, Actividades primarias"
_rama = _hijo(_sector_primario, "11 Agricultura")
_subrama = _hijo(_rama, "111 Agricultura")
_clase_cultivo = _hijo(_subrama, "1111 Cultivo")
_subclase_soya = _hijo(_clase_cultivo, "11111 Cultivo de soya")
_clase_soya = _hijo(_subclase_soya, "111110 Cultivo de soya")
_hoja_soya = _hijo(_clase_soya, "001 Soya y otras oleaginosas")

_sector_terciario = "INPP, Producción total, según actividad económica, Actividades terciarias"
_clase_laboratorio = _hijo(
    _sector_terciario,
    "621511 Laboratorios médicos y de diagnóstico del sector privado",
)
_hoja_gabinete = _hijo(_clase_laboratorio, "535 Estudios de gabinete")
_hoja_clinicos = _hijo(_clase_laboratorio, "536 Análisis clínicos")
_hoja_huerfana_622 = _hijo(_clase_laboratorio, "622 Hospitales")

_TITULOS_AE = [
    _sector_primario,
    _rama,
    _subrama,
    _clase_cultivo,
    _subclase_soya,
    _clase_soya,
    _hoja_soya,
    _clase_laboratorio,
    _hoja_gabinete,
    _hoja_clinicos,
    _hoja_huerfana_622,
]

df_jerarquico = pd.DataFrame(
    {
        "Título": _TITULOS_AE,
        "Cifra": ["Indices"] * len(_TITULOS_AE),
        "Serie": [str(90000 + i) for i in range(len(_TITULOS_AE))],
        "Ene 2019": ["100.00"] * len(_TITULOS_AE),
        "Feb 2019": ["101.00"] * len(_TITULOS_AE),
    }
)


def _escribir_csv(ruta: Path, df: pd.DataFrame) -> None:
    contenido = _ENCABEZADO_BIE + df.to_csv(index=False)
    ruta.write_text(contenido, encoding="utf-8")


# ---------- Errores de archivo ----------


def test_archivo_no_encontrado(tmp_path: Path) -> None:
    ruta = tmp_path / "no_existe.csv"
    with pytest.raises(ArchivoNoEncontrado):
        LectorSeriesCsv().leer(ruta)


def test_archivo_vacio(tmp_path: Path) -> None:
    ruta = tmp_path / "vacio.csv"
    ruta.touch()
    with pytest.raises(ArchivoVacio):
        LectorSeriesCsv().leer(ruta)


def test_archivo_corrupto_sin_columna_titulo(tmp_path: Path) -> None:
    ruta = tmp_path / "corrupto.csv"
    df_sin_titulo = df_plano.rename(columns={"Título": "Encabezado"})
    _escribir_csv(ruta, df_sin_titulo)
    with pytest.raises(ArchivoCorrupto):
        LectorSeriesCsv().leer(ruta)


def test_orientacion_no_detectable(tmp_path: Path) -> None:
    ruta = tmp_path / "sin_orientacion.csv"
    df_sin_cifra = df_plano.drop(columns=["Cifra", "Serie"])
    _escribir_csv(ruta, df_sin_cifra)
    with pytest.raises(OrientacionNoDetectable):
        LectorSeriesCsv().leer(ruta)


def test_serie_vacia_sin_genericos(tmp_path: Path) -> None:
    ruta = tmp_path / "sin_genericos.csv"
    df_sin_genericos = df_plano.copy()
    df_sin_genericos["Título"] = ["Alimentos", "Bebidas", "Lacteos"]
    _escribir_csv(ruta, df_sin_genericos)
    with pytest.raises(SerieVacia):
        LectorSeriesCsv().leer(ruta)


def test_prefijos_de_recorte_mezclados_falla(tmp_path: Path) -> None:
    ruta = tmp_path / "mezclado.csv"
    df_mezclado = df_plano.copy()
    df_mezclado.loc[0, "Título"] = (
        "Índices de precios de genéricos para mercado de exportación, 2001 Soya"
    )
    _escribir_csv(ruta, df_mezclado)
    with pytest.raises(ArchivoCorrupto):
        LectorSeriesCsv().leer(ruta)


# ---------- Extracción plana ----------


def test_extraccion_plana_codigo_nombre_y_valores(tmp_path: Path) -> None:
    ruta = tmp_path / "plano.csv"
    _escribir_csv(ruta, df_plano)
    resultado = LectorSeriesCsv().leer(ruta)

    assert list(resultado.index) == ["001", "002", "003"]
    assert list(resultado["generico"]) == ["soya", "frijol", "garbanzo"]
    fila_frijol = resultado.loc["002"].to_dict()
    assert fila_frijol[PeriodoMensual(2019, 2)] == 102.0


def test_version_detectada_sin_patron_base_es_none(tmp_path: Path) -> None:
    # df_plano no trae "Base <mes> <AAAA>=100" en el Título (formato real completo
    # reservado a este test aparte, para no acoplar los demás al patrón exacto).
    ruta = tmp_path / "plano.csv"
    _escribir_csv(ruta, df_plano)
    resultado = LectorSeriesCsv().leer(ruta)
    assert resultado.attrs["version_detectada"] is None


@pytest.mark.parametrize(
    "prefijo_base,año_esperado",
    [("Base junio 2012=100", 2012), ("Base Julio 2019=100", 2019), ("Base Julio 2025=100", 2025)],
)
def test_version_detectada_extrae_año_de_la_base(
    tmp_path: Path, prefijo_base: str, año_esperado: int
) -> None:
    df = df_plano.copy()
    df["Título"] = (
        "Índice nacional de precios productor. " + prefijo_base + " (SCIAN 2013), " + df["Título"]
    )
    ruta = tmp_path / "con_base.csv"
    _escribir_csv(ruta, df)
    resultado = LectorSeriesCsv().leer(ruta)
    assert resultado.attrs["version_detectada"] == año_esperado


@pytest.mark.parametrize(
    "prefijo,recorte_esperado",
    [
        ("1", "mercado_nacional"),
        ("2", "mercado_exportacion"),
        ("3", "produccion_total"),
        ("4", "bienes_finales"),
    ],
)
def test_recorte_autodetectado_por_prefijo(
    tmp_path: Path, prefijo: str, recorte_esperado: str
) -> None:
    ruta = tmp_path / f"prefijo_{prefijo}.csv"
    df = df_plano.copy()
    df["Título"] = df["Título"].str.replace(", 1", f", {prefijo}", regex=False)
    _escribir_csv(ruta, df)
    resultado = LectorSeriesCsv().leer(ruta)
    assert resultado.attrs["recorte"] == recorte_esperado


def test_orientacion_vertical_da_mismo_resultado_que_horizontal(tmp_path: Path) -> None:
    ruta_h = tmp_path / "plano_h.csv"
    _escribir_csv(ruta_h, df_plano)
    resultado_h = LectorSeriesCsv().leer(ruta_h)

    # Formato vertical real: primera columna "Título" con las etiquetas de fila
    # (Cifra/Serie/periodos), una columna por genérico con el Título completo
    # como encabezado.
    filas_etiqueta = ["Cifra", "Serie", "Ene 2019", "Feb 2019"]
    datos_v: dict[str, list[str]] = {"Título": filas_etiqueta}
    for _, fila in df_plano.iterrows():
        datos_v[fila["Título"]] = [fila["Cifra"], fila["Serie"], fila["Ene 2019"], fila["Feb 2019"]]
    df_v = pd.DataFrame(datos_v)
    ruta_v = tmp_path / "plano_v.csv"
    _escribir_csv(ruta_v, df_v)
    resultado_v = LectorSeriesCsv().leer(ruta_v)

    pd.testing.assert_frame_equal(resultado_h.sort_index(), resultado_v.sort_index())


# ---------- Extracción jerárquica (`ae`) ----------


def test_extraccion_jerarquica_solo_hojas(tmp_path: Path) -> None:
    ruta = tmp_path / "jerarquico.csv"
    _escribir_csv(ruta, df_jerarquico)
    resultado = LectorSeriesCsv().leer(ruta)

    # 001 (soya), 535, 536 -- NO 622 (huérfano excluido a mano)
    assert set(resultado.index) == {"001", "535", "536"}
    assert resultado.attrs["recorte"] == "produccion_total"


def test_codigo_huerfano_622_excluido(tmp_path: Path) -> None:
    ruta = tmp_path / "jerarquico_622.csv"
    _escribir_csv(ruta, df_jerarquico)
    resultado = LectorSeriesCsv().leer(ruta)
    assert "622" not in resultado.index


# ---------- Datos reales ----------


@pytest.mark.requires_data
@pytest.mark.parametrize(
    "archivo,filas_esperadas,recorte_esperado",
    [
        ("produccion_total/s19_h_nm_nae.CSV", 560, "produccion_total"),
        ("produccion_total/s19_h_nm_ae.CSV", 560, "produccion_total"),
        ("mercado_nacional/s19_h_nm.CSV", 560, "mercado_nacional"),
        ("bienes_finales/s19_h_nm.CSV", 540, "bienes_finales"),
        ("mercado_exportacion/s19_h_nm.CSV", 447, "mercado_exportacion"),
    ],
)
def test_lector_series_csv_real_s19(
    archivo: str, filas_esperadas: int, recorte_esperado: str
) -> None:
    resultado = LectorSeriesCsv().leer(DATA_DIR / archivo)
    assert len(resultado) == filas_esperadas
    assert resultado.attrs["recorte"] == recorte_esperado
    assert not resultado.index.duplicated().any()
    assert "622" not in resultado.index


@pytest.mark.requires_data
@pytest.mark.parametrize(
    "archivo,filas_esperadas,recorte_esperado",
    [
        ("produccion_total/s12_h_nm_nae.CSV", 567, "produccion_total"),
        ("produccion_total/s12_h_nm_ae.CSV", 567, "produccion_total"),
        ("mercado_nacional/s12_h_nm.CSV", 567, "mercado_nacional"),
        ("bienes_finales/s12_h_nm.CSV", 556, "bienes_finales"),
        ("mercado_exportacion/s12_h_nm.CSV", 468, "mercado_exportacion"),
    ],
)
def test_lector_series_csv_real_s12(
    archivo: str, filas_esperadas: int, recorte_esperado: str
) -> None:
    resultado = LectorSeriesCsv().leer(DATA_DIR / archivo)
    assert len(resultado) == filas_esperadas
    assert resultado.attrs["recorte"] == recorte_esperado
    assert resultado.attrs["version_detectada"] == 2012
    assert not resultado.index.duplicated().any()


@pytest.mark.requires_data
def test_lector_series_csv_real_horizontal_igual_vertical() -> None:
    h = LectorSeriesCsv().leer(DATA_DIR / "mercado_nacional/s19_h_nm.CSV")
    v = LectorSeriesCsv().leer(DATA_DIR / "mercado_nacional/s19_v_nm.CSV")
    pd.testing.assert_frame_equal(h.sort_index(), v.sort_index())
