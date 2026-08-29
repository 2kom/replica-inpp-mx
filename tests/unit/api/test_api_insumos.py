from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import replica_inpp as rep
from replica_inpp.api import insumos
from replica_inpp.dominio.errores import ArchivoCorrupto, InvarianteViolado, VersionNoCoincide
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.periodos import PeriodoMensual

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "inputs"


def _canasta_dummy() -> CanastaINPP:
    columnas: dict[str, list[Any]] = {
        "generico": ["soya"],
        "codigo sector": ["11"],
        "sector": ["11 agricultura"],
        "codigo subsector": ["111"],
        "subsector": ["111 agricultura"],
        "codigo rama": ["1111"],
        "rama": ["1111 cultivo"],
        "codigo subrama": ["11111"],
        "subrama": ["11111 cultivo de soya"],
        "codigo clase": ["111110"],
        "clase": ["111110 cultivo de soya"],
        "produccion total": [100.0],
        "bienes intermedios": [100.0],
        "bienes finales": [100.0],
        "demanda interna total": [100.0],
        "demanda interna consumo": [100.0],
        "demanda interna capital": [100.0],
        "exportaciones": [100.0],
        "encadenamiento total": [None],
        "encadenamiento produccion nacional": [None],
        "encadenamiento exportacion": [None],
        "encadenamiento uso final": [None],
    }
    df = pd.DataFrame(columnas, index=pd.Index(["001"], name="codigo"))
    return CanastaINPP(df, version=2019)


def _df_lector(
    recorte: str = "mercado_nacional", version_detectada: int | None = None
) -> pd.DataFrame:
    df = pd.DataFrame(
        {"generico": ["soya"], PeriodoMensual(2019, 1): [100.0]},
        index=pd.Index(["001"], name="codigo"),
    )
    df.attrs["recorte"] = recorte
    df.attrs["origen"] = Path("x.csv")
    df.attrs["version_detectada"] = version_detectada
    return df


# -- validación de versión --------------------------------------------------


@pytest.mark.parametrize("version", [2009, 2020, 0, 9999])
def test_cargar_serie_version_invalida(version: int) -> None:
    with pytest.raises(InvarianteViolado):
        insumos.cargar_serie("x.csv", version)  # type: ignore[arg-type]


@pytest.mark.parametrize("version", [2012, 2019, 2025])
def test_cargar_serie_version_valida_no_falla_por_version(mocker, version: int) -> None:
    lector = mocker.patch.object(insumos, "LectorSeriesCsv")
    lector.return_value.leer.return_value = _df_lector()

    resultado = insumos.cargar_serie("x.csv", version)

    assert isinstance(resultado, SerieNormalizada)
    assert resultado.df.attrs["version"] == version


# -- delegación y armado de SerieNormalizada ---------------------------------


def test_cargar_serie_delega_al_lector_con_path(mocker) -> None:
    lector = mocker.patch.object(insumos, "LectorSeriesCsv")
    leer = lector.return_value.leer
    leer.return_value = _df_lector()

    insumos.cargar_serie("data/s.csv", 2019)

    leer.assert_called_once_with(Path("data/s.csv"))


def test_cargar_serie_arma_serienormalizada_con_recorte_del_lector(mocker) -> None:
    lector = mocker.patch.object(insumos, "LectorSeriesCsv")
    lector.return_value.leer.return_value = _df_lector(recorte="mercado_exportacion")

    resultado = insumos.cargar_serie("x.csv", 2019)

    assert isinstance(resultado, SerieNormalizada)
    assert resultado.recorte == "mercado_exportacion"


def test_cargar_serie_propaga_invariante_violado_del_dominio(mocker) -> None:
    lector = mocker.patch.object(insumos, "LectorSeriesCsv")
    df_invalido = _df_lector()
    df_invalido[PeriodoMensual(2019, 1)] = -1.0  # valor negativo -- invariante del dominio
    lector.return_value.leer.return_value = df_invalido

    with pytest.raises(InvarianteViolado):
        insumos.cargar_serie("x.csv", 2019)


# -- version vs. base detectada en el Título ---------------------------------


def test_cargar_serie_version_no_coincide_con_base_detectada_falla(mocker) -> None:
    lector = mocker.patch.object(insumos, "LectorSeriesCsv")
    lector.return_value.leer.return_value = _df_lector(version_detectada=2012)

    with pytest.raises(VersionNoCoincide):
        insumos.cargar_serie("x.csv", 2025)


def test_cargar_serie_version_coincide_con_base_detectada_no_falla(mocker) -> None:
    lector = mocker.patch.object(insumos, "LectorSeriesCsv")
    lector.return_value.leer.return_value = _df_lector(version_detectada=2019)

    resultado = insumos.cargar_serie("x.csv", 2019)

    assert resultado.df.attrs["version"] == 2019


def test_cargar_serie_sin_version_detectada_no_falla(mocker) -> None:
    # version_detectada=None (Título sin el patrón "Base ... =100") no debe
    # bloquear la carga -- no debería pasar con un archivo real, pero no es
    # motivo para reventar si pasa.
    lector = mocker.patch.object(insumos, "LectorSeriesCsv")
    lector.return_value.leer.return_value = _df_lector(version_detectada=None)

    resultado = insumos.cargar_serie("x.csv", 2019)

    assert resultado.df.attrs["version"] == 2019


# -- fachada pública ----------------------------------------------------------


def test_cargar_serie_esta_en_all_de_la_fachada() -> None:
    assert "cargar_serie" in rep.__all__


def test_rep_cargar_serie_es_la_misma_funcion_que_insumos_cargar_serie() -> None:
    assert rep.cargar_serie is insumos.cargar_serie


def test_rep_cargar_serie_funciona_con_lector_controlado(mocker) -> None:
    lector = mocker.patch.object(insumos, "LectorSeriesCsv")
    lector.return_value.leer.return_value = _df_lector(
        recorte="produccion_total", version_detectada=2019
    )

    resultado = rep.cargar_serie("x.csv", 2019)

    assert isinstance(resultado, SerieNormalizada)
    assert resultado.recorte == "produccion_total"
    assert resultado.df.attrs["version"] == 2019


# -- cruce real contra datos reales -------------------------------------------


@pytest.mark.requires_data
def test_cargar_serie_real_2012_declarado_2025_falla() -> None:
    ruta = DATA_DIR / "produccion_total" / "s12_h_nm_nae.CSV"
    with pytest.raises(VersionNoCoincide):
        insumos.cargar_serie(str(ruta), 2025)


@pytest.mark.requires_data
def test_cargar_serie_real_2012_declarado_2012_no_falla() -> None:
    ruta = DATA_DIR / "produccion_total" / "s12_h_nm_nae.CSV"
    resultado = insumos.cargar_serie(str(ruta), 2012)
    assert resultado.df.attrs["version"] == 2012


# -- version vs. base detectada, camino jerárquico `ae` ----------------------
# (negociación 2026-08-28: la validación de versión no tenía cobertura sobre el
# camino jerárquico -- solo se había probado contra archivos planos `nae`)


@pytest.mark.requires_data
def test_cargar_serie_real_ae_2019_declarado_2025_falla() -> None:
    ruta = DATA_DIR / "produccion_total" / "s19_h_nm_ae.CSV"
    with pytest.raises(VersionNoCoincide):
        insumos.cargar_serie(str(ruta), 2025)


@pytest.mark.requires_data
def test_cargar_serie_real_ae_2019_declarado_2019_no_falla() -> None:
    ruta = DATA_DIR / "produccion_total" / "s19_h_nm_ae.CSV"
    resultado = insumos.cargar_serie(str(ruta), 2019)
    assert resultado.df.attrs["version"] == 2019


# -- bases mezcladas en un mismo archivo (no solo la primera fila) ----------


def test_cargar_serie_bases_mezcladas_en_el_archivo_falla(tmp_path: Path) -> None:
    encabezado = "Instituto Nacional de Estadística y Geografía\n\n\n\n\n"
    df = pd.DataFrame(
        {
            "Título": [
                "Índice nacional de precios productor. Base Julio 2019=100 (SCIAN 2013), "
                "Índices de precios de genéricos para mercado nacional, 1001 Soya",
                "Índice nacional de precios productor. Base Julio 2025=100 (SCIAN 2018), "
                "Índices de precios de genéricos para mercado nacional, 1002 Frijol",
            ],
            "Cifra": ["Indices", "Indices"],
            "Serie": ["11111", "22222"],
            "Ene 2019": ["100.00", "200.00"],
        }
    )
    ruta = tmp_path / "bases_mezcladas.csv"
    ruta.write_text(encabezado + df.to_csv(index=False), encoding="utf-8")

    with pytest.raises(ArchivoCorrupto):
        rep.cargar_serie(str(ruta), 2019)


# -- cargar_canasta: validación de versión -----------------------------------


@pytest.mark.parametrize("version", [2010, 2013, 2018, 2024, 0, 9999])
def test_cargar_canasta_version_invalida(version: int) -> None:
    with pytest.raises(InvarianteViolado):
        insumos.cargar_canasta("x.csv", version)  # type: ignore[arg-type]


@pytest.mark.parametrize("version", [2012, 2019, 2025])
def test_cargar_canasta_version_valida_no_falla_por_version(mocker, version: int) -> None:
    lector = mocker.patch.object(insumos, "LectorCanastaCsv")
    lector.return_value.leer.return_value = _canasta_dummy()

    resultado = insumos.cargar_canasta("x.csv", version)  # type: ignore[arg-type]

    assert isinstance(resultado, CanastaINPP)


# -- cargar_canasta: delegación al lector ------------------------------------


def test_cargar_canasta_delega_al_lector_con_path_y_version(mocker) -> None:
    lector = mocker.patch.object(insumos, "LectorCanastaCsv")
    leer = lector.return_value.leer
    leer.return_value = _canasta_dummy()

    insumos.cargar_canasta("data/c.csv", 2019)

    leer.assert_called_once_with(Path("data/c.csv"), 2019)


def test_cargar_canasta_devuelve_lo_que_retorna_el_lector(mocker) -> None:
    lector = mocker.patch.object(insumos, "LectorCanastaCsv")
    esperado = _canasta_dummy()
    lector.return_value.leer.return_value = esperado

    resultado = insumos.cargar_canasta("x.csv", 2019)

    assert resultado is esperado


# -- cargar_canasta: invariante de índice a través de la API pública ---------
# (negociación 2026-08-26: codigo ausente se colaba como índice "nan")


def test_cargar_canasta_codigo_ausente_falla_via_api_publica(tmp_path: Path) -> None:
    columnas_peso = [
        "produccion total",
        "bienes intermedios",
        "bienes finales",
        "demanda interna total",
        "demanda interna consumo",
        "demanda interna capital",
        "exportaciones",
    ]
    df = pd.DataFrame(
        {
            "codigo": ["", "002"],
            "generico": ["soya", "frijol"],
            "sector": ["11 agricultura", "11 agricultura"],
            "subsector": ["111 agricultura", "111 agricultura"],
            "rama": ["1111 cultivo", "1111 cultivo"],
            "subrama": ["11111 cultivo de soya", "11111 cultivo de soya"],
            "clase": ["111110 cultivo de soya", "111131 cultivo de frijol"],
            **{col: [50.0, 50.0] for col in columnas_peso},
            "encadenamiento total": [None, None],
            "encadenamiento produccion nacional": [None, None],
            "encadenamiento exportacion": [None, None],
            "encadenamiento uso final": [None, None],
        }
    )
    ruta = tmp_path / "codigo_ausente.csv"
    df.to_csv(ruta, index=False)

    with pytest.raises(InvarianteViolado):
        insumos.cargar_canasta(str(ruta), 2019)


# -- cargar_canasta: fachada pública ------------------------------------------


def test_cargar_canasta_esta_en_all_de_la_fachada() -> None:
    assert "cargar_canasta" in rep.__all__


def test_rep_cargar_canasta_es_la_misma_funcion_que_insumos_cargar_canasta() -> None:
    assert rep.cargar_canasta is insumos.cargar_canasta


def test_rep_cargar_canasta_funciona_con_lector_controlado(mocker) -> None:
    lector = mocker.patch.object(insumos, "LectorCanastaCsv")
    esperado = _canasta_dummy()
    lector.return_value.leer.return_value = esperado

    resultado = rep.cargar_canasta("x.csv", 2019)

    assert resultado is esperado


# -- cargar_canasta: cruce real contra datos reales ---------------------------


@pytest.mark.requires_data
@pytest.mark.parametrize(
    "carpeta,version,filas_esperadas",
    [
        ("canasta", 2012, 567),
        ("canasta", 2019, 560),
        ("canasta", 2025, 570),
        ("ponderadores", 2012, 567),
        ("ponderadores", 2019, 560),
        ("ponderadores", 2025, 570),
    ],
)
def test_cargar_canasta_real(carpeta: str, version: int, filas_esperadas: int) -> None:
    ruta = DATA_DIR / carpeta / f"ponderadores_{version}.csv"
    resultado = insumos.cargar_canasta(str(ruta), version)  # type: ignore[arg-type]

    assert isinstance(resultado, CanastaINPP)
    assert resultado.version == version
    assert len(resultado.df) == filas_esperadas
