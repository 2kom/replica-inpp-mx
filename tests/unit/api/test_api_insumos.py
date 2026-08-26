from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import replica_inpp as rep
from replica_inpp.api import insumos
from replica_inpp.dominio.errores import ArchivoCorrupto, InvarianteViolado, VersionNoCoincide
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.periodos import PeriodoMensual

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "inputs"


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
