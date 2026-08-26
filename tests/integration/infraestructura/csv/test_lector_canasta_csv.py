from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from replica_inpp.dominio.errores import (
    ArchivoCorrupto,
    ArchivoNoEncontrado,
    ArchivoVacio,
    ColumnasMinFaltantes,
)
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.infraestructura.csv.lector_canasta_csv import LectorCanastaCsv

DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "inputs"

_COLUMNAS_PESO = (
    "produccion total",
    "bienes intermedios",
    "bienes finales",
    "demanda interna total",
    "demanda interna consumo",
    "demanda interna capital",
    "exportaciones",
)

# CSV sintético mínimo con nombre combinado ("11 agricultura...") en las columnas
# jerárquicas -- reproduce la variante `canasta/` real.
_df_combinado = pd.DataFrame(
    {
        "codigo": ["001", "002"],
        "generico": ["soya", "frijol"],
        "sector": [
            "11 agricultura cria y explotacion de animales",
            "11 agricultura cria y explotacion de animales",
        ],
        "subsector": ["111 agricultura", "111 agricultura"],
        "rama": ["1111 cultivo de semillas oleaginosas", "1111 cultivo de semillas oleaginosas"],
        "subrama": ["11111 cultivo de soya", "11113 cultivo de leguminosas"],
        "clase": ["111110 cultivo de soya", "111131 cultivo de frijol"],
        **{col: [50.0, 50.0] for col in _COLUMNAS_PESO},
        "encadenamiento total": [None, None],
        "encadenamiento produccion nacional": [None, None],
        "encadenamiento exportacion": [None, None],
        "encadenamiento uso final": [None, None],
    }
)

# Misma info, variante `ponderadores/` real: columnas jerárquicas bare (solo código).
_df_bare = _df_combinado.assign(
    sector=["11", "11"],
    subsector=["111", "111"],
    rama=["1111", "1111"],
    subrama=["11111", "11113"],
    clase=["111110", "111131"],
)


def _escribir(ruta: Path, df: pd.DataFrame) -> None:
    df.to_csv(ruta, index=False)


# ---------- Errores de archivo ----------


def test_archivo_no_encontrado() -> None:
    with pytest.raises(ArchivoNoEncontrado):
        LectorCanastaCsv().leer(Path("no_existe.csv"), 2019)


def test_archivo_vacio(tmp_path: Path) -> None:
    ruta = tmp_path / "vacio.csv"
    ruta.touch()
    with pytest.raises(ArchivoVacio):
        LectorCanastaCsv().leer(ruta, 2019)


def test_columnas_min_faltantes(tmp_path: Path) -> None:
    ruta = tmp_path / "incompleto.csv"
    _escribir(ruta, _df_combinado.drop(columns=["exportaciones"]))
    with pytest.raises(ColumnasMinFaltantes):
        LectorCanastaCsv().leer(ruta, 2019)


# ---------- Separación código/nombre jerárquico ----------


def test_columna_combinada_agrega_codigo_a_la_izquierda(tmp_path: Path) -> None:
    ruta = tmp_path / "combinado.csv"
    _escribir(ruta, _df_combinado)
    resultado = LectorCanastaCsv().leer(ruta, 2019)

    assert list(resultado.df["codigo sector"]) == ["11", "11"]
    assert resultado.df.loc["001", "sector"] == "11 agricultura cria y explotacion de animales"


def test_columna_bare_copia_el_codigo_sin_tocar_la_original(tmp_path: Path) -> None:
    ruta = tmp_path / "bare.csv"
    _escribir(ruta, _df_bare)
    resultado = LectorCanastaCsv().leer(ruta, 2019)

    assert list(resultado.df["codigo sector"]) == ["11", "11"]
    assert resultado.df.loc["001", "sector"] == "11"


def test_columnas_jerarquicas_se_leen_como_str_preserva_cero_izquierda(tmp_path: Path) -> None:
    ruta = tmp_path / "cero_izquierda.csv"
    df = _df_bare.assign(sector=["01", "01"])
    _escribir(ruta, df)
    resultado = LectorCanastaCsv().leer(ruta, 2019)

    assert resultado.df["codigo sector"].tolist() == ["01", "01"]


def test_combinado_y_bare_dan_el_mismo_codigo(tmp_path: Path) -> None:
    ruta_combinado = tmp_path / "combinado.csv"
    ruta_bare = tmp_path / "bare.csv"
    _escribir(ruta_combinado, _df_combinado)
    _escribir(ruta_bare, _df_bare)

    combinado = LectorCanastaCsv().leer(ruta_combinado, 2019)
    bare = LectorCanastaCsv().leer(ruta_bare, 2019)

    assert list(combinado.df["codigo sector"]) == list(bare.df["codigo sector"])
    assert list(combinado.df["codigo clase"]) == list(bare.df["codigo clase"])


def test_columna_mezcla_combinado_y_bare_falla(tmp_path: Path) -> None:
    ruta = tmp_path / "mezclado.csv"
    df = _df_combinado.copy()
    df.loc[1, "sector"] = "11"  # bare, mientras la fila 0 sigue combinada
    _escribir(ruta, df)
    with pytest.raises(ArchivoCorrupto):
        LectorCanastaCsv().leer(ruta, 2019)


def test_columna_con_valor_irreconocible_falla(tmp_path: Path) -> None:
    ruta = tmp_path / "irreconocible.csv"
    df = _df_combinado.copy()
    df.loc[1, "clase"] = "sin codigo"
    _escribir(ruta, df)
    with pytest.raises(ArchivoCorrupto):
        LectorCanastaCsv().leer(ruta, 2019)


# ---------- Marcador N/A de encadenamiento ----------


def test_na_en_encadenamiento_parcial_se_lee_como_nan(tmp_path: Path) -> None:
    ruta = tmp_path / "con_na.csv"
    df = _df_combinado.assign(**{"encadenamiento exportacion": ["1.5", "N/A"]})
    _escribir(ruta, df)
    resultado = LectorCanastaCsv().leer(ruta, 2025)

    assert resultado.df["encadenamiento exportacion"].isna().tolist() == [False, True]


# ---------- Devuelve CanastaINPP ----------


def test_leer_devuelve_canastainpp(tmp_path: Path) -> None:
    ruta = tmp_path / "combinado.csv"
    _escribir(ruta, _df_combinado)
    resultado = LectorCanastaCsv().leer(ruta, 2019)

    assert isinstance(resultado, CanastaINPP)
    assert resultado.version == 2019


# ---------- Datos reales ----------


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
def test_lector_canasta_csv_real(carpeta: str, version: int, filas_esperadas: int) -> None:
    ruta = DATA_DIR / carpeta / f"ponderadores_{version}.csv"
    resultado = LectorCanastaCsv().leer(ruta, version)  # type: ignore[arg-type]

    assert isinstance(resultado, CanastaINPP)
    assert len(resultado.df) == filas_esperadas
    assert not resultado.df.index.duplicated().any()


@pytest.mark.requires_data
def test_lector_canasta_csv_real_combinado_y_bare_dan_mismo_codigo_2019() -> None:
    combinado = LectorCanastaCsv().leer(DATA_DIR / "canasta" / "ponderadores_2019.csv", 2019)
    bare = LectorCanastaCsv().leer(DATA_DIR / "ponderadores" / "ponderadores_2019.csv", 2019)

    assert list(combinado.df["codigo sector"].sort_index()) == list(
        bare.df["codigo sector"].sort_index()
    )
