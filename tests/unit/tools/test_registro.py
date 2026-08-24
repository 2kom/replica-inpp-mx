from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from canasta_inpp.registro import escribir_registro

# -- helpers ------------------------------------------------------------


def _df(
    codigos: list[str],
    *,
    con_encadenamiento: bool = False,
    sectores: list[str] | None = None,
) -> pd.DataFrame:
    n = len(codigos)
    datos: dict[str, object] = {
        "codigo": codigos,
        "generico": [f"generico {c}" for c in codigos],
        "sector": sectores or ["11 agricultura"] * n,
        "subsector": ["111 cultivo"] * n,
        "rama": ["1111 arroz"] * n,
        "subrama": ["11111 arroz"] * n,
        "clase": ["111110 arroz"] * n,
        "produccion total": ["1"] * n,
        "bienes intermedios": ["1"] * n,
        "bienes finales": [float("nan")] * n,
        "demanda interna total": ["1"] * n,
        "demanda interna consumo": ["1"] * n,
        "demanda interna capital": ["1"] * n,
        "exportaciones": [float("nan")] * n,
    }
    if con_encadenamiento:
        datos["encadenamiento total"] = ["1.01"] * n
        datos["encadenamiento produccion nacional"] = ["1.01"] * n
        datos["encadenamiento exportacion"] = [float("nan")] * n
        datos["encadenamiento uso final"] = ["1.01"] * n
    return pd.DataFrame(datos)


def _escribir(
    tmp_path: Path,
    df: pd.DataFrame,
    *,
    version: int = 2019,
    xlsx_canasta: Path | None = None,
    xlsx_encadenamientos: Path | None = None,
) -> Path:
    return escribir_registro(
        df,
        version=version,  # type: ignore[arg-type]
        xlsx_ponderadores=Path("entrada/ponderadores.xlsx"),
        xlsx_canasta=xlsx_canasta,
        xlsx_encadenamientos=xlsx_encadenamientos,
        ruta_csv=tmp_path / f"ponderadores_{version}.csv",
        ruta_salida=tmp_path,
    )


def _leer_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


# -- nombre de archivo ---------------------------------------------------


def test_nombre_de_archivo_con_version_timestamp_y_uuid(tmp_path: Path) -> None:
    ruta = _escribir(tmp_path, _df(["100"]), version=2019)
    assert re.match(r"^canasta_2019_\d{8}_\d{6}_\d{6}_[0-9a-f]{8}\.json$", ruta.name)


def test_dos_corridas_seguidas_no_se_pisan(tmp_path: Path) -> None:
    _escribir(tmp_path, _df(["100"]))
    _escribir(tmp_path, _df(["100"]))
    assert len(list(tmp_path.glob("canasta_*.json"))) == 2


# -- campos basicos -------------------------------------------------------


def test_campos_basicos_del_json(tmp_path: Path) -> None:
    ruta = _escribir(tmp_path, _df(["100", "200"]), version=2012)
    registro = _leer_json(ruta)
    assert registro["xlsx_ponderadores"] == str(Path("entrada/ponderadores.xlsx"))
    assert registro["csv"] == str(tmp_path / "ponderadores_2012.csv")
    assert registro["version"] == 2012
    assert registro["genericos"] == 2


def test_xlsx_canasta_none_si_no_vino_el_flag(tmp_path: Path) -> None:
    ruta = _escribir(tmp_path, _df(["100"]), xlsx_canasta=None)
    assert _leer_json(ruta)["xlsx_canasta"] is None


def test_xlsx_canasta_presente_si_vino_el_flag(tmp_path: Path) -> None:
    ruta = _escribir(tmp_path, _df(["100"]), xlsx_canasta=Path("entrada/canasta.xlsx"))
    assert _leer_json(ruta)["xlsx_canasta"] == str(Path("entrada/canasta.xlsx"))


def test_xlsx_encadenamientos_none_si_no_vino_el_flag(tmp_path: Path) -> None:
    ruta = _escribir(tmp_path, _df(["100"]), xlsx_encadenamientos=None)
    assert _leer_json(ruta)["xlsx_encadenamientos"] is None


def test_xlsx_encadenamientos_presente_si_vino_el_flag(tmp_path: Path) -> None:
    ruta = _escribir(
        tmp_path,
        _df(["100"], con_encadenamiento=True),
        xlsx_encadenamientos=Path("entrada/encadenamiento.xlsx"),
    )
    assert _leer_json(ruta)["xlsx_encadenamientos"] == str(Path("entrada/encadenamiento.xlsx"))


def test_devuelve_la_ruta_del_json(tmp_path: Path) -> None:
    ruta = _escribir(tmp_path, _df(["100"]))
    assert ruta.exists()
    assert ruta.parent == tmp_path


# -- pesos ------------------------------------------------------------------


def test_pesos_cuenta_no_vacios_por_columna(tmp_path: Path) -> None:
    # "bienes finales"/"exportaciones" vienen NaN en el helper _df -- deben
    # contarse en 0, el resto (con valor real) en 1.
    ruta = _escribir(tmp_path, _df(["100"]))
    pesos = _leer_json(ruta)["pesos"]
    assert pesos == {
        "produccion total": 1,
        "bienes intermedios": 1,
        "bienes finales": 0,
        "demanda interna total": 1,
        "demanda interna consumo": 1,
        "demanda interna capital": 1,
        "exportaciones": 0,
    }


def test_pesos_siempre_presente_aunque_no_venga_canasta_ni_encadenamientos(
    tmp_path: Path,
) -> None:
    # las 7 columnas de peso salen de --ponderadores solo, no dependen de
    # --canasta/--encadenamientos
    ruta = _escribir(tmp_path, _df(["100"]), xlsx_canasta=None, xlsx_encadenamientos=None)
    assert set(_leer_json(ruta)["pesos"]) == {
        "produccion total",
        "bienes intermedios",
        "bienes finales",
        "demanda interna total",
        "demanda interna consumo",
        "demanda interna capital",
        "exportaciones",
    }


# -- encadenamiento -----------------------------------------------------------


def test_encadenamiento_none_si_no_vino_el_flag(tmp_path: Path) -> None:
    ruta = _escribir(tmp_path, _df(["100"]), xlsx_encadenamientos=None)
    assert _leer_json(ruta)["encadenamiento"] is None


def test_encadenamiento_cuenta_no_vacios_por_columna_si_vino_el_flag(tmp_path: Path) -> None:
    # "encadenamiento exportacion" viene NaN en el helper _df (N/A legitimo) --
    # debe contarse en 0, el resto en 1.
    ruta = _escribir(
        tmp_path,
        _df(["100"], con_encadenamiento=True),
        xlsx_encadenamientos=Path("entrada/encadenamiento.xlsx"),
    )
    assert _leer_json(ruta)["encadenamiento"] == {
        "encadenamiento total": 1,
        "encadenamiento produccion nacional": 1,
        "encadenamiento exportacion": 0,
        "encadenamiento uso final": 1,
    }


# -- clasificaciones ----------------------------------------------------------


def test_clasificaciones_trae_las_5_columnas_jerarquicas(tmp_path: Path) -> None:
    ruta = _escribir(tmp_path, _df(["100"]))
    assert set(_leer_json(ruta)["clasificaciones"]) == {
        "sector",
        "subsector",
        "rama",
        "subrama",
        "clase",
    }


def test_clasificaciones_cuenta_genericos_por_categoria(tmp_path: Path) -> None:
    df = _df(
        ["100", "200", "300"],
        sectores=["11 agricultura", "11 agricultura", "31-33 industrias"],
    )
    ruta = _escribir(tmp_path, df)
    assert _leer_json(ruta)["clasificaciones"]["sector"] == {
        "11 agricultura": {"genericos": 2},
        "31-33 industrias": {"genericos": 1},
    }


def test_clasificaciones_sin_metodo_ni_origen(tmp_path: Path) -> None:
    # a diferencia de replica-inpc-mx (cruce texto<->texto con decision entre
    # 2 fuentes), aca no hay ninguna decision que auditar por categoria
    ruta = _escribir(tmp_path, _df(["100"]))
    categoria = next(iter(_leer_json(ruta)["clasificaciones"]["sector"].values()))
    assert set(categoria) == {"genericos"}


def test_clasificaciones_ignora_jerarquia_vacia_o_solo_espacios(tmp_path: Path) -> None:
    # "" es el mismo sentinel con el que extraer_canasta inicializa su
    # maquina de estados si un generico aparece antes de su fila de
    # sector/subsector/etc -- no debe aparecer como si fuera una categoria real.
    df = _df(
        ["100", "200", "300"],
        sectores=["11 agricultura", "", "   "],
    )
    ruta = _escribir(tmp_path, df)
    sector = _leer_json(ruta)["clasificaciones"]["sector"]
    assert "" not in sector
    assert "   " not in sector
    assert sector == {"11 agricultura": {"genericos": 1}}


# -- codificacion / stdout -----------------------------------------------------


def test_preserva_acentos_sin_escapar(tmp_path: Path) -> None:
    df = _df(["100"], sectores=["31-33 fabricación de alimentos, bebidas y tabaco"])
    ruta = _escribir(tmp_path, df)
    assert "fabricación" in ruta.read_text(encoding="utf-8")


def test_imprime_resumen_a_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ruta_csv = tmp_path / "ponderadores_2019.csv"
    escribir_registro(
        _df(["100"]),
        version=2019,
        xlsx_ponderadores=Path("entrada/ponderadores.xlsx"),
        xlsx_canasta=None,
        xlsx_encadenamientos=None,
        ruta_csv=ruta_csv,
        ruta_salida=tmp_path,
    )
    salida = capsys.readouterr().out
    assert "version 2019: 1 genericos extraidos" in salida
    assert str(ruta_csv) in salida


def test_df_vacio_no_lanza(tmp_path: Path) -> None:
    ruta = _escribir(tmp_path, _df([]))
    registro = _leer_json(ruta)
    assert registro["genericos"] == 0
    assert registro["pesos"] == {
        "produccion total": 0,
        "bienes intermedios": 0,
        "bienes finales": 0,
        "demanda interna total": 0,
        "demanda interna consumo": 0,
        "demanda interna capital": 0,
        "exportaciones": 0,
    }
    assert registro["clasificaciones"]["sector"] == {}
