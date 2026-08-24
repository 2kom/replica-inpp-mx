from __future__ import annotations

from pathlib import Path
from typing import cast

import canasta_inpp.extraccion_xlsx as extraccion_xlsx
import canasta_inpp.registro as registro
import canasta_inpp.utilidades as utilidades
import pandas as pd
import pytest
from generar_canasta import VERSION_ENCADENAMIENTO_OBLIGATORIO, main, parsear_args

# -- helpers ---------------------------------------------------------------


def _xlsx(tmp_path: Path, nombre: str = "ponderadores.xlsx") -> Path:
    ruta = tmp_path / nombre
    ruta.write_bytes(b"fake")
    return ruta


def _dir(tmp_path: Path, nombre: str = "no_es_archivo") -> Path:
    ruta = tmp_path / nombre
    ruta.mkdir()
    return ruta


def _error(capsys: pytest.CaptureFixture[str], argv: list[str]) -> str:
    with pytest.raises(SystemExit) as exc:
        parsear_args(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


def _argv_base(tmp_path: Path, version: int = 2019, **extra: str) -> list[str]:
    """Arma un argv válido (--version/--ponderadores/-o), con overrides opcionales."""
    ponderadores = extra.pop("ponderadores", None) or str(_xlsx(tmp_path))
    salida = extra.pop("salida", None) or str(tmp_path / "salida")
    argv = ["--version", str(version), "--ponderadores", ponderadores, "-o", salida]
    for flag, valor in extra.items():
        argv += [f"--{flag}", valor]
    return argv


# -- parseo válido -----------------------------------------------------------


@pytest.mark.parametrize("version", [2012, 2019, 2025])
def test_version_valida_sin_encadenamiento_obligatorio_o_con_el(
    tmp_path: Path, version: int
) -> None:
    ponderadores = _xlsx(tmp_path)
    salida = tmp_path / "salida"
    argv = ["--version", str(version), "--ponderadores", str(ponderadores), "-o", str(salida)]
    if version == VERSION_ENCADENAMIENTO_OBLIGATORIO:
        argv += ["--encadenamientos", str(_xlsx(tmp_path, "encadenamiento.xlsx"))]

    args = parsear_args(argv)

    assert args.version == version
    assert args.ponderadores == ponderadores
    assert args.salida == salida
    assert args.canasta is None


def test_canasta_opcional_se_acepta(tmp_path: Path) -> None:
    canasta = _xlsx(tmp_path, "canasta.xlsx")
    args = parsear_args(_argv_base(tmp_path, canasta=str(canasta)))
    assert args.canasta == canasta


# -- obligatorios (--version, --ponderadores, -o) -----------------------------


def test_falta_version(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    xlsx = _xlsx(tmp_path)
    err = _error(capsys, ["--ponderadores", str(xlsx), "-o", str(tmp_path)])
    assert "--version" in err
    assert "required" in err


def test_falta_ponderadores(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    err = _error(capsys, ["--version", "2019", "-o", str(tmp_path)])
    assert "--ponderadores" in err
    assert "required" in err


def test_falta_salida(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    xlsx = _xlsx(tmp_path)
    err = _error(capsys, ["--version", "2019", "--ponderadores", str(xlsx)])
    assert "-o" in err
    assert "required" in err


@pytest.mark.parametrize("version", [1999, 2003])
def test_version_fuera_de_choices(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, version: int
) -> None:
    # 2003 explícito -- usaba esquema previo a SCIAN (P/GD/DIV/R/SG), se descartó del
    # todo en vez de diferirlo. Sin este caso, reintroducir 2003 en VERSIONES no
    # rompería ningún test.
    err = _error(capsys, _argv_base(tmp_path, version=version))
    assert "invalid choice" in err


# -- --ponderadores: existencia/tipo -------------------------------------


def test_ponderadores_no_existe(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    err = _error(capsys, _argv_base(tmp_path, ponderadores=str(tmp_path / "no_existe.xlsx")))
    assert "No se encontró --ponderadores" in err


def test_ponderadores_es_directorio(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    directorio = _dir(tmp_path)
    err = _error(capsys, _argv_base(tmp_path, ponderadores=str(directorio)))
    assert "--ponderadores" in err
    assert "directorio" in err


# -- --canasta: existencia/tipo (opcional, pero si viene se valida) ----------


def test_canasta_no_existe(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    err = _error(capsys, _argv_base(tmp_path, canasta=str(tmp_path / "no.xlsx")))
    assert "No se encontró --canasta" in err


def test_canasta_es_directorio(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    directorio = _dir(tmp_path)
    err = _error(capsys, _argv_base(tmp_path, canasta=str(directorio)))
    assert "--canasta" in err
    assert "directorio" in err


# -- --encadenamientos: obligatoriedad condicional a --version 2025 ----------


def test_encadenamientos_obligatorio_falta_en_2025(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    err = _error(capsys, _argv_base(tmp_path, version=2025))
    assert "--encadenamientos es obligatorio" in err
    assert str(VERSION_ENCADENAMIENTO_OBLIGATORIO) in err


@pytest.mark.parametrize("version", [2012, 2019])
def test_encadenamientos_prohibido_fuera_de_2025(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, version: int
) -> None:
    encadenamientos = _xlsx(tmp_path, "encadenamiento.xlsx")
    err = _error(
        capsys, _argv_base(tmp_path, version=version, encadenamientos=str(encadenamientos))
    )
    assert "--encadenamientos solo aplica" in err
    assert str(VERSION_ENCADENAMIENTO_OBLIGATORIO) in err


def test_encadenamientos_no_existe_en_2025(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    err = _error(
        capsys,
        _argv_base(tmp_path, version=2025, encadenamientos=str(tmp_path / "no_existe.xlsx")),
    )
    assert "No se encontró --encadenamientos" in err


def test_encadenamientos_es_directorio_en_2025(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    directorio = _dir(tmp_path)
    err = _error(capsys, _argv_base(tmp_path, version=2025, encadenamientos=str(directorio)))
    assert "--encadenamientos" in err
    assert "directorio" in err


# -- -o: existencia como archivo es error -------------------------------


def test_salida_existe_como_archivo_es_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    salida = tmp_path / "salida_es_archivo.txt"
    salida.write_text("ya existo")
    err = _error(capsys, _argv_base(tmp_path, salida=str(salida)))
    assert "-o" in err
    assert "directorio" in err


def test_salida_como_directorio_ya_existente_es_valida(tmp_path: Path) -> None:
    salida = _dir(tmp_path, "salida")
    args = parsear_args(_argv_base(tmp_path, salida=str(salida)))
    assert args.salida == salida


# -- main(): crea -o y despacha a la extracción --------------------------------


def test_main_falla_parseo_no_crea_directorio_de_salida(tmp_path: Path) -> None:
    salida = tmp_path / "salida"
    with pytest.raises(SystemExit):
        main(["--version", "2019", "-o", str(salida)])  # falta --ponderadores
    assert not salida.exists()


# -- main(): wiring del pipeline (ponderadores base + --canasta/--encadenamientos
# aditivos) -- mockea los 7 pasos de canasta_inpp para probar el flujo de main()
# sin depender de xlsx reales. ---------------------------------------------------


def _df_ponderadores(codigos: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "codigo": codigos,
            "generico": [f"generico base {c}" for c in codigos],
            "sector": [f"sector base {c}" for c in codigos],
            "subsector": [f"subsector base {c}" for c in codigos],
            "rama": [f"rama base {c}" for c in codigos],
            "subrama": [f"subrama base {c}" for c in codigos],
            "clase": [f"clase base {c}" for c in codigos],
            "produccion total": ["1"] * len(codigos),
        }
    )


def _df_canasta(codigos: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "codigo": codigos,
            "generico": [f"generico canasta {c}" for c in codigos],
            "sector": [f"sector canasta {c}" for c in codigos],
            "subsector": [f"subsector canasta {c}" for c in codigos],
            "rama": [f"rama canasta {c}" for c in codigos],
            "subrama": [f"subrama canasta {c}" for c in codigos],
            "clase": [f"clase canasta {c}" for c in codigos],
        }
    )


def _df_encadenamiento(codigos: list[str]) -> pd.DataFrame:
    # valor distinto por código (no una constante) -- para que un test que compare
    # valores exactos detecte un merge que trae la fila equivocada.
    return pd.DataFrame(
        {"codigo": codigos, "encadenamiento total": [f"{codigo}.5" for codigo in codigos]}
    )


class _Pipeline:
    """Mockea los 7 pasos que `main()` importa localmente y registra cómo los llama.

    `resolver_sector_agrupado`/`normalizar_columnas_*` se mockean como identidad
    (devuelven el df tal cual) para aislar el wiring del propio `main()` de la
    lógica de esas funciones, que ya tiene sus tests en `test_utilidades.py`.
    """

    ruta_ponderadores: Path
    ruta_canasta: Path
    ruta_encadenamiento: Path
    columnas_normalizadas_codigo: list[str]
    columnas_normalizadas_texto: list[str]
    registro: dict[str, object]

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.llamadas: list[str] = []
        self.guardado: dict[str, object] = {}
        self._ponderadores = _df_ponderadores(["100"])
        self._canasta = _df_canasta(["100"])
        self._encadenamiento = _df_encadenamiento(["100"])

        def extraer_ponderadores(ruta: Path, _version: int) -> pd.DataFrame:
            self.llamadas.append("extraer_ponderadores")
            self.ruta_ponderadores = ruta
            return self._ponderadores.copy()

        def extraer_canasta(ruta: Path, _version: int) -> pd.DataFrame:
            self.llamadas.append("extraer_canasta")
            self.ruta_canasta = ruta
            return self._canasta.copy()

        def extraer_encadenamiento(ruta: Path) -> pd.DataFrame:
            self.llamadas.append("extraer_encadenamiento")
            self.ruta_encadenamiento = ruta
            return self._encadenamiento.copy()

        def resolver_sector_agrupado(df: pd.DataFrame) -> pd.DataFrame:
            self.llamadas.append("resolver_sector_agrupado")
            return df

        def normalizar_columnas_con_codigo(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
            self.llamadas.append("normalizar_columnas_con_codigo")
            self.columnas_normalizadas_codigo = list(columnas)
            return df

        def normalizar_columnas_texto(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
            self.llamadas.append("normalizar_columnas_texto")
            self.columnas_normalizadas_texto = list(columnas)
            return df

        def guardar_csv(df: pd.DataFrame, ruta: Path, version: int) -> None:
            self.llamadas.append("guardar_csv")
            self.guardado = {"df": df, "ruta": ruta, "version": version}

        def escribir_registro(df: pd.DataFrame, **kwargs: object) -> Path:
            self.llamadas.append("escribir_registro")
            self.registro = {"df": df, **kwargs}
            return cast(Path, kwargs["ruta_salida"]) / "registro.json"

        monkeypatch.setattr(extraccion_xlsx, "extraer_ponderadores", extraer_ponderadores)
        monkeypatch.setattr(extraccion_xlsx, "extraer_canasta", extraer_canasta)
        monkeypatch.setattr(extraccion_xlsx, "extraer_encadenamiento", extraer_encadenamiento)
        monkeypatch.setattr(utilidades, "resolver_sector_agrupado", resolver_sector_agrupado)
        monkeypatch.setattr(
            utilidades, "normalizar_columnas_con_codigo", normalizar_columnas_con_codigo
        )
        monkeypatch.setattr(utilidades, "normalizar_columnas_texto", normalizar_columnas_texto)
        monkeypatch.setattr(utilidades, "guardar_csv", guardar_csv)
        monkeypatch.setattr(registro, "escribir_registro", escribir_registro)


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> _Pipeline:
    return _Pipeline(monkeypatch)


@pytest.mark.usefixtures("pipeline")
def test_main_crea_directorio_de_salida_anidado(tmp_path: Path) -> None:
    salida = tmp_path / "salida" / "anidada"
    assert not salida.exists()

    main(_argv_base(tmp_path, salida=str(salida)))

    assert salida.is_dir()


def test_main_solo_ponderadores_no_llama_canasta_ni_encadenamiento(
    tmp_path: Path, pipeline: _Pipeline
) -> None:
    main(_argv_base(tmp_path, version=2019))

    assert pipeline.llamadas == [
        "extraer_ponderadores",
        "resolver_sector_agrupado",
        "normalizar_columnas_con_codigo",
        "normalizar_columnas_texto",
        "guardar_csv",
        "escribir_registro",
    ]
    assert pipeline.columnas_normalizadas_codigo == [
        "sector",
        "subsector",
        "rama",
        "subrama",
        "clase",
    ]
    assert pipeline.columnas_normalizadas_texto == ["generico"]
    assert pipeline.guardado["ruta"] == tmp_path / "salida" / "ponderadores_2019.csv"
    assert pipeline.guardado["version"] == 2019
    assert pipeline.registro["version"] == 2019
    assert pipeline.registro["xlsx_ponderadores"] == pipeline.ruta_ponderadores
    assert pipeline.registro["xlsx_canasta"] is None
    assert pipeline.registro["xlsx_encadenamientos"] is None
    assert pipeline.registro["ruta_csv"] == pipeline.guardado["ruta"]
    assert pipeline.registro["ruta_salida"] == tmp_path / "salida"
    assert pipeline.registro["df"] is pipeline.guardado["df"]


def test_main_con_canasta_reconcilia_114_a_113_solo_en_2019(
    tmp_path: Path, pipeline: _Pipeline
) -> None:
    pipeline._ponderadores = _df_ponderadores(["100", "114"])
    pipeline._canasta = _df_canasta(["100", "113"])
    canasta = _xlsx(tmp_path, "canasta.xlsx")

    main(_argv_base(tmp_path, version=2019, canasta=str(canasta)))

    assert pipeline.llamadas[:2] == ["extraer_ponderadores", "extraer_canasta"]
    df = cast(pd.DataFrame, pipeline.guardado["df"])
    assert sorted(df["codigo"]) == ["100", "113"]
    # la fila que era 114 se cruzó como 113 -- trae la jerarquía de --canasta, no la
    # de --ponderadores (esas columnas se descartan antes del merge).
    fila = df.loc[df["codigo"] == "113"].iloc[0]
    assert fila["generico"] == "generico canasta 113"
    # el peso (columna que no es de jerarquía) se preserva desde --ponderadores.
    assert list(df["produccion total"]) == ["1", "1"]


def test_main_con_canasta_no_reconcilia_fuera_de_2019(tmp_path: Path, pipeline: _Pipeline) -> None:
    pipeline._ponderadores = _df_ponderadores(["100", "114"])
    pipeline._canasta = _df_canasta(["100", "114"])
    canasta = _xlsx(tmp_path, "canasta.xlsx")

    main(_argv_base(tmp_path, version=2012, canasta=str(canasta)))

    df = cast(pd.DataFrame, pipeline.guardado["df"])
    assert sorted(df["codigo"]) == ["100", "114"]


def test_main_con_canasta_codigos_faltantes_en_canasta_falla(
    tmp_path: Path, pipeline: _Pipeline
) -> None:
    # ej. real: --canasta apunta al xlsx de otro año por error -- nada en el CLI
    # valida que --ponderadores/--canasta sean de la misma versión.
    pipeline._ponderadores = _df_ponderadores(["100", "999"])
    pipeline._canasta = _df_canasta(["100"])  # sin "999"
    canasta = _xlsx(tmp_path, "canasta.xlsx")

    with pytest.raises(ValueError, match="999"):
        main(_argv_base(tmp_path, version=2012, canasta=str(canasta)))

    assert "guardar_csv" not in pipeline.llamadas


def test_main_con_canasta_codigos_sobrantes_en_canasta_falla(
    tmp_path: Path, pipeline: _Pipeline
) -> None:
    # el lado contrario: --canasta trae un código que --ponderadores no tiene -- un
    # left merge lo descartaría en silencio si solo se chequeara el sentido faltante.
    pipeline._ponderadores = _df_ponderadores(["100"])
    pipeline._canasta = _df_canasta(["100", "999"])  # "999" no está en --ponderadores
    canasta = _xlsx(tmp_path, "canasta.xlsx")

    with pytest.raises(ValueError, match="999"):
        main(_argv_base(tmp_path, version=2012, canasta=str(canasta)))

    assert "guardar_csv" not in pipeline.llamadas


def test_main_con_encadenamientos_hace_merge_aditivo(tmp_path: Path, pipeline: _Pipeline) -> None:
    pipeline._ponderadores = _df_ponderadores(["100", "200"])
    pipeline._encadenamiento = _df_encadenamiento(["100", "200"])
    encadenamientos = _xlsx(tmp_path, "encadenamiento.xlsx")

    main(_argv_base(tmp_path, version=2025, encadenamientos=str(encadenamientos)))

    assert pipeline.llamadas == [
        "extraer_ponderadores",
        "extraer_encadenamiento",
        "resolver_sector_agrupado",
        "normalizar_columnas_con_codigo",
        "normalizar_columnas_texto",
        "guardar_csv",
        "escribir_registro",
    ]
    df = cast(pd.DataFrame, pipeline.guardado["df"])
    assert len(df) == 2
    assert dict(zip(df["codigo"], df["encadenamiento total"], strict=True)) == {
        "100": "100.5",
        "200": "200.5",
    }
    assert pipeline.guardado["ruta"] == tmp_path / "salida" / "ponderadores_2025.csv"
    assert pipeline.registro["version"] == 2025
    assert pipeline.registro["xlsx_ponderadores"] == pipeline.ruta_ponderadores
    assert pipeline.registro["xlsx_canasta"] is None
    assert pipeline.registro["xlsx_encadenamientos"] == pipeline.ruta_encadenamiento
    assert pipeline.registro["ruta_csv"] == pipeline.guardado["ruta"]
    assert pipeline.registro["ruta_salida"] == tmp_path / "salida"
    assert pipeline.registro["df"] is pipeline.guardado["df"]


def test_main_con_encadenamientos_codigos_sobrantes_falla(
    tmp_path: Path, pipeline: _Pipeline
) -> None:
    # mismo chequeo bidireccional que --canasta: un código exclusivo de
    # --encadenamientos no debe descartarse en silencio por el left merge.
    pipeline._ponderadores = _df_ponderadores(["100"])
    pipeline._encadenamiento = _df_encadenamiento(["100", "999"])
    encadenamientos = _xlsx(tmp_path, "encadenamiento.xlsx")

    with pytest.raises(ValueError, match="999"):
        main(_argv_base(tmp_path, version=2025, encadenamientos=str(encadenamientos)))

    assert "guardar_csv" not in pipeline.llamadas


def test_main_con_canasta_y_encadenamientos_aplica_canasta_primero(
    tmp_path: Path, pipeline: _Pipeline
) -> None:
    pipeline._ponderadores = _df_ponderadores(["100", "200"])
    pipeline._canasta = _df_canasta(["100", "200"])
    pipeline._encadenamiento = _df_encadenamiento(["100", "200"])
    canasta = _xlsx(tmp_path, "canasta.xlsx")
    encadenamientos = _xlsx(tmp_path, "encadenamiento.xlsx")

    main(
        _argv_base(
            tmp_path, version=2025, canasta=str(canasta), encadenamientos=str(encadenamientos)
        )
    )

    assert pipeline.llamadas == [
        "extraer_ponderadores",
        "extraer_canasta",
        "extraer_encadenamiento",
        "resolver_sector_agrupado",
        "normalizar_columnas_con_codigo",
        "normalizar_columnas_texto",
        "guardar_csv",
        "escribir_registro",
    ]
    df = cast(pd.DataFrame, pipeline.guardado["df"])
    assert len(df) == 2
    assert dict(zip(df["codigo"], df["generico"], strict=True)) == {
        "100": "generico canasta 100",
        "200": "generico canasta 200",
    }
    assert dict(zip(df["codigo"], df["encadenamiento total"], strict=True)) == {
        "100": "100.5",
        "200": "200.5",
    }
    assert pipeline.registro["version"] == 2025
    assert pipeline.registro["xlsx_ponderadores"] == pipeline.ruta_ponderadores
    assert pipeline.registro["xlsx_canasta"] == pipeline.ruta_canasta
    assert pipeline.registro["xlsx_encadenamientos"] == pipeline.ruta_encadenamiento
    assert pipeline.registro["ruta_csv"] == pipeline.guardado["ruta"]
    assert pipeline.registro["ruta_salida"] == tmp_path / "salida"
    assert pipeline.registro["df"] is pipeline.guardado["df"]
