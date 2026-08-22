from __future__ import annotations

from pathlib import Path

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


@pytest.mark.parametrize("version", [2003, 2012, 2019, 2025])
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


def test_version_fuera_de_choices(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    err = _error(capsys, _argv_base(tmp_path, version=1999))
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


@pytest.mark.parametrize("version", [2003, 2012, 2019])
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


# -- main(): crea -o y despacha a la extracción (todavía sin implementar) ----


def test_main_crea_directorio_de_salida_antes_de_fallar(tmp_path: Path) -> None:
    salida = tmp_path / "salida" / "anidada"
    assert not salida.exists()

    with pytest.raises(NotImplementedError):
        main(_argv_base(tmp_path, salida=str(salida)))

    assert salida.is_dir()


def test_main_falla_parseo_no_crea_directorio_de_salida(tmp_path: Path) -> None:
    salida = tmp_path / "salida"
    with pytest.raises(SystemExit):
        main(["--version", "2019", "-o", str(salida)])  # falta --ponderadores
    assert not salida.exists()
