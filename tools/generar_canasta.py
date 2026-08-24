"""
Generador de canastas INPP.

Extrae datos de archivos xlsx del INEGI y genera un archivo CSV intermedio
(ponderadores_<version>.csv) para el pipeline de réplica del INPP.

Uso:
    python tools/generar_canasta.py --version 2003 --ponderadores ruta.xlsx -o salida/
    python tools/generar_canasta.py --version 2019 --ponderadores ruta.xlsx --canasta ruta.xlsx -o salida/
    python tools/generar_canasta.py --version 2025 --ponderadores ruta.xlsx --canasta ruta.xlsx \\
        --encadenamientos ruta.xlsx -o salida/

Ver: docs/requerimientos/explicacion.md (procedimiento de encadenamiento).
"""

import argparse
from pathlib import Path

VERSIONES = (2003, 2012, 2019, 2025)

# Única versión que trae su propio archivo de factor de encadenamiento
# (docs/requerimientos/xlsx/2025/factor_de_encadenamiento_ti.xlsx) — ver
# docs/requerimientos/explicacion.md §5.
VERSION_ENCADENAMIENTO_OBLIGATORIO = 2025


def parsear_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y parsea los flags del CLI, valida su combinación."""
    parser = argparse.ArgumentParser(
        description="Genera archivos CSV de canastas INPP a partir de fuentes xlsx del INEGI.",
    )

    parser.add_argument(
        "--version",
        type=int,
        choices=VERSIONES,
        required=True,
        help="Versión de canasta a extraer.",
    )
    parser.add_argument(
        "--ponderadores",
        type=Path,
        required=True,
        help=(
            "Ruta al xlsx con código SCIAN + ponderador por genérico "
            "(en 2003 también trae el nombre completo del genérico)."
        ),
    )
    parser.add_argument(
        "--canasta",
        type=Path,
        help=(
            "Opcional. Ruta al xlsx con el árbol SCIAN completo (nombres de "
            "Sector/Subsector/Rama/Subrama/Clase) — no aplica a 2003, esa "
            "versión ya trae el nombre del genérico en --ponderadores."
        ),
    )
    parser.add_argument(
        "--encadenamientos",
        type=Path,
        help=(
            f"Ruta al xlsx con el factor de encadenamiento por genérico. "
            f"Obligatorio únicamente con --version {VERSION_ENCADENAMIENTO_OBLIGATORIO}."
        ),
    )
    parser.add_argument(
        "-o",
        type=Path,
        dest="salida",
        required=True,
        help="Directorio de salida para el CSV.",
    )

    args = parser.parse_args(argv)
    _validar_args(args, parser)
    return args


def _validar_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Valida existencia/tipo de rutas y la obligatoriedad condicional de --encadenamientos."""
    if not args.ponderadores.exists():
        parser.error(f"No se encontró --ponderadores: {args.ponderadores}")
    if not args.ponderadores.is_file():
        parser.error(f"--ponderadores debe ser un archivo, no un directorio: {args.ponderadores}")

    if args.canasta is not None:
        if not args.canasta.exists():
            parser.error(f"No se encontró --canasta: {args.canasta}")
        if not args.canasta.is_file():
            parser.error(f"--canasta debe ser un archivo, no un directorio: {args.canasta}")

    if args.version == VERSION_ENCADENAMIENTO_OBLIGATORIO and args.encadenamientos is None:
        parser.error(
            f"--encadenamientos es obligatorio con --version {VERSION_ENCADENAMIENTO_OBLIGATORIO}."
        )
    if args.encadenamientos is not None:
        if args.version != VERSION_ENCADENAMIENTO_OBLIGATORIO:
            parser.error(
                f"--encadenamientos solo aplica a --version {VERSION_ENCADENAMIENTO_OBLIGATORIO}."
            )
        if not args.encadenamientos.exists():
            parser.error(f"No se encontró --encadenamientos: {args.encadenamientos}")
        if not args.encadenamientos.is_file():
            parser.error(
                f"--encadenamientos debe ser un archivo, no un directorio: {args.encadenamientos}"
            )

    if args.salida.exists() and not args.salida.is_dir():
        parser.error(f"-o debe ser un directorio: {args.salida}")


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada del CLI: parsea args, extrae, cruza y guarda el CSV.

    Base siempre igual: extrae `--ponderadores`. `--canasta`/`--encadenamientos`
    son aditivos encima de esa base, no modos separados.
    """
    args = parsear_args(argv)
    args.salida.mkdir(parents=True, exist_ok=True)

    from canasta_inpp.extraccion_xlsx import (
        extraer_canasta,
        extraer_encadenamiento,
        extraer_ponderadores,
    )
    from canasta_inpp.utilidades import (
        guardar_csv,
        normalizar_columnas_con_codigo,
        normalizar_columnas_texto,
        resolver_sector_agrupado,
    )

    df = extraer_ponderadores(args.ponderadores, args.version)

    if args.canasta is not None:
        if args.version == 2019:
            # discrepancia real de fuente: "Chocolate en tableta y en polvo" es 113 en
            # --canasta pero 114 en --ponderadores -- confirmado con la Tabla de
            # correspondencia SCIAN 2013-2007 de INEGI (fusión de 113+114 hacia 113).
            df["codigo"] = df["codigo"].replace({"114": "113"})
        df_canasta = extraer_canasta(args.canasta, args.version)
        columnas_jerarquia = ["generico", "sector", "subsector", "rama", "subrama", "clase"]
        df = df.drop(columns=columnas_jerarquia).merge(df_canasta, on="codigo", how="left")

    if args.encadenamientos is not None:
        df_encadenamiento = extraer_encadenamiento(args.encadenamientos)
        df = df.merge(df_encadenamiento, on="codigo", how="left")

    df = resolver_sector_agrupado(df)
    df = normalizar_columnas_con_codigo(df, ["sector", "subsector", "rama", "subrama", "clase"])
    df = normalizar_columnas_texto(df, ["generico"])
    guardar_csv(df, args.salida / f"ponderadores_{args.version}.csv", args.version)


if __name__ == "__main__":
    main()
