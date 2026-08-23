# aqui van las funciones pequeñas que no ameritan un archivo aparte, pero que se usan en varios lugares

import re
from collections.abc import Sequence

import pandas as pd

_TRANS_TILDES = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")
_PATRON_ESPACIOS = re.compile(r"\s+")


def normalizar_texto(texto: str) -> str:
    """Minusculas, sin tildes (conserva la ñ), sin puntuacion, espacios simples.

    Mismo estandar que usa `replica-inpc-mx` (`tools/canasta_inpc/utilidades.py`),
    reimplementado acá porque `tools/` es standalone y no puede importar de otro repo.
    """
    texto = texto.translate(_TRANS_TILDES).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    return _PATRON_ESPACIOS.sub(" ", texto).strip()


def normalizar_columnas_texto(df: pd.DataFrame, columnas: Sequence[str]) -> pd.DataFrame:
    """Aplica `normalizar_texto` a `columnas`, deja el resto del df intacto. Devuelve copia.

    A diferencia de `replica-inpc-mx` (donde `normalizar_texto` corre inline
    dentro de `extraer_xlsx`), acá corre como paso aparte -- DESPUÉS de
    `extraer_ponderadores`/`extraer_canasta` (que devuelven texto crudo tal
    cual el xlsx, ej. `"Genérico A"`, `"11 Sector A"`) y ANTES de
    `guardar_csv`. Decisión deliberada: separa la correctitud estructural de
    la extracción (parsing/offsets de columna/state machine, cubierta por
    `test_extraccion_xlsx.py` con texto crudo) del formato final de texto
    (cubierto acá) -- evita acoplar dos fuentes de fallo distintas en la
    misma función y no rompe los tests de extracción existentes.

    Columna típica a normalizar acá: `generico` -- texto libre, sin código
    pegado. `sector`/`subsector`/`rama`/`subrama`/`clase` traen código+nombre
    combinado (ej. `"31-33 Industrias manufactureras"`) y usan
    `normalizar_columnas_con_codigo` en cambio, ver esa función -- aplicar
    `normalizar_texto` directo ahí destruye rangos SCIAN con guion (`31-33`
    colapsa a `3133`, forma indistinguible de un código de rama real de 4
    dígitos). NUNCA `codigo` ni las columnas de peso/encadenamiento (esas van
    tal cual el xlsx, sin normalizar, ver `esquema.COLUMNAS_BASE`).
    """
    df = df.copy()
    for columna in columnas:
        df[columna] = df[columna].apply(normalizar_texto)
    return df


def normalizar_texto_con_codigo(texto: str) -> str:
    """Normaliza preservando intacto el código SCIAN al inicio del texto.

    Para celdas donde código y nombre vienen combinados en un solo string
    (`sector`/`subsector`/`rama`/`subrama`/`clase` de `extraer_canasta`, ej.
    `"31-33 Industrias manufactureras"`, `"11 Agricultura, cría..."`) --
    `normalizar_texto` sobre el texto completo trata el guion de un rango
    SCIAN como puntuación descartable: `"31-33"` colapsa a `"3133"`, forma
    indistinguible de un código de rama real de 4 dígitos (`"1111 Cultivo de
    soya"` → `"1111 cultivo de soya"`). `31-33`/`48-49` son notación SCIAN
    oficial (agrupación de sectores, no texto libre) -- se preservan tal
    cual, sin parsear ni expandir los códigos que agrupan.

    Separa por el primer espacio: ningún código real trae espacio interno
    (confirmado contra los xlsx reales -- rangos `31-33`/`48-49`, código
    simple `11`; la etiqueta de header partido `G-11` de 2012 ya se filtra
    antes de llegar acá, ver `_es_codigo_generico`). Si no hay espacio
    (texto vacío o solo código, sin nombre), devuelve el texto tal cual.
    """
    codigo, _, nombre = texto.partition(" ")
    if not nombre:
        return codigo
    return f"{codigo} {normalizar_texto(nombre)}"


def normalizar_columnas_con_codigo(df: pd.DataFrame, columnas: Sequence[str]) -> pd.DataFrame:
    """Aplica `normalizar_texto_con_codigo` a `columnas`, deja el resto del df intacto. Devuelve copia.

    Usar para `sector`/`subsector`/`rama`/`subrama`/`clase` -- código+nombre
    combinado, ver `normalizar_texto_con_codigo`. NUNCA para `generico`
    (texto libre sin código, usa `normalizar_columnas_texto`) ni `codigo`.
    """
    df = df.copy()
    for columna in columnas:
        df[columna] = df[columna].apply(normalizar_texto_con_codigo)
    return df


def resolver_sector_agrupado(df: pd.DataFrame) -> pd.DataFrame:
    """Resuelve `sector` agrupado (rango SCIAN, ej. `"31-33"`) al código concreto vía `subsector`.

    SCIAN agrupa sector en un rango cuando un solo código de 2 dígitos no
    alcanza (`31-33` Industrias manufactureras, `48-49` Transportes...) --
    confirmado contra los 3 xlsx reales, solo esos 2 rangos existen. Para
    `dominio/calculo` (agrupar por sector) un rango no sirve como clave --
    hace falta el código concreto (`31`/`32`/`33`) que le corresponde a cada
    genérico puntual.

    `subsector` siempre trae el código concreto en sus primeros 2 dígitos
    (`311 Industria alimentaria` → sector `31`, `492 Servicios de
    mensajería...` → sector `49`) -- confirmado sin ambigüedad contra los 3
    xlsx reales (2012/2019/2025), 0 casos donde el subsector no alcance a
    resolver el rango. El nombre de `sector` (si lo trae, ej. con
    `--canasta`) se preserva tal cual -- INEGI usa el mismo nombre para las
    3 ramas del rango (`"Industrias manufactureras"` para 31/32/33 por
    igual, no hay nombre distinto por código individual), solo cambia el
    código.

    Funciona igual con `sector`/`subsector` bare (solo `--ponderadores`,
    sin nombre) o código+nombre combinado (con `--canasta`) -- ver
    `partition`, deja `nombre` vacío en el caso bare.

    Filas donde `sector` no es un rango (no trae guion) quedan intactas.
    Correr ANTES de `normalizar_columnas_con_codigo` -- opera sobre el
    código crudo, aunque el orden no afecta el resultado (el segmento de
    código no lo toca la normalización de texto).

    Lanza `ValueError` en vez de devolver una clasificación silenciosamente
    falsa cuando la jerarquía es inconsistente -- solo se dispara al
    resolver un rango, nunca en filas con `sector` simple: `subsector` sin
    código de 3 dígitos numéricos, `sector` con rango mal formado (no
    "NN-NN"), o `subsector` cuyos primeros 2 dígitos caen fuera del rango
    declarado por `sector` (ej. `subsector="481..."` con `sector="31-33..."`
    -- 48 no pertenece a 31-33). No se observa en los xlsx reales
    (2012/2019/2025 siempre consistentes) -- protege contra una extracción
    o cruce futuro inconsistente, que sin esto degradaría en clasificación
    falsa en vez de fallar.
    """
    df = df.copy()

    def _resolver(fila: pd.Series) -> str:
        codigo, separador, nombre = str(fila["sector"]).partition(" ")
        if "-" not in codigo:
            return str(fila["sector"])

        identificador = (
            f"código {fila['codigo']}" if "codigo" in fila.index else f"fila {fila.name}"
        )

        codigo_subsector = str(fila["subsector"]).partition(" ")[0]
        if not (len(codigo_subsector) == 3 and codigo_subsector.isdigit()):
            raise ValueError(
                f"No se puede resolver sector agrupado '{fila['sector']}' ({identificador}): "
                f"subsector '{fila['subsector']}' no trae un código de 3 dígitos."
            )

        lo_str, _, hi_str = codigo.partition("-")
        if not (len(lo_str) == 2 and lo_str.isdigit() and len(hi_str) == 2 and hi_str.isdigit()):
            raise ValueError(
                f"Rango de sector '{codigo}' ({identificador}) con formato inesperado -- se "
                f"esperaban 2 códigos de 2 dígitos separados por guion."
            )

        lo, hi = int(lo_str), int(hi_str)
        codigo_resuelto = codigo_subsector[:2]
        if not (lo <= int(codigo_resuelto) <= hi):
            raise ValueError(
                f"Subsector '{fila['subsector']}' (código {codigo_resuelto}, {identificador}) no "
                f"pertenece al rango de sector '{fila['sector']}' ({lo}-{hi})."
            )

        return f"{codigo_resuelto}{separador}{nombre}"

    df["sector"] = df.apply(_resolver, axis=1)
    return df
