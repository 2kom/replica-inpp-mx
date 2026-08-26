"""Lector de series de índices del INPP (CSV del BIE de INEGI)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from replica_inpp.dominio.errores import (
    ArchivoCorrupto,
    ArchivoNoEncontrado,
    ArchivoVacio,
    EncodingNoLegible,
    OrientacionNoDetectable,
    SerieVacia,
)
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.dominio.tipos import RECORTE_POR_PREFIJO, RecorteINPP
from replica_inpp.infraestructura.csv._utils import _normalizar

# Fila plana: ", <1 dígito recorte><3 dígitos código> <nombre>" al final del Título.
# Confirmado 100% en nae de las 4 carpetas × s12/s19/s25, cero excepciones.
_PATRON_PLANO = re.compile(r",\s*(\d)(\d{3})\s+(.+)$")

# Fila hoja de un archivo jerárquico (`ae`): código de 3 dígitos SIN prefijo de recorte.
_PATRON_HOJA = re.compile(r"^(\d{3})\s+(.+)$")

# "Base <mes> <AAAA>=100" al inicio del Título -- el año de la base coincide con la
# versión de canasta (2012/2019/2025), verificado contra los xlsx/CSV reales.
_PATRON_BASE = re.compile(r"Base\s+\w+\s+(\d{4})=100")

# Huérfano conocido de origen en `ae` de produccion_total (verificado s19): la fila
# "...621511 Laboratorios médicos y de diagnóstico del sector privado, 622 Hospitales"
# no tiene hijos correctamente encadenados en esta exportación de INEGI. `622` es
# código de SubSector SCIAN, NO existe como código de genérico en ponderadores.csv
# (ahí `622` solo aparece en la columna `subsector` de genéricos reales, ej.
# "hospitalizacion" código 537). El algoritmo de padres-por-prefijo-de-comas no
# puede distinguirlo estructuralmente porque el dato mismo está mal formado, no el
# patrón — se excluye a mano.
_CODIGOS_HUERFANOS_CONOCIDOS = {"622"}

_Extraccion = tuple[str, str, pd.Series]  # (codigo, nombre, valores)


class LectorSeriesCsv:
    def leer(self, ruta: Path) -> pd.DataFrame:
        crudo = self._leer_csv(ruta)
        if crudo.columns[0] != "Título":
            raise ArchivoCorrupto(
                f"La primera columna sin importar orientación debe ser 'Título', "
                f"pero se encontró: {crudo.columns[0]}"
            )

        if "Cifra" in crudo.columns:
            data = self._horizontal(crudo)
        elif "Cifra" in crudo.iloc[:, 0].values:
            data = self._vertical(crudo)
        else:
            raise OrientacionNoDetectable(
                "No se pudo detectar orientación de la serie, se esperaba encontrar "
                "'Serie' y 'Cifra' como columnas o filas"
            )

        version_detectada = self._detectar_version(data.index)

        extracciones, recorte = self._extraer(data)
        if not extracciones:
            raise SerieVacia("Error al procesar serie, no se encontraron genéricos en el título")

        periodos = [PeriodoMensual.desde_str(str(c)) for c in data.columns]

        codigos, nombres, valores = zip(*extracciones)

        df_serie = pd.DataFrame(valores, columns=data.columns).apply(pd.to_numeric, errors="coerce")
        df_serie.index = pd.Index(codigos, name="codigo")
        df_serie.columns = periodos
        df_serie.insert(0, "generico", nombres)
        df_serie.attrs["origen"] = ruta
        df_serie.attrs["recorte"] = recorte
        df_serie.attrs["version_detectada"] = version_detectada

        return df_serie

    def _detectar_version(self, titulos: pd.Index) -> int | None:
        """Extrae el año de "Base <mes> <AAAA>=100" de TODOS los títulos -- coincide
        con la versión de canasta. `None` si ningún título trae ese patrón (no
        debería pasar con un archivo real del BIE).

        Raises:
            ArchivoCorrupto: los títulos traen más de un año de base distinto.
        """
        años = {int(m.group(1)) for t in titulos if (m := _PATRON_BASE.search(str(t)))}
        if len(años) > 1:
            raise ArchivoCorrupto(f"El archivo mezcla más de una base de versión: {sorted(años)}")
        return años.pop() if años else None

    def _leer_csv(self, ruta: Path) -> pd.DataFrame:
        for encoding in ["utf-8", "cp1252"]:
            try:
                return pd.read_csv(ruta, skiprows=5, dtype=str, encoding=encoding)
            except FileNotFoundError:
                raise ArchivoNoEncontrado(f"No se encontró el archivo: {ruta}")
            except pd.errors.EmptyDataError:
                raise ArchivoVacio(f"El archivo está vacío: {ruta}")
            except pd.errors.ParserError:
                raise ArchivoCorrupto(f"El archivo está corrupto o no es un CSV válido: {ruta}")
            except UnicodeDecodeError:
                continue

        try:
            return pd.read_csv(ruta, skiprows=5, dtype=str, encoding="latin-1")
        except FileNotFoundError:
            raise ArchivoNoEncontrado(f"No se encontró el archivo: {ruta}")
        except pd.errors.EmptyDataError:
            raise ArchivoVacio(f"El archivo está vacío: {ruta}")
        except pd.errors.ParserError:
            raise ArchivoCorrupto(f"El archivo está corrupto o no es un CSV válido: {ruta}")
        except UnicodeDecodeError:
            raise EncodingNoLegible(
                f"No se pudo decodificar el archivo con los encodings soportados: {ruta}"
            )

    def _columnas_periodo_validas(self, columnas: pd.Index) -> list[str]:
        validas = []
        for col in columnas:
            try:
                PeriodoMensual.desde_str(str(col))
            except Exception:
                continue
            validas.append(str(col))
        return validas

    def _horizontal(self, serie: pd.DataFrame) -> pd.DataFrame:
        columnas_validas = self._columnas_periodo_validas(serie.columns)
        return serie.set_index("Título")[columnas_validas]

    def _vertical(self, serie: pd.DataFrame) -> pd.DataFrame:
        primera_col = serie.iloc[:, 0]
        validas = set(self._columnas_periodo_validas(pd.Index(primera_col.dropna().unique())))
        filas_validas = serie[primera_col.isin(validas)]
        return filas_validas.set_index(serie.columns[0]).T

    def _extraer(self, data: pd.DataFrame) -> tuple[list[_Extraccion], RecorteINPP]:
        planas = self._extraer_plano(data)
        if planas is not None:
            return planas

        return self._extraer_jerarquico(data)

    def _extraer_plano(self, data: pd.DataFrame) -> tuple[list[_Extraccion], RecorteINPP] | None:
        """Camino rápido: todas las filas traen ', <prefijo><código> <nombre>'.

        Si UNA sola fila no matchea, el archivo no es plano (es jerárquico, `ae`) —
        se aborta este camino entero en vez de mezclar extracciones parciales.
        """
        extracciones: list[_Extraccion] = []
        prefijos_vistos: set[str] = set()

        for titulo in data.index:
            match = _PATRON_PLANO.search(str(titulo))
            if match is None:
                return None
            prefijo, codigo, nombre = match.groups()
            prefijos_vistos.add(prefijo)
            extracciones.append((codigo, _normalizar(nombre), data.loc[titulo]))

        if len(prefijos_vistos) != 1:
            raise ArchivoCorrupto(
                f"El archivo mezcla prefijos de recorte distintos en una misma tabla: {prefijos_vistos}"
            )
        (prefijo,) = prefijos_vistos
        if prefijo not in RECORTE_POR_PREFIJO:
            raise ArchivoCorrupto(f"Prefijo de recorte desconocido: '{prefijo}'")

        return extracciones, RECORTE_POR_PREFIJO[prefijo]

    def _extraer_jerarquico(self, data: pd.DataFrame) -> tuple[list[_Extraccion], RecorteINPP]:
        """Camino jerárquico (`ae`): filtra agregados, se queda solo con las hojas.

        Una fila es "padre" (agregado, no genérico) si su Título completo es,
        por construcción del archivo, prefijo-por-coma exacto del Título de otra
        fila más profunda. Mismo algoritmo que `LectorSeriesCsv._extraer_por_jerarquia`
        en replica-inpc-mx — reusado tal cual, solo cambia cómo se parsea la hoja.
        """
        titulos = [str(t) for t in data.index]
        titulos_set = set(titulos)

        padres_set: set[str] = set()
        for titulo in titulos:
            partes = titulo.split(",")
            for i in range(1, len(partes)):
                prefijo = ",".join(partes[:i])
                if prefijo in titulos_set:
                    padres_set.add(prefijo)

        extracciones: list[_Extraccion] = []
        for pos, titulo in enumerate(titulos):
            if titulo in padres_set:
                continue

            partes = titulo.split(",")
            inicio_hoja = None
            for i in range(len(partes) - 1, 0, -1):
                if ",".join(partes[:i]) in titulos_set:
                    inicio_hoja = i
                    break
            if inicio_hoja is None:
                continue

            resto = ",".join(partes[inicio_hoja:]).strip().lstrip(",").strip()
            match = _PATRON_HOJA.match(resto)
            if match is None:
                continue

            codigo, nombre = match.groups()
            if codigo in _CODIGOS_HUERFANOS_CONOCIDOS:
                continue
            extracciones.append((codigo, _normalizar(nombre), data.iloc[pos]))

        # `ae`/`nae` solo existe como variante en produccion_total — si se llegó a
        # este camino, el recorte es ese, sin ambigüedad.
        return extracciones, "produccion_total"
