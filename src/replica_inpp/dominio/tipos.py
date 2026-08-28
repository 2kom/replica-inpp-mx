"""Tipos compartidos del dominio del INPP."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from replica_inpp.dominio.periodos import PeriodoMensual

# Recorte de destino/etapa — eje ortogonal a la clasificación SCIAN, sin equivalente
# en el INPC. Confirmado embebido en el `Título` de las series como prefijo
# numérico de 1 dígito, consistente en s12/s19/s25, cero excepciones.
RecorteINPP = Literal[
    "produccion_total", "mercado_nacional", "bienes_finales", "mercado_exportacion"
]

# Versión de canasta/ponderadores soportada — las únicas 3 con LAYOUTS_XLSX/LAYOUTS_CANASTA
# en tools/canasta_inpp/esquema.py. Identifica la vintage de ponderadores/SCIAN, y coincide
# con la base DECLARADA en el Título de la serie ("Base <mes> <AAAA>=100", lo que valida
# cargar_serie contra VersionNoCoincide): s19 declara "Base Julio 2019=100", s25 declara
# "Base Julio 2025=100" -- son etiquetas distintas, sí cambia entre versiones. Lo que NO
# cambió todavía es el valor NUMÉRICO de la serie 2025 (sigue en la base vieja, ~107-188,
# no ~100) -- el rebase real es el paso de encadenamiento pendiente: la etiqueta ya lo
# anuncia pero el dato aún no cambió.
VersionCanasta = Literal[2012, 2019, 2025]

RECORTE_POR_PREFIJO: dict[str, RecorteINPP] = {
    "1": "mercado_nacional",
    "2": "mercado_exportacion",
    "3": "produccion_total",
    "4": "bienes_finales",
}

# Rango de vigencia por versión de canasta, igual patrón que `RANGOS_CANASTAS` de
# replica-inpc-mx. El fin de una versión coincide con el inicio de la siguiente
# (periodo de traslape compartido) aunque solo 2019→2025 encadena de verdad; 2012
# arranca en jun 2012 porque esa vintage tiene su propia base (Jun 2012=100). 2019
# y 2025 declaran etiquetas de base DISTINTAS en el Título ("Base Julio 2019=100"
# vs "Base Julio 2025=100", ver VersionCanasta arriba) pero comparten, por ahora,
# la misma ESCALA NUMÉRICA: el rebase real de 2025 a su propia base es el paso de
# encadenamiento todavía pendiente — hasta que corra, el valor numérico de la
# serie 2025 sigue anclado a Jul 2019=100.
# `None` = vigente hasta hoy.
RANGOS_CANASTAS: dict[VersionCanasta, tuple[PeriodoMensual, PeriodoMensual | None]] = {
    2012: (PeriodoMensual(2012, 6), PeriodoMensual(2019, 7)),
    2019: (PeriodoMensual(2019, 7), PeriodoMensual(2025, 7)),
    2025: (PeriodoMensual(2025, 7), None),
}

# --- Agregación: nivel de agrupación SCIAN, o general/Mercancías-Servicios ---

# Constante general — equivalente a TIPO_INPC de INPC ("sin desglose, la serie
# completa que se pasó"): si `serie.recorte` es produccion_total, esto da el
# índice de producción total; si es bienes_finales, el de bienes finales, etc.
TIPO_INPP: str = "INPP"

# Alias válidos de agregación por nivel SCIAN -> columna real de `CanastaINPP`.
# Acepta nombre completo o abreviatura oficial ("S: Sector; SB: Subsector;
# R: Rama; SR: Subrama; C: Clase de actividad económica", ver xlsx de
# ponderadores). Siempre en mayúsculas.
AGREGACION_A_COLUMNA: dict[str, str] = {
    "SECTOR": "codigo sector",
    "S": "codigo sector",
    "SUBSECTOR": "codigo subsector",
    "SB": "codigo subsector",
    "RAMA": "codigo rama",
    "R": "codigo rama",
    "SUBRAMA": "codigo subrama",
    "SR": "codigo subrama",
    "CLASE": "codigo clase",
    "C": "codigo clase",
}

# Nombre canónico por alias — normaliza la abreviatura oficial ("S", "SB", "R",
# "SR", "C") al nombre completo, para que `ResultadoIndice`/`ManifestCalculo`
# siempre guarden el mismo valor sin importar qué alias se haya pasado a
# `calcular_indice`. `TIPO_INPP` y `"MERCANCIAS_SERVICIOS"` no están acá porque
# ya son su propio nombre canónico (sin abreviatura que normalizar).
AGREGACION_A_NOMBRE: dict[str, str] = {
    "SECTOR": "SECTOR",
    "S": "SECTOR",
    "SUBSECTOR": "SUBSECTOR",
    "SB": "SUBSECTOR",
    "RAMA": "RAMA",
    "R": "RAMA",
    "SUBRAMA": "SUBRAMA",
    "SR": "SUBRAMA",
    "CLASE": "CLASE",
    "C": "CLASE",
}

# Agregaciones válidas que NO son nivel SCIAN — no tienen columna 1:1 en la
# canasta, se calculan aparte en dominio/calculo (Mercancías/Servicios agrupa
# por SECTORES_MERCANCIAS vs complemento).
AGREGACIONES_ESPECIALES: frozenset[str] = frozenset({TIPO_INPP, "MERCANCIAS_SERVICIOS"})

AGREGACIONES_VALIDAS: frozenset[str] = frozenset(AGREGACION_A_COLUMNA) | AGREGACIONES_ESPECIALES

# --- Rubro: columna de peso / destino de la producción ---

RUBRO_A_COLUMNA_PESO: dict[str, str] = {
    "produccion_total": "produccion total",
    "bienes_intermedios": "bienes intermedios",
    "bienes_finales": "bienes finales",
    "demanda_interna_total": "demanda interna total",
    "demanda_interna_consumo": "demanda interna consumo",
    "demanda_interna_capital": "demanda interna capital",
    "exportaciones": "exportaciones",
}

RUBROS_VALIDOS: frozenset[str] = frozenset(RUBRO_A_COLUMNA_PESO)

# Qué `rubro` son válidos para cada `recorte` de serie -- verificado contra la
# nota b/ al pie de cada hoja del xlsx de ponderadores de 2019 y 2025 (2025
# coincide EXACTO, palabra por palabra, con 2019 en las 5 notas), que dice
# EXPLÍCITO con qué precios se combina cada columna de peso. Antes esta tabla
# asumía que toda columna de peso salvo "produccion total" se combinaba con
# los precios de produccion_total -- falso: la nota b/ de DemandaInterna dice
# "combinadas con los precios para mercado nacional", no producción total.
# 2012 tiene una anomalía sin resolver: las notas b/ de sus hojas "Bienes
# Finales" y "Exportaciones" son texto IDÉNTICO palabra por palabra entre sí y
# mencionan "demanda interna" -- tema ajeno a ambas hojas, evidencia interna
# de copy-paste en el xlsx fuente. No se toma como señal confiable (2019 y
# 2025, independientes entre sí, coinciden en el texto correcto por tema) --
# no justifica versionar este mapa por `version` de canasta.
# Corregido y confirmado numéricamente contra el BIE real (validación
# 2026-08-27): demanda_interna_total/consumo/capital con la serie
# mercado_nacional coinciden, dentro del error de punto flotante (max_abs ≤
# 3.27e-13; 73/73 dentro de tolerancia 0.0009), con los indicadores del BIE
# 1380015/1380016/1380017 -- antes (con produccion_total) diff hasta 0.5.
# `bienes_intermedios` también trae nota b/ "mercado nacional", pero probado
# contra el BIE (1750002) sigue sin dar exacto (max 3.09/3.10 con cualquiera
# de las dos series) -- se deja tal cual, en produccion_total, mecanismo real
# sin identificar todavía.
# `bienes_finales` trae su propia nota b/ ("combinadas con los precios para
# mercado de producción total"), probada contra produccion_total (1700001) y
# tampoco da exacto (max 1.18, igual que con la serie bienes_finales) -- mismo
# caso, sin tocar.
RUBROS_POR_RECORTE: dict[RecorteINPP, frozenset[str]] = {
    "produccion_total": frozenset({"produccion_total", "bienes_intermedios"}),
    "mercado_nacional": frozenset(
        {"demanda_interna_total", "demanda_interna_consumo", "demanda_interna_capital"}
    ),
    "bienes_finales": frozenset({"bienes_finales"}),
    "mercado_exportacion": frozenset({"exportaciones"}),
}

# --- Encadenamiento: solo aplica a partir de 2025 ---

# Columna de `CanastaINPP` con el factor de encadenamiento por recorte —
# verificado numéricamente contra un genérico con precios muy divergentes
# entre recortes (Lámina de acero, código 339: los 4 valores de encadenamiento
# resultaron distintos entre sí), confirmando que el eje del encadenamiento es
# por SERIE (recorte), no por rubro/columna de peso.
COLUMNA_ENCADENAMIENTO_POR_RECORTE: dict[RecorteINPP, str] = {
    "produccion_total": "encadenamiento total",
    "mercado_nacional": "encadenamiento produccion nacional",
    "mercado_exportacion": "encadenamiento exportacion",
    "bienes_finales": "encadenamiento uso final",
}

# --- Mercancías / Servicios ---

# Sectores que integran "Mercancías" — verificado numéricamente contra la
# canasta 2019 real: la suma de estos sectores reproduce exacto 66.46585
# (fila "Mercancías" del xlsx de ponderadores). El complemento (todo lo que
# no está acá) es "Servicios" (33.53415).
#
# "31", "32", "33" van SEPARADOS, no como "31-33": `codigo sector` de
# `CanastaINPP` nunca trae el rango combinado — INEGI publica "31-33
# Industrias manufactureras" como una sola serie en el BIE, pero en la
# canasta real cada genérico ya viene resuelto a su sector específico
# (31, 32 o 33) vía `resolver_sector_agrupado` (`tools/canasta_inpp/`).
# Un `"31-33"` acá nunca matchea nada — filtraba manufacturas entero del
# grupo "Mercancías" (bug real, encontrado comparando contra INEGI: daba
# 20.98/79.02 en vez de 66.46585/33.53415).
SECTORES_MERCANCIAS: frozenset[str] = frozenset({"11", "21", "22", "23", "31", "32", "33"})

# --- Petróleo ---

# Código de genérico "Petróleo crudo" — verificado estable en 2012/2019/2025
# (a diferencia del caso Chocolate 113/114 en ponderadores↔canasta, sin drift
# entre versiones).
CODIGO_PETROLEO_CRUDO: str = "070"


@dataclass
class ManifestCalculo:
    """Registro de una corrida elemental de cálculo — uno por combinación version/agregación/rubro."""

    version: VersionCanasta
    agregacion: str
    rubro: str
    sin_petroleo: bool
    calculador: Literal["LaspeyresDirecto", "LaspeyresEncadenado"]
    ruta_canasta: Path | None = None
    ruta_series: Path | None = None
    fecha: datetime = field(default_factory=datetime.now)
