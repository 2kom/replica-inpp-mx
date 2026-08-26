"""Tipos compartidos del dominio del INPP."""

from __future__ import annotations

from typing import Literal

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
# no ~100) -- el rebase real es el paso de encadenamiento pendiente (ver CLAUDE.md, sección
# "Encadenamiento"): la etiqueta ya lo anuncia pero el dato aún no cambió.
VersionCanasta = Literal[2012, 2019, 2025]

RECORTE_POR_PREFIJO: dict[str, RecorteINPP] = {
    "1": "mercado_nacional",
    "2": "mercado_exportacion",
    "3": "produccion_total",
    "4": "bienes_finales",
}
