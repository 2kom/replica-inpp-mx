"""Tipos compartidos del dominio del INPP."""

from __future__ import annotations

from typing import Literal

# Recorte de destino/etapa — eje ortogonal a la clasificación SCIAN, sin equivalente
# en el INPC. Confirmado embebido en el `Título` de las series como prefijo
# numérico de 1 dígito, consistente en s12/s19/s25, cero excepciones.
RecorteINPP = Literal[
    "produccion_total", "mercado_nacional", "bienes_finales", "mercado_exportacion"
]

RECORTE_POR_PREFIJO: dict[str, RecorteINPP] = {
    "1": "mercado_nacional",
    "2": "mercado_exportacion",
    "3": "produccion_total",
    "4": "bienes_finales",
}
