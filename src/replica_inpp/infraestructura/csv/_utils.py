import re

_TRANS_TILDES = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")
_PATRON_ESPACIOS = re.compile(r"\s+")


def _normalizar(nombre: str) -> str:

    nombre = nombre.translate(_TRANS_TILDES)
    nombre = re.sub(r"[^\w\s]", "", nombre)
    nombre = _PATRON_ESPACIOS.sub(" ", nombre).strip().lower()
    return nombre
