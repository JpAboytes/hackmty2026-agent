"""Recognize requests to prepare forms; saving is exclusively a client event."""

import re
import unicodedata


def requested_form(query: str) -> str | None:
    text = "".join(
        c for c in unicodedata.normalize("NFKD", query.lower()) if not unicodedata.combining(c)
    )
    if re.search(r"\b(no|nunca|como|que es)\b", text):
        return None
    domain = (
        "budget"
        if "presupuesto" in text
        else "savings_goal"
        if "meta" in text and "ahorr" in text
        else None
    )
    if domain is None:
        return None
    if re.search(
        r"\b(edita|editar|modifica|modificar|cambia|cambiar|actualiza|actualizar)\b", text
    ):
        return f"{domain}.load"
    if re.search(r"\b(crea|crear|nuevo|nueva|agrega|agregar)\b", text):
        return f"{domain}.create"
    return None
