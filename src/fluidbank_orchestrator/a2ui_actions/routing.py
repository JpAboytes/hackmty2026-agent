"""Recognize requests to prepare forms; saving is exclusively a client event."""

import re
import unicodedata


def requested_form(query: str) -> str | None:
    text = "".join(
        c for c in unicodedata.normalize("NFKD", query.lower()) if not unicodedata.combining(c)
    )
    if re.search(r"\b(no|nunca|como|que es|cuando|cuanto|debo)\b", text):
        return None
    if re.search(r"\b(transfiere|transferir|envia|enviar|manda|mandar|mueve|mover)\b", text) or (
        "transferencia" in text
        and re.search(r"\b(quiero|hacer|haz|realiza|realizar|nueva)\b", text)
    ):
        return "transfer.execute"
    if "tarjeta" in text and re.search(
        r"\b(paga|pagar|quiero pagar|abona|abonar|liquida|liquidar)\b", text
    ):
        return "credit_card.pay"
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


def requested_form_arguments(query: str, name: str) -> dict[str, str | float]:
    """Extract only obvious form defaults; the visible form remains authoritative."""
    text = " ".join(query.strip().split())
    arguments: dict[str, str | float] = {}
    number = r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)"
    amount_match = re.search(rf"\$\s*{number}", text, re.IGNORECASE)
    if amount_match is None:
        amount_match = re.search(rf"\b{number}\s*(?:mxn|pesos?)\b", text, re.IGNORECASE)
    if amount_match is None and name == "transfer.execute":
        amount_match = re.search(
            rf"\b(?:transfiere|envia|manda|mueve)\s+{number}\s+(?:a|para)\b",
            text,
            re.IGNORECASE,
        )
    if amount_match:
        amount = float(amount_match.group(1).replace(",", ""))
        if 0 < amount <= 100_000:
            arguments["initial_amount"] = amount
    if name == "transfer.execute":
        recipient_match = re.search(
            r"\b(?:a|para)\s+([^,.;!?]+?)\s*$",
            text,
            re.IGNORECASE,
        )
        if recipient_match:
            recipient = recipient_match.group(1).strip()
            if 0 < len(recipient) <= 120:
                arguments["initial_recipient"] = recipient
    return arguments
