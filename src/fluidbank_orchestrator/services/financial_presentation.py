"""Deterministic selection and trusted construction of Finance v2 presentations."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any, cast

from fluidbank_orchestrator.schemas.a2ui import (
    A2UI_FINANCE_V2_CATALOG,
    A2UIBundle,
    validate_complete_sequence,
)
from fluidbank_orchestrator.schemas.banking_view import (
    FINANCIAL_INTENTS,
    FinancialIntent,
    validate_banking_view,
)
from fluidbank_orchestrator.state import UserProfile

_ZONE = timezone(timedelta(hours=-6), name="America/Monterrey")
_ACCOUNT_NAMES = {
    "checking": "Cuenta de cheques",
    "savings": "Cuenta de ahorro",
    "credit": "Tarjeta de crédito",
}
_CATEGORY_MAP = {
    "groceries": "food",
    "dining": "food",
    "food": "food",
    "transport": "transport",
    "transportation": "transport",
    "entertainment": "entertainment",
    "utilities": "utilities",
    "health": "health",
    "shopping": "shopping",
    "salary": "income",
    "income": "income",
    "transfer": "transfer",
}
_TITLES: dict[FinancialIntent, str] = {
    "financial-summary": "Tu panorama financiero",
    "transactions": "Tus movimientos",
    "spending-analysis": "Así estás gastando",
    "cash-flow": "Tu flujo de efectivo",
    "budgets": "Tus presupuestos",
    "recurring-payments": "Tus próximos cobros",
    "credit-card": "Tu tarjeta de crédito",
    "debts": "Tus deudas",
    "transfers": "Tus transferencias",
    "card-security": "Seguridad de tu tarjeta",
    "savings-goals": "Tus metas de ahorro",
    "banking-information": "Tu información bancaria",
    "financial-education": "Guía financiera",
}
_FINANCIAL_VIEW_SURFACE_ID = "financial-view"
_FINANCIAL_VIEW_RESOURCE_URI = "a2ui://finance/view"
_FINANCIAL_ACTION_NAME = "request_financial_view"
_FINANCIAL_ACTION_COMPONENT_ID = "request_financial_view_button"


def _follow_up(intent: FinancialIntent) -> tuple[str, FinancialIntent]:
    if intent == "financial-summary":
        return "Ver gastos del último mes", "transactions"
    return "Volver al panorama financiero", "financial-summary"


@dataclass(frozen=True, slots=True)
class FinancialPresentation:
    intent: FinancialIntent
    message: str
    data: dict[str, Any]
    a2ui: A2UIBundle


def _normalized(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def normalize_action_intent(value: object) -> FinancialIntent | None:
    for intent in FINANCIAL_INTENTS:
        if value == intent:
            return intent
    return None


def classify_financial_request(query: str) -> FinancialIntent | None:
    """Bound natural language to the finite presentation vocabulary."""
    text = _normalized(query)
    rules: tuple[tuple[FinancialIntent, tuple[str, ...]], ...] = (
        (
            "financial-summary",
            (
                "como van mis finanzas",
                "como estan mis finanzas",
                "resumen financiero",
                "cuanto dinero tengo",
                "cual es mi saldo",
                "mi saldo disponible",
                "how much money do i have",
                "account balance",
                "how are my finances",
                "financial overview",
            ),
        ),
        (
            "spending-analysis",
            (
                "mis gastos",
                "en que se me fue",
                "como han cambiado mis gastos",
                "que dias gasto mas",
                "spending analysis",
                "my spending",
            ),
        ),
        (
            "transactions",
            ("movimientos", "transacciones", "transactions", "gaste ayer", "compras de"),
        ),
        (
            "cash-flow",
            (
                "flujo de efectivo",
                "me alcanzara",
                "cash flow",
                "ingresos y gastos",
                "income and expenses",
            ),
        ),
        ("budgets", ("presupuesto", "limite semanal", "budget")),
        (
            "recurring-payments",
            (
                "pagos recurrentes",
                "suscripciones",
                "proximos cobros",
                "proximos pagos",
                "upcoming payments",
                "recurring payments",
            ),
        ),
        ("credit-card", ("tarjeta de credito", "pago minimo", "credit card")),
        ("debts", ("deuda", "liquidar", "debts")),
        ("transfers", ("transferir", "transferencia", "transfer")),
        (
            "card-security",
            ("no reconozco", "aclaracion", "disputa", "bloquear tarjeta", "card security"),
        ),
        (
            "savings-goals",
            ("meta de ahorro", "meta para", "quiero ahorrar", "savings goal"),
        ),
        (
            "banking-information",
            (
                "estado de cuenta",
                "clabe",
                "informacion bancaria",
                "bank statement",
                "beneficiario",
                "beneficiary",
            ),
        ),
        (
            "financial-education",
            ("que pasa si pago", "explicame", "educacion financiera", "financial education"),
        ),
    )
    for intent, phrases in rules:
        if any(phrase in text for phrase in phrases):
            return intent
    return None


def select_presentation_intent(
    requested: FinancialIntent,
    observations: Sequence[Mapping[str, Any]],
    *,
    query: str,
    action_requested: bool = False,
) -> FinancialIntent:
    """Choose presentation semantics only after retrieval observations exist."""
    if requested != "transactions" or action_requested:
        return requested
    text = _normalized(query)
    if any(term in text for term in ("gasto", "spending", "categoria", "dias gasto")):
        if any(_rows_for_table(observations, "transactions")):
            return "spending-analysis"
    return requested


def _rows_for_table(observations: Sequence[Mapping[str, Any]], table: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for observation in observations:
        if observation.get("is_error") is True:
            continue
        arguments = observation.get("arguments")
        data = observation.get("data")
        raw_rows: object = None
        if observation.get("name") == "select_rows":
            if not isinstance(arguments, Mapping) or arguments.get("table") != table:
                continue
            raw_rows = data.get("rows") if isinstance(data, Mapping) else None
        elif observation.get("name") == "get_transactions" and table == "transactions":
            raw_rows = data.get("transactions") if isinstance(data, Mapping) else None
        if not isinstance(raw_rows, list):
            continue
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                continue
            row = deepcopy(dict(raw))
            key = str(row.get("id", repr(sorted(row.items()))))
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    try:
        parsed = float(value)
    except (ValueError, OverflowError):
        return None
    return parsed if isfinite(parsed) else None


def _currency(profile: UserProfile) -> str | None:
    balances = profile.get("owned_balances", {})
    if len(balances) == 1:
        currency = next(iter(balances))
        return currency if currency in {"MXN", "USD"} else None
    return None


def _empty_view(intent: FinancialIntent, description: str, currency: str = "MXN") -> dict[str, Any]:
    return {
        "intent": intent,
        "title": _TITLES[intent],
        "currency": currency,
        "state": "empty",
        "description": description,
    }


def _summary_view(
    observations: Sequence[Mapping[str, Any]], profile: UserProfile
) -> tuple[dict[str, Any], dict[str, Any], str]:
    raw_accounts = _rows_for_table(observations, "accounts")
    currencies = {
        row.get("currency")
        for row in raw_accounts
        if row.get("account_type") in {"checking", "savings"}
        and row.get("currency") in {"MXN", "USD"}
        and _number(row.get("available_balance")) is not None
    }
    if len(currencies) > 1:
        view = _empty_view(
            "financial-summary",
            "Tus cuentas usan más de una moneda; no se combinaron saldos sin una "
            "conversión explícita.",
        )
        return view, {"presentation_intent": "financial-summary"}, view["description"]
    if not currencies:
        view = _empty_view(
            "financial-summary", "No hay saldos de cuentas verificables para mostrar."
        )
        return view, {"presentation_intent": "financial-summary"}, view["description"]

    currency = cast("str", next(iter(currencies)))
    accounts: list[dict[str, Any]] = []
    owned_balance = 0.0
    for row in raw_accounts:
        account_type = row.get("account_type")
        balance = _number(row.get("available_balance"))
        account_id = row.get("id")
        if (
            not isinstance(account_id, str)
            or account_type not in _ACCOUNT_NAMES
            or row.get("currency") != currency
            or balance is None
        ):
            continue
        accounts.append(
            {
                "accountId": account_id,
                "accountName": _ACCOUNT_NAMES[cast("str", account_type)],
                "accountType": account_type,
                "availableBalance": balance,
            }
        )
        if account_type in {"checking", "savings"}:
            owned_balance += balance
    if not accounts:
        view = _empty_view(
            "financial-summary", "No hay saldos de cuentas verificables para mostrar.", currency
        )
        return view, {"presentation_intent": "financial-summary"}, view["description"]

    view = {
        "intent": "financial-summary",
        "title": _TITLES["financial-summary"],
        "currency": currency,
        "totalOwnedBalance": owned_balance,
        "accounts": accounts,
    }
    data = {
        "presentation_intent": "financial-summary",
        "owned_balance": owned_balance,
        "currency": currency,
    }
    return view, data, f"Tienes {owned_balance:,.2f} {currency} entre cheques y ahorro."


def _transaction_rows(
    observations: Sequence[Mapping[str, Any]], profile: UserProfile
) -> tuple[list[dict[str, Any]], str | None]:
    currency = _currency(profile)
    mapped: list[dict[str, Any]] = []
    source_rows = _rows_for_table(observations, "transactions")
    row_currencies = {
        row.get("currency") for row in source_rows if row.get("currency") in {"MXN", "USD"}
    }
    if len(row_currencies) == 1:
        currency = cast("str", next(iter(row_currencies)))
    elif len(row_currencies) > 1:
        currency = None
    for row in source_rows:
        identifier = row.get("id")
        merchant = row.get("merchant")
        occurred_at = row.get("occurred_at")
        amount = _number(row.get("amount"))
        direction = row.get("direction")
        normalized_direction = (
            "credit" if direction == "income" else "debit" if direction == "expense" else direction
        )
        if (
            not isinstance(identifier, str)
            or not isinstance(merchant, str)
            or not merchant.strip()
            or not isinstance(occurred_at, str)
            or amount is None
            or normalized_direction not in {"credit", "debit"}
        ):
            continue
        try:
            parsed = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            continue
        category = _CATEGORY_MAP.get(str(row.get("category")), "other")
        mapped.append(
            {
                "transactionId": identifier,
                "title": merchant.strip()[:120],
                "amount": abs(amount) if normalized_direction == "credit" else -abs(amount),
                "occurredAt": occurred_at,
                "category": category,
                "status": "completed",
            }
        )
    mapped.sort(key=lambda item: item["occurredAt"], reverse=True)
    return mapped[:100], currency


def _transactions_view(
    observations: Sequence[Mapping[str, Any]], profile: UserProfile
) -> tuple[dict[str, Any], dict[str, Any], str]:
    transactions, currency = _transaction_rows(observations, profile)
    if currency is None:
        view = _empty_view(
            "transactions",
            "No se pueden combinar movimientos sin identificar una sola moneda para la consulta.",
        )
        return view, {"presentation_intent": "transactions"}, view["description"]
    if not transactions:
        view = _empty_view("transactions", "No hay movimientos en el periodo consultado.", currency)
        return view, {"presentation_intent": "transactions"}, view["description"]
    local_days = [
        datetime.fromisoformat(item["occurredAt"].replace("Z", "+00:00"))
        .astimezone(_ZONE)
        .date()
        .isoformat()
        for item in transactions
    ]
    view = {
        "intent": "transactions",
        "title": _TITLES["transactions"],
        "currency": currency,
        "startDate": min(local_days),
        "endDate": max(local_days),
        "timeZone": "America/Monterrey",
        "transactions": transactions,
    }
    return (
        view,
        {"presentation_intent": "transactions", "transaction_count": len(transactions)},
        f"Encontré {len(transactions)} movimientos.",
    )


def _spending_view(
    observations: Sequence[Mapping[str, Any]], profile: UserProfile
) -> tuple[dict[str, Any], dict[str, Any], str]:
    domain = next(
        (
            observation.get("data")
            for observation in reversed(observations)
            if observation.get("name") == "analyze_spending"
            and observation.get("is_error") is not True
            and isinstance(observation.get("data"), Mapping)
        ),
        None,
    )
    if isinstance(domain, Mapping):
        summaries = domain.get("summary_by_currency")
        categories = domain.get("by_category")
        daily = domain.get("daily_series")
        if (
            isinstance(summaries, list)
            and len(summaries) == 1
            and isinstance(categories, list)
            and isinstance(daily, list)
        ):
            currency = summaries[0].get("currency") if isinstance(summaries[0], Mapping) else None
            total = (
                _number(summaries[0].get("expenses")) if isinstance(summaries[0], Mapping) else None
            )
            if currency in {"MXN", "USD"} and total is not None:
                # Several database categories collapse onto one visual category:
                # `groceries` and `dining` are both food, and `subscription`,
                # `credit_card` and `housing` are all other. The amounts have to be
                # summed before the view is built, because the contract allows one
                # row per visual category and rejects a repeated one - mapping the
                # rows one by one made every spending answer fail validation.
                category_totals: defaultdict[str, float] = defaultdict(float)
                for item in categories:
                    if not isinstance(item, Mapping) or item.get("currency") != currency:
                        continue
                    amount = _number(item.get("amount"))
                    if amount is None or amount < 0:
                        continue
                    name = _CATEGORY_MAP.get(str(item.get("category")), "other")
                    # Spending views describe expenses; an income bucket is not one.
                    if name == "income":
                        continue
                    category_totals[name] += amount
                category_data = [
                    {"category": name, "amount": amount}
                    for name, amount in sorted(
                        category_totals.items(), key=lambda entry: (-entry[1], entry[0])
                    )[:8]
                ]
                daily_data = [
                    {"date": str(item.get("date")), "value": _number(item.get("amount")) or 0}
                    for item in daily[-366:]
                    if isinstance(item, Mapping) and item.get("currency") == currency
                ]
                if daily_data:
                    view = {
                        "intent": "spending-analysis",
                        "title": _TITLES["spending-analysis"],
                        "currency": currency,
                        "totalSpent": total,
                        "categories": category_data,
                        "trend": {
                            "title": "Evolución de gastos",
                            "data": [
                                {"label": item["date"], "values": [item["value"]]}
                                for item in daily_data[:240]
                            ],
                            "series": [{"id": "expenses", "label": "Gastos", "tone": "orange"}],
                        },
                        "activity": {
                            "title": "Tus días de mayor gasto",
                            "data": daily_data,
                            "initialDate": daily_data[-1]["date"],
                            "initialView": "month",
                            "tone": "orange",
                        },
                    }
                    return (
                        view,
                        {"presentation_intent": "spending-analysis", "domain_result": dict(domain)},
                        f"Analicé {total:,.2f} {currency} de gastos.",
                    )
    transactions, currency = _transaction_rows(observations, profile)
    expenses = [item for item in transactions if item["amount"] < 0]
    if currency is None or not expenses:
        view = _empty_view(
            "spending-analysis",
            "No hay gastos verificables en una sola moneda para analizar.",
            currency or "MXN",
        )
        return view, {"presentation_intent": "spending-analysis"}, view["description"]

    categories: defaultdict[str, float] = defaultdict(float)
    daily: defaultdict[str, float] = defaultdict(float)
    for item in expenses:
        amount = -float(item["amount"])
        categories[item["category"]] += amount
        day = (
            datetime.fromisoformat(item["occurredAt"].replace("Z", "+00:00"))
            .astimezone(_ZONE)
            .date()
            .isoformat()
        )
        daily[day] += amount
    category_data = [
        {"category": name, "amount": amount}
        for name, amount in sorted(categories.items(), key=lambda item: (-item[1], item[0]))[:8]
        if name != "income"
    ]
    ordered_days = sorted(daily)
    view = {
        "intent": "spending-analysis",
        "title": _TITLES["spending-analysis"],
        "currency": currency,
        "totalSpent": sum(categories.values()),
        "categories": category_data,
        "trend": {
            "title": "Evolución de gastos",
            "data": [{"label": day, "values": [daily[day]]} for day in ordered_days[:240]],
            "series": [{"id": "expenses", "label": "Gastos", "tone": "orange"}],
        },
        "activity": {
            "title": "Tus días de mayor gasto",
            "data": [{"date": day, "value": daily[day]} for day in ordered_days[-366:]],
            "initialDate": ordered_days[-1],
            "initialView": "month",
            "tone": "orange",
        },
    }
    total = sum(categories.values())
    return (
        view,
        {
            "presentation_intent": "spending-analysis",
            "expense_total": total,
            "currency": currency,
            "source_result_count": sum(
                1
                for observation in observations
                if observation.get("name") == "select_rows"
                and isinstance(observation.get("arguments"), Mapping)
                and observation["arguments"].get("table") == "transactions"
            ),
        },
        f"Analicé {total:,.2f} {currency} de gastos.",
    )


def _recurring_view(
    observations: Sequence[Mapping[str, Any]], profile: UserProfile
) -> tuple[dict[str, Any], dict[str, Any], str]:
    currency = _currency(profile)
    payments: list[dict[str, Any]] = []
    for row in _rows_for_table(observations, "subscriptions"):
        identifier = row.get("id")
        name = row.get("name")
        amount = _number(row.get("amount"))
        next_date = row.get("next_charge_date")
        cycle = row.get("billing_cycle")
        status = row.get("status")
        if (
            isinstance(identifier, str)
            and isinstance(name, str)
            and name.strip()
            and amount is not None
            and amount >= 0
            and isinstance(next_date, str)
            and cycle in {"weekly", "monthly", "yearly"}
            and status in {"active", "paused"}
        ):
            payments.append(
                {
                    "id": identifier,
                    "name": name[:120],
                    "amount": amount,
                    "nextDate": next_date,
                    "cycle": cycle,
                    "status": status,
                }
            )
    if currency is None or not payments:
        view = _empty_view(
            "recurring-payments",
            "No hay pagos recurrentes verificables en una sola moneda para mostrar.",
            currency or "MXN",
        )
        return view, {"presentation_intent": "recurring-payments"}, view["description"]
    view = {
        "intent": "recurring-payments",
        "title": _TITLES["recurring-payments"],
        "currency": currency,
        "payments": payments[:30],
    }
    return view, {"presentation_intent": "recurring-payments"}, "Estos son tus próximos cobros."


def _build_bundle(intent: FinancialIntent, view: Mapping[str, Any]) -> A2UIBundle:
    validated_view = validate_banking_view(view)
    action_label, request_intent = _follow_up(intent)
    components: list[dict[str, Any]] = [
        {
            "id": "root",
            "component": "Column",
            "children": ["banking_view", _FINANCIAL_ACTION_COMPONENT_ID],
        },
        {"id": "banking_view", "component": "BankingView", "view": {"path": "/view"}},
        {
            "id": "request_financial_view_label",
            "component": "Text",
            "text": {"path": "/actionLabel"},
        },
        {
            "id": _FINANCIAL_ACTION_COMPONENT_ID,
            "component": "Button",
            "child": "request_financial_view_label",
            "variant": "primary",
            "action": {
                "event": {
                    "name": _FINANCIAL_ACTION_NAME,
                    "context": {"intent": {"path": "/requestIntent"}},
                }
            },
        },
    ]
    messages = [
        {
            "version": "v0.9.1",
            "createSurface": {
                "surfaceId": _FINANCIAL_VIEW_SURFACE_ID,
                "catalogId": A2UI_FINANCE_V2_CATALOG,
            },
        },
        {
            "version": "v0.9.1",
            "updateComponents": {
                "surfaceId": _FINANCIAL_VIEW_SURFACE_ID,
                "components": components,
            },
        },
        {
            "version": "v0.9.1",
            "updateDataModel": {
                "surfaceId": _FINANCIAL_VIEW_SURFACE_ID,
                "path": "/",
                "value": {
                    "view": validated_view,
                    "actionLabel": action_label,
                    "requestIntent": request_intent,
                },
            },
        },
    ]
    validate_complete_sequence(messages, _FINANCIAL_VIEW_SURFACE_ID)
    return A2UIBundle(resource_uri=_FINANCIAL_VIEW_RESOURCE_URI, messages=messages)


def build_financial_presentation(
    intent: FinancialIntent,
    observations: Sequence[Mapping[str, Any]],
    profile: UserProfile,
) -> FinancialPresentation:
    """Interpret retained data and construct one validated Finance v2 response."""
    domain_observations = [
        observation
        for observation in observations
        if observation.get("name")
        in {
            "get_financial_overview",
            "get_accounts",
            "get_transactions",
            "analyze_spending",
            "get_cash_flow",
            "get_budget_progress",
            "get_savings_progress",
            "get_debt_overview",
            "get_upcoming_payments",
            "get_financial_alerts",
            "get_bank_statements",
            "get_payment_activity",
            "get_beneficiaries",
            "get_transaction_disputes",
            "compare_debt_scenarios",
        }
    ]
    if domain_observations and domain_observations[-1].get("is_error") is True:
        view = _empty_view(
            intent,
            "No pude verificar los datos financieros solicitados; no mostraré cifras estimadas.",
            _currency(profile) or "MXN",
        )
        data = {
            "presentation_intent": intent,
            "tool_error": deepcopy(domain_observations[-1].get("data", {})),
        }
        message = view["description"]
    elif intent == "financial-summary":
        view, data, message = _summary_view(observations, profile)
    elif intent == "transactions":
        view, data, message = _transactions_view(observations, profile)
    elif intent == "spending-analysis":
        view, data, message = _spending_view(observations, profile)
    elif intent == "recurring-payments":
        view, data, message = _recurring_view(observations, profile)
    else:
        currency = _currency(profile) or "MXN"
        view = _empty_view(
            intent,
            "La consulta está soportada, pero faltan datos verificables para construir esta vista.",
            currency,
        )
        data = {"presentation_intent": intent}
        message = view["description"]
    if domain_observations and domain_observations[-1].get("is_error") is not True:
        domain_data = domain_observations[-1].get("data")
        if isinstance(domain_data, Mapping):
            data.setdefault("domain_result", deepcopy(dict(domain_data)))
    return FinancialPresentation(
        intent=intent,
        message=message,
        data=data,
        a2ui=_build_bundle(intent, view),
    )
