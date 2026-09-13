"""Turning verified rows into one Finance v2 view payload per intent.

Each ``*_view`` function returns ``(view, data, message)``: the view payload the
trusted builder validates, the structured ``data`` echoed to the client, and the
conversational line that accompanies it. They share three rules:

* only data that came through ``verified_rows`` may appear;
* a currency that cannot be established unambiguously produces an empty view,
  never a mixed total;
* a row that fails its contract checks is dropped, never repaired.

To support a new intent, add its builder here and dispatch to it in ``builder``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from ...schemas.banking_view import FinancialIntent
from ...state import UserProfile
from .intents import TITLES
from .verified_rows import number, profile_currency, rows_for_table

_ZONE = timezone(timedelta(hours=-6), name="America/Monterrey")
_ACCOUNT_NAMES = {
    "checking": "Cuenta de cheques",
    "savings": "Cuenta de ahorro",
    "credit": "Tarjeta de crédito",
}
#: Several database categories collapse onto one visual category; anything
#: unmapped becomes "other".
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
_CARD_NETWORKS = {"visa", "mastercard", "amex", "other"}
_CARD_STATUSES = {"active", "blocked", "inactive"}

_MAX_CARDS = 12
_MAX_TRANSACTIONS = 100
_MAX_CATEGORIES = 8
_MAX_TREND_POINTS = 240
_MAX_ACTIVITY_CELLS = 366
_MAX_RECURRING_PAYMENTS = 30

#: ``(view, structured data, message)`` - what every builder in this module returns.
ViewResult = tuple[dict[str, Any], dict[str, Any], str]


def empty_view(intent: FinancialIntent, description: str, currency: str = "MXN") -> dict[str, Any]:
    """The honest answer when nothing verifiable supports the requested view."""
    return {
        "intent": intent,
        "title": TITLES[intent],
        "currency": currency,
        "state": "empty",
        "description": description,
    }


def _empty_result(intent: FinancialIntent, description: str, currency: str = "MXN") -> ViewResult:
    view = empty_view(intent, description, currency)
    return view, {"presentation_intent": intent}, view["description"]


def _expiry(row: Mapping[str, Any]) -> str | None:
    """`YYYY-MM` from the two stored integers, or nothing. Never a full date."""
    month = row.get("expires_month")
    year = row.get("expires_year")
    if not isinstance(month, int) or not isinstance(year, int):
        return None
    if isinstance(month, bool) or isinstance(year, bool) or not 1 <= month <= 12:
        return None
    if not 1900 <= year <= 2100:
        return None
    return f"{year:04d}-{month:02d}"


def _payment_cards(
    observations: Sequence[Mapping[str, Any]], owned_account_ids: set[str]
) -> list[dict[str, Any]]:
    """Map `cards` rows to the masked contract, dropping anything unverifiable.

    Only the four-digit tail travels. A card whose account is not part of the
    summary is dropped rather than shown next to somebody else's balance.
    """
    cards: list[dict[str, Any]] = []
    for row in rows_for_table(observations, "cards"):
        identifier = row.get("id")
        account_id = row.get("account_id")
        name = row.get("display_name")
        card_type = row.get("card_type")
        network = row.get("network")
        last_four = row.get("last_four")
        status = row.get("status")
        if (
            not isinstance(identifier, str)
            or not isinstance(name, str)
            or not name.strip()
            or card_type not in {"debit", "credit"}
            or network not in _CARD_NETWORKS
            or not isinstance(last_four, str)
            or not last_four.isdigit()
            or len(last_four) != 4
            or status not in _CARD_STATUSES
        ):
            continue
        if not isinstance(account_id, str) or account_id not in owned_account_ids:
            continue
        card = {
            "cardId": identifier,
            "cardName": name.strip()[:120],
            "cardType": card_type,
            "network": network,
            "lastFour": last_four,
            "status": status,
            "accountId": account_id,
        }
        expires = _expiry(row)
        if expires is not None:
            card["expires"] = expires
        cards.append(card)
    return cards[:_MAX_CARDS]


def summary_view(observations: Sequence[Mapping[str, Any]], profile: UserProfile) -> ViewResult:
    """Owned cash across accounts in one currency, with the masked cards behind it."""
    del profile
    raw_accounts = rows_for_table(observations, "accounts")
    currencies = {
        row.get("currency")
        for row in raw_accounts
        if row.get("account_type") in {"checking", "savings"}
        and row.get("currency") in {"MXN", "USD"}
        and number(row.get("available_balance")) is not None
    }
    if len(currencies) > 1:
        return _empty_result(
            "financial-summary",
            "Tus cuentas usan más de una moneda; no se combinaron saldos sin una "
            "conversión explícita.",
        )
    if not currencies:
        return _empty_result(
            "financial-summary", "No hay saldos de cuentas verificables para mostrar."
        )

    currency = cast("str", next(iter(currencies)))
    accounts: list[dict[str, Any]] = []
    owned_balance = 0.0
    for row in raw_accounts:
        account_type = row.get("account_type")
        balance = number(row.get("available_balance"))
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
        return _empty_result(
            "financial-summary", "No hay saldos de cuentas verificables para mostrar.", currency
        )

    view = {
        "intent": "financial-summary",
        "title": TITLES["financial-summary"],
        "currency": currency,
        "totalOwnedBalance": owned_balance,
        "accounts": accounts,
    }
    cards = _payment_cards(observations, {item["accountId"] for item in accounts})
    if cards:
        view["cards"] = cards
    data = {
        "presentation_intent": "financial-summary",
        "owned_balance": owned_balance,
        "currency": currency,
    }
    return view, data, f"Tienes {owned_balance:,.2f} {currency} entre cheques y ahorro."


def _transaction_rows(
    observations: Sequence[Mapping[str, Any]], profile: UserProfile
) -> tuple[list[dict[str, Any]], str | None]:
    """Verified transactions in one currency, newest first. `None` if mixed."""
    currency = profile_currency(profile)
    mapped: list[dict[str, Any]] = []
    source_rows = rows_for_table(observations, "transactions")
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
        amount = number(row.get("amount"))
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
    return mapped[:_MAX_TRANSACTIONS], currency


def _local_day(occurred_at: str) -> str:
    return (
        datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        .astimezone(_ZONE)
        .date()
        .isoformat()
    )


def transactions_view(
    observations: Sequence[Mapping[str, Any]], profile: UserProfile
) -> ViewResult:
    """The movement list, bounded to one currency and one local date range."""
    transactions, currency = _transaction_rows(observations, profile)
    if currency is None:
        return _empty_result(
            "transactions",
            "No se pueden combinar movimientos sin identificar una sola moneda para la consulta.",
        )
    if not transactions:
        return _empty_result(
            "transactions", "No hay movimientos en el periodo consultado.", currency
        )
    local_days = [_local_day(item["occurredAt"]) for item in transactions]
    view = {
        "intent": "transactions",
        "title": TITLES["transactions"],
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


def _spending_from_domain_result(observations: Sequence[Mapping[str, Any]]) -> ViewResult | None:
    """Prefer `analyze_spending`: it already returns chart-ready aggregates."""
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
    if not isinstance(domain, Mapping):
        return None
    summaries = domain.get("summary_by_currency")
    categories = domain.get("by_category")
    daily = domain.get("daily_series")
    if not (
        isinstance(summaries, list)
        and len(summaries) == 1
        and isinstance(categories, list)
        and isinstance(daily, list)
    ):
        return None
    currency = summaries[0].get("currency") if isinstance(summaries[0], Mapping) else None
    total = number(summaries[0].get("expenses")) if isinstance(summaries[0], Mapping) else None
    if currency not in {"MXN", "USD"} or total is None:
        return None

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
        amount = number(item.get("amount"))
        if amount is None or amount < 0:
            continue
        name = _CATEGORY_MAP.get(str(item.get("category")), "other")
        # Spending views describe expenses; an income bucket is not one.
        if name == "income":
            continue
        category_totals[name] += amount
    daily_data = [
        {"date": str(item.get("date")), "value": number(item.get("amount")) or 0}
        for item in daily[-_MAX_ACTIVITY_CELLS:]
        if isinstance(item, Mapping) and item.get("currency") == currency
    ]
    if not daily_data:
        return None
    view = _spending_payload(
        currency=cast("str", currency),
        total=total,
        categories=_ranked_categories(category_totals),
        trend=[
            {"label": item["date"], "values": [item["value"]]}
            for item in daily_data[:_MAX_TREND_POINTS]
        ],
        activity=daily_data,
    )
    return (
        view,
        {"presentation_intent": "spending-analysis", "domain_result": dict(domain)},
        f"Analicé {total:,.2f} {currency} de gastos.",
    )


def _ranked_categories(totals: Mapping[str, float]) -> list[dict[str, Any]]:
    return [
        {"category": name, "amount": amount}
        for name, amount in sorted(totals.items(), key=lambda entry: (-entry[1], entry[0]))[
            :_MAX_CATEGORIES
        ]
    ]


def _spending_payload(
    *,
    currency: str,
    total: float,
    categories: list[dict[str, Any]],
    trend: list[dict[str, Any]],
    activity: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "intent": "spending-analysis",
        "title": TITLES["spending-analysis"],
        "currency": currency,
        "totalSpent": total,
        "categories": categories,
        "trend": {
            "title": "Evolución de gastos",
            "data": trend,
            "series": [{"id": "expenses", "label": "Gastos", "tone": "orange"}],
        },
        "activity": {
            "title": "Tus días de mayor gasto",
            "data": activity,
            "initialDate": activity[-1]["date"],
            "initialView": "month",
            "tone": "orange",
        },
    }


def _spending_from_transactions(
    observations: Sequence[Mapping[str, Any]], profile: UserProfile
) -> ViewResult:
    """Fall back to aggregating the verified transaction rows ourselves."""
    transactions, currency = _transaction_rows(observations, profile)
    expenses = [item for item in transactions if item["amount"] < 0]
    if currency is None or not expenses:
        return _empty_result(
            "spending-analysis",
            "No hay gastos verificables en una sola moneda para analizar.",
            currency or "MXN",
        )

    categories: defaultdict[str, float] = defaultdict(float)
    daily: defaultdict[str, float] = defaultdict(float)
    for item in expenses:
        amount = -float(item["amount"])
        categories[item["category"]] += amount
        daily[_local_day(item["occurredAt"])] += amount
    ordered_days = sorted(daily)
    total = sum(categories.values())
    view = _spending_payload(
        currency=currency,
        total=total,
        categories=[
            entry for entry in _ranked_categories(categories) if entry["category"] != "income"
        ],
        trend=[{"label": day, "values": [daily[day]]} for day in ordered_days[:_MAX_TREND_POINTS]],
        activity=[
            {"date": day, "value": daily[day]} for day in ordered_days[-_MAX_ACTIVITY_CELLS:]
        ],
    )
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


def spending_view(observations: Sequence[Mapping[str, Any]], profile: UserProfile) -> ViewResult:
    """Category split, trend and daily activity, from the best source available."""
    from_domain = _spending_from_domain_result(observations)
    if from_domain is not None:
        return from_domain
    return _spending_from_transactions(observations, profile)


def recurring_view(observations: Sequence[Mapping[str, Any]], profile: UserProfile) -> ViewResult:
    """Upcoming subscription charges, in the user's single owned currency."""
    currency = profile_currency(profile)
    payments: list[dict[str, Any]] = []
    for row in rows_for_table(observations, "subscriptions"):
        identifier = row.get("id")
        name = row.get("name")
        amount = number(row.get("amount"))
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
        return _empty_result(
            "recurring-payments",
            "No hay pagos recurrentes verificables en una sola moneda para mostrar.",
            currency or "MXN",
        )
    view = {
        "intent": "recurring-payments",
        "title": TITLES["recurring-payments"],
        "currency": currency,
        "payments": payments[:_MAX_RECURRING_PAYMENTS],
    }
    return view, {"presentation_intent": "recurring-payments"}, "Estos son tus próximos cobros."
