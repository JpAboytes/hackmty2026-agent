"""Strict Finance v2 ``BankingView`` data contract.

This mirrors the shared mobile contract.  The model may select one of the
enumerated intents, but only trusted application code creates these values.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

FinancialIntent = Literal[
    "financial-summary",
    "transactions",
    "spending-analysis",
    "cash-flow",
    "budgets",
    "recurring-payments",
    "credit-card",
    "debts",
    "transfers",
    "card-security",
    "savings-goals",
    "banking-information",
    "financial-education",
]

FINANCIAL_INTENTS: tuple[FinancialIntent, ...] = (
    "financial-summary",
    "transactions",
    "spending-analysis",
    "cash-flow",
    "budgets",
    "recurring-payments",
    "credit-card",
    "debts",
    "transfers",
    "card-security",
    "savings-goals",
    "banking-information",
    "financial-education",
)

Label = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=120),
]
Description = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=600)
]
ItemId = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)]
Money = Annotated[float, Field(ge=0, le=1e12, allow_inf_nan=False)]
PositiveMoney = Annotated[float, Field(gt=0, le=1e12, allow_inf_nan=False)]
SignedMoney = Annotated[float, Field(ge=-1e12, le=1e12, allow_inf_nan=False)]
Day = Annotated[str, StringConstraints(strict=True, pattern=r"^\d{4}-\d{2}-\d{2}$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, populate_by_name=True)


def _valid_day(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("invalid calendar date") from exc
    if parsed.isoformat() != value or not date(1900, 1, 1) <= parsed <= date(2100, 12, 31):
        raise ValueError("date is outside the supported range")
    return value


def _require_unique(items: list[Any], attribute: str) -> None:
    if len({getattr(item, attribute) for item in items}) != len(items):
        raise ValueError("item identifiers must be unique")


class Account(_StrictModel):
    account_id: ItemId = Field(alias="accountId")
    account_name: Label = Field(alias="accountName")
    account_type: Literal["checking", "savings", "credit"] = Field(alias="accountType")
    account_last_four: Annotated[str, StringConstraints(strict=True, pattern=r"^\d{4}$")] | None = (
        Field(default=None, alias="accountLastFour")
    )
    available_balance: SignedMoney = Field(alias="availableBalance")
    status: Literal["active", "blocked", "inactive"] = "active"


LastFour = Annotated[str, StringConstraints(strict=True, pattern=r"^\d{4}$")]
# Percentage points, like `credit_card_terms.annual_interest_rate`; never a fraction.
Rate = Annotated[float, Field(ge=0, le=1000, allow_inf_nan=False)]


class PaymentCard(_StrictModel):
    """Masked projection of `public.cards`.

    There is deliberately no field for a full card number, CVV, expiry day or
    cardholder document: the contract cannot carry one even if a caller has it.
    """

    card_id: ItemId = Field(alias="cardId")
    card_name: Label = Field(alias="cardName")
    card_type: Literal["debit", "credit"] = Field(alias="cardType")
    network: Literal["visa", "mastercard", "amex", "other"]
    last_four: LastFour = Field(alias="lastFour")
    status: Literal["active", "blocked", "inactive"] = "active"
    expires: (
        Annotated[str, StringConstraints(strict=True, pattern=r"^\d{4}-(0[1-9]|1[0-2])$")] | None
    ) = None
    account_id: ItemId | None = Field(default=None, alias="accountId")


class Transaction(_StrictModel):
    transaction_id: ItemId = Field(alias="transactionId")
    title: Label
    description: Description | None = None
    amount: SignedMoney
    occurred_at: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=64)] = (
        Field(alias="occurredAt")
    )
    category: Literal[
        "food",
        "transport",
        "entertainment",
        "utilities",
        "health",
        "shopping",
        "income",
        "transfer",
        "other",
    ]
    status: Literal["pending", "completed", "declined"] = "completed"

    @field_validator("occurred_at")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("occurredAt must be an ISO timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("occurredAt must include an offset")
        return value


class AreaSeries(_StrictModel):
    id: Annotated[
        str,
        StringConstraints(
            strict=True, min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9._:-]*$"
        ),
    ]
    label: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=80)]
    tone: Literal["blue", "violet", "green", "orange"] | None = None


class AreaPoint(_StrictModel):
    label: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=80)]
    values: Annotated[
        list[Annotated[float, Field(ge=-1e15, le=1e15, allow_inf_nan=False)]],
        Field(min_length=1, max_length=4),
    ]


class AreaProps(_StrictModel):
    title: Annotated[str, StringConstraints(strict=True, max_length=100)] | None = None
    subtitle: Annotated[str, StringConstraints(strict=True, max_length=200)] | None = None
    data: Annotated[list[AreaPoint], Field(max_length=240)]
    series: Annotated[list[AreaSeries], Field(min_length=1, max_length=4)]
    height: Annotated[float, Field(ge=180, le=500, allow_inf_nan=False)] | None = None
    currency: Literal["MXN", "USD"] | None = None
    status: Literal["ready", "loading"] | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> AreaProps:
        if len({item.id for item in self.series}) != len(self.series):
            raise ValueError("series identifiers must be unique")
        if any(len(point.values) != len(self.series) for point in self.data):
            raise ValueError("each point must contain one value per series")
        return self


class HeatmapCell(_StrictModel):
    date: Day
    value: Annotated[float, Field(ge=0, le=1e15, allow_inf_nan=False)]

    _validate_date = field_validator("date")(_valid_day)


class HeatmapProps(_StrictModel):
    title: Annotated[str, StringConstraints(strict=True, max_length=100)] | None = None
    subtitle: Annotated[str, StringConstraints(strict=True, max_length=200)] | None = None
    data: Annotated[list[HeatmapCell], Field(max_length=4000)]
    initial_date: Day | None = Field(default=None, alias="initialDate")
    initial_view: Literal["year", "month", "week"] | None = Field(default=None, alias="initialView")
    tone: Literal["green", "blue", "violet", "orange"] | None = None
    currency: Literal["MXN", "USD"] | None = None
    status: Literal["ready", "loading"] | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> HeatmapProps:
        if len({item.date for item in self.data}) != len(self.data):
            raise ValueError("heatmap dates must be unique")
        if self.initial_date is not None:
            _valid_day(self.initial_date)
        return self


class ScheduleItem(_StrictModel):
    id: ItemId
    name: Label
    date: Day
    amount: Money
    direction: Literal["income", "expense"]

    _validate_date = field_validator("date")(_valid_day)


class Scenario(_StrictModel):
    id: ItemId
    name: Label
    monthly_payment: Money = Field(alias="monthlyPayment")
    months: Annotated[int, Field(ge=1, le=600)]
    total_interest: Money = Field(alias="totalInterest")


class Goal(_StrictModel):
    id: ItemId
    name: Label
    saved: Money
    target: PositiveMoney
    target_date: Day = Field(alias="targetDate")
    monthly_contribution: Money = Field(alias="monthlyContribution")

    _validate_date = field_validator("target_date")(_valid_day)


class _ViewBase(_StrictModel):
    title: Label
    subtitle: Description | None = None
    currency: Literal["MXN", "USD"] = "MXN"


class FinancialSummaryView(_ViewBase):
    intent: Literal["financial-summary"]
    total_owned_balance: SignedMoney = Field(alias="totalOwnedBalance")
    accounts: Annotated[list[Account], Field(max_length=30)]
    income: Money | None = None
    expenses: Money | None = None
    cards: Annotated[list[PaymentCard], Field(max_length=12)] | None = None

    @model_validator(mode="after")
    def unique_accounts(self) -> FinancialSummaryView:
        if len({item.account_id for item in self.accounts}) != len(self.accounts):
            raise ValueError("account identifiers must be unique")
        cards = self.cards or []
        if len({item.card_id for item in cards}) != len(cards):
            raise ValueError("card identifiers must be unique")
        owned = {item.account_id for item in self.accounts}
        for card in cards:
            if card.account_id is not None and card.account_id not in owned:
                raise ValueError("card does not belong to an account in this summary")
        return self


class TransactionsView(_ViewBase):
    intent: Literal["transactions"]
    start_date: Day = Field(alias="startDate")
    end_date: Day = Field(alias="endDate")
    time_zone: Literal["America/Monterrey"] = Field(alias="timeZone")
    transactions: Annotated[list[Transaction], Field(max_length=100)]

    @model_validator(mode="after")
    def validate_period(self) -> TransactionsView:
        start = _valid_day(self.start_date)
        end = _valid_day(self.end_date)
        if start > end:
            raise ValueError("startDate must not follow endDate")
        if len({item.transaction_id for item in self.transactions}) != len(self.transactions):
            raise ValueError("transaction identifiers must be unique")
        # Monterrey has observed UTC-06 without daylight-saving changes since 2022;
        # the product's supported financial dates are current/future demo dates.
        zone = timezone(timedelta(hours=-6))
        for item in self.transactions:
            occurred = datetime.fromisoformat(item.occurred_at.replace("Z", "+00:00"))
            local_day = occurred.astimezone(zone).date().isoformat()
            if local_day < start or local_day > end:
                raise ValueError("transaction is outside the requested period")
        return self


class CategoryAmount(_StrictModel):
    category: Literal[
        "food",
        "transport",
        "entertainment",
        "utilities",
        "health",
        "shopping",
        "transfer",
        "other",
    ]
    amount: Money


class SpendingAnalysisView(_ViewBase):
    intent: Literal["spending-analysis"]
    total_spent: Money = Field(alias="totalSpent")
    categories: Annotated[list[CategoryAmount], Field(max_length=8)]
    previous_total: Money | None = Field(default=None, alias="previousTotal")
    insight: Description | None = None
    trend: AreaProps | None = None
    activity: HeatmapProps | None = None

    @model_validator(mode="after")
    def validate_categories(self) -> SpendingAnalysisView:
        if len({item.category for item in self.categories}) != len(self.categories):
            raise ValueError("categories must be unique")
        if self.activity is not None and len(self.activity.data) > 366:
            raise ValueError("spending activity is limited to 366 days")
        # The headline may exceed the breakdown when categories are truncated, but
        # the visible slices can never add up to more than the total they belong to.
        if sum(item.amount for item in self.categories) > self.total_spent + 0.01:
            raise ValueError("categories add up to more than totalSpent")
        return self


class CashFlowView(_ViewBase):
    intent: Literal["cash-flow"]
    projected_balance: SignedMoney = Field(alias="projectedBalance")
    target_date: Day = Field(alias="targetDate")
    assumptions: Description
    projection: AreaProps
    upcoming: Annotated[list[ScheduleItem], Field(max_length=30)]

    _validate_date = field_validator("target_date")(_valid_day)

    @model_validator(mode="after")
    def unique_upcoming(self) -> CashFlowView:
        _require_unique(self.upcoming, "id")
        return self


class Budget(_StrictModel):
    id: ItemId
    name: Label
    spent: Money
    limit: PositiveMoney
    period: Label
    status: Literal["active", "paused"]


class BudgetsView(_ViewBase):
    intent: Literal["budgets"]
    budgets: Annotated[list[Budget], Field(max_length=30)]

    @model_validator(mode="after")
    def unique_budgets(self) -> BudgetsView:
        _require_unique(self.budgets, "id")
        return self


class RecurringPayment(_StrictModel):
    id: ItemId
    name: Label
    amount: Money
    next_date: Day = Field(alias="nextDate")
    cycle: Literal["weekly", "monthly", "yearly"]
    status: Literal["active", "paused"]

    _validate_date = field_validator("next_date")(_valid_day)


class RecurringPaymentsView(_ViewBase):
    intent: Literal["recurring-payments"]
    payments: Annotated[list[RecurringPayment], Field(max_length=30)]

    @model_validator(mode="after")
    def unique_payments(self) -> RecurringPaymentsView:
        _require_unique(self.payments, "id")
        return self


class CreditCardView(_ViewBase):
    intent: Literal["credit-card"]
    card_name: Label = Field(alias="cardName")
    last_four: LastFour | None = Field(default=None, alias="lastFour")
    debt: Money
    available_credit: Money = Field(alias="availableCredit")
    minimum_payment: Money = Field(alias="minimumPayment")
    interest_free_payment: Money = Field(alias="interestFreePayment")
    due_date: Day = Field(alias="dueDate")
    card: PaymentCard | None = None
    credit_limit: PositiveMoney | None = Field(default=None, alias="creditLimit")
    statement_balance: Money | None = Field(default=None, alias="statementBalance")
    cutoff_date: Day | None = Field(default=None, alias="cutoffDate")
    annual_interest_rate: Rate | None = Field(default=None, alias="annualInterestRate")
    cat_percentage: Rate | None = Field(default=None, alias="catPercentage")

    _validate_date = field_validator("due_date")(_valid_day)

    @field_validator("cutoff_date")
    @classmethod
    def validate_cutoff(cls, value: str | None) -> str | None:
        return None if value is None else _valid_day(value)

    @model_validator(mode="after")
    def validate_terms(self) -> CreditCardView:
        if self.credit_limit is not None and self.credit_limit < self.available_credit:
            raise ValueError("availableCredit must not exceed creditLimit")
        if self.cutoff_date is not None and self.cutoff_date > self.due_date:
            raise ValueError("cutoffDate must not follow dueDate")
        if self.card is not None:
            if self.card.card_type != "credit":
                raise ValueError("a credit-card view only accepts a credit card")
            if self.last_four is not None and self.card.last_four != self.last_four:
                raise ValueError("lastFour contradicts the card")
        return self


class DebtsView(_ViewBase):
    intent: Literal["debts"]
    outstanding: Money
    assumptions: Description
    scenarios: Annotated[list[Scenario], Field(min_length=1, max_length=4)]

    @model_validator(mode="after")
    def unique_scenarios(self) -> DebtsView:
        _require_unique(self.scenarios, "id")
        return self


class TransfersView(_ViewBase):
    intent: Literal["transfers"]
    source: Label
    recipient: Label
    amount: PositiveMoney
    fee: Money
    scheduled_date: Day | None = Field(default=None, alias="scheduledDate")

    @field_validator("scheduled_date")
    @classmethod
    def validate_scheduled_date(cls, value: str | None) -> str | None:
        return None if value is None else _valid_day(value)


class CardSecurityView(_ViewBase):
    intent: Literal["card-security"]
    card_name: Label = Field(alias="cardName")
    last_four: LastFour | None = Field(default=None, alias="lastFour")
    status: Literal["active", "blocked", "inactive"]
    reported_transaction: Transaction | None = Field(default=None, alias="reportedTransaction")
    guidance: Description
    card: PaymentCard | None = None

    @model_validator(mode="after")
    def validate_card(self) -> CardSecurityView:
        if self.card is not None:
            if self.card.status != self.status:
                raise ValueError("the card status contradicts the view status")
            if self.last_four is not None and self.card.last_four != self.last_four:
                raise ValueError("lastFour contradicts the card")
        return self


class SavingsGoalsView(_ViewBase):
    intent: Literal["savings-goals"]
    goals: Annotated[list[Goal], Field(max_length=30)]

    @model_validator(mode="after")
    def unique_goals(self) -> SavingsGoalsView:
        _require_unique(self.goals, "id")
        return self


class Document(_StrictModel):
    id: ItemId
    name: Label
    period: Label
    status: Literal["available", "processing"]


class BankingInformationView(_ViewBase):
    intent: Literal["banking-information"]
    bank_name: Label = Field(alias="bankName")
    holder: Label
    masked_clabe: (
        Annotated[str, StringConstraints(strict=True, pattern=r"^[•*]{14}\d{4}$")] | None
    ) = Field(default=None, alias="maskedClabe")
    documents: Annotated[list[Document], Field(max_length=30)]

    @model_validator(mode="after")
    def unique_documents(self) -> BankingInformationView:
        _require_unique(self.documents, "id")
        return self


class FinancialEducationView(_ViewBase):
    intent: Literal["financial-education"]
    concept: Label
    explanation: Description
    takeaways: Annotated[list[Description], Field(min_length=1, max_length=5)]
    scenarios: Annotated[list[Scenario], Field(max_length=4)] | None = None

    @model_validator(mode="after")
    def unique_scenarios(self) -> FinancialEducationView:
        if self.scenarios is not None:
            _require_unique(self.scenarios, "id")
        return self


class EmptyBankingView(_ViewBase):
    intent: FinancialIntent
    state: Literal["empty"]
    description: Description


BankingViewValue = (
    FinancialSummaryView
    | TransactionsView
    | SpendingAnalysisView
    | CashFlowView
    | BudgetsView
    | RecurringPaymentsView
    | CreditCardView
    | DebtsView
    | TransfersView
    | CardSecurityView
    | SavingsGoalsView
    | BankingInformationView
    | FinancialEducationView
    | EmptyBankingView
)

_BANKING_VIEW_ADAPTER: TypeAdapter[Any] = TypeAdapter(BankingViewValue)


def validate_banking_view(value: Any) -> dict[str, Any]:
    """Validate one view and return a detached, alias-preserving JSON object."""
    parsed = _BANKING_VIEW_ADAPTER.validate_python(value, strict=True)
    return cast(
        "dict[str, Any]",
        parsed.model_dump(mode="json", by_alias=True, exclude_none=True),
    )
