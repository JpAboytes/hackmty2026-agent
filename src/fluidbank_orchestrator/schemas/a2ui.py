"""Strict A2UI v0.9.1 validation for the Expo transport boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date
from importlib.resources import files
from math import isfinite
from typing import Annotated, Any, Literal

from a2ui.basic_catalog.provider import BasicCatalog  # type: ignore[import-untyped]
from a2ui.inference_formats.direct_json import (  # type: ignore[import-untyped]
    DirectJsonFormat,
)
from a2ui.schema.catalog import CatalogConfig  # type: ignore[import-untyped]
from a2ui.schema.catalog_provider import A2uiCatalogProvider  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .banking_view import FINANCIAL_INTENTS, validate_banking_view

A2UI_VERSION = "v0.9.1"
A2UI_SDK_VERSION = "0.9.1"
A2UI_MIME_TYPE = "application/a2ui+json"
A2UI_BASIC_CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
A2UI_FINANCE_CATALOG = "https://fluidbank.app/a2ui/catalogs/finance/v1"
A2UI_FINANCE_V2_CATALOG = "https://fluidbank.app/a2ui/catalogs/finance/v2"

MAX_RESOURCE_BYTES = 262_144
MAX_MESSAGES = 100
MAX_COMPONENTS = 500
MAX_STRING_LENGTH = 4_000
MAX_JSON_DEPTH = 32

Identifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z][A-Za-z0-9._:-]*$",
    ),
]
BoundedString = Annotated[
    str,
    StringConstraints(strict=True, max_length=MAX_STRING_LENGTH),
]
ContextKey = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)]


class A2UIValidationError(ValueError):
    """An A2UI payload is unsafe or incompatible with the Expo renderer."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, populate_by_name=True)


class _Binding(_StrictModel):
    path: Annotated[str, StringConstraints(strict=True, max_length=512)]


DynamicString = BoundedString | _Binding
DynamicValue = (
    BoundedString
    | StrictInt
    | StrictFloat
    | StrictBool
    | Annotated[list[JsonValue], Field(max_length=100)]
    | _Binding
)


class _Accessibility(_StrictModel):
    label: DynamicString | None = None
    description: DynamicString | None = None


class _ActionEvent(_StrictModel):
    name: Identifier
    context: dict[ContextKey, DynamicValue] | None = None


class _ServerAction(_StrictModel):
    event: _ActionEvent


class _ComponentBase(_StrictModel):
    id: Identifier
    weight: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)] | None = None
    accessibility: _Accessibility | None = None


class _TextComponent(_ComponentBase):
    component: Literal["Text"]
    text: DynamicString
    variant: Literal["h1", "h2", "h3", "h4", "h5", "caption", "body"] | None = None


class _ButtonComponent(_ComponentBase):
    component: Literal["Button"]
    child: Identifier
    variant: Literal["default", "primary", "borderless"] | None = None
    action: _ServerAction


class _TextFieldComponent(_ComponentBase):
    component: Literal["TextField"]
    label: DynamicString
    value: _Binding
    variant: Literal["shortText", "longText", "number", "obscured"] | None = None


class _DateInputComponent(_ComponentBase):
    component: Literal["DateTimeInput"]
    label: DynamicString | None = None
    value: _Binding
    enable_date: Literal[True] = Field(alias="enableDate")
    enable_time: Literal[False] | None = Field(default=None, alias="enableTime")


class _SliderComponent(_ComponentBase):
    component: Literal["Slider"]
    label: DynamicString | None = None
    value: _Binding
    min: StrictInt | StrictFloat = 0
    max: StrictInt | StrictFloat

    @model_validator(mode="after")
    def valid_range(self) -> _SliderComponent:
        if not isfinite(self.min) or not isfinite(self.max) or self.max <= self.min:
            raise ValueError("invalid slider range")
        return self


class _CardComponent(_ComponentBase):
    component: Literal["Card"]
    child: Identifier


class _ColumnComponent(_ComponentBase):
    component: Literal["Column"]
    children: Annotated[list[Identifier], Field(max_length=MAX_COMPONENTS)]
    justify: (
        Literal[
            "start",
            "center",
            "end",
            "spaceBetween",
            "spaceAround",
            "spaceEvenly",
            "stretch",
        ]
        | None
    ) = None
    align: Literal["start", "center", "end", "stretch"] | None = None


class _AreaSeries(_StrictModel):
    id: Annotated[
        str,
        StringConstraints(
            strict=True, min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9._:-]*$"
        ),
    ]
    label: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=80)]
    tone: Literal["blue", "violet", "green", "orange"] | None = None


class _AreaPoint(_StrictModel):
    label: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=80)]
    values: Annotated[
        list[Annotated[float, Field(ge=-1e15, le=1e15, allow_inf_nan=False)]],
        Field(min_length=1, max_length=4),
    ]


class _AreaProps(_StrictModel):
    data: Annotated[list[_AreaPoint], Field(max_length=240)]
    series: Annotated[list[_AreaSeries], Field(min_length=1, max_length=4)]
    title: Annotated[str, StringConstraints(strict=True, max_length=100)] | None = None
    subtitle: Annotated[str, StringConstraints(strict=True, max_length=200)] | None = None
    height: Annotated[float, Field(ge=180, le=500, allow_inf_nan=False)] | None = None
    currency: Literal["MXN", "USD"] | None = None
    status: Literal["ready", "loading"] | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> _AreaProps:
        if len({item.id for item in self.series}) != len(self.series):
            raise ValueError("series identifiers must be unique")
        if any(len(point.values) != len(self.series) for point in self.data):
            raise ValueError("each point must contain one value per series")
        if len({point.label for point in self.data}) != len(self.data):
            raise ValueError("area point labels must be unique")
        return self


class _HeatmapCell(_StrictModel):
    date: Annotated[str, StringConstraints(strict=True, pattern=r"^\d{4}-\d{2}-\d{2}$")]
    value: Annotated[float, Field(ge=0, le=1e15, allow_inf_nan=False)]

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("invalid calendar date") from exc
        if parsed.isoformat() != value or not date(1900, 1, 1) <= parsed <= date(2100, 12, 31):
            raise ValueError("date is outside the supported range")
        return value


class _HeatmapProps(_StrictModel):
    data: Annotated[list[_HeatmapCell], Field(max_length=500)]
    title: Annotated[str, StringConstraints(strict=True, max_length=100)] | None = None
    subtitle: Annotated[str, StringConstraints(strict=True, max_length=200)] | None = None
    initial_date: (
        Annotated[str, StringConstraints(strict=True, pattern=r"^\d{4}-\d{2}-\d{2}$")] | None
    ) = Field(default=None, alias="initialDate")
    initial_view: Literal["year", "month", "week"] | None = Field(default=None, alias="initialView")
    tone: Literal["green", "blue", "violet", "orange"] | None = None
    currency: Literal["MXN", "USD"] | None = None
    status: Literal["ready", "loading"] | None = None

    @model_validator(mode="after")
    def require_unique_dates(self) -> _HeatmapProps:
        if len({item.date for item in self.data}) != len(self.data):
            raise ValueError("heatmap dates must be unique")
        if self.initial_date is not None:
            _HeatmapCell(date=self.initial_date, value=0)
        return self


class _AreaChartValue(_StrictModel):
    kind: Literal["area"]
    accessible_summary: (
        Annotated[str, StringConstraints(strict=True, min_length=1, max_length=500)] | None
    ) = Field(default=None, alias="accessibleSummary")
    props: _AreaProps


class _HeatmapChartValue(_StrictModel):
    kind: Literal["heatmap"]
    accessible_summary: (
        Annotated[str, StringConstraints(strict=True, min_length=1, max_length=500)] | None
    ) = Field(default=None, alias="accessibleSummary")
    props: _HeatmapProps


ChartValue = _Binding | Annotated[_AreaChartValue | _HeatmapChartValue, Field(discriminator="kind")]
_CHART_VALUE_ADAPTER: TypeAdapter[Any] = TypeAdapter(
    Annotated[_AreaChartValue | _HeatmapChartValue, Field(discriminator="kind")]
)


class _ChartComponent(_ComponentBase):
    component: Literal["Chart"]
    chart: ChartValue


class _BankingViewComponent(_ComponentBase):
    component: Literal["BankingView"]
    view: JsonValue | _Binding


Component = (
    _TextFieldComponent
    | _DateInputComponent
    | _SliderComponent
    | _TextComponent
    | _ButtonComponent
    | _CardComponent
    | _ColumnComponent
    | _ChartComponent
    | _BankingViewComponent
)


class _Theme(_StrictModel):
    primary_color: Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")] | None = Field(
        default=None, alias="primaryColor"
    )
    icon_url: Annotated[str, StringConstraints(strict=True, max_length=2_048)] | None = Field(
        default=None, alias="iconUrl"
    )
    agent_display_name: Annotated[str, StringConstraints(strict=True, max_length=128)] | None = (
        Field(default=None, alias="agentDisplayName")
    )


class _CreateSurfaceBody(_StrictModel):
    surface_id: Identifier = Field(alias="surfaceId")
    catalog_id: Literal[
        "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json",
        "https://fluidbank.app/a2ui/catalogs/finance/v1",
        "https://fluidbank.app/a2ui/catalogs/finance/v2",
    ] = Field(alias="catalogId")
    theme: _Theme | None = None
    send_data_model: StrictBool | None = Field(default=None, alias="sendDataModel")


class _UpdateComponentsBody(_StrictModel):
    surface_id: Identifier = Field(alias="surfaceId")
    components: Annotated[list[Component], Field(min_length=1, max_length=MAX_COMPONENTS)]


class _UpdateDataModelBody(_StrictModel):
    surface_id: Identifier = Field(alias="surfaceId")
    path: Annotated[str, StringConstraints(strict=True, max_length=512)] | None = None
    value: JsonValue | None = None


class _DeleteSurfaceBody(_StrictModel):
    surface_id: Identifier = Field(alias="surfaceId")


class _CreateSurfaceMessage(_StrictModel):
    version: Literal["v0.9.1"]
    create_surface: _CreateSurfaceBody = Field(alias="createSurface")


class _UpdateComponentsMessage(_StrictModel):
    version: Literal["v0.9.1"]
    update_components: _UpdateComponentsBody = Field(alias="updateComponents")


class _UpdateDataModelMessage(_StrictModel):
    version: Literal["v0.9.1"]
    update_data_model: _UpdateDataModelBody = Field(alias="updateDataModel")


class _DeleteSurfaceMessage(_StrictModel):
    version: Literal["v0.9.1"]
    delete_surface: _DeleteSurfaceBody = Field(alias="deleteSurface")


_MESSAGE_ADAPTER: TypeAdapter[Any] = TypeAdapter(
    _CreateSurfaceMessage
    | _UpdateComponentsMessage
    | _UpdateDataModelMessage
    | _DeleteSurfaceMessage
)


class _MappingCatalogProvider(A2uiCatalogProvider):  # type: ignore[misc]
    def __init__(self, catalog: Mapping[str, Any]) -> None:
        self._catalog = deepcopy(dict(catalog))

    def load(self) -> dict[str, Any]:
        return deepcopy(self._catalog)


def _finance_catalog_config(*, version: Literal["v1", "v2"] = "v1") -> CatalogConfig:
    resource = files("fluidbank_orchestrator.a2ui_catalogs").joinpath("finance_v1.json")
    try:
        loaded = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("The finance A2UI catalog could not be loaded") from exc
    if (
        not isinstance(loaded, Mapping)
        or loaded.get("catalogId") != A2UI_FINANCE_CATALOG
        or loaded.get("$id") != A2UI_FINANCE_CATALOG
    ):
        raise RuntimeError("The finance A2UI catalog has an unexpected identifier")
    loaded = deepcopy(dict(loaded))
    if version == "v2":
        loaded["$id"] = A2UI_FINANCE_V2_CATALOG
        loaded["catalogId"] = A2UI_FINANCE_V2_CATALOG
        loaded["title"] = "Fluidbank Finance Catalog v2"
        components = loaded.get("components")
        definitions = loaded.get("$defs")
        if not isinstance(components, dict) or not isinstance(definitions, dict):
            raise RuntimeError("The finance A2UI catalog is malformed")
        components["BankingView"] = {
            "type": "object",
            "allOf": [
                {
                    "$ref": (
                        "https://a2ui.org/specification/v0_9/common_types.json"
                        "#/$defs/ComponentCommon"
                    )
                },
                {"$ref": "#/$defs/CatalogComponentCommon"},
                {
                    "type": "object",
                    "properties": {
                        "component": {"const": "BankingView"},
                        "view": {
                            "anyOf": [
                                {"$ref": "#/$defs/DataBinding"},
                                {"type": "object"},
                            ]
                        },
                    },
                    "required": ["component", "view"],
                },
            ],
            "unevaluatedProperties": False,
        }
        any_component = definitions.get("anyComponent")
        if not isinstance(any_component, dict) or not isinstance(any_component.get("oneOf"), list):
            raise RuntimeError("The finance A2UI catalog is malformed")
        any_component["oneOf"].append({"$ref": "#/components/BankingView"})
    return CatalogConfig(
        name=f"fluidbank-finance-{version}",
        provider=_MappingCatalogProvider(loaded),
    )


def _catalog_validator(config: CatalogConfig, catalog_id: str) -> Any:
    try:
        validator = (
            DirectJsonFormat(
                version=A2UI_SDK_VERSION,
                catalogs=[config],
                accepts_inline_catalogs=False,
            )
            .get_selected_catalog()
            .validator
        )
        validator.validate(
            [
                {
                    "version": A2UI_VERSION,
                    "createSurface": {
                        "surfaceId": "catalog-validation",
                        "catalogId": catalog_id,
                    },
                },
                {
                    "version": A2UI_VERSION,
                    "updateComponents": {
                        "surfaceId": "catalog-validation",
                        "components": [{"id": "root", "component": "Text", "text": "validation"}],
                    },
                },
            ]
        )
        return validator
    except Exception as exc:
        raise RuntimeError("An allowlisted A2UI catalog failed startup validation") from exc


_SDK_VALIDATORS = {
    A2UI_BASIC_CATALOG: _catalog_validator(
        BasicCatalog.get_config(A2UI_SDK_VERSION), A2UI_BASIC_CATALOG
    ),
    A2UI_FINANCE_CATALOG: _catalog_validator(_finance_catalog_config(), A2UI_FINANCE_CATALOG),
    A2UI_FINANCE_V2_CATALOG: _catalog_validator(
        _finance_catalog_config(version="v2"), A2UI_FINANCE_V2_CATALOG
    ),
}
_COMPONENTS_BY_CATALOG = {
    A2UI_BASIC_CATALOG: frozenset(
        {"Text", "Button", "Card", "Column", "TextField", "DateTimeInput", "Slider"}
    ),
    A2UI_FINANCE_CATALOG: frozenset({"Text", "Button", "Card", "Column", "Chart"}),
    A2UI_FINANCE_V2_CATALOG: frozenset(
        {"Text", "Button", "Card", "Column", "Chart", "BankingView"}
    ),
}
_OPERATIONS = ("createSurface", "updateComponents", "updateDataModel", "deleteSurface")


class A2UIBundle(BaseModel):
    """Self-contained ordered A2UI sequence returned to Expo."""

    model_config = ConfigDict(extra="forbid")

    resource_uri: Annotated[str, StringConstraints(min_length=1, max_length=2_048)]
    messages: Annotated[list[dict[str, Any]], Field(min_length=1, max_length=MAX_MESSAGES)]


def _validate_json_tree(value: Any, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise A2UIValidationError("A2UI JSON exceeds the nesting limit")
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise A2UIValidationError("A2UI JSON contains an oversized string")
        return
    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise A2UIValidationError("A2UI JSON contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_tree(item, depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > MAX_STRING_LENGTH:
                raise A2UIValidationError("A2UI JSON contains an invalid object key")
            _validate_json_tree(item, depth=depth + 1)
        return
    raise A2UIValidationError("A2UI content must contain JSON values only")


def validate_json_pointer(path: str | None) -> None:
    """Validate the absolute JSON Pointer subset accepted by Expo."""
    if path is None or path == "/":
        return
    if not path.startswith("/") or "#" in path:
        raise A2UIValidationError("A2UI data paths must be absolute JSON Pointers")
    segments = path[1:].split("/")
    for segment in segments:
        index = 0
        while index < len(segment):
            if segment[index] == "~":
                if index + 1 >= len(segment) or segment[index + 1] not in {"0", "1"}:
                    raise A2UIValidationError("A2UI data path has an invalid escape")
                index += 1
            index += 1
        decoded = segment.replace("~1", "/").replace("~0", "~")
        if decoded in {"__proto__", "prototype", "constructor"}:
            raise A2UIValidationError("A2UI data path contains a forbidden segment")


def _operation(message: Mapping[str, Any]) -> str:
    operations = [name for name in _OPERATIONS if name in message]
    if len(operations) != 1:
        raise A2UIValidationError("A2UI messages must contain exactly one operation")
    return operations[0]


def message_surface_id(message: Mapping[str, Any]) -> str:
    operation = _operation(message)
    body = message.get(operation)
    if not isinstance(body, Mapping) or not isinstance(body.get("surfaceId"), str):
        raise A2UIValidationError("A2UI message has no valid surface ID")
    surface_id = body["surfaceId"]
    assert isinstance(surface_id, str)
    return surface_id


def validate_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    expected_surface_id: str | None = None,
    catalog_id: str | None = None,
) -> list[dict[str, Any]]:
    """Validate messages and return detached copies without schema translation."""
    if not messages or len(messages) > MAX_MESSAGES:
        raise A2UIValidationError("A2UI message count is outside the supported bounds")

    validated: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise A2UIValidationError("Every A2UI message must be a JSON object")
        _validate_json_tree(message)
        try:
            parsed = _MESSAGE_ADAPTER.validate_python(message, strict=True)
        except Exception as exc:
            raise A2UIValidationError("A2UI message is malformed or unsupported") from exc

        if isinstance(parsed, _UpdateComponentsMessage):
            ids = [component.id for component in parsed.update_components.components]
            if len(ids) != len(set(ids)):
                raise A2UIValidationError("A2UI component IDs must be unique")
            if catalog_id is not None:
                allowed = _COMPONENTS_BY_CATALOG.get(catalog_id)
                if allowed is None or any(
                    component.component not in allowed
                    for component in parsed.update_components.components
                ):
                    raise A2UIValidationError("A2UI component is not defined by its catalog")
        if isinstance(parsed, _UpdateDataModelMessage):
            validate_json_pointer(parsed.update_data_model.path)

        surface_id = message_surface_id(message)
        if expected_surface_id is not None and surface_id != expected_surface_id:
            raise A2UIValidationError("A2UI message targets an unexpected surface")
        validated.append(deepcopy(dict(message)))
    return validated


def validate_static_template(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Validate the required ordered createSurface/updateComponents resource."""
    if (
        len(messages) != 2
        or _operation(messages[0]) != "createSurface"
        or _operation(messages[1]) != "updateComponents"
    ):
        raise A2UIValidationError(
            "A2UI templates must contain createSurface followed by updateComponents"
        )
    surface_id = message_surface_id(messages[0])
    creation = messages[0].get("createSurface")
    if not isinstance(creation, Mapping) or creation.get("catalogId") not in _SDK_VALIDATORS:
        raise A2UIValidationError("A2UI template uses an unsupported catalog")
    catalog_id = creation["catalogId"]
    assert isinstance(catalog_id, str)
    validated = validate_messages(messages, expected_surface_id=surface_id, catalog_id=catalog_id)
    components = validated[1]["updateComponents"]["components"]
    if not any(component.get("id") == "root" for component in components):
        raise A2UIValidationError("A2UI templates must declare a root component")
    _sdk_validate(validated, catalog_id)
    return surface_id, validated


def validate_dynamic_updates(
    messages: Sequence[Mapping[str, Any]], *, expected_surface_id: str
) -> list[dict[str, Any]]:
    """Validate embedded data-model updates for a selected surface."""
    if not messages or any(_operation(message) != "updateDataModel" for message in messages):
        raise A2UIValidationError("Embedded A2UI content must contain updateDataModel messages")
    return validate_messages(messages, expected_surface_id=expected_surface_id)


def validate_complete_sequence(messages: Sequence[Mapping[str, Any]], surface_id: str) -> None:
    """Run local and official SDK validation over the complete ordered stream."""
    creations = [message.get("createSurface") for message in messages if "createSurface" in message]
    if len(creations) != 1 or not isinstance(creations[0], Mapping):
        raise A2UIValidationError("A2UI sequence must declare exactly one surface catalog")
    catalog_id = creations[0].get("catalogId")
    if not isinstance(catalog_id, str) or catalog_id not in _SDK_VALIDATORS:
        raise A2UIValidationError("A2UI sequence uses an unsupported catalog")
    validated = validate_messages(messages, expected_surface_id=surface_id, catalog_id=catalog_id)
    if catalog_id in {A2UI_FINANCE_CATALOG, A2UI_FINANCE_V2_CATALOG}:
        chart_components = [
            component
            for message in validated
            for component in message.get("updateComponents", {}).get("components", [])
            if component.get("component") == "Chart"
        ]
        data_model: Any = None
        for message in validated:
            update = message.get("updateDataModel")
            if isinstance(update, Mapping):
                data_model = _apply_data_update(data_model, update)
        for component in chart_components:
            chart = component.get("chart")
            candidate = (
                _resolve_data_path(data_model, chart["path"])
                if isinstance(chart, Mapping) and isinstance(chart.get("path"), str)
                else chart
            )
            try:
                _CHART_VALUE_ADAPTER.validate_python(candidate, strict=True)
            except Exception as exc:
                raise A2UIValidationError("Finance chart data model is invalid") from exc
    if catalog_id == A2UI_FINANCE_V2_CATALOG:
        finance_components = [
            component
            for message in validated
            for component in message.get("updateComponents", {}).get("components", [])
        ]
        banking_components = [
            component
            for component in finance_components
            if component.get("component") == "BankingView"
        ]
        if not banking_components:
            raise A2UIValidationError("Finance v2 must contain a BankingView")
        banking_data_model: Any = None
        for message in validated:
            update = message.get("updateDataModel")
            if isinstance(update, Mapping):
                banking_data_model = _apply_data_update(banking_data_model, update)
        for component in banking_components:
            view = component.get("view")
            candidate = (
                _resolve_data_path(banking_data_model, view["path"])
                if isinstance(view, Mapping) and isinstance(view.get("path"), str)
                else view
            )
            if candidate is _MISSING:
                raise A2UIValidationError("Finance BankingView data binding could not be resolved")
            try:
                validate_banking_view(candidate)
            except Exception as exc:
                raise A2UIValidationError("Finance BankingView data model is invalid") from exc
        for component in finance_components:
            if component.get("component") != "Button":
                continue
            event = component.get("action", {}).get("event", {})
            context = event.get("context") if isinstance(event, Mapping) else None
            intent_value = context.get("intent") if isinstance(context, Mapping) else None
            resolved_intent = (
                _resolve_data_path(banking_data_model, intent_value["path"])
                if isinstance(intent_value, Mapping) and isinstance(intent_value.get("path"), str)
                else intent_value
            )
            if (
                not isinstance(event, Mapping)
                or event.get("name") != "request_financial_view"
                or not isinstance(context, Mapping)
                or set(context) != {"intent"}
                or resolved_intent not in FINANCIAL_INTENTS
            ):
                raise A2UIValidationError("Finance v2 contains an unsupported action")
    _sdk_validate(validated, catalog_id)


_MISSING = object()


def _pointer_segments(path: str | None) -> list[str]:
    resolved = path or "/"
    validate_json_pointer(resolved)
    if resolved == "/":
        return []
    return [segment.replace("~1", "/").replace("~0", "~") for segment in resolved[1:].split("/")]


def _array_index(segment: str, length: int, *, allow_end: bool) -> int:
    if not segment.isascii() or not segment.isdigit() or (len(segment) > 1 and segment[0] == "0"):
        raise A2UIValidationError("A2UI data path has an invalid array index")
    index = int(segment)
    if index > length or (index == length and not allow_end):
        raise A2UIValidationError("A2UI data path has an invalid array index")
    return index


def _apply_data_update(current: Any, update: Mapping[str, Any]) -> Any:
    segments = _pointer_segments(
        update.get("path") if isinstance(update.get("path"), str) else None
    )
    has_value = "value" in update
    value = deepcopy(update.get("value"))
    if not segments:
        return value if has_value else None
    root = deepcopy(current) if isinstance(current, dict | list) else {}
    parent: Any = root
    for segment in segments[:-1]:
        if isinstance(parent, list):
            child = parent[_array_index(segment, len(parent), allow_end=False)]
        elif isinstance(parent, dict):
            child = parent.setdefault(segment, {})
        else:
            raise A2UIValidationError("A2UI data path has no container")
        if not isinstance(child, dict | list):
            raise A2UIValidationError("A2UI data path has no container")
        parent = child
    leaf = segments[-1]
    if isinstance(parent, list):
        index = _array_index(leaf, len(parent), allow_end=has_value)
        if has_value and index == len(parent):
            parent.append(value)
        elif has_value:
            parent[index] = value
        else:
            parent.pop(index)
    elif isinstance(parent, dict):
        if has_value:
            parent[leaf] = value
        else:
            parent.pop(leaf, None)
    else:
        raise A2UIValidationError("A2UI data path has no container")
    return root


def _resolve_data_path(model: Any, path: str) -> Any:
    value = model
    for segment in _pointer_segments(path):
        if isinstance(value, list):
            try:
                value = value[_array_index(segment, len(value), allow_end=False)]
            except A2UIValidationError:
                return _MISSING
        elif isinstance(value, Mapping) and segment in value:
            value = value[segment]
        else:
            return _MISSING
    return value


def _sdk_validate(messages: list[dict[str, Any]], catalog_id: str) -> None:
    try:
        _SDK_VALIDATORS[catalog_id].validate(messages)
    except Exception as exc:
        raise A2UIValidationError("A2UI messages fail the official v0.9.1 validator") from exc
