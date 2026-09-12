"""Strict A2UI v0.9.1 validation for the Expo transport boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isfinite
from typing import Annotated, Any, Literal

from a2ui.basic_catalog.provider import BasicCatalog  # type: ignore[import-untyped]
from a2ui.inference_formats.direct_json import (  # type: ignore[import-untyped]
    DirectJsonFormat,
)
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
)

A2UI_VERSION = "v0.9.1"
A2UI_SDK_VERSION = "0.9.1"
A2UI_MIME_TYPE = "application/a2ui+json"
A2UI_BASIC_CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"

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


Component = _TextComponent | _ButtonComponent | _CardComponent | _ColumnComponent


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
    catalog_id: Literal["https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"] = (
        Field(alias="catalogId")
    )
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
_SDK_VALIDATOR = (
    DirectJsonFormat(
        version=A2UI_SDK_VERSION,
        catalogs=[BasicCatalog.get_config(A2UI_SDK_VERSION)],
        accepts_inline_catalogs=False,
    )
    .get_selected_catalog()
    .validator
)
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
    messages: Sequence[Mapping[str, Any]], *, expected_surface_id: str | None = None
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
    validated = validate_messages(messages, expected_surface_id=surface_id)
    components = validated[1]["updateComponents"]["components"]
    if not any(component.get("id") == "root" for component in components):
        raise A2UIValidationError("A2UI templates must declare a root component")
    _sdk_validate(validated)
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
    validated = validate_messages(messages, expected_surface_id=surface_id)
    _sdk_validate(validated)


def _sdk_validate(messages: list[dict[str, Any]]) -> None:
    try:
        _SDK_VALIDATOR.validate(messages)
    except Exception as exc:
        raise A2UIValidationError("A2UI messages fail the official v0.9.1 validator") from exc
