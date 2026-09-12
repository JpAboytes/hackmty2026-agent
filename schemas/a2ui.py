"""Pydantic models for the A2UI v1 wire protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

A2UI_VERSION = "a2ui/v1"
AccessibilityFontScale = Literal["sm", "md", "lg", "xl"]
AccessibilityContrast = Literal["normal", "high"]
AccessibilityHitTarget = Literal["normal", "large"]
ComponentType = Literal["Banner", "Button", "InteractiveSlider", "MetricCard"]
Layout = Literal["vertical_stack"]
ActionType = Literal["A2UI_DISPATCH"]
ActionIntent = Literal[
    "REQUEST_CREDIT",
    "MANAGE_SUBSCRIPTIONS",
    "CONFIRM_SIMULATION",
    "VIEW_DETAILS",
]
TemplateId = Literal["Template_Crisis_Flujo", "Template_Subscriptions", "Template_Projection"]


class A2UIAction(BaseModel):
    """A validated action event emitted by an interactive component."""

    model_config = ConfigDict(extra="forbid")

    type: ActionType
    intent: ActionIntent
    payload: dict[str, object] = Field(default_factory=dict)


class A2UIComponent(BaseModel):
    """A registered, non-executable component description."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    type: ComponentType
    tags: list[str] = Field(default_factory=list)
    props: dict[str, object] = Field(default_factory=dict)


class A2UISurface(BaseModel):
    """The layout and component tree rendered by the client."""

    model_config = ConfigDict(extra="forbid")

    layout: Layout
    components: list[A2UIComponent] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_component_ids(self) -> A2UISurface:
        ids = [component.id for component in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("component ids must be unique within a surface")
        return self


class A2UIAccessibility(BaseModel):
    """Accessibility settings applied consistently by the renderer."""

    model_config = ConfigDict(extra="forbid")

    font_scale: AccessibilityFontScale
    contrast: AccessibilityContrast
    hit_target: AccessibilityHitTarget


class A2UIMeta(BaseModel):
    """Narrative and accessibility context for a generated surface."""

    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(min_length=1)
    accessibility: A2UIAccessibility


class A2UIPayload(BaseModel):
    """Top-level A2UI v1 response."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["a2ui/v1"] = A2UI_VERSION
    template_id: TemplateId
    applied_tags: list[str] = Field(default_factory=list)
    meta: A2UIMeta
    surface: A2UISurface
