"""OTel span patterns for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class SpanStatusCode(str, Enum):
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


@dataclass
class SpanStatus:
    code: SpanStatusCode = SpanStatusCode.UNSET
    description: str = ""

    def is_error(self) -> bool:
        return self.code == SpanStatusCode.ERROR

    @classmethod
    def ok(cls) -> SpanStatus:
        return cls(SpanStatusCode.OK)

    @classmethod
    def error(cls, description: str) -> SpanStatus:
        return cls(SpanStatusCode.ERROR, description)


@dataclass
class SpanContext:
    trace_id: str
    span_id: str
    trace_flags: int = 1
    is_remote: bool = False

    def is_sampled(self) -> bool:
        return bool(self.trace_flags & 1)

    def to_traceparent(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"


@dataclass
class SpanEvent:
    name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMSpan:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    status: SpanStatus = field(default_factory=SpanStatus)
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, event: SpanEvent) -> None:
        self.events.append(event)

    def finish(self, status: SpanStatus | None = None) -> None:
        self.end_time = datetime.now(UTC)
        if status is not None:
            self.status = status

    def is_finished(self) -> bool:
        return self.end_time is not None

    def input_tokens(self) -> int | None:
        v = self.attributes.get("gen_ai.usage.input_tokens")
        return int(v) if v is not None else None

    def output_tokens(self) -> int | None:
        v = self.attributes.get("gen_ai.usage.output_tokens")
        return int(v) if v is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "status": self.status.code.value,
            "attributes": dict(self.attributes),
            "events": [{"name": e.name, "attributes": e.attributes} for e in self.events],
            "duration_ms": self.duration_ms(),
        }


class LLMSpanBuilder:
    """Fluent builder for LLM spans."""

    _counter: int = 0

    def __init__(self, name: str) -> None:
        LLMSpanBuilder._counter += 1
        self._span = LLMSpan(
            name=name,
            trace_id=f"trace{LLMSpanBuilder._counter:032x}",
            span_id=f"span{LLMSpanBuilder._counter:016x}",
        )

    def with_parent(self, parent_span_id: str) -> LLMSpanBuilder:
        self._span.parent_span_id = parent_span_id
        return self

    def with_attribute(self, key: str, value: Any) -> LLMSpanBuilder:
        self._span.set_attribute(key, value)
        return self

    def with_model(self, model: str) -> LLMSpanBuilder:
        self._span.set_attribute("gen_ai.request.model", model)
        return self

    def with_provider(self, provider: str) -> LLMSpanBuilder:
        self._span.set_attribute("gen_ai.system", provider)
        return self

    def with_tokens(self, input_tokens: int, output_tokens: int) -> LLMSpanBuilder:
        self._span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        self._span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        self._span.set_attribute("gen_ai.usage.total_tokens", input_tokens + output_tokens)
        return self

    def build(self) -> LLMSpan:
        return self._span
