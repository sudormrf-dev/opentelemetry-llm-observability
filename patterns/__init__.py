"""OpenTelemetry patterns for LLM observability."""

from .conventions import (
    LLMAttributes,
    LLMErrorType,
    LLMFramework,
    LLMOperation,
    LLMProvider,
    SpanKind,
)
from .metrics import (
    Counter,
    Gauge,
    Histogram,
    LLMMetrics,
    MetricPoint,
    MetricRegistry,
)
from .sampling import (
    SamplingDecision,
    SamplingRule,
    SamplingStrategy,
    TraceSampler,
)
from .spans import (
    LLMSpan,
    LLMSpanBuilder,
    SpanContext,
    SpanEvent,
    SpanStatus,
    SpanStatusCode,
)

__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "LLMAttributes",
    "LLMErrorType",
    "LLMFramework",
    "LLMMetrics",
    "LLMOperation",
    "LLMProvider",
    "LLMSpan",
    "LLMSpanBuilder",
    "MetricPoint",
    "MetricRegistry",
    "SamplingDecision",
    "SamplingRule",
    "SamplingStrategy",
    "SpanContext",
    "SpanEvent",
    "SpanKind",
    "SpanStatus",
    "SpanStatusCode",
    "TraceSampler",
]
