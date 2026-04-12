"""OTel metrics patterns for LLM observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class MetricKind(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricPoint:
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    kind: MetricKind = MetricKind.COUNTER

    def with_label(self, key: str, value: str) -> MetricPoint:
        return MetricPoint(
            name=self.name,
            value=self.value,
            labels={**self.labels, key: value},
            timestamp=self.timestamp,
            kind=self.kind,
        )


class Counter:
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._points: list[MetricPoint] = []

    def add(self, value: float, labels: dict[str, str] | None = None) -> None:
        self._value += value
        self._points.append(MetricPoint(self.name, value, labels or {}, kind=MetricKind.COUNTER))

    def total(self) -> float:
        return self._value

    def points(self) -> list[MetricPoint]:
        return list(self._points)


class Gauge:
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._points: list[MetricPoint] = []

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        self._value = value
        self._points.append(MetricPoint(self.name, value, labels or {}, kind=MetricKind.GAUGE))

    def current(self) -> float:
        return self._value

    def points(self) -> list[MetricPoint]:
        return list(self._points)


class Histogram:
    def __init__(
        self, name: str, description: str = "", buckets: list[float] | None = None
    ) -> None:
        self.name = name
        self.description = description
        self._buckets = buckets or [1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000]
        self._observations: list[float] = []
        self._points: list[MetricPoint] = []

    def record(self, value: float, labels: dict[str, str] | None = None) -> None:
        self._observations.append(value)
        self._points.append(MetricPoint(self.name, value, labels or {}, kind=MetricKind.HISTOGRAM))

    def count(self) -> int:
        return len(self._observations)

    def sum(self) -> float:
        return sum(self._observations)

    def mean(self) -> float:
        if not self._observations:
            return 0.0
        return self.sum() / self.count()

    def percentile(self, p: float) -> float:
        if not self._observations:
            return 0.0
        sorted_obs = sorted(self._observations)
        idx = int(len(sorted_obs) * p / 100)
        return sorted_obs[min(idx, len(sorted_obs) - 1)]

    def bucket_counts(self) -> dict[float, int]:
        counts: dict[float, int] = {}
        for b in self._buckets:
            counts[b] = sum(1 for v in self._observations if v <= b)
        return counts

    def points(self) -> list[MetricPoint]:
        return list(self._points)


class LLMMetrics:
    """Standard LLM observability metrics collection."""

    def __init__(self) -> None:
        self.token_usage = Counter("gen_ai.client.token.usage", "Token usage by operation")
        self.request_duration = Histogram(
            "gen_ai.client.operation.duration",
            "LLM request duration in ms",
            buckets=[100, 500, 1000, 2000, 5000, 10000, 30000],
        )
        self.error_count = Counter("gen_ai.client.error.count", "LLM error count")
        self.active_requests = Gauge("gen_ai.client.active_requests", "Active LLM requests")
        self.cost_estimate = Counter("gen_ai.client.cost.estimate", "Estimated cost in USD")

    def record_request(
        self,
        duration_ms: float,
        input_tokens: int,
        output_tokens: int,
        model: str,
        provider: str,
        error: str | None = None,
    ) -> None:
        labels = {"model": model, "provider": provider}
        self.request_duration.record(duration_ms, labels)
        self.token_usage.add(input_tokens, {**labels, "token_type": "input"})
        self.token_usage.add(output_tokens, {**labels, "token_type": "output"})
        if error:
            self.error_count.add(1, {**labels, "error_type": error})

    def summary(self) -> dict[str, Any]:
        return {
            "total_tokens": self.token_usage.total(),
            "request_count": self.request_duration.count(),
            "mean_duration_ms": self.request_duration.mean(),
            "p95_duration_ms": self.request_duration.percentile(95),
            "error_count": self.error_count.total(),
        }


class MetricRegistry:
    """Registry of named metrics."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}

    def counter(self, name: str, description: str = "") -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name, description)
        return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description)
        return self._gauges[name]

    def histogram(self, name: str, description: str = "") -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = Histogram(name, description)
        return self._histograms[name]

    def all_names(self) -> list[str]:
        return sorted(list(self._counters) + list(self._gauges) + list(self._histograms))
