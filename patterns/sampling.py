"""OTel sampling patterns for LLM traces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SamplingDecision(str, Enum):
    DROP = "DROP"
    RECORD_ONLY = "RECORD_ONLY"
    RECORD_AND_SAMPLE = "RECORD_AND_SAMPLE"


class SamplingStrategy(str, Enum):
    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"
    TRACE_ID_RATIO = "trace_id_ratio"
    PARENT_BASED = "parent_based"
    RULE_BASED = "rule_based"


@dataclass
class SamplingRule:
    """A single sampling rule."""

    attribute_key: str
    attribute_value: str
    decision: SamplingDecision
    priority: int = 0

    def matches(self, attributes: dict[str, Any]) -> bool:
        return str(attributes.get(self.attribute_key, "")) == self.attribute_value


class TraceSampler:
    """Rule-based trace sampler for LLM spans."""

    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.TRACE_ID_RATIO,
        ratio: float = 1.0,
    ) -> None:
        self._strategy = strategy
        self._ratio = max(0.0, min(1.0, ratio))
        self._rules: list[SamplingRule] = []

    @property
    def strategy(self) -> SamplingStrategy:
        return self._strategy

    @property
    def ratio(self) -> float:
        return self._ratio

    def add_rule(self, rule: SamplingRule) -> TraceSampler:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        return self

    def should_sample(self, span_name: str, attributes: dict[str, Any]) -> SamplingDecision:
        if self._strategy == SamplingStrategy.ALWAYS_ON:
            return SamplingDecision.RECORD_AND_SAMPLE
        if self._strategy == SamplingStrategy.ALWAYS_OFF:
            return SamplingDecision.DROP
        if self._strategy == SamplingStrategy.RULE_BASED:
            for rule in self._rules:
                if rule.matches(attributes):
                    return rule.decision
            return SamplingDecision.RECORD_AND_SAMPLE
        if self._strategy == SamplingStrategy.TRACE_ID_RATIO:
            # Simple deterministic sampling using span_name hash
            h = hash(span_name) % 1000
            threshold = int(self._ratio * 1000)
            if h < threshold:
                return SamplingDecision.RECORD_AND_SAMPLE
            return SamplingDecision.DROP
        return SamplingDecision.RECORD_AND_SAMPLE

    def is_sampled(self, span_name: str, attributes: dict[str, Any] | None = None) -> bool:
        decision = self.should_sample(span_name, attributes or {})
        return decision == SamplingDecision.RECORD_AND_SAMPLE

    def rule_count(self) -> int:
        return len(self._rules)

    @classmethod
    def always_on(cls) -> TraceSampler:
        return cls(SamplingStrategy.ALWAYS_ON)

    @classmethod
    def always_off(cls) -> TraceSampler:
        return cls(SamplingStrategy.ALWAYS_OFF)

    @classmethod
    def ratio_based(cls, ratio: float) -> TraceSampler:
        return cls(SamplingStrategy.TRACE_ID_RATIO, ratio)
