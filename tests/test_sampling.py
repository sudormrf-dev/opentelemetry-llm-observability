"""Tests for sampling.py."""

from __future__ import annotations

from patterns.sampling import SamplingDecision, SamplingRule, SamplingStrategy, TraceSampler


class TestTraceSampler:
    def test_always_on(self):
        s = TraceSampler.always_on()
        assert s.is_sampled("span") is True

    def test_always_off(self):
        s = TraceSampler.always_off()
        assert s.is_sampled("span") is False

    def test_ratio_1_0(self):
        s = TraceSampler.ratio_based(1.0)
        assert s.ratio == 1.0

    def test_ratio_0_0_drops(self):
        s = TraceSampler.ratio_based(0.0)
        assert s.should_sample("x", {}) == SamplingDecision.DROP

    def test_strategy_property(self):
        s = TraceSampler.always_on()
        assert s.strategy == SamplingStrategy.ALWAYS_ON

    def test_ratio_clamped_high(self):
        s = TraceSampler(ratio=2.0)
        assert s.ratio == 1.0

    def test_ratio_clamped_low(self):
        s = TraceSampler(ratio=-1.0)
        assert s.ratio == 0.0

    def test_rule_based_match(self):
        s = TraceSampler(SamplingStrategy.RULE_BASED)
        rule = SamplingRule("gen_ai.system", "openai", SamplingDecision.DROP)
        s.add_rule(rule)
        decision = s.should_sample("chat", {"gen_ai.system": "openai"})
        assert decision == SamplingDecision.DROP

    def test_rule_based_no_match_samples(self):
        s = TraceSampler(SamplingStrategy.RULE_BASED)
        rule = SamplingRule("gen_ai.system", "openai", SamplingDecision.DROP)
        s.add_rule(rule)
        decision = s.should_sample("chat", {"gen_ai.system": "anthropic"})
        assert decision == SamplingDecision.RECORD_AND_SAMPLE

    def test_rule_priority_ordering(self):
        s = TraceSampler(SamplingStrategy.RULE_BASED)
        low = SamplingRule("key", "val", SamplingDecision.DROP, priority=0)
        high = SamplingRule("key", "val", SamplingDecision.RECORD_AND_SAMPLE, priority=10)
        s.add_rule(low)
        s.add_rule(high)
        decision = s.should_sample("x", {"key": "val"})
        assert decision == SamplingDecision.RECORD_AND_SAMPLE

    def test_rule_count(self):
        s = TraceSampler(SamplingStrategy.RULE_BASED)
        s.add_rule(SamplingRule("a", "b", SamplingDecision.DROP))
        assert s.rule_count() == 1

    def test_add_rule_returns_self(self):
        s = TraceSampler(SamplingStrategy.RULE_BASED)
        r = SamplingRule("a", "b", SamplingDecision.DROP)
        assert s.add_rule(r) is s

    def test_sampling_rule_matches(self):
        rule = SamplingRule("model", "gpt-4", SamplingDecision.RECORD_AND_SAMPLE)
        assert rule.matches({"model": "gpt-4"}) is True
        assert rule.matches({"model": "other"}) is False
