"""Tests for spans.py."""

from __future__ import annotations

import time

from patterns.spans import (
    LLMSpan,
    LLMSpanBuilder,
    SpanContext,
    SpanEvent,
    SpanStatus,
    SpanStatusCode,
)


class TestSpanStatus:
    def test_ok(self):
        s = SpanStatus.ok()
        assert s.code == SpanStatusCode.OK

    def test_error(self):
        s = SpanStatus.error("boom")
        assert s.is_error() is True
        assert s.description == "boom"

    def test_unset_not_error(self):
        assert SpanStatus().is_error() is False


class TestSpanContext:
    def test_is_sampled(self):
        ctx = SpanContext("a" * 32, "b" * 16, trace_flags=1)
        assert ctx.is_sampled() is True

    def test_not_sampled(self):
        ctx = SpanContext("a" * 32, "b" * 16, trace_flags=0)
        assert ctx.is_sampled() is False

    def test_traceparent_format(self):
        ctx = SpanContext("a" * 32, "b" * 16)
        tp = ctx.to_traceparent()
        assert tp.startswith("00-")
        assert len(tp.split("-")) == 4


class TestLLMSpan:
    def setup_method(self):
        self.span = LLMSpan(name="chat", trace_id="t1", span_id="s1")

    def test_set_attribute(self):
        self.span.set_attribute("key", "val")
        assert self.span.attributes["key"] == "val"

    def test_add_event(self):
        self.span.add_event(SpanEvent("prompt_sent"))
        assert len(self.span.events) == 1

    def test_finish(self):
        self.span.finish()
        assert self.span.is_finished() is True

    def test_finish_with_status(self):
        self.span.finish(SpanStatus.ok())
        assert self.span.status.code == SpanStatusCode.OK

    def test_duration_ms_none_before_finish(self):
        assert self.span.duration_ms() is None

    def test_duration_ms_after_finish(self):
        time.sleep(0.01)
        self.span.finish()
        assert self.span.duration_ms() is not None
        assert self.span.duration_ms() >= 0  # type: ignore[operator]

    def test_input_tokens_none(self):
        assert self.span.input_tokens() is None

    def test_input_tokens_set(self):
        self.span.set_attribute("gen_ai.usage.input_tokens", 100)
        assert self.span.input_tokens() == 100

    def test_output_tokens_set(self):
        self.span.set_attribute("gen_ai.usage.output_tokens", 50)
        assert self.span.output_tokens() == 50

    def test_to_dict(self):
        d = self.span.to_dict()
        assert d["name"] == "chat"
        assert "attributes" in d


class TestLLMSpanBuilder:
    def test_build_returns_span(self):
        span = LLMSpanBuilder("test").build()
        assert span.name == "test"

    def test_with_model(self):
        span = LLMSpanBuilder("test").with_model("gpt-4o").build()
        assert span.attributes["gen_ai.request.model"] == "gpt-4o"

    def test_with_provider(self):
        span = LLMSpanBuilder("test").with_provider("openai").build()
        assert span.attributes["gen_ai.system"] == "openai"

    def test_with_tokens(self):
        span = LLMSpanBuilder("test").with_tokens(100, 50).build()
        assert span.input_tokens() == 100
        assert span.output_tokens() == 50
        assert span.attributes["gen_ai.usage.total_tokens"] == 150

    def test_with_parent(self):
        span = LLMSpanBuilder("test").with_parent("parent_id").build()
        assert span.parent_span_id == "parent_id"

    def test_with_attribute(self):
        span = LLMSpanBuilder("test").with_attribute("custom", "val").build()
        assert span.attributes["custom"] == "val"

    def test_chaining(self):
        b = LLMSpanBuilder("x")
        assert b.with_model("m") is b
