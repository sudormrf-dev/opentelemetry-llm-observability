"""Tests for conventions.py."""

from __future__ import annotations

from patterns.conventions import (
    LLMAttributes,
    LLMErrorType,
    LLMFramework,
    LLMOperation,
    LLMProvider,
    SpanKind,
)


class TestLLMProvider:
    def test_openai_value(self):
        assert LLMProvider.OPENAI == "openai"

    def test_anthropic_value(self):
        assert LLMProvider.ANTHROPIC == "anthropic"

    def test_all_have_values(self):
        assert all(p.value for p in LLMProvider)


class TestLLMOperation:
    def test_chat(self):
        assert LLMOperation.CHAT == "chat"

    def test_embedding(self):
        assert LLMOperation.EMBEDDING == "embeddings"


class TestLLMFramework:
    def test_langchain(self):
        assert LLMFramework.LANGCHAIN == "langchain"

    def test_raw(self):
        assert LLMFramework.RAW == "raw"


class TestLLMErrorType:
    def test_rate_limit(self):
        assert LLMErrorType.RATE_LIMIT == "rate_limit_error"

    def test_context_length(self):
        assert LLMErrorType.CONTEXT_LENGTH == "context_length_exceeded"


class TestSpanKind:
    def test_client(self):
        assert SpanKind.CLIENT == "CLIENT"

    def test_internal(self):
        assert SpanKind.INTERNAL == "INTERNAL"


class TestLLMAttributes:
    def test_system_key(self):
        assert LLMAttributes.SYSTEM == "gen_ai.system"

    def test_request_model_key(self):
        assert LLMAttributes.REQUEST_MODEL == "gen_ai.request.model"

    def test_usage_input_tokens(self):
        assert "input_tokens" in LLMAttributes.USAGE_INPUT_TOKENS

    def test_all_keys_non_empty(self):
        keys = LLMAttributes.all_keys()
        assert len(keys) > 0
        assert all(isinstance(k, str) for k in keys)

    def test_token_attrs(self):
        token_keys = LLMAttributes.token_attrs()
        assert len(token_keys) == 3
        assert LLMAttributes.USAGE_INPUT_TOKENS in token_keys

    def test_tool_name(self):
        assert "tool" in LLMAttributes.TOOL_NAME

    def test_agent_name(self):
        assert "agent" in LLMAttributes.AGENT_NAME
