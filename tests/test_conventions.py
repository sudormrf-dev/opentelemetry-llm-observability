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
    def test_openai_value(self) -> None:
        assert LLMProvider.OPENAI.value == "openai"

    def test_anthropic_value(self) -> None:
        assert LLMProvider.ANTHROPIC.value == "anthropic"

    def test_all_have_values(self) -> None:
        assert all(p.value for p in LLMProvider)


class TestLLMOperation:
    def test_chat(self) -> None:
        assert LLMOperation.CHAT.value == "chat"

    def test_embedding(self) -> None:
        assert LLMOperation.EMBEDDING.value == "embeddings"


class TestLLMFramework:
    def test_langchain(self) -> None:
        assert LLMFramework.LANGCHAIN.value == "langchain"

    def test_raw(self) -> None:
        assert LLMFramework.RAW.value == "raw"


class TestLLMErrorType:
    def test_rate_limit(self) -> None:
        assert LLMErrorType.RATE_LIMIT.value == "rate_limit_error"

    def test_context_length(self) -> None:
        assert LLMErrorType.CONTEXT_LENGTH.value == "context_length_exceeded"


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
