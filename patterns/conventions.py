"""OpenTelemetry semantic conventions for LLM observability.

Based on OpenTelemetry GenAI semantic conventions (draft).
Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/

Patterns:
  - LLMProvider: provider identifiers (openai, anthropic, etc.)
  - LLMOperation: operation types (chat, completion, embedding, etc.)
  - LLMAttributes: attribute name constants
  - SpanKind: OTel span kind enum
"""

from __future__ import annotations

from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    BEDROCK = "aws.bedrock"
    VERTEX = "gcp.vertex_ai"
    AZURE_OPENAI = "azure.openai"


class LLMOperation(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embeddings"
    IMAGE_GENERATION = "image_generation"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    FINE_TUNING = "fine_tuning"
    RERANK = "rerank"


class LLMFramework(str, Enum):
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"
    HAYSTACK = "haystack"
    AUTOGEN = "autogen"
    CREWAI = "crewai"
    INSTRUCTOR = "instructor"
    DSPY = "dspy"
    RAW = "raw"


class LLMErrorType(str, Enum):
    RATE_LIMIT = "rate_limit_error"
    CONTEXT_LENGTH = "context_length_exceeded"
    INVALID_REQUEST = "invalid_request_error"
    AUTHENTICATION = "authentication_error"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    CONTENT_FILTER = "content_filter"
    QUOTA_EXCEEDED = "quota_exceeded"


class SpanKind(str, Enum):
    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


class LLMAttributes:
    """OpenTelemetry attribute name constants for LLM spans.

    Usage::

        span.set_attribute(LLMAttributes.SYSTEM, "openai")
        span.set_attribute(LLMAttributes.REQUEST_MODEL, "gpt-4o")
    """

    # GenAI root namespace
    SYSTEM = "gen_ai.system"
    OPERATION = "gen_ai.operation.name"

    # Request attributes
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_TOP_K = "gen_ai.request.top_k"
    REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"

    # Response attributes
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    RESPONSE_ID = "gen_ai.response.id"

    # Token usage
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

    # RAG / retrieval
    RETRIEVAL_QUERY = "gen_ai.retrieval.query"
    RETRIEVAL_DOCUMENTS = "gen_ai.retrieval.documents"
    RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"

    # Agent / tool
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    AGENT_NAME = "gen_ai.agent.name"

    # Error
    ERROR_TYPE = "error.type"

    # Infrastructure
    SERVER_ADDRESS = "server.address"
    SERVER_PORT = "server.port"

    @classmethod
    def all_keys(cls) -> list[str]:
        """Return all attribute key constants."""
        return [
            v
            for k, v in vars(cls).items()
            if not k.startswith("_") and isinstance(v, str) and k != k.lower()
        ]

    @classmethod
    def token_attrs(cls) -> list[str]:
        """Return token usage attribute keys."""
        return [cls.USAGE_INPUT_TOKENS, cls.USAGE_OUTPUT_TOKENS, cls.USAGE_TOTAL_TOKENS]
