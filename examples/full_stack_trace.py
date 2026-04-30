"""End-to-end LLM pipeline trace simulation with OpenTelemetry patterns.

Demonstrates a complete request lifecycle:
  1. API request span  (FastAPI gateway)
  2. RAG retrieval span (vector DB lookup)
  3. LLM call span     (with token counts and cost)
  4. Tool call span    (external API invocation)
  5. Response span     (serialization + delivery)

All spans share a single trace_id and form a parent-child tree.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from patterns.conventions import LLMAttributes, LLMOperation, LLMProvider
from patterns.spans import LLMSpan, LLMSpanBuilder, SpanStatus

# ---------------------------------------------------------------------------
# Cost table (USD per 1 000 tokens)
# ---------------------------------------------------------------------------

MODEL_COST: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku": {"input": 0.0008, "output": 0.004},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
}


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a model call."""
    rates = MODEL_COST.get(model, {"input": 0.002, "output": 0.008})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000


def _sleep_ms(ms: float) -> None:
    """Busy-sleep for *ms* milliseconds (simulation only)."""
    time.sleep(ms / 1_000)


# ---------------------------------------------------------------------------
# Trace context helpers
# ---------------------------------------------------------------------------

def new_trace_id() -> str:
    """Generate a 32-hex-char W3C trace-id."""
    return uuid.uuid4().hex + uuid.uuid4().hex[:0]  # 32 chars


def new_span_id() -> str:
    """Generate a 16-hex-char W3C span-id."""
    return uuid.uuid4().hex[:16]


@dataclass
class TraceContext:
    """Propagates trace_id across the pipeline."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex + uuid.uuid4().hex[:0])
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def child_span_id(self) -> str:
        return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Span factories for each pipeline stage
# ---------------------------------------------------------------------------

def build_api_request_span(ctx: TraceContext) -> LLMSpan:
    """Simulate FastAPI ingress span (~5-15 ms)."""
    span_id = ctx.child_span_id()
    span = (
        LLMSpanBuilder("api.request")
        .with_attribute("http.method", "POST")
        .with_attribute("http.route", "/v1/chat/completions")
        .with_attribute("http.status_code", 200)
        .with_attribute("http.scheme", "https")
        .with_attribute("server.address", "api.example.com")
        .with_attribute("server.port", 443)
        .with_attribute("user_agent.original", "openai-python/1.30.0")
        .with_attribute("request.id", ctx.request_id)
        .with_attribute("span.kind", "SERVER")
        .build()
    )
    span.trace_id = ctx.trace_id
    span.span_id = span_id
    duration = random.uniform(5, 15)
    _sleep_ms(duration)
    span.finish(SpanStatus.ok())
    return span


def build_rag_retrieval_span(ctx: TraceContext, parent_span_id: str) -> LLMSpan:
    """Simulate vector DB retrieval span (~20-80 ms)."""
    query = random.choice([
        "What is the refund policy for enterprise subscriptions?",
        "How to configure SSO with Okta?",
        "API rate limits for the Pro tier",
        "Webhook event payload schema",
    ])
    top_k = random.randint(3, 8)
    docs_found = random.randint(top_k - 1, top_k)

    span_id = ctx.child_span_id()
    span = (
        LLMSpanBuilder("rag.retrieval")
        .with_parent(parent_span_id)
        .with_attribute(LLMAttributes.RETRIEVAL_QUERY, query)
        .with_attribute(LLMAttributes.RETRIEVAL_TOP_K, top_k)
        .with_attribute(LLMAttributes.RETRIEVAL_DOCUMENTS, docs_found)
        .with_attribute("db.system", "pgvector")
        .with_attribute("db.collection", "knowledge_base")
        .with_attribute("vector.dimensions", 1536)
        .with_attribute("vector.similarity_metric", "cosine")
        .with_attribute("retrieval.latency_ms", round(random.uniform(20, 80), 2))
        .with_attribute("span.kind", "CLIENT")
        .build()
    )
    span.trace_id = ctx.trace_id
    span.span_id = span_id

    duration = random.uniform(20, 80)
    _sleep_ms(duration)
    span.finish(SpanStatus.ok())
    return span


def build_llm_call_span(ctx: TraceContext, parent_span_id: str, model: str, provider: str) -> LLMSpan:
    """Simulate LLM inference span with token counts and cost (~300-2000 ms)."""
    input_tokens = random.randint(500, 3000)
    output_tokens = random.randint(50, 500)
    cost = _compute_cost(model, input_tokens, output_tokens)
    finish_reason = random.choice(["stop", "stop", "stop", "length"])

    span_id = ctx.child_span_id()
    span = (
        LLMSpanBuilder("llm.call")
        .with_parent(parent_span_id)
        .with_provider(provider)
        .with_model(model)
        .with_tokens(input_tokens, output_tokens)
        .with_attribute(LLMAttributes.OPERATION, LLMOperation.CHAT.value)
        .with_attribute(LLMAttributes.RESPONSE_MODEL, model)
        .with_attribute(LLMAttributes.RESPONSE_ID, f"chatcmpl-{uuid.uuid4().hex[:24]}")
        .with_attribute(LLMAttributes.RESPONSE_FINISH_REASONS, finish_reason)
        .with_attribute(LLMAttributes.REQUEST_MAX_TOKENS, 1024)
        .with_attribute(LLMAttributes.REQUEST_TEMPERATURE, round(random.uniform(0.0, 1.0), 2))
        .with_attribute("gen_ai.cost.usd", round(cost, 6))
        .with_attribute("gen_ai.cost.input_usd", round(input_tokens * MODEL_COST.get(model, {"input": 0.002})["input"] / 1_000, 6))
        .with_attribute("gen_ai.cost.output_usd", round(output_tokens * MODEL_COST.get(model, {"output": 0.008})["output"] / 1_000, 6))
        .with_attribute("span.kind", "CLIENT")
        .build()
    )
    span.trace_id = ctx.trace_id
    span.span_id = span_id

    # TTFT + streaming tokens
    ttft_ms = random.uniform(200, 600)
    tokens_per_sec = random.uniform(30, 80)
    stream_ms = (output_tokens / tokens_per_sec) * 1_000
    _sleep_ms(ttft_ms + stream_ms)
    span.finish(SpanStatus.ok())
    return span


def build_tool_call_span(ctx: TraceContext, parent_span_id: str) -> LLMSpan:
    """Simulate an external tool call span (~50-200 ms)."""
    tool_name = random.choice(["get_weather", "search_docs", "run_query", "send_email"])
    call_id = f"call_{uuid.uuid4().hex[:8]}"

    span_id = ctx.child_span_id()
    span = (
        LLMSpanBuilder("tool.call")
        .with_parent(parent_span_id)
        .with_attribute(LLMAttributes.TOOL_NAME, tool_name)
        .with_attribute(LLMAttributes.TOOL_CALL_ID, call_id)
        .with_attribute("tool.input_bytes", random.randint(64, 512))
        .with_attribute("tool.output_bytes", random.randint(128, 4096))
        .with_attribute("tool.success", True)
        .with_attribute("http.method", "GET" if "get" in tool_name or "search" in tool_name else "POST")
        .with_attribute("span.kind", "CLIENT")
        .build()
    )
    span.trace_id = ctx.trace_id
    span.span_id = span_id

    duration = random.uniform(50, 200)
    _sleep_ms(duration)
    span.finish(SpanStatus.ok())
    return span


def build_response_span(ctx: TraceContext, parent_span_id: str, output_tokens: int) -> LLMSpan:
    """Simulate response serialization + delivery span (~2-10 ms)."""
    span_id = ctx.child_span_id()
    span = (
        LLMSpanBuilder("api.response")
        .with_parent(parent_span_id)
        .with_attribute("http.status_code", 200)
        .with_attribute("response.format", "application/json")
        .with_attribute("response.output_tokens", output_tokens)
        .with_attribute("response.bytes", output_tokens * 4)  # rough estimate
        .with_attribute("response.streaming", True)
        .with_attribute("span.kind", "SERVER")
        .build()
    )
    span.trace_id = ctx.trace_id
    span.span_id = span_id

    duration = random.uniform(2, 10)
    _sleep_ms(duration)
    span.finish(SpanStatus.ok())
    return span


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------

@dataclass
class PipelineTrace:
    """Complete trace for one LLM pipeline request."""

    context: TraceContext
    spans: list[LLMSpan] = field(default_factory=list)

    def total_duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time for s in self.spans if s.end_time)
        return (end - start).total_seconds() * 1_000

    def root_span(self) -> LLMSpan | None:
        for s in self.spans:
            if s.parent_span_id is None:
                return s
        return None

    def token_totals(self) -> dict[str, int]:
        input_t = output_t = 0
        for span in self.spans:
            input_t += span.attributes.get("gen_ai.usage.input_tokens", 0)
            output_t += span.attributes.get("gen_ai.usage.output_tokens", 0)
        return {"input": input_t, "output": output_t, "total": input_t + output_t}

    def estimated_cost(self) -> float:
        return sum(s.attributes.get("gen_ai.cost.usd", 0.0) for s in self.spans)


def run_pipeline(model: str = "gpt-4o", provider: str = LLMProvider.OPENAI.value) -> PipelineTrace:
    """Execute a simulated full-stack LLM pipeline and return its trace."""
    ctx = TraceContext()
    trace = PipelineTrace(context=ctx)

    # 1. API ingress
    api_span = build_api_request_span(ctx)
    trace.spans.append(api_span)

    # 2. RAG retrieval (child of API span)
    rag_span = build_rag_retrieval_span(ctx, api_span.span_id)
    trace.spans.append(rag_span)

    # 3. LLM call (child of API span)
    llm_span = build_llm_call_span(ctx, api_span.span_id, model, provider)
    trace.spans.append(llm_span)

    # 4. Tool call (child of LLM span, only ~40% of requests use tools)
    if random.random() < 0.4:
        tool_span = build_tool_call_span(ctx, llm_span.span_id)
        trace.spans.append(tool_span)

    # 5. Response serialization (child of API span)
    output_tokens: int = llm_span.attributes.get("gen_ai.usage.output_tokens", 100)
    resp_span = build_response_span(ctx, api_span.span_id, output_tokens)
    trace.spans.append(resp_span)

    return trace


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def print_trace(trace: PipelineTrace) -> None:
    """Print a human-readable summary of the pipeline trace."""
    print(f"\n{'='*70}")
    print(f"TRACE  {trace.context.trace_id}")
    print(f"  request_id : {trace.context.request_id}")
    print(f"  total_ms   : {trace.total_duration_ms():.1f}")
    tokens = trace.token_totals()
    print(f"  tokens     : {tokens['input']} in / {tokens['output']} out / {tokens['total']} total")
    print(f"  cost_usd   : ${trace.estimated_cost():.6f}")
    print(f"{'='*70}")
    for span in trace.spans:
        indent = "  " if span.parent_span_id else ""
        child_mark = "└─ " if span.parent_span_id else "   "
        dur = span.duration_ms()
        dur_str = f"{dur:.1f} ms" if dur is not None else "ongoing"
        model = span.attributes.get("gen_ai.request.model", "")
        model_str = f" [{model}]" if model else ""
        print(f"{indent}{child_mark}{span.name:<25}{dur_str:>10}{model_str}")
    print()


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Run three pipeline traces with different models and print results."""
    scenarios: list[tuple[str, str]] = [
        ("gpt-4o", LLMProvider.OPENAI.value),
        ("claude-3-5-sonnet", LLMProvider.ANTHROPIC.value),
        ("gemini-1.5-pro", LLMProvider.GOOGLE.value),
    ]

    print("OpenTelemetry LLM Pipeline — Full-Stack Trace Simulation")
    print(f"Started at {datetime.now(UTC).isoformat()}\n")

    for model, provider in scenarios:
        trace = run_pipeline(model=model, provider=provider)
        print_trace(trace)


if __name__ == "__main__":
    main()
