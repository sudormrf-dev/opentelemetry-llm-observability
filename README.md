# opentelemetry-llm-observability

OpenTelemetry patterns for LLM observability: semantic conventions, span building, metrics collection, and trace sampling.

## Patterns

- **conventions** — `LLMAttributes`, `LLMProvider`, `LLMOperation`, `SpanKind` (GenAI semconv)
- **spans** — `LLMSpan`, `LLMSpanBuilder`, `SpanContext`, `SpanStatus`
- **metrics** — `Counter`, `Gauge`, `Histogram`, `LLMMetrics`, `MetricRegistry`
- **sampling** — `TraceSampler` with rule-based, ratio, always-on/off strategies

## Install

```bash
pip install -e ".[dev]"
pytest
```
