"""Sampling strategy overhead benchmark.

Compares four sampling strategies on three dimensions:
  - Latency added per span decision (ns)
  - Memory consumed (bytes per 1 000 spans)
  - Throughput (spans/sec)

Strategies benchmarked:
  1. Always sample   (100 %)
  2. Head-based      (10 % ratio)
  3. Tail-based      (error spans only)
  4. Adaptive        (dynamic ratio based on error rate)

No third-party dependencies — uses stdlib: time, random, gc, tracemalloc.
"""

from __future__ import annotations

import gc
import random
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from patterns.sampling import SamplingDecision, SamplingRule, SamplingStrategy, TraceSampler

# ---------------------------------------------------------------------------
# Synthetic span factory
# ---------------------------------------------------------------------------

_SPAN_NAMES = [
    "llm.call", "rag.retrieval", "tool.call", "api.request",
    "embedding.create", "rerank.query", "cache.lookup",
]

_ERROR_RATE = 0.05  # 5 % of spans are errors


def make_span_attrs(force_error: bool = False) -> dict[str, Any]:
    """Return a minimal attributes dict for a synthetic span."""
    is_error = force_error or (random.random() < _ERROR_RATE)
    return {
        "gen_ai.system": random.choice(["openai", "anthropic", "google"]),
        "gen_ai.request.model": random.choice(["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"]),
        "error": "true" if is_error else "false",
    }


# ---------------------------------------------------------------------------
# Adaptive sampler (not in patterns/sampling.py — defined here)
# ---------------------------------------------------------------------------

class AdaptiveSampler:
    """Dynamically adjusts ratio based on observed error rate.

    Keeps a sliding window of recent decisions; when error rate exceeds the
    high-water mark, it bumps the sampling ratio to capture more context.
    """

    def __init__(self, base_ratio: float = 0.1, window: int = 200) -> None:
        self._base_ratio = base_ratio
        self._ratio = base_ratio
        self._window: list[bool] = []  # True = error
        self._window_size = window

    def should_sample(self, span_name: str, attributes: dict[str, Any]) -> SamplingDecision:
        is_error = attributes.get("error") == "true"
        self._window.append(is_error)
        if len(self._window) > self._window_size:
            self._window.pop(0)

        # Recompute ratio every 50 spans
        if len(self._window) % 50 == 0 and self._window:
            error_rate = sum(self._window) / len(self._window)
            if error_rate > 0.10:
                self._ratio = min(1.0, self._base_ratio * 4)
            elif error_rate < 0.02:
                self._ratio = self._base_ratio
            else:
                self._ratio = self._base_ratio * 2

        if is_error:
            return SamplingDecision.RECORD_AND_SAMPLE

        h = abs(hash(span_name)) % 1000
        threshold = int(self._ratio * 1000)
        if h < threshold:
            return SamplingDecision.RECORD_AND_SAMPLE
        return SamplingDecision.DROP


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Results for a single sampling strategy."""

    strategy_name: str
    n_spans: int
    sampled_count: int
    dropped_count: int
    latency_ns_mean: float
    latency_ns_p99: float
    memory_bytes_per_1k: float
    spans_per_sec: float

    def sample_rate(self) -> float:
        return self.sampled_count / max(self.n_spans, 1)


SamplerFn = Callable[[str, dict[str, Any]], SamplingDecision]


def _benchmark_strategy(
    name: str,
    sampler_fn: SamplerFn,
    n_spans: int = 10_000,
) -> BenchmarkResult:
    """Run the benchmark for one strategy and return metrics."""
    span_names = [random.choice(_SPAN_NAMES) for _ in range(n_spans)]
    attrs_list = [make_span_attrs() for _ in range(n_spans)]

    gc.collect()
    gc.disable()

    # --- Latency measurement ---
    latencies: list[float] = []
    sampled = dropped = 0

    for span_name, attrs in zip(span_names, attrs_list):
        t0 = time.perf_counter_ns()
        decision = sampler_fn(span_name, attrs)
        t1 = time.perf_counter_ns()
        latencies.append(t1 - t0)
        if decision == SamplingDecision.RECORD_AND_SAMPLE:
            sampled += 1
        else:
            dropped += 1

    gc.enable()

    latencies.sort()
    mean_ns = sum(latencies) / len(latencies)
    p99_idx = int(len(latencies) * 0.99)
    p99_ns = latencies[min(p99_idx, len(latencies) - 1)]

    # --- Memory measurement (per 1 000 spans) ---
    gc.collect()
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    dummy: list[dict[str, Any]] = []
    for span_name, attrs in zip(span_names[:1000], attrs_list[:1000]):
        decision = sampler_fn(span_name, attrs)
        if decision == SamplingDecision.RECORD_AND_SAMPLE:
            dummy.append({"name": span_name, **attrs})  # simulate storing sampled span
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    mem_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)

    # --- Throughput ---
    t_start = time.perf_counter()
    for span_name, attrs in zip(span_names, attrs_list):
        sampler_fn(span_name, attrs)
    t_end = time.perf_counter()
    elapsed = max(t_end - t_start, 1e-9)
    spans_per_sec = n_spans / elapsed

    return BenchmarkResult(
        strategy_name=name,
        n_spans=n_spans,
        sampled_count=sampled,
        dropped_count=dropped,
        latency_ns_mean=mean_ns,
        latency_ns_p99=p99_ns,
        memory_bytes_per_1k=float(mem_bytes),
        spans_per_sec=spans_per_sec,
    )


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

def _build_strategies() -> list[tuple[str, SamplerFn]]:
    """Return (name, sampler_fn) pairs for all four strategies."""

    # 1. Always sample
    always = TraceSampler.always_on()

    # 2. Head-based 10 %
    head = TraceSampler.ratio_based(0.10)

    # 3. Tail-based (error-only rule)
    tail = TraceSampler(strategy=SamplingStrategy.RULE_BASED)
    tail.add_rule(SamplingRule(attribute_key="error", attribute_value="true",
                               decision=SamplingDecision.RECORD_AND_SAMPLE, priority=10))
    tail.add_rule(SamplingRule(attribute_key="error", attribute_value="false",
                               decision=SamplingDecision.DROP, priority=1))

    # 4. Adaptive
    adaptive = AdaptiveSampler(base_ratio=0.10)

    return [
        ("Always (100%)",      always.should_sample),
        ("Head-based (10%)",   head.should_sample),
        ("Tail-based (errors)", tail.should_sample),
        ("Adaptive",           adaptive.should_sample),
    ]


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------

def print_results(results: list[BenchmarkResult]) -> None:
    """Print an aligned comparison table."""
    col = [30, 10, 10, 12, 12, 14, 14]
    headers = ["Strategy", "Sampled%", "Spans/sec", "Lat mean ns", "Lat p99 ns", "Mem /1k B", "Dropped"]
    sep = "  "

    print(f"\n{'='*110}")
    print("  Sampling Strategy Overhead Benchmark")
    print(f"{'='*110}")
    header_row = sep.join(h.ljust(c) for h, c in zip(headers, col))
    print(f"  {header_row}")
    print(f"  {'-' * 105}")

    for r in results:
        row = sep.join([
            r.strategy_name.ljust(col[0]),
            f"{r.sample_rate() * 100:.1f}%".ljust(col[1]),
            f"{r.spans_per_sec:,.0f}".ljust(col[2]),
            f"{r.latency_ns_mean:.1f}".ljust(col[3]),
            f"{r.latency_ns_p99:.1f}".ljust(col[4]),
            f"{r.memory_bytes_per_1k:,.0f}".ljust(col[5]),
            str(r.dropped_count).ljust(col[6]),
        ])
        print(f"  {row}")

    print(f"{'='*110}")

    # Relative latency vs always-on baseline
    baseline = next((r for r in results if "Always" in r.strategy_name), None)
    if baseline:
        print("\n  Latency overhead vs always-on baseline:")
        for r in results:
            if r is baseline:
                continue
            delta = r.latency_ns_mean - baseline.latency_ns_mean
            sign = "+" if delta >= 0 else ""
            print(f"    {r.strategy_name:<28} {sign}{delta:.1f} ns/span")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all sampling benchmarks and print the results table."""
    random.seed(0)
    n_spans = 10_000

    print(f"Running sampling overhead benchmark ({n_spans:,} spans per strategy)...")
    strategies = _build_strategies()

    results: list[BenchmarkResult] = []
    for name, fn in strategies:
        print(f"  Benchmarking: {name}...")
        result = _benchmark_strategy(name, fn, n_spans=n_spans)
        results.append(result)

    print_results(results)


if __name__ == "__main__":
    main()
