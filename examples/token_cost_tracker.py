"""Real-time token cost tracker for LLM pipelines.

Simulates 100 LLM requests across multiple models and sessions, then:
  - Tracks cost per request, per session, and per model
  - Fires alerts when a session exceeds a configurable threshold
  - Prints a monthly cost dashboard

Uses only stdlib: random, time, dataclasses, collections.
"""

from __future__ import annotations

import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

# ---------------------------------------------------------------------------
# Cost table — USD per 1 000 tokens (input / output)
# ---------------------------------------------------------------------------

MODEL_RATES: dict[str, tuple[float, float]] = {
    "gpt-4o":             (0.005,   0.015),
    "gpt-4o-mini":        (0.00015, 0.0006),
    "claude-3-5-sonnet":  (0.003,   0.015),
    "claude-3-5-haiku":   (0.0008,  0.004),
    "gemini-1.5-pro":     (0.00125, 0.005),
    "gemini-1.5-flash":   (0.000075, 0.0003),
    "mistral-large":      (0.002,   0.006),
}

ALL_MODELS = list(MODEL_RATES.keys())

# Distribution weights (heavier traffic on cheaper models)
MODEL_WEIGHTS = [10, 25, 8, 20, 6, 18, 13]

# Alert threshold per session (USD)
SESSION_COST_ALERT_USD = 0.05


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LLMRequest:
    """One recorded LLM request."""

    request_id: str
    session_id: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    timestamp: datetime
    cost_usd: float = 0.0
    error: bool = False

    def __post_init__(self) -> None:
        rate_in, rate_out = MODEL_RATES.get(self.model, (0.002, 0.008))
        self.cost_usd = (self.input_tokens * rate_in + self.output_tokens * rate_out) / 1_000


@dataclass
class SessionStats:
    """Aggregated stats for a single user session."""

    session_id: str
    requests: list[LLMRequest] = field(default_factory=list)
    alert_fired: bool = False

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.requests)

    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.requests)

    def request_count(self) -> int:
        return len(self.requests)

    def mean_latency_ms(self) -> float:
        if not self.requests:
            return 0.0
        return sum(r.latency_ms for r in self.requests) / len(self.requests)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TokenCostTracker:
    """Collects LLM request metrics and drives alerting."""

    def __init__(self, alert_threshold_usd: float = SESSION_COST_ALERT_USD) -> None:
        self._threshold = alert_threshold_usd
        self._sessions: dict[str, SessionStats] = {}
        self._all_requests: list[LLMRequest] = []
        self._alerts: list[str] = []

    def record(self, request: LLMRequest) -> None:
        """Ingest a request, check session alerts."""
        self._all_requests.append(request)

        if request.session_id not in self._sessions:
            self._sessions[request.session_id] = SessionStats(session_id=request.session_id)

        session = self._sessions[request.session_id]
        session.requests.append(request)

        if not session.alert_fired and session.total_cost() >= self._threshold:
            session.alert_fired = True
            msg = (
                f"[ALERT] session={request.session_id[:8]}  "
                f"cost=${session.total_cost():.4f} >= threshold=${self._threshold:.4f}"
            )
            self._alerts.append(msg)
            print(msg)

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self._all_requests)

    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self._all_requests)

    def cost_by_model(self) -> dict[str, float]:
        by_model: dict[str, float] = defaultdict(float)
        for r in self._all_requests:
            by_model[r.model] += r.cost_usd
        return dict(sorted(by_model.items(), key=lambda kv: kv[1], reverse=True))

    def cost_by_session(self) -> dict[str, float]:
        return {sid: s.total_cost() for sid, s in self._sessions.items()}

    def top_sessions(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.cost_by_session().items(), key=lambda kv: kv[1], reverse=True)[:n]

    def error_rate(self) -> float:
        if not self._all_requests:
            return 0.0
        errors = sum(1 for r in self._all_requests if r.error)
        return errors / len(self._all_requests)

    def p95_latency_ms(self) -> float:
        lats = sorted(r.latency_ms for r in self._all_requests)
        if not lats:
            return 0.0
        idx = int(len(lats) * 0.95)
        return lats[min(idx, len(lats) - 1)]


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate_request(session_id: str) -> LLMRequest:
    """Generate one synthetic LLM request."""
    model = random.choices(ALL_MODELS, weights=MODEL_WEIGHTS, k=1)[0]
    # Larger models → more tokens, more latency
    token_scale = 1.0 if "mini" in model or "flash" in model or "haiku" in model else 2.5
    input_tokens = int(random.gauss(800, 300) * token_scale)
    input_tokens = max(50, min(input_tokens, 8000))
    output_tokens = int(random.gauss(250, 100) * token_scale)
    output_tokens = max(20, min(output_tokens, 2000))

    base_latency = {"gpt-4o": 900, "gpt-4o-mini": 300, "claude-3-5-sonnet": 800,
                    "claude-3-5-haiku": 350, "gemini-1.5-pro": 700,
                    "gemini-1.5-flash": 250, "mistral-large": 600}.get(model, 500)
    latency_ms = max(80.0, random.gauss(base_latency, base_latency * 0.25))

    error = random.random() < 0.02  # 2 % error rate

    return LLMRequest(
        request_id=str(uuid.uuid4()),
        session_id=session_id,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.now(UTC),
        error=error,
    )


def run_simulation(n_requests: int = 100, n_sessions: int = 12) -> TokenCostTracker:
    """Simulate *n_requests* spread across *n_sessions* user sessions."""
    tracker = TokenCostTracker()
    session_ids = [str(uuid.uuid4()) for _ in range(n_sessions)]

    print(f"Simulating {n_requests} LLM requests across {n_sessions} sessions...\n")

    for i in range(n_requests):
        session_id = random.choice(session_ids)
        req = simulate_request(session_id)
        tracker.record(req)

        # Light progress indicator every 10 requests
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1:>3}/{n_requests}]  running_cost=${tracker.total_cost():.4f}")

    return tracker


# ---------------------------------------------------------------------------
# Dashboard printer
# ---------------------------------------------------------------------------

def print_dashboard(tracker: TokenCostTracker) -> None:
    """Print a monthly cost dashboard from tracked data."""
    divider = "=" * 65

    print(f"\n{divider}")
    print("  LLM COST DASHBOARD — Monthly Projection")
    print(divider)

    total = tracker.total_cost()
    tokens = tracker.total_tokens()
    req_count = len(tracker._all_requests)  # noqa: SLF001

    print(f"  Total cost          : ${total:.4f}")
    print(f"  Total tokens        : {tokens:,}")
    print(f"  Total requests      : {req_count}")
    print(f"  Cost per request    : ${total / max(req_count, 1):.6f}")
    print(f"  Cost per 1K tokens  : ${total / max(tokens / 1_000, 0.001):.6f}")
    print(f"  Error rate          : {tracker.error_rate() * 100:.1f}%")
    print(f"  p95 latency         : {tracker.p95_latency_ms():.0f} ms")
    print(f"  Alerts fired        : {len(tracker._alerts)}")  # noqa: SLF001

    # Annualised projection
    monthly_projection = total * 30  # 100 requests ~ 1 day of traffic
    print(f"\n  Monthly projection  : ${monthly_projection:.2f}  (×30 scale)")
    print(f"  Annual projection   : ${monthly_projection * 12:.2f}")

    print(f"\n{divider}")
    print("  Cost by Model")
    print(divider)
    for model, cost in tracker.cost_by_model().items():
        bar_len = int(cost / max(total, 0.0001) * 40)
        bar = "█" * bar_len
        pct = cost / max(total, 0.0001) * 100
        print(f"  {model:<22} ${cost:.4f}  {bar:<40} {pct:.1f}%")

    print(f"\n{divider}")
    print("  Top 5 Most Expensive Sessions")
    print(divider)
    for rank, (session_id, cost) in enumerate(tracker.top_sessions(5), 1):
        session = tracker._sessions[session_id]  # noqa: SLF001
        print(
            f"  #{rank}  {session_id[:8]}...  "
            f"${cost:.4f}  {session.request_count()} reqs  "
            f"avg {session.mean_latency_ms():.0f} ms"
        )

    print(f"\n{divider}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the token cost tracking demo."""
    random.seed(42)
    start = time.monotonic()

    tracker = run_simulation(n_requests=100, n_sessions=12)
    print_dashboard(tracker)

    elapsed = time.monotonic() - start
    print(f"Simulation completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
