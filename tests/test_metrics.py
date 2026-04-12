"""Tests for metrics.py."""

from __future__ import annotations

from patterns.metrics import Counter, Gauge, Histogram, LLMMetrics, MetricRegistry


class TestCounter:
    def test_add(self):
        c = Counter("c")
        c.add(5)
        assert c.total() == 5

    def test_add_multiple(self):
        c = Counter("c")
        c.add(3)
        c.add(7)
        assert c.total() == 10

    def test_points(self):
        c = Counter("c")
        c.add(1, {"model": "gpt-4"})
        assert len(c.points()) == 1
        assert c.points()[0].labels["model"] == "gpt-4"


class TestGauge:
    def test_set(self):
        g = Gauge("g")
        g.set(42.0)
        assert g.current() == 42.0

    def test_overwrite(self):
        g = Gauge("g")
        g.set(1.0)
        g.set(99.0)
        assert g.current() == 99.0

    def test_points(self):
        g = Gauge("g")
        g.set(5.0)
        assert len(g.points()) == 1


class TestHistogram:
    def test_record(self):
        h = Histogram("h")
        h.record(100)
        assert h.count() == 1

    def test_sum(self):
        h = Histogram("h")
        h.record(100)
        h.record(200)
        assert h.sum() == 300

    def test_mean(self):
        h = Histogram("h")
        h.record(100)
        h.record(200)
        assert h.mean() == 150

    def test_mean_empty(self):
        assert Histogram("h").mean() == 0.0

    def test_percentile(self):
        h = Histogram("h")
        for i in range(1, 101):
            h.record(float(i))
        p50 = h.percentile(50)
        assert 45 <= p50 <= 55

    def test_percentile_empty(self):
        assert Histogram("h").percentile(95) == 0.0

    def test_bucket_counts(self):
        h = Histogram("h", buckets=[10, 100])
        h.record(5)
        h.record(50)
        h.record(200)
        counts = h.bucket_counts()
        assert counts[10] == 1
        assert counts[100] == 2


class TestLLMMetrics:
    def setup_method(self):
        self.m = LLMMetrics()

    def test_record_request(self):
        self.m.record_request(500, 100, 50, "gpt-4", "openai")
        assert self.m.request_duration.count() == 1
        assert self.m.token_usage.total() == 150

    def test_record_error(self):
        self.m.record_request(100, 0, 0, "gpt-4", "openai", error="timeout")
        assert self.m.error_count.total() == 1

    def test_no_error_no_error_count(self):
        self.m.record_request(100, 10, 5, "gpt-4", "openai")
        assert self.m.error_count.total() == 0

    def test_summary_keys(self):
        self.m.record_request(300, 50, 25, "claude-3", "anthropic")
        s = self.m.summary()
        assert "total_tokens" in s
        assert "request_count" in s
        assert s["request_count"] == 1


class TestMetricRegistry:
    def setup_method(self):
        self.reg = MetricRegistry()

    def test_counter(self):
        c = self.reg.counter("hits")
        c.add(1)
        assert self.reg.counter("hits").total() == 1

    def test_gauge(self):
        g = self.reg.gauge("active")
        g.set(5.0)
        assert self.reg.gauge("active").current() == 5.0

    def test_histogram(self):
        h = self.reg.histogram("latency")
        h.record(100)
        assert self.reg.histogram("latency").count() == 1

    def test_all_names(self):
        self.reg.counter("a")
        self.reg.gauge("b")
        self.reg.histogram("c")
        assert set(self.reg.all_names()) == {"a", "b", "c"}

    def test_idempotent(self):
        c1 = self.reg.counter("x")
        c2 = self.reg.counter("x")
        assert c1 is c2
