"""What a server did over a window of real traffic, from its Prometheus counters alone.

``sonar watch URL --duration S`` (or ``-- COMMAND``, timing the window around it) reads
vLLM's ``/metrics`` before and after the window and samples its gauges every
``--interval`` seconds. It sends no requests, so it can sit beside production traffic or
beside another load generator and describe what the server saw: requests finished and why,
preemptions, prompt and output tokens (and the prefix cache's share of the prompt),
latency quantiles from the server's histograms, speculative acceptance by draft position,
the share of engine steps that carried prefill, and the load gauges' maxima and means.
"""

import argparse
import json
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

MetricSeries = dict[tuple[str, tuple[tuple[str, str], ...]], float]

GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:scheduler_local_prefill_backlog_tokens",
)

LATENCIES = {
    "queue": "vllm:request_queue_time_seconds",
    "first_token": "vllm:time_to_first_token_seconds",
    "inter_token": "vllm:inter_token_latency_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
    "end_to_end": "vllm:e2e_request_latency_seconds",
}

# A step above this many tokens carried prefill: decode steps carry at most
# max-num-seqs x (1 + drafts) tokens, so set it just above that for the server watched.
DECODE_ONLY_STEP_TOKENS = 64


def parse_metric_series(text: str) -> MetricSeries:
    """Every sample keyed by name and labels, summed over engine and model."""
    series: MetricSeries = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        name, _, labels = head.partition("{")
        pairs = tuple(sorted(
            (key, label)
            for key, label in re.findall(r'(\w+)="([^"]*)"', labels)
            if key not in ("engine", "model_name")
        ))
        try:
            series[(name, pairs)] = series.get((name, pairs), 0.0) + float(value)
        except ValueError:
            continue
    return series


def scrape_metric_series(base: str) -> MetricSeries:
    with urllib.request.urlopen(f"{base}/metrics", timeout=10) as response:
        return parse_metric_series(response.read().decode("utf-8", errors="replace"))


def series_sum(series: MetricSeries, name: str, **labels: str) -> float:
    return sum(
        value
        for (key, pairs), value in series.items()
        if key == name and all(dict(pairs).get(label) == want for label, want in labels.items())
    )


def series_by_label(series: MetricSeries, name: str, label: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for (key, pairs), value in series.items():
        if key == name and label in dict(pairs):
            values[dict(pairs)[label]] = values.get(dict(pairs)[label], 0.0) + value
    return values


def histogram_buckets(series: MetricSeries, name: str) -> list[tuple[float, float]]:
    cumulative = {
        float("inf") if bound == "+Inf" else float(bound): count
        for bound, count in series_by_label(series, f"{name}_bucket", "le").items()
    }
    return sorted(cumulative.items())


def histogram_quantile(series: MetricSeries, name: str, quantile: float) -> float | None:
    """Prometheus-style estimate, linear inside the bucket that crosses the quantile."""
    buckets = histogram_buckets(series, name)
    if not buckets or buckets[-1][1] <= 0:
        return None
    target = quantile * buckets[-1][1]
    lower, below = 0.0, 0.0
    for upper, count in buckets:
        if count >= target:
            if upper == float("inf"):
                return lower
            share = (target - below) / (count - below) if count > below else 1.0
            return lower + (upper - lower) * share
        lower, below = upper, count
    return lower


def histogram_above(series: MetricSeries, name: str, bound: float) -> float:
    buckets = histogram_buckets(series, name)
    if not buckets:
        return 0.0
    at_or_below = max((count for upper, count in buckets if upper <= bound), default=0.0)
    return buckets[-1][1] - at_or_below


def workload_summary(delta: MetricSeries, seconds: float, gauges: dict[str, list[float]],
                     decode_step_tokens: int = DECODE_ONLY_STEP_TOKENS) -> dict:
    """What the server did over one window: traffic shape, latency, drafts, load."""
    def total(name: str, **labels: str) -> float:
        return series_sum(delta, name, **labels)

    def ratio(numerator: float, denominator: float, digits: int = 3) -> float | None:
        return round(numerator / denominator, digits) if denominator else None

    def quantiles(name: str) -> dict[str, float | None]:
        estimates = {}
        for quantile in (0.5, 0.9, 0.99):
            value = histogram_quantile(delta, name, quantile)
            estimates[f"p{round(quantile * 100)}"] = None if value is None else round(value, 3)
        return estimates

    reasons = {
        reason: int(round(count))
        for reason, count in series_by_label(delta, "vllm:request_success_total", "finished_reason").items()
        if round(count)
    }
    finished = sum(reasons.values())
    hours = seconds / 3600
    prompt = total("vllm:prompt_tokens_total")
    generated = total("vllm:generation_tokens_total")
    drafts = total("vllm:spec_decode_num_drafts_total")
    accepted_at = series_by_label(delta, "vllm:spec_decode_num_accepted_tokens_per_pos_total", "position")
    drafted_at = series_by_label(delta, "vllm:spec_decode_num_draft_tokens_per_pos_total", "position")
    steps = total("vllm:iteration_tokens_total_count")
    return {
        "seconds": round(seconds, 1),
        "requests": {
            "finished": finished,
            "by_reason": reasons,
            "per_hour": ratio(finished, hours, 1),
            "preemptions": int(round(total("vllm:num_preemptions_total"))),
        },
        "prompt": {
            "tokens": int(prompt),
            "cached_share": ratio(total("vllm:prompt_tokens_cached_total"), prompt),
            "per_request": quantiles("vllm:request_prompt_tokens"),
        },
        "output": {
            "tokens": int(generated),
            "tok_per_s": ratio(generated, seconds, 1),
            "per_request": quantiles("vllm:request_generation_tokens"),
        },
        "latency_s": {label: quantiles(name) for label, name in LATENCIES.items()},
        "speculative": {
            "accepted_per_draft": ratio(total("vllm:spec_decode_num_accepted_tokens_total"), drafts),
            "verified_per_draft": ratio(total("vllm:spec_decode_num_draft_tokens_total"), drafts),
            "acceptance_by_position": [
                ratio(accepted_at.get(position, 0.0), drafted_at[position])
                for position in sorted(drafted_at, key=int)
            ],
        },
        "steps": {
            "count": int(steps),
            "carrying_prefill_share": ratio(
                histogram_above(delta, "vllm:iteration_tokens_total", decode_step_tokens), steps
            ),
        },
        "load": {
            name.removeprefix("vllm:"): {
                "max": round(max(values), 3),
                "mean": round(sum(values) / len(values), 3),
            }
            for name, values in gauges.items()
            if values
        },
    }


def format_workload(summary: dict) -> str:
    requests, prompt, output = summary["requests"], summary["prompt"], summary["output"]
    latency, speculative = summary["latency_s"], summary["speculative"]

    def spread(values: dict, unit: str = "") -> str:
        return " / ".join("-" if value is None else f"{value:g}{unit}" for value in values.values())

    lines = [f"window {summary['seconds']:.0f} s: {requests['finished']} requests finished "
             f"({requests['per_hour'] or 0:g}/h), "
             f"{requests['preemptions']} preemptions, reasons {requests['by_reason'] or {}}",
             f"prompt tokens {prompt['tokens']} ({(prompt['cached_share'] or 0) * 100:.0f}% from the "
             f"prefix cache), per request p50/p90/p99 {spread(prompt['per_request'])}",
             f"output tokens {output['tokens']} ({output['tok_per_s'] or 0:g} tok/s over the window), "
             f"per request p50/p90/p99 {spread(output['per_request'])}"]
    for label, values in latency.items():
        lines.append(f"  {label.replace('_', ' '):<12} p50/p90/p99 {spread(values, ' s')}")
    lines.append(f"drafts: {speculative['accepted_per_draft']} accepted of {speculative['verified_per_draft']} "
                 f"verified per step, by position {speculative['acceptance_by_position']}")
    share = summary["steps"]["carrying_prefill_share"]
    lines.append(f"engine steps {summary['steps']['count']}, "
                 f"{'-' if share is None else f'{share * 100:.1f}%'} carrying prefill")
    for name, values in summary["load"].items():
        lines.append(f"  {name:<40} max {values['max']:g}  mean {values['mean']:g}")
    return "\n".join(lines)


def watch(base: str, *, duration: float | None = None, interval: float = 5.0, run: list[str] | None = None,
          decode_step_tokens: int = DECODE_ONLY_STEP_TOKENS) -> tuple[dict, int]:
    """Watch ``base`` for ``duration`` seconds, or for as long as ``run`` (a command) runs.
    Returns the summary and the command's exit code (0 without one)."""
    gauges: dict[str, list[float]] = {name: [] for name in GAUGES}

    def sample() -> MetricSeries:
        series = scrape_metric_series(base)
        for name in GAUGES:
            if any(key == name for key, _ in series):
                gauges[name].append(series_sum(series, name))
        return series

    before = sample()
    started = time.monotonic()
    child = subprocess.Popen(run) if run else None
    deadline = started + duration if duration else None
    returncode = 0
    try:
        while True:
            if child is not None and child.poll() is not None:
                returncode = child.returncode
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            remaining = deadline - time.monotonic() if deadline else interval
            time.sleep(max(0.1, min(interval, remaining)))
            sample()
    except KeyboardInterrupt:
        if child is not None:
            child.send_signal(signal.SIGINT)
            returncode = child.wait()
    seconds = time.monotonic() - started
    after = sample()
    delta = {key: value - before.get(key, 0.0) for key, value in after.items()}
    if any(value < 0 for (key, _), value in delta.items() if key.endswith(("_total", "_count", "_bucket"))):
        raise SystemExit("a counter went backwards: the server restarted inside the window")
    return workload_summary(delta, seconds, gauges, decode_step_tokens), returncode


def add_parser(sub) -> None:
    parser = sub.add_parser("watch", help="describe a window of a server's traffic from its counters alone")
    parser.add_argument("url", help="base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--duration", type=float, help="window length in seconds")
    parser.add_argument("--interval", type=float, default=5.0, help="gauge sampling period")
    parser.add_argument("--decode-step-tokens", type=int, default=DECODE_ONLY_STEP_TOKENS,
                        help="steps of more tokens than this carried prefill (default %(default)s)")
    parser.add_argument("--label", help="name recorded in the JSON report")
    parser.add_argument("--json", metavar="PATH", help="also write the summary as JSON")
    parser.add_argument("run", nargs=argparse.REMAINDER, help="-- command to time the window around")
    parser.set_defaults(func=command)


def command(args: argparse.Namespace) -> int:
    run = args.run[1:] if args.run and args.run[0] == "--" else args.run
    summary, returncode = watch(args.url.rstrip("/"), duration=args.duration, interval=args.interval,
                                run=run or None, decode_step_tokens=args.decode_step_tokens)
    summary["label"] = args.label
    summary["command"] = run or None
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(format_workload(summary))
    return returncode
