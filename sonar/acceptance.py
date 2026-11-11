"""Speculative-decoding acceptance over many texts: each distinct prompt alone, one after
another.

A change that moves temperature-0 outputs (a more precise kernel, a different activation
order) changes the text, and one prompt's acceptance then moves with its continuation, not
with the change. Here every prompt of ``streams.PROMPTS`` runs alone (fresh cache salt,
temperature 0) and the server's speculative-decoding counters (vLLM's ``spec_decode_*``)
are read before and after it. One JSON line per prompt (accepted tokens per draft, verified
tokens per draft, decode tok/s), then the summary: means over prompts, so arms compare
prompt by prompt over the same set.
"""

import argparse
import json
import statistics
import urllib.request

from .streams import PROMPTS, model_id, stream

COUNTERS = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total")


def counters(text: str) -> dict[str, float]:
    totals = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in totals:
            totals[name] += float(line.rsplit(" ", 1)[1])
    return totals


def scrape(base: str) -> dict[str, float]:
    with urllib.request.urlopen(base + "/metrics", timeout=60) as response:
        return counters(response.read().decode())


def one(base: str, model: str, index: int, tokens: int, extra_body: dict | None = None) -> dict:
    before = scrape(base)
    result = {}
    stream(base, model, PROMPTS[index], tokens, result, extra_body)
    after = scrape(base)
    delta = {name: after[name] - before[name] for name in COUNTERS}
    drafts = delta[COUNTERS[0]]
    if not drafts or result.get("first") is None or result["last"] <= result["first"]:
        raise SystemExit(f"prompt {index}: no drafts or no decode window")
    return {"prompt": index, "tokens": result["tokens"],
            "accepted_per_draft": round(delta[COUNTERS[2]] / drafts, 4),
            "verified_per_draft": round(delta[COUNTERS[1]] / drafts, 4),
            "tps": round((result["tokens"] - 1) / (result["last"] - result["first"]), 2)}


def summary(rows: list[dict]) -> dict:
    def mean_ci(key):
        values = [r[key] for r in rows]
        mean = statistics.mean(values)
        half = 1.96 * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
        return round(mean, 4), round(100 * half / mean, 2)
    accepted, accepted_ci = mean_ci("accepted_per_draft")
    tps, tps_ci = mean_ci("tps")
    return {"summary": "acceptance sweep", "prompts": len(rows), "accepted_per_draft_mean": accepted,
            "accepted_ci95_pct": accepted_ci, "tps_mean": tps, "tps_ci95_pct": tps_ci}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("url", help="base URL of a server with speculative decoding")
    parser.add_argument("--prompts", type=int, default=len(PROMPTS), help=f"first N prompts (1-{len(PROMPTS)})")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--model", help="model id (default: the server's first)")
    parser.add_argument("--extra-body", help="JSON merged into every request body")


def command(args: argparse.Namespace) -> int:
    if not 1 <= args.prompts <= len(PROMPTS):
        raise SystemExit(f"--prompts must be between 1 and {len(PROMPTS)}")
    base = args.url.rstrip("/")
    model = args.model or model_id(base)
    extra = json.loads(args.extra_body) if args.extra_body else {}
    rows = []
    for index in range(args.prompts):
        rows.append(one(base, model, index, args.tokens, extra))
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps(summary(rows)), flush=True)
    return 0
