"""Quick latency and speed probes, each a few minutes: one stream over many texts
(``single``), short-prompt time to first token (``ttft``) and latency under mixed traffic
(``mixed``). They print one JSON line per measurement; ``single`` and ``mixed`` end with
a summary line.

They are quick looks between the steady-state measurements (``run``, ``suite``): no
measurement window, no intervals over sub-windows, no pairing.
"""

import argparse
import glob
import json
import os
import statistics
import threading
import time
import urllib.request
import uuid

SINGLE_PROMPTS = [
    "Explain how a hash map handles collisions, with an example in Python.",
    "Describe a walk through a spice market in Marrakech in the morning.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Compare TCP and UDP for real-time multiplayer games and when to use each.",
    "Give a detailed recipe for lentil soup with timings and substitutions.",
    "Summarize the causes and consequences of the printing press in Europe.",
    "Explain the difference between supervised and unsupervised learning to a student.",
    "Plan a three-day hiking trip in the Alps with packing advice.",
    "Return a JSON object describing five fictional cities with name, population and founding year.",
    "Write a SQL query that finds the top three customers by revenue per region, and explain it.",
    "List the planets of the solar system with one interesting fact about each.",
    "Draft a polite email asking a landlord to fix a leaking kitchen tap.",
    "Explain how public-key cryptography lets two strangers agree on a secret.",
    "Describe the life cycle of a star from nebula to remnant.",
    "Write a Python function that merges overlapping intervals, with tests.",
    "Give advice for a first-time manager running their first one-on-one meetings.",
    "Summarize the plot of a heist film you invent, in five acts.",
    "Explain why the sky is blue and sunsets are red.",
    "Return a JSON array of six books with title, author, year and genre.",
    "Compare electric and petrol cars on cost, emissions and maintenance.",
    "Write a limerick about a cat who learns to code, then explain the meter.",
    "Explain how a bill becomes a law in a parliamentary system.",
    "Describe how to train for a first half marathon over twelve weeks.",
    "Explain gradient descent with a worked numerical example.",
]
TTFT_WORDS = ("amber basil cedar delta ember fjord grove harbor indigo juniper kestrel lantern meadow "
              "nectar orchid pebble quartz raven saffron thistle umber velvet willow xenon yarrow zephyr "
              "anchor birch cobalt dune eagle fern garnet heron iris jade kelp lotus maple nutmeg").split()
MIXED_STREAMS = SINGLE_PROMPTS[:4]
MIXED_SHORT = ["What is the capital of Australia?", "Name three prime numbers above fifty.",
               "Give one synonym for quick.", "How many legs does a spider have?"]


def model_id(base: str) -> str:
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        return json.load(response)["data"][0]["id"]


def _chat(base: str, body: dict) -> urllib.request.Request:
    return urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})


# single: one stream at a time over many texts.

def single_prompts(count: int) -> list[str]:
    """``count`` prompts at an even stride, keeping the mix of prose, code and JSON."""
    return SINGLE_PROMPTS[:: max(1, len(SINGLE_PROMPTS) // count)][:count]


def single_one(base: str, model: str, prompt: str, tokens: int, extra_body: dict) -> float:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "stream": True, "ignore_eos": True, "cache_salt": uuid.uuid4().hex}
    body.update(extra_body)
    body["stream_options"] = {"include_usage": True}
    first = last = None
    count = 0
    with urllib.request.urlopen(_chat(base, body), timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                count = chunk["usage"]["completion_tokens"]
            if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                now = time.monotonic()
                first = first or now
                last = now
    return (count - 1) / (last - first)


def single_summary(rates: dict[int, list[float]]) -> dict:
    every = [x for v in rates.values() for x in v]
    mean = statistics.mean(every)
    half = 1.96 * statistics.stdev(every) / len(every) ** 0.5
    return {"summary": "c1 distinct prompts", "tps_mean": round(mean, 2), "ci95_pct": round(100 * half / mean, 2),
            "requests": len(every),
            "per_prompt_median_mean": round(statistics.mean(statistics.median(v) for v in rates.values()), 2)}


def command_single(args: argparse.Namespace) -> int:
    base = args.url.rstrip("/")
    model = args.model or model_id(base)
    extra = json.loads(args.extra_body) if args.extra_body else {}
    prompts = single_prompts(args.prompts)
    rates = {i: [] for i in range(len(prompts))}
    for r in range(args.rounds):
        for i, prompt in enumerate(prompts):
            tps = single_one(base, model, prompt, args.tokens, extra)
            rates[i].append(tps)
            print(json.dumps({"round": r, "prompt": i, "tps": round(tps, 2)}), flush=True)
    print(json.dumps(single_summary(rates)), flush=True)
    return 0


# ttft: short prefills, one request at a time.

def ttft_request(base: str, model: str, prompt: str, extra_body: dict) -> tuple[float, int]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
            "temperature": 0, "cache_salt": uuid.uuid4().hex}
    body.update(extra_body)
    start = time.perf_counter()
    with urllib.request.urlopen(_chat(base, body), timeout=300) as response:
        data = json.load(response)
    return time.perf_counter() - start, data["usage"]["prompt_tokens"]


def command_ttft(args: argparse.Namespace) -> int:
    base = args.url.rstrip("/")
    model = args.model or model_id(base)
    extra = json.loads(args.extra_body) if args.extra_body else {}
    ttft_request(base, model, "warm up", extra)
    for words in (1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40):
        prompt = "Repeat: " + " ".join(TTFT_WORDS[:words])
        times, tokens = [], None
        for _ in range(args.reps):
            elapsed, tokens = ttft_request(base, model, prompt, extra)
            times.append(1000 * elapsed)
        print(json.dumps({"words": words, "prompt_tokens": tokens, "median_ms": round(statistics.median(times), 2),
                          "min_ms": round(min(times), 2), "reps": args.reps}), flush=True)
    return 0


# mixed: decoding streams, long prompts and short requests on a fixed schedule.

def long_texts(source_dir: str) -> list[str]:
    """Two cold ~4000-token prompts: the Python files of ``source_dir``, concatenated in name
    order (up to about 40,000 characters), cut into two 12,400-character pieces."""
    source = ""
    for path in sorted(glob.glob(os.path.join(source_dir, "*.py"))):
        source += open(path).read() + "\n"
        if len(source) > 40000:
            break
    if len(source) < 24800:
        raise SystemExit(f"--long-source {source_dir}: fewer than 24,800 characters of Python source")
    return [source[:12400], source[12400:24800]]


def mixed_stream(base: str, model: str, prompt: str, tokens: int, out: dict, ignore_eos: bool,
                 extra_body: dict) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "stream": True, "ignore_eos": ignore_eos, "cache_salt": uuid.uuid4().hex}
    body.update(extra_body)
    out["start"] = time.monotonic()
    out["chunks"] = []
    with urllib.request.urlopen(_chat(base, body), timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                    out["chunks"].append(time.monotonic())


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float("nan")


def mixed_round(base: str, model: str, long: list[str], extra_body: dict, index: int) -> dict:
    plan = [(0.0, p, 384, "stream", True) for p in MIXED_STREAMS]
    plan += [(1.0, long[0], 16, "long", False), (4.0, long[1], 16, "long", False)]
    plan += [(t, MIXED_SHORT[i], 32, "short", False) for i, t in enumerate((2.0, 2.5, 3.0, 5.0))]
    results, threads = [], []
    t0 = time.monotonic()
    for at, prompt, tokens, kind, ignore in sorted(plan, key=lambda p: p[0]):
        delay = at - (time.monotonic() - t0)
        if delay > 0:
            time.sleep(delay)
        out = {"kind": kind}
        results.append(out)
        thread = threading.Thread(target=mixed_stream, args=(base, model, prompt, tokens, out, ignore, extra_body))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    gaps, rates = [], []
    for o in results:
        if o["kind"] == "stream" and len(o["chunks"]) > 1:
            gaps += [1000 * (b - a) for a, b in zip(o["chunks"], o["chunks"][1:])]
            rates.append(len(o["chunks"]) / (o["chunks"][-1] - o["chunks"][0]))
    ttft = {k: [1000 * (o["chunks"][0] - o["start"]) for o in results if o["kind"] == k and o["chunks"]]
            for k in ("short", "long")}
    return {"round": index, "short_ttft_ms_median": round(statistics.median(ttft["short"]), 1),
            "short_ttft_ms_max": round(max(ttft["short"]), 1),
            "long_ttft_ms_mean": round(statistics.mean(ttft["long"]), 1),
            "stream_gap_ms_p50": round(pct(gaps, 0.5), 1), "stream_gap_ms_p95": round(pct(gaps, 0.95), 1),
            "stream_gap_ms_p99": round(pct(gaps, 0.99), 1),
            "stream_chunks_per_s": round(statistics.mean(rates), 2)}


def command_mixed(args: argparse.Namespace) -> int:
    base = args.url.rstrip("/")
    model = args.model or model_id(base)
    extra = json.loads(args.extra_body) if args.extra_body else {}
    long = long_texts(args.long_source or os.path.dirname(os.path.abspath(__file__)))
    rows = []
    for r in range(args.rounds):
        rows.append(mixed_round(base, model, long, extra, r))
        print(json.dumps(rows[-1]), flush=True)
        time.sleep(2)
    keys = [k for k in rows[0] if k != "round"]
    print(json.dumps({"summary": "mixed traffic latency", "rounds": args.rounds,
                      **{k: round(statistics.mean(s[k] for s in rows), 1) for k in keys}}), flush=True)
    return 0


def add_parsers(sub) -> None:
    single = sub.add_parser("single", help="one stream at a time over distinct prompts (quick)")
    single.add_argument("url", help="base URL, e.g. http://127.0.0.1:8000")
    single.add_argument("--tokens", type=int, default=256)
    single.add_argument("--rounds", type=int, default=1)
    single.add_argument("--prompts", type=int, default=len(SINGLE_PROMPTS),
                        help=f"N of the {len(SINGLE_PROMPTS)} prompts at an even stride")
    single.add_argument("--model", help="model id (default: the server's first)")
    single.add_argument("--extra-body", help="JSON merged into every request body")
    single.set_defaults(func=command_single)
    ttft = sub.add_parser("ttft", help="time to first token of short prompts (1-40 words), one at a time")
    ttft.add_argument("url", help="base URL, e.g. http://127.0.0.1:8000")
    ttft.add_argument("--reps", type=int, default=7)
    ttft.add_argument("--model", help="model id (default: the server's first)")
    ttft.add_argument("--extra-body", help="JSON merged into every request body")
    ttft.set_defaults(func=command_ttft)
    mixed = sub.add_parser("mixed", help="latency under mixed traffic: streams, long prompts, short requests")
    mixed.add_argument("url", help="base URL, e.g. http://127.0.0.1:8000")
    mixed.add_argument("--rounds", type=int, default=3)
    mixed.add_argument("--long-source", metavar="DIR",
                       help="directory whose Python files make the two long prompts (default: Sonar's own)")
    mixed.add_argument("--model", help="model id (default: the server's first)")
    mixed.add_argument("--extra-body", help="JSON merged into every request body")
    mixed.set_defaults(func=command_mixed)
