"""Temperature-0 determinism at the output: the same prompts in different company.

Eight distinct prompts (the first eight of ``streams.PROMPTS``), temperature 0, a fixed
number of tokens (``ignore_eos``), fresh cache salts, served:

- alone: one at a time (the reference);
- c8: all eight at once (decode steps of eight streams);
- staggered: arriving 100 ms apart (steps of every size from one to eight);
- mixed: all eight at once with a long prompt arriving 0.3 s later, so decode rows
  share steps with its prefill chunks; the long prompt's own output is compared with
  it served alone;
- cached: each prompt twice with one salt; the second reuses the first's KV and
  computes only its last token;
- mixed15 (``--wide``): fifteen prompts at once (the eight and seven more) with the long
  prompt 0.3 s later, so decode steps carry nearly twice the rows of eight streams;
  each compared with it alone;
- again: one at a time once more (run to run, after the traffic above).

Each request asks for the chosen tokens' logprobs. Tokens must match the reference
exactly; logprobs are compared bit for bit as well (equal tokens with unequal logprobs
is a numeric difference that has not flipped a token yet). ``--no-logprobs`` compares
texts instead, which on some servers exercises a different (greedy) verification path.

One JSON line per scenario, then the summary: ``identical`` is true only when every token
(and, where the server returns them, every logprob) matched. The summary carries SHA-256
digests of the references (the eight prompts alone, and the long prompt alone), with and
without logprobs, so two builds can be compared bit for bit. Exits 0 either way: the run
records, the caller judges.
"""

import argparse
import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request
import uuid

from .streams import PROMPTS as STREAM_PROMPTS

PROMPTS = STREAM_PROMPTS[:8]
# The wide scenario's seven more prompts (fifteen streams leave a slot for the long one).
MORE_PROMPTS = [
    "Write a Python function that merges overlapping intervals, with tests.",
    "Explain how a CPU cache hierarchy affects the speed of matrix multiplication.",
    "Describe the water cycle for a ten-year-old, step by step.",
    "Implement binary search in Rust and explain each edge case.",
    "Outline a beginner's four-week strength training plan.",
    "Explain how public-key cryptography lets strangers agree on a secret.",
    "Write a SQL query for the top three products by revenue per month, and explain it.",
]


def long_prompt(lines: int) -> str:
    """A fixed, roughly 15-token-per-line field log, then a question about it."""
    rng = random.Random(1210)
    colours = ["amber", "slate", "crimson", "olive", "ivory", "teal", "umber", "violet"]
    things = ["heron", "lantern", "ferry", "orchard", "glacier", "market", "bridge", "kiln"]
    places = ["north ridge", "harbour", "old mill", "east field", "lower lock", "quarry"]
    body = "\n".join(f"Entry {i}: a {rng.choice(colours)} {rng.choice(things)} near the "
                     f"{rng.choice(places)}, count {rng.randint(1, 99)}, wind {rng.randint(0, 40)} knots."
                     for i in range(lines))
    return f"Field log:\n{body}\n\nWhich entries mention a glacier near the harbour? List their numbers."


class Client:
    def __init__(self, base: str, logprobs: bool = True, extra_body: dict | None = None,
                 model: str | None = None):
        self.base = base
        if model is None:
            with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
                model = json.load(response)["data"][0]["id"]
        self.model = model
        self.logprobs = logprobs
        self.extra_body = dict(extra_body or {})

    def complete(self, prompt: str, tokens: int, salt: str | None = None) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
                "temperature": 0, "ignore_eos": True, "cache_salt": salt or uuid.uuid4().hex}
        body.update(self.extra_body)
        if self.logprobs:
            body["logprobs"] = True
        request = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                choice = json.load(response)["choices"][0]
        except urllib.error.HTTPError as error:
            if self.logprobs and error.code == 400:  # the server does not return logprobs: tokens only
                self.logprobs = False
                return self.complete(prompt, tokens, salt)
            raise
        content = (choice.get("logprobs") or {}).get("content")
        text = choice["message"]["content"]
        if content:
            return {"tokens": [c["token"] for c in content], "logprobs": [c["logprob"] for c in content],
                    "text": text}
        return {"tokens": None, "text": text, "logprobs": None}


def together(client: Client, jobs: list[tuple[float, str, int, str | None]]) -> list[dict]:
    """Run (delay seconds, prompt, tokens, salt) jobs concurrently; results in job order."""
    results: list[dict] = [{} for _ in jobs]
    start = time.monotonic()

    def run(index, delay, prompt, tokens, salt):
        time.sleep(max(0.0, start + delay - time.monotonic()))
        results[index].update(client.complete(prompt, tokens, salt))

    threads = [threading.Thread(target=run, args=(i, *job)) for i, job in enumerate(jobs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if any(not r for r in results):
        raise SystemExit("a request returned nothing")
    return results


def compare(result: dict, reference: dict) -> dict:
    """First differing token (None: all equal) and whether every logprob matched."""
    if result["tokens"] is None or reference["tokens"] is None:
        a, b = result.get("text") or "", reference.get("text") or ""
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
        return {"first_token_diff": None, "first_char_diff": first, "logprobs_equal": None}
    a, b = result["tokens"], reference["tokens"]
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
    shared = len(a) if first is None else first
    lp_equal = result["logprobs"][:shared] == reference["logprobs"][:shared]
    lp_first = next((i for i in range(shared) if result["logprobs"][i] != reference["logprobs"][i]), None)
    return {"first_token_diff": first, "logprobs_equal": lp_equal, "first_logprob_diff": lp_first}


def digest(results: list[dict]) -> str:
    """SHA-256 of the results' tokens and logprobs (text where tokens are absent)."""
    payload = [[r["tokens"], r["logprobs"]] if r["tokens"] is not None else r.get("text") for r in results]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def text_digest(results: list[dict]) -> str:
    """SHA-256 of the results' texts, the same with or without logprobs."""
    return hashlib.sha256(json.dumps([r.get("text") for r in results]).encode()).hexdigest()


def _diffs(rows: list[dict]) -> dict:
    return {"token_diffs": sum(r.get("first_token_diff") is not None or r.get("first_char_diff") is not None
                               for r in rows),
            "logprob_diffs": sum(r["logprobs_equal"] is False for r in rows)}


def run(client: Client, tokens: int = 192, long_lines: int = 400, wide: bool = False, emit=print) -> dict:
    """Every scenario; ``emit`` receives one JSON line per scenario, then the summary line.
    Returns the summary."""
    long_text = long_prompt(long_lines)
    reference = [client.complete(p, tokens) for p in PROMPTS]
    long_reference = client.complete(long_text, 64)
    scenarios = {
        "c8": together(client, [(0.0, p, tokens, None) for p in PROMPTS]),
        "staggered": together(client, [(0.1 * i, p, tokens, None) for i, p in enumerate(PROMPTS)]),
    }
    mixed = together(client, [(0.0, p, tokens, None) for p in PROMPTS] + [(0.3, long_text, 64, None)])
    scenarios["mixed"], long_mixed = mixed[:-1], mixed[-1]
    widened = []
    if wide:
        more_reference = [client.complete(p, tokens) for p in MORE_PROMPTS]
        widened = together(client, [(0.0, p, tokens, None) for p in PROMPTS + MORE_PROMPTS]
                           + [(0.3, long_text, 64, None)])
    cached = []
    for prompt in PROMPTS:
        salt = uuid.uuid4().hex
        client.complete(prompt, tokens, salt)
        cached.append(client.complete(prompt, tokens, salt))
    scenarios["cached"] = cached
    scenarios["again"] = [client.complete(p, tokens) for p in PROMPTS]

    identical, report = True, {}
    for name, results in scenarios.items():
        rows = [dict(compare(r, ref), prompt=i) for i, (r, ref) in enumerate(zip(results, reference))]
        if name == "mixed":
            rows.append(dict(compare(long_mixed, long_reference), prompt="long"))
        report[name] = _diffs(rows)
        identical &= report[name]["token_diffs"] == 0 and report[name]["logprob_diffs"] == 0
        emit(json.dumps({"scenario": name, **report[name], "requests": rows}))
    if widened:
        references = reference + more_reference
        rows = [dict(compare(r, ref), prompt=i) for i, (r, ref) in enumerate(zip(widened[:-1], references))]
        rows.append(dict(compare(widened[-1], long_reference), prompt="long"))
        report["mixed15"] = _diffs(rows)
        identical &= report["mixed15"]["token_diffs"] == 0 and report["mixed15"]["logprob_diffs"] == 0
        emit(json.dumps({"scenario": "mixed15", **report["mixed15"], "requests": rows}))
    summary = {"summary": "temperature-0 determinism", "identical": identical,
               "logprobs": client.logprobs, "tokens": tokens, "scenarios": report,
               "reference_sha256": digest(reference), "long_sha256": digest([long_reference]),
               "reference_text_sha256": text_digest(reference),
               "long_text_sha256": text_digest([long_reference])}
    emit(json.dumps(summary))
    return summary


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("url", help="base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--long-lines", type=int, default=400)
    parser.add_argument("--wide", action="store_true", help="add the fifteen-stream mixed scenario")
    parser.add_argument("--no-logprobs", action="store_true", help="compare texts, without logprobs")
    parser.add_argument("--model", help="model id (default: the server's first)")
    parser.add_argument("--extra-body", help="JSON merged into every request body "
                                             "(for example the chat-template switch that turns reasoning off)")


def command(args: argparse.Namespace) -> int:
    extra = json.loads(args.extra_body) if args.extra_body else {}
    client = Client(args.url.rstrip("/"), logprobs=not args.no_logprobs, extra_body=extra, model=args.model)
    run(client, tokens=args.tokens, long_lines=args.long_lines, wide=args.wide,
        emit=lambda line: print(line, flush=True))
    return 0
