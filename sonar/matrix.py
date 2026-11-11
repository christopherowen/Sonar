"""The measurement matrix: decode points (case x concurrency) sampled until each is precise,
beside quality, structured-output compliance, sampled acceptance, cold prefill, prefix-cache
replay and long-context admission checks.

Each decode sample is one burst: every stream of the point starts at once, and the sample
ends when the last finishes. A point keeps sampling, in shuffled rounds, until the metric it
converges on (tok/s, step time or accepted per verified draft) has a 95% interval within
``--precision``. Every sample waits for an idle server and is repeated when someone else's
traffic overlapped it (the server's finished-request counter and its running gauge say so).

A burst holds a ramp and a tail, and one prompt repeated on N streams writes one text on a
batch-invariant server, so these points show where a build moves rather than by how much;
``sonar run`` and ``sonar suite`` measure the steady state. The matrix stays because its
cases, prompts and reports are fixed: a report from one build compares with a report from
another, point by point (``rows``).

Callers that watch the hardware pass a ``guard`` factory to ``Sampler``: it is called with the
sampler's halt callback and must return an object whose ``check()`` raises to stop the run
(memory or temperature limits, for example).

Reasoning: cases marked ``reasoning_off`` merge ``options.reasoning_off`` into their request
bodies, the server's own switch for that (for example ``{"chat_template_kwargs":
{"thinking": false}}``); Sonar sends nothing model-specific unless told.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import functools
import hashlib
import json
import random
import re
import sysconfig
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import NamedTuple

SCHEMA = "sonar-matrix/1"
SUITES = ("quality", "compliance", "decode", "sampled", "prefill", "prefix", "admission")
# --full runs the promotion report; compliance is selected explicitly with --suites
# until constrained decoding beside speculative decoding is qualified on the server.
FULL_SUITES = tuple(suite for suite in SUITES if suite != "compliance")
TIMEOUT_SECONDS = 1800
# These prompts and settings are fixed: reports compare point by point only while they
# stay the same (the workload records their digests).
DECODE_PROMPTS = {
    "prose": (
        "Explain, in precise technical prose, how speculative decoding can "
        "increase language-model serving throughput while preserving the "
        "target model's output distribution. Discuss acceptance rate, "
        "verification cost, batching, and latency."
    ),
    "code": (
        "Write a complete Python implementation of an LRU cache using only "
        "the standard library. Include type hints, O(1) get and put methods, "
        "capacity validation, and a short executable example. Explain the "
        "complexity after the code."
    ),
}
DECODE_TOKENS = 256


class DecodeCase(NamedTuple):
    """One decode workload.

    prompt: one prompt, or a tuple of distinct prompts given to successive
        streams (cycled, with a stream tag once they repeat).
    per_stream: tag a single prompt with each stream's number, so concurrent
        streams neither share a prefix-cache block nor produce one output.
    fixed_length: generate exactly max_tokens tokens (min_tokens and
        ignore_eos), so every stack and sample measures the same length.
    max_tokens: the output budget per request; the portable protocol's 256.
    min_samples / max_samples: sampling floors for this case. The command-line
        values apply when they are larger.
    converge_on: the per-sample metric whose confidence interval ends the
        point's sampling (tps, step_ms or accepted_per_verified); None defers
        to --converge-on, then tps.
    response_format: an OpenAI response_format object, for the server's
        structured output.
    json_schema: score every output as a JSON array of objects against this
        item schema (a tuple pairs one schema with each prompt): completed
        objects, whether the truncated prefix parses once closed, and how many
        objects conform. Outputs are also written to the report directory.
    """

    prompt: str | tuple[str, ...]
    reasoning_off: bool = False
    fixed_length: bool = False
    per_stream: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = DECODE_TOKENS
    min_samples: int | None = None
    max_samples: int | None = None
    converge_on: str | None = None
    response_format: dict | None = None
    json_schema: dict | tuple[dict, ...] | None = None


# Extra decode cases, selected with --decode-cases. The server reasons by
# default, so most of what "prose" and "code" measure is reasoning text; the
# -nothink cases turn reasoning off to measure the answer itself, where draft
# acceptance differs most.
DECODE_CASES = {
    "prose-nothink": DecodeCase(DECODE_PROMPTS["prose"], True),
    "code-nothink": DecodeCase(DECODE_PROMPTS["code"], True),
    "json-nothink": DecodeCase(
        "Return only a JSON array of twelve objects describing fictional "
        "library books, with no prose and no code fence. Each object has "
        "the keys title, author, year, genres (an array of two strings), "
        "isbn, pages, available (a boolean), and rating (a number from 1 "
        "to 5).",
        True,
    ),
}
DECODE_CASES.update((case, DecodeCase(prompt)) for case, prompt in DECODE_PROMPTS.items())
# Portable cases follow the protocol common public serving benchmarks use, so
# figures line up with theirs: reasoning off, exactly 256 tokens, a distinct
# prompt per stream, and workloads from highly predictable (a count) to
# free-running prose and sampled chat.
PORTABLE_CODE_TAIL = (
    "Reply with Python source only: no comments, no docstrings, no Markdown "
    "fences. After it, write unit tests and any helpers they need, and keep "
    "writing code."
)
PORTABLE_CODE_TASKS = tuple(
    f"{name}\n{spec}\n{PORTABLE_CODE_TAIL}"
    for name, spec in (
        ("flatten", "def flatten(items) -> list: flatten arbitrarily nested lists of ints, preserving order."),
        ("stack", "class Stack: push, pop, peek and is_empty over a list, raising IndexError on underflow."),
        ("palindrome", "def is_palindrome(text) -> bool: ignore case and every non-alphanumeric character."),
        ("word_counts", "def word_counts(text) -> dict: count words case-insensitively, ignoring punctuation."),
        ("matmul", "def matmul(a, b) -> list: multiply two matrices given as lists of rows, checking shapes."),
        ("trie", "class Trie: insert(word), contains(word) and starts_with(prefix) over nested dicts."),
        ("roman", "def roman_to_int(text) -> int: convert a Roman numeral, including subtractive pairs."),
        ("rotate", "def rotate(nums, k) -> list: rotate a list right by k positions, for any integer k."),
    )
)
PORTABLE_CASES = {
    "count": "Count upward from 1 to 400. Write only the numbers, separated by single spaces, and nothing else.",
    "explain": (
        "Explain step by step how a B-tree index works in a database: node "
        "layout, search, insertion with node splits, deletion with merges, and "
        "the cost of each operation. Be thorough and precise."
    ),
    "tasks": PORTABLE_CODE_TASKS,
    "rows": (
        "Output only a JSON array of invented server telemetry records. Each "
        "record has hostname, rack, cpuPct, memPct, diskIops, netMbps and "
        "status. Produce as many records as you can, with no Markdown and no "
        "commentary."
    ),
    "math": (
        "Solve this step by step, showing every calculation: a train leaves a "
        "station at 9:40 at 84 km/h, and a second train follows on the same "
        "track at 10:15 at 112 km/h. When and how far from the station does "
        "the second train catch the first? Then write and solve three similar "
        "problems with different numbers."
    ),
    "chat": (
        "Explain to a curious twelve-year-old how the heart pumps blood around "
        "the body, why it beats faster during exercise, and what blood pressure "
        "means. Be friendly and concrete."
    ),
    "essay": (
        "Write a flowing essay of about 900 words on the history of bridges, "
        "from rope bridges and Roman arches to modern cable-stayed spans. Use "
        "full paragraphs only: no headings, no lists and no Markdown."
    ),
    "story": (
        "Write continuous literary prose of about 900 words following a "
        "mountain town through one winter, seen by a baker, a schoolteacher "
        "and a snowplough driver. No headings and no lists."
    ),
}
DECODE_CASES.update(
    (case, DecodeCase(prompt, True, fixed_length=True, per_stream=True))
    for case, prompt in PORTABLE_CASES.items()
)
DECODE_CASES["chat-sampled"] = DecodeCase(
    PORTABLE_CASES["chat"], True, fixed_length=True, per_stream=True,
    temperature=0.7, top_p=0.95,
)
# Names --decode-cases expands.
DECODE_CASE_GROUPS = {"portable": [*PORTABLE_CASES, "chat-sampled"]}
# The internal precision set resolves small differences between arms. It keeps
# the portable protocol's settings (reasoning off, temperature 0) but gives
# every sample four times the decode steps, uses prompts whose natural answer
# exceeds the budget so nothing is forced to continue past its end, takes at
# least five samples per point, and converges on step time rather than tok/s,
# which inherits the acceptance swing of whichever text the run produced.
PRECISION_TOKENS = 1024
PRECISION_MIN_SAMPLES = 5
PRECISION_MAX_SAMPLES = 12
CONVERGENCE_METRICS = ("tps", "step_ms", "accepted_per_verified")
PRECISION_PROSE = (
    "Write a continuous essay of about 2,500 words on how cities have managed "
    "fresh water, from Roman aqueducts and medieval wells to modern treatment "
    "plants and desalination. Use full paragraphs only: no headings, no lists "
    "and no Markdown. Do not stop early."
)


def record_schema(**properties: dict) -> dict:
    """An object schema requiring exactly these keys."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def strings(count: int) -> dict:
    return {"type": "array", "items": {"type": "string"}, "minItems": count, "maxItems": count}


def percent() -> dict:
    return {"type": "number", "minimum": 0, "maximum": 100}


# One schema per record type. The prompt text is generated from the schema, so
# what the model is asked for and what its output is checked against cannot
# drift apart.
PRECISION_JSON_RECORDS = (
    ("library books", record_schema(
        title={"type": "string"}, author={"type": "string"}, year={"type": "integer"},
        genres=strings(2), isbn={"type": "string"}, pages={"type": "integer"},
        available={"type": "boolean"}, rating={"type": "number", "minimum": 1, "maximum": 5},
    )),
    ("server telemetry records", record_schema(
        hostname={"type": "string"}, rack={"type": "string"}, cpuPct=percent(), memPct=percent(),
        diskIops={"type": "number"}, netMbps={"type": "number"}, status={"type": "string"},
        sampledAt={"type": "string"},
    )),
    ("employees", record_schema(
        employeeId={"type": "string"}, firstName={"type": "string"}, lastName={"type": "string"},
        department={"type": "string"}, title={"type": "string"}, hiredOn={"type": "string"},
        salary={"type": "number"}, remote={"type": "boolean"},
    )),
    ("retail products", record_schema(
        sku={"type": "string"}, name={"type": "string"}, category={"type": "string"},
        priceCents={"type": "integer"}, currency={"type": "string"}, inStock={"type": "boolean"},
        tags=strings(3), weightGrams={"type": "number"},
    )),
    ("airline flights", record_schema(
        flightNumber={"type": "string"}, origin={"type": "string"}, destination={"type": "string"},
        departs={"type": "string"}, arrives={"type": "string"}, aircraft={"type": "string"},
        seatsSold={"type": "integer"}, onTime={"type": "boolean"},
    )),
    ("weather observations", record_schema(
        station={"type": "string"}, observedAt={"type": "string"}, temperatureC={"type": "number"},
        humidityPct=percent(), windKph={"type": "number"}, windDirection={"type": "string"},
        pressureHpa={"type": "number"}, conditions={"type": "string"},
    )),
    ("invoices", record_schema(
        invoiceId={"type": "string"}, customer={"type": "string"}, issuedOn={"type": "string"},
        dueOn={"type": "string"},
        lineItems={
            "type": "array", "minItems": 2, "maxItems": 2,
            "items": record_schema(description={"type": "string"}, amountCents={"type": "integer"}),
        },
        paid={"type": "boolean"},
    )),
    ("IoT sensor readings", record_schema(
        sensorId={"type": "string"}, site={"type": "string"}, kind={"type": "string"},
        value={"type": "number"}, unit={"type": "string"}, batteryPct=percent(),
        firmware={"type": "string"}, lastSeen={"type": "string"},
    )),
)
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def schema_key_text(schema: dict) -> str:
    """How a prompt describes an object schema's keys and value types."""
    def phrase(prop: dict) -> str:
        kind = prop.get("type")
        if kind == "array":
            items = prop.get("items", {})
            noun = {"string": "strings", "integer": "integers", "number": "numbers",
                    "boolean": "booleans", "object": "objects"}.get(items.get("type"), "values")
            inner = f" with {schema_key_text(items)}" if items.get("type") == "object" else ""
            count = prop.get("minItems") if prop.get("minItems") == prop.get("maxItems") else None
            size = f"{NUMBER_WORDS.get(count, count)} " if count else ""
            return f"an array of {size}{noun}{inner}"
        if kind == "number" and "minimum" in prop and "maximum" in prop:
            return f"a number from {prop['minimum']:g} to {prop['maximum']:g}"
        return {"string": "a string", "integer": "an integer", "number": "a number", "boolean": "a boolean"}[kind]

    parts = [f"{name} ({phrase(prop)})" for name, prop in schema["properties"].items()]
    return ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]


def json_records_prompt(subject: str, schema: dict, count: int = 80) -> str:
    return (
        f"Output only a JSON array of {count} invented {subject}. Each object has exactly the keys "
        f"{schema_key_text(schema)}. Use realistic, varied values. No Markdown, no code fence and no "
        f"commentary: begin with [ and keep adding objects until the array has {count} entries."
    )


PRECISION_JSON_PROMPTS = tuple(json_records_prompt(subject, schema) for subject, schema in PRECISION_JSON_RECORDS)
# Code tasks whose natural answer runs well past the budget: an implementation,
# a complete test suite and a command-line interface for each. The portable
# tasks are single functions and can stop early.
PRECISION_CODE_TAIL = (
    "Reply with Python source only: no comments, no docstrings, no Markdown fences. Write the full "
    "implementation, then a complete unittest suite covering every public method and edge case, then "
    "an argparse command-line interface for it. Keep writing code until all three parts are complete."
)
PRECISION_CODE_TASKS = tuple(
    f"{name}\n{spec}\n{PRECISION_CODE_TAIL}"
    for name, spec in (
        ("library", "A library system: Book, Member and Library classes with borrowing limits, due dates, "
                    "late fees, reservations and title/author search."),
        ("inventory", "An inventory manager: products with SKUs, stock levels, reorder thresholds, suppliers, "
                      "CSV import/export and a low-stock report."),
        ("scheduler", "A task scheduler: tasks with dependencies, priorities and durations, topological ordering, "
                      "cycle detection and a critical-path calculation."),
        ("ledger", "A bank ledger: accounts, deposits, withdrawals, transfers with overdraft rules, monthly "
                   "interest and statement generation."),
        ("expressions", "An arithmetic expression engine: tokenizer, recursive-descent parser to an AST, "
                        "evaluator with variables and functions, and a pretty printer."),
        ("matrix", "A matrix library over lists of lists: construction, addition, multiplication, transpose, "
                   "determinant, inverse and solving linear systems, with shape checking."),
        ("cache", "A caching package: LRU, LFU and TTL caches behind one interface, with statistics, "
                  "eviction callbacks and a decorator for memoizing functions."),
        ("adventure", "A text-adventure engine: rooms, exits, items, inventory, a command parser, save and "
                      "load of game state, and a scripted sample world."),
    )
)
PRECISION_JSON_SCHEMAS = tuple(schema for _, schema in PRECISION_JSON_RECORDS)
PRECISION_JSON_SCHEMA = {
    "name": "records",
    "schema": {
        "type": "array",
        "items": record_schema(
            id={"type": "integer"}, name={"type": "string"}, category={"type": "string"},
            value={"type": "number"}, active={"type": "boolean"}, tags=strings(2),
        ),
    },
}
PRECISION_CASES = {
    "prose-1k": DecodeCase(
        PRECISION_PROSE, True, per_stream=True, max_tokens=PRECISION_TOKENS,
        min_samples=PRECISION_MIN_SAMPLES, max_samples=PRECISION_MAX_SAMPLES, converge_on="step_ms",
    ),
    "code-1k": DecodeCase(
        PRECISION_CODE_TASKS, True, max_tokens=PRECISION_TOKENS,
        min_samples=PRECISION_MIN_SAMPLES, max_samples=PRECISION_MAX_SAMPLES, converge_on="step_ms",
    ),
    "json-1k": DecodeCase(
        PRECISION_JSON_PROMPTS, True, max_tokens=PRECISION_TOKENS,
        min_samples=PRECISION_MIN_SAMPLES, max_samples=PRECISION_MAX_SAMPLES, converge_on="step_ms",
        json_schema=PRECISION_JSON_SCHEMAS,
    ),
}
DECODE_CASES.update(PRECISION_CASES)
DECODE_CASE_GROUPS["precision"] = list(PRECISION_CASES)
# Structured output through the server's grammar-constrained decoding. Opt in
# with --decode-cases json-schema-1k. It stays out of the precision group:
# constrained decoding changes the sampler path and needs its own
# qualification beside speculative decoding before its numbers mean anything.
DECODE_CASES["json-schema-1k"] = DecodeCase(
    json_records_prompt("inventory records", PRECISION_JSON_SCHEMA["schema"]["items"]),
    True, per_stream=True, max_tokens=PRECISION_TOKENS,
    min_samples=PRECISION_MIN_SAMPLES, max_samples=PRECISION_MAX_SAMPLES, converge_on="step_ms",
    response_format={"type": "json_schema", "json_schema": PRECISION_JSON_SCHEMA},
    json_schema=PRECISION_JSON_SCHEMA["schema"]["items"],
)
SAMPLED_PROMPTS = {
    "prose": (
        "Write a short essay about how rivers shape the land and the people "
        "who live near them."
    ),
    "code": (
        "Write a Python module with a thread-safe LRU cache class (get, put, "
        "delete, clear) and unit tests using unittest."
    ),
}
LRU_PROMPT = (
    "Write a short Python LRU cache class with get and put methods using "
    "OrderedDict. Give code and one sentence on complexity."
)
NOVEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
FILLER_WORDS = (
    "river stone lantern orchard meadow harbor canyon glacier prairie "
    "summit valley forest desert island marsh delta ridge basin"
).split()
METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:request_success_total",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
# Two-sided 95% Student t quantiles by degrees of freedom (lower bound).
T_95 = (
    (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571), (6, 2.447),
    (7, 2.365), (8, 2.306), (9, 2.262), (10, 2.228), (12, 2.179), (15, 2.131),
    (20, 2.086), (30, 2.042), (60, 2.000), (120, 1.980),
)


class Abort(RuntimeError):
    """Stop issuing requests; the partial report is still written."""


def t_95(degrees: float) -> float:
    quantile = T_95[0][1]
    for bound, value in T_95:
        if degrees >= bound:
            quantile = value
    return quantile


def describe(values: list[float]) -> dict:
    count = len(values)
    if not count:
        return {"n": 0}
    mean = sum(values) / count
    ordered = sorted(values)
    middle = count // 2
    median = ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    summary = {
        "n": count,
        "mean": round(mean, 3),
        "median": round(median, 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "sd": None,
        "ci95": None,
        "ci95_pct": None,
    }
    if count > 1:
        sd = (sum((value - mean) ** 2 for value in values) / (count - 1)) ** 0.5
        half = t_95(count - 1) * sd / count**0.5
        summary.update(sd=round(sd, 3), ci95=round(half, 3))
        if mean:
            summary["ci95_pct"] = round(100 * half / mean, 2)
    return summary


def difference(new: dict, reference: dict) -> tuple[float, float, float] | None:
    """Percent change of the new mean and its Welch 95% interval."""
    if not new.get("n") or not reference.get("n") or not reference.get("mean"):
        return None
    new_var = (new.get("sd") or 0.0) ** 2 / new["n"]
    ref_var = (reference.get("sd") or 0.0) ** 2 / reference["n"]
    denominator = 0.0
    if new["n"] > 1:
        denominator += new_var**2 / (new["n"] - 1)
    if reference["n"] > 1:
        denominator += ref_var**2 / (reference["n"] - 1)
    degrees = (new_var + ref_var) ** 2 / denominator if denominator else 120
    change = new["mean"] / reference["mean"] - 1
    half = t_95(degrees) * (new_var + ref_var) ** 0.5 / reference["mean"]
    return 100 * change, 100 * (change - half), 100 * (change + half)


def scrape_metrics(base: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base}/metrics", timeout=10) as response:
        text = response.read().decode("utf-8", errors="replace")
    return parse_metrics(text)


def parse_metrics(text: str) -> dict[str, float]:
    values = dict.fromkeys(METRICS, 0.0)
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in values:
            values[name] += float(line.rsplit(" ", 1)[1])
    return values


def stream_request(
    url: str, payload: dict, stop: threading.Event, keep_content: bool = False
) -> dict:
    """Stream one request; every failure stays in the receipt."""
    body = dict(payload, stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first: float | None = None
    last: float | None = None
    status: int | None = None
    usage: dict = {}
    finish: str | None = None
    content: list[str] = []
    reasoning_chars = 0
    content_chars = 0
    digest = hashlib.sha256()
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status = response.status
            for raw in response:
                if stop.is_set():
                    error = "aborted by bench"
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                event = json.loads(line[6:])
                usage = event.get("usage") or usage
                for choice in event.get("choices") or []:
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    pieces = (
                        delta.get("reasoning_content"),
                        delta.get("reasoning"),
                        delta.get("content"),
                        choice.get("text"),
                    )
                    emitted = "".join(piece for piece in pieces if isinstance(piece, str))
                    # Completions stream their first token directly; chat
                    # streams open with an empty role delta.
                    if first is None and (emitted or "text" in choice):
                        first = time.perf_counter()
                    if not emitted:
                        continue
                    last = time.perf_counter()
                    digest.update(emitted.encode())
                    for key in ("reasoning_content", "reasoning"):
                        if isinstance(delta.get(key), str):
                            reasoning_chars += len(delta[key])
                    if isinstance(delta.get("content"), str):
                        content_chars += len(delta["content"])
                        content.append(delta["content"])
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    ended = time.perf_counter()
    tokens = usage.get("completion_tokens")
    decode = None
    if tokens and tokens > 1 and first is not None and ended > first:
        decode = (tokens - 1) / (ended - first)
    result = {
        "ok": error is None and status == 200 and first is not None,
        "ttft_s": None if first is None else round(first - started, 4),
        "elapsed_s": round(ended - started, 4),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": tokens,
        # "stop" is a natural end, "length" the budget; None when the server sent neither.
        "finish_reason": finish,
        "decode_tps": None if decode is None else round(decode, 2),
        "output_sha256": digest.hexdigest()[:16],
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "error": error,
        # Absolute clock readings; Sampler.run turns them into offsets from the
        # sample's start (first_s, last_s).
        "first_at": first,
        "last_at": last,
    }
    if keep_content:
        result["content"] = "".join(content)
    return result


class PeakWatch:
    """Peak KV use and request counts while one sample runs."""

    def __init__(self, base: str) -> None:
        self.base = base
        self.peak = {"kv_cache_usage": 0.0, "running": 0.0, "active": 0.0}
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def _poll(self) -> None:
        while not self.done.wait(1.0):
            try:
                metrics = scrape_metrics(self.base)
            except Exception:
                continue
            running = metrics["vllm:num_requests_running"]
            active = running + metrics["vllm:num_requests_waiting"]
            self.peak["kv_cache_usage"] = max(self.peak["kv_cache_usage"], metrics["vllm:kv_cache_usage_perc"])
            self.peak["running"] = max(self.peak["running"], running)
            self.peak["active"] = max(self.peak["active"], active)

    def finish(self) -> dict:
        self.done.set()
        self.thread.join()
        return self.peak


class Sampler:
    def __init__(self, base: str, guard=None) -> None:
        self.base = base
        self.stop = threading.Event()
        self.reason: str | None = None
        self.sent = 0
        self.foreign = 0
        self.start_total: float | None = None
        self.discarded: list[dict] = []
        # The caller's hardware watch (see the module docstring), or None.
        self.guard = guard(self.halt) if guard is not None else None

    def halt(self, reason: str) -> None:
        if not self.stop.is_set():
            self.reason = reason
            self.stop.set()

    def wait_idle(self, timeout: float = 300) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            if self.stop.is_set():
                raise Abort(self.reason)
            if self.guard is not None:
                self.guard.check()
            metrics = scrape_metrics(self.base)
            if not metrics["vllm:num_requests_running"] and not metrics["vllm:num_requests_waiting"]:
                # Let the success counter settle before attributing requests.
                time.sleep(0.5)
                metrics = scrape_metrics(self.base)
                if self.start_total is None:
                    self.start_total = metrics["vllm:request_success_total"]
                finished = metrics["vllm:request_success_total"] - self.start_total
                self.foreign = max(self.foreign, int(finished) - self.sent)
                return metrics
            if time.monotonic() > deadline:
                raise Abort("the service stayed busy with other requests")
            time.sleep(2)

    def run(
        self,
        label: str,
        payloads: list[tuple[str, dict]],
        concurrency: int | None = None,
        keep_content: bool = False,
        retries: int = 3,
    ) -> dict:
        """Send one sample; repeat it when other traffic overlapped."""
        concurrency = concurrency or len(payloads)
        for attempt in range(retries + 1):
            before = self.wait_idle()
            peak = PeakWatch(self.base)
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                try:
                    results = list(
                        pool.map(
                            lambda item: stream_request(
                                f"{self.base}{item[0]}", item[1], self.stop, keep_content
                            ),
                            payloads,
                        )
                    )
                except BaseException:
                    self.halt("interrupted")
                    raise
            wall = time.perf_counter() - started
            for result in results:
                for key in ("first", "last"):
                    at = result.pop(f"{key}_at", None)
                    result[f"{key}_s"] = None if at is None else round(at - started, 4)
            peaks = peak.finish()
            self.sent += len(payloads)
            if self.stop.is_set():
                raise Abort(self.reason)
            after = scrape_metrics(self.base)
            finished = after["vllm:request_success_total"] - self.start_total
            foreign = int(finished) - self.sent - self.foreign
            # A closed loop may briefly show a finished request beside its successor.
            slack = 1 if len(payloads) > concurrency else 0
            if foreign > 0 or peaks["active"] > concurrency + slack:
                self.foreign += max(foreign, 0)
                self.discarded.append({"label": label, "attempt": attempt, "foreign": foreign})
                print(f"  {label}: other traffic overlapped; repeating", flush=True)
                continue
            metrics = {
                name.removeprefix("vllm:").removesuffix("_total"): after[name] - before[name]
                for name in METRICS
                if name.endswith("_total")
            }
            return {"wall_s": wall, "requests": results, "metrics": metrics, "peak": peaks}
        raise Abort(f"{label}: other traffic overlapped {retries + 1} attempts")


def reasoning_body(options: argparse.Namespace) -> dict:
    """What turns the server's reasoning off (merged into reasoning-off cases' bodies)."""
    return dict(getattr(options, "reasoning_off", None) or {})


def chat_payload(model: str, prompt: str, max_tokens: int, **extra) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    payload.update(extra)
    return "/v1/chat/completions", payload


def decode_payloads(model: str, case: str, concurrency: int,
                    reasoning_off: dict | None = None) -> list[tuple[str, dict]]:
    spec = DECODE_CASES[case]
    extra: dict = {}
    if spec.reasoning_off:
        extra.update(reasoning_off or {})
    if spec.fixed_length:
        extra.update(min_tokens=spec.max_tokens, ignore_eos=True)
    if spec.top_p is not None:
        extra["top_p"] = spec.top_p
    if spec.response_format is not None:
        extra["response_format"] = spec.response_format
    payloads = []
    for stream in range(concurrency):
        if isinstance(spec.prompt, tuple):
            prompt = spec.prompt[stream % len(spec.prompt)]
            if stream >= len(spec.prompt):
                prompt = f"[stream {stream + 1}]\n{prompt}"
        elif spec.per_stream and concurrency > 1:
            prompt = f"{spec.prompt} (stream {stream + 1}/{concurrency})"
        else:
            prompt = spec.prompt
        seed = 42 + stream if spec.temperature else 42
        payloads.append(
            chat_payload(model, prompt, spec.max_tokens, temperature=spec.temperature, seed=seed, **extra)
        )
    return payloads


def decode_window_tps(requests: list[dict]) -> float | None:
    """Tokens after each stream's first, over the span from the earliest first
    token to the latest last token: aggregate decode without prefill or the
    start stagger of a concurrent wave."""
    ok = [
        request for request in requests
        if request.get("ok") and request.get("first_s") is not None
        and request.get("last_s") is not None and (request.get("completion_tokens") or 0) > 1
    ]
    if not ok:
        return None
    window = max(request["last_s"] for request in ok) - min(request["first_s"] for request in ok)
    tokens = sum(request["completion_tokens"] - 1 for request in ok)
    return round(tokens / window, 3) if window > 0 else None


def filler(tokens: int, seed: int) -> str:
    # About one token per word for these plain lowercase words.
    body = " ".join(FILLER_WORDS[(i * 7 + seed) % len(FILLER_WORDS)] for i in range(tokens))
    return f"Document {seed}:\n{body}"


def novel_text(tokens: int, seed: int) -> str:
    """Random pseudo-words: n-grams no cache keyed on text has seen.

    The filler repeats 18 words, so anything a server keeps by n-gram (an n-gram
    memory table read from disk, say) stays warm and "cold" prefill never reads it.
    Words of two to four random letters average 1.9 tokens each (measured on a
    large-vocabulary tokenizer at 4K-64K).
    """
    rng = random.Random(seed)
    words = (
        "".join(rng.choice(NOVEL_LETTERS) for _ in range(rng.randint(2, 4)))
        for _ in range(int(tokens / 1.9))
    )
    return f"Document {seed}:\n" + " ".join(words)


# Python standard-library source averages about 3.8 characters per token on
# a large-vocabulary tokenizer (2.6-4.3 depending on the files, measured at 4K-64K).
SOURCE_CHARS_PER_TOKEN = 3.8


@functools.cache
def standard_library_sources() -> tuple[Path, ...]:
    root = Path(sysconfig.get_paths()["stdlib"])
    return tuple(sorted(path for path in root.glob("*.py") if path.stat().st_size > 4096))


def source_text(tokens: int, seed: int) -> str:
    """Real text: the Python standard library's source and docstrings, read
    from a seeded file onward, like the documentation and code prompts
    public benchmarks prefill with. The first line keeps every prompt's
    prefix unique."""
    files = standard_library_sources()
    index = random.Random(seed).randrange(len(files))
    chars = int(tokens * SOURCE_CHARS_PER_TOKEN)
    parts: list[str] = []
    size = 0
    while size < chars:
        text = files[index % len(files)].read_text(encoding="utf-8", errors="replace")
        parts.append(text)
        size += len(text)
        index += 1
    return f"Document {seed}:\n" + "".join(parts)[:chars]


def acceptance(metrics: dict) -> float | None:
    drafts = metrics["spec_decode_num_drafts"]
    return round(metrics["spec_decode_num_accepted_tokens"] / drafts, 4) if drafts else None


def step_times_ms(samples: list[dict]) -> list[float]:
    """Milliseconds per decode step for single-stream samples.

    Each step emits one token plus the accepted drafts, so the step time is
    (1 + accepted per draft) / decode tok/s. It is independent of how many
    drafts the text lets the model accept, which makes it far steadier than
    tok/s for changes to kernels or host overhead.
    """
    times = []
    for sample in samples:
        accepted = sample.get("accepted_per_draft")
        requests = sample.get("requests") or []
        if accepted is None or len(requests) != 1:
            continue
        rate = requests[0].get("decode_tps")
        if requests[0].get("ok") and rate:
            times.append(1000 * (1 + accepted) / rate)
    return times


def step_time_ms(requests: list[dict], accepted: float | None, concurrency: int) -> float | None:
    """Milliseconds per engine step for one sample.

    Each step gives every active stream one token plus its accepted drafts. A
    single stream is exact from that request's decode rate. Concurrent streams
    use the aggregate decode window; they finish a few steps apart, so the
    window's tail runs with fewer streams and the value is a slight
    overestimate.
    """
    ok = [request for request in requests if request.get("ok")]
    if accepted is None or len(ok) != len(requests) or len(ok) != concurrency:
        return None
    if concurrency == 1:
        rate = ok[0].get("decode_tps")
        return round(1000 * (1 + accepted) / rate, 3) if rate else None
    window = decode_window_tps(requests)
    return round(1000 * (1 + accepted) * concurrency / window, 3) if window else None


def used_whole_budget(request: dict, max_tokens: int) -> bool:
    """Whether a request ran to its token budget: the server's finish reason
    when it sent one, otherwise the token count."""
    finish = request.get("finish_reason")
    if finish is not None:
        return finish == "length"
    return (request.get("completion_tokens") or 0) >= max_tokens


def sample_step_times(entries: list[dict]) -> list[float]:
    """Step times stored with each sample, or derived from older reports that
    kept only single-stream requests."""
    stored = [entry["step_ms"] for entry in entries if entry.get("step_ms")]
    return stored if stored else step_times_ms(entries)


def convergence_values(entries: list[dict], metric: str) -> list[float]:
    values = sample_step_times(entries) if metric == "step_ms" else [entry.get(metric) for entry in entries]
    return [value for value in values if value is not None]


def sample_metrics(result: dict, concurrency: int) -> dict:
    """One decode sample: the primary numbers, measured and derived throughput,
    and the compact per-request receipts."""
    requests = result["requests"]
    ok = [request for request in requests if request["ok"]]
    tokens = sum(request["completion_tokens"] or 0 for request in ok)
    metrics = result["metrics"]
    drafts = metrics["spec_decode_num_drafts"]
    draft_tokens = metrics["spec_decode_num_draft_tokens"]
    accepted = acceptance(metrics)
    entry = {
        "tps": round(tokens / result["wall_s"], 3),
        "decode_window_tps": decode_window_tps(requests),
        "accepted_per_draft": accepted,
        # Adaptive verification trims draft rows per step; step time follows
        # the rows it keeps, so record them with each sample.
        "verified_per_draft": round(draft_tokens / drafts, 4) if drafts else None,
        # Primary numbers: the draft's hit rate on the rows it was given, the
        # tokens each engine step produced, and how long a step took. The
        # first two follow the generated text; step time stays stable.
        "accepted_per_verified": (
            round(metrics["spec_decode_num_accepted_tokens"] / draft_tokens, 4) if draft_tokens else None
        ),
        "tokens_per_step": None if accepted is None else round(1 + accepted, 4),
        "step_ms": step_time_ms(requests, accepted, concurrency),
        "derived_tps": None,
        "requests": compact(requests),
    }
    if entry["step_ms"] and entry["tokens_per_step"]:
        # What the primaries imply: every stream contributes its tokens each step.
        entry["derived_tps"] = round(1000 * entry["tokens_per_step"] * concurrency / entry["step_ms"], 3)
    return entry


_SCHEMA_VALIDATORS: dict[str, object] = {}


def schema_validator(schema: dict):
    """A cached Draft 2020-12 validator from python-jsonschema, the reference
    implementation. The schema itself is checked the first time."""
    try:
        import jsonschema
    except ImportError as error:
        raise SystemExit(
            "JSON decode cases need the jsonschema package: on Debian and Ubuntu "
            "`sudo apt install python3-jsonschema`; elsewhere "
            "`python3 -m pip install --require-hashes -r requirements/jsonschema.txt` (in Sonar's checkout)"
        ) from error
    key = json.dumps(schema, sort_keys=True)
    validator = _SCHEMA_VALIDATORS.get(key)
    if validator is None:
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = _SCHEMA_VALIDATORS[key] = jsonschema.Draft202012Validator(schema)
    return validator


def json_schema_violations(value, schema: dict) -> list[str]:
    """Schema violations as ``$.path: message`` lines, in document order."""
    def order(error):
        return [(0, part) if isinstance(part, int) else (1, part) for part in error.absolute_path]

    lines = []
    for error in sorted(schema_validator(schema).iter_errors(value), key=order):
        path = "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path)
        lines.append(f"{path}: {error.message}")
    return lines


def json_schema_for(spec: DecodeCase, index: int) -> dict | None:
    """The item schema for a case's index-th stream, following prompt cycling."""
    schemas = spec.json_schema
    if isinstance(schemas, tuple):
        return schemas[index % len(schemas)]
    return schemas


def json_array_progress(content: str, item_schema: dict | None = None) -> dict:
    """Completed elements of a (possibly truncated) JSON array, decoded one at a
    time with the standard library parser; whether the prefix up to the last
    complete element parses once closed; and, given an item schema, how many
    elements conform. The closed prefix is returned under ``json_prefix`` for
    the caller to save."""
    start = content.find("[")
    if start < 0:
        return {"json_objects": 0, "json_prefix_valid": False}
    decoder = json.JSONDecoder()
    index = start + 1
    items: list = []
    end = start + 1
    while True:
        while index < len(content) and content[index] in " \t\r\n,":
            index += 1
        if index >= len(content) or content[index] == "]":
            break
        try:
            value, index = decoder.raw_decode(content, index)
        except ValueError:
            break
        items.append(value)
        end = index
    result = {"json_objects": sum(1 for item in items if isinstance(item, dict)), "json_prefix_valid": bool(items)}
    if items:
        result["json_prefix"] = content[start:end] + "]"
    if item_schema is not None:
        violations = [json_schema_violations(item, item_schema) for item in items]
        result["json_valid_objects"] = sum(1 for found in violations if not found)
        result["json_violation"] = next(
            (f"$[{position}]{found[0][1:]}" for position, found in enumerate(violations) if found), None
        )
    return result


def save_output(directory: Path | None, point: str, tag: str, index: int, request: dict) -> None:
    """Keep a JSON case's raw text and, when it parses, the closed prefix for
    offline inspection (jq, a full schema validator)."""
    if directory is None:
        return
    folder = directory / "outputs" / point
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{tag}-stream{index + 1}"
    (folder / f"{name}.txt").write_text(request.get("content") or "", encoding="utf-8")
    if request.get("json_prefix"):
        (folder / f"{name}.json").write_text(request["json_prefix"] + "\n", encoding="utf-8")


def compact(results: list[dict]) -> list[dict]:
    keys = ("ok", "ttft_s", "first_s", "last_s", "prompt_tokens", "completion_tokens", "decode_tps",
            "output_sha256", "reasoning_chars", "content_chars", "error", "finish_reason", "json_objects",
            "json_prefix_valid", "json_valid_objects", "json_violation")
    return [{key: result[key] for key in keys if key in result} for result in results]


def assess_lru(content: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", content, re.DOTALL)
    if match is None:
        return "missing Python code block"
    try:
        tree = ast.parse(match.group(1))
    except SyntaxError as error:
        return f"Python syntax error: {error.msg} at line {error.lineno}"
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {item.name: item for item in node.body if isinstance(item, ast.FunctionDef)}
        if {"get", "put"} <= methods.keys():
            arguments = [argument.arg for argument in methods["put"].args.args]
            if arguments != ["self", "key", "value"]:
                return f"put signature uses {arguments!r}"
            return "pass"
    return "missing LRU class with get and put methods"


def suite_quality(bench: Sampler, options: argparse.Namespace, rng: random.Random) -> dict:
    """The fixed LRU request, five times; every answer must pass."""
    runs = []
    for index in range(5):
        sample = bench.run(
            f"quality {index + 1}/5",
            [chat_payload(options.model, LRU_PROMPT, 256, temperature=0, seed=42,
                          **reasoning_body(options))],
            keep_content=True,
        )
        result = sample["requests"][0]
        verdict = assess_lru(result["content"]) if result["ok"] else f"request failed: {result['error']}"
        runs.append({"verdict": verdict, "content": result["content"]})
        print(f"  quality {index + 1}/5: {verdict}", flush=True)
    passed = sum(run["verdict"] == "pass" for run in runs)
    return {"passed": passed, "runs": runs, "ok": passed == len(runs)}


COMPLIANCE_OBJECTS = 12
COMPLIANCE_MAX_TOKENS = 4096
COMPLIANCE_CONCURRENCY = 4


def compliance_payloads(model: str, constrained: bool, reasoning_off: dict | None = None) -> list[tuple[str, dict]]:
    """One request per record type, asking for a complete array of
    COMPLIANCE_OBJECTS objects, with room to finish naturally. The constrained
    arm sends the same prompt with the array schema as response_format."""
    payloads = []
    for subject, schema in PRECISION_JSON_RECORDS:
        extra: dict = dict(reasoning_off or {})
        if constrained:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "records",
                    "schema": {"type": "array", "items": schema,
                               "minItems": COMPLIANCE_OBJECTS, "maxItems": COMPLIANCE_OBJECTS},
                },
            }
        payloads.append(chat_payload(
            model, json_records_prompt(subject, schema, COMPLIANCE_OBJECTS), COMPLIANCE_MAX_TOKENS,
            temperature=0, seed=42, **extra,
        ))
    return payloads


def assess_json_records(content: str, schema: dict, count: int) -> dict:
    """Whole-output compliance: exactly a JSON array of ``count`` objects, each
    conforming to ``schema``, with no fence or commentary around it."""
    text = content.strip()
    fenced = text.startswith("```")
    if fenced:
        text = re.sub(r"^```[A-Za-z]*\s*|\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except ValueError as error:
        return {"parsed": False, "fenced": fenced, "objects": 0, "conforming": 0,
                "violation": f"not JSON: {error}", "ok": False}
    if not isinstance(parsed, list):
        return {"parsed": True, "fenced": fenced, "objects": 0, "conforming": 0,
                "violation": "top level is not an array", "ok": False}
    violations = [json_schema_violations(item, schema) for item in parsed]
    conforming = sum(1 for found in violations if not found)
    violation = next((f"$[{index}]{found[0][1:]}" for index, found in enumerate(violations) if found), None)
    if violation is None and len(parsed) != count:
        violation = f"{len(parsed)} objects, {count} requested"
    return {"parsed": True, "fenced": fenced, "objects": len(parsed), "conforming": conforming,
            "violation": violation, "ok": not fenced and violation is None}


def suite_compliance(bench: Sampler, options: argparse.Namespace, rng: random.Random) -> dict:
    """Structured-output compliance on natural completion.

    Two arms over the same eight record prompts: prompt-only, then the same
    requests constrained by the array schema through response_format. A
    request passes when it stopped on its own, parsed as exactly the requested
    number of objects, and every object conforms. This measures correctness,
    not speed; the decode suite's json cases measure speed.
    """
    output_dir = getattr(options, "output_dir", None)
    arms = {}
    for arm, constrained in (("prompt_only", False), ("constrained", True)):
        result = bench.run(f"compliance {arm}", compliance_payloads(options.model, constrained, reasoning_body(options)),
                           concurrency=COMPLIANCE_CONCURRENCY, keep_content=True)
        rows = []
        for (subject, schema), request in zip(PRECISION_JSON_RECORDS, result["requests"]):
            row = {"subject": subject, "ok": False, "finish_reason": request.get("finish_reason"),
                   "completion_tokens": request.get("completion_tokens"), "elapsed_s": request.get("elapsed_s")}
            if request["ok"]:
                row.update(assess_json_records(request.get("content") or "", schema, COMPLIANCE_OBJECTS))
                if request.get("finish_reason") == "length":
                    row["ok"] = False
                    row["violation"] = row["violation"] or "hit the token budget"
            else:
                row["error"] = request.get("error")
            rows.append(row)
            if output_dir is not None:
                folder = output_dir / "outputs" / f"compliance-{arm}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{subject.replace(' ', '-')}.txt").write_text(request.get("content") or "", encoding="utf-8")
            print(f"  compliance {arm} {subject}: {'pass' if row['ok'] else row.get('violation') or row.get('error')}", flush=True)
        metrics = result["metrics"]
        draft_tokens = metrics["spec_decode_num_draft_tokens"]
        arms[arm] = {
            "requests": rows,
            "passed": sum(row["ok"] for row in rows),
            "failed_requests": sum(1 for request in result["requests"] if not request["ok"]),
            "accepted_per_draft": acceptance(metrics),
            "accepted_per_verified": (
                round(metrics["spec_decode_num_accepted_tokens"] / draft_tokens, 4) if draft_tokens else None
            ),
            "elapsed_s": describe([row["elapsed_s"] for row in rows if row.get("elapsed_s")]),
        }
    return {"arms": arms, "objects_requested": COMPLIANCE_OBJECTS, "max_tokens": COMPLIANCE_MAX_TOKENS,
            "ok": all(arm["passed"] == len(arm["requests"]) for arm in arms.values())}


def decode_plan(options: argparse.Namespace, point: str) -> tuple[DecodeCase, int, str, int, int]:
    """A point's case, concurrency, convergence metric and sample bounds.

    --converge-on wins when given; otherwise the case decides, then tps. The
    case's sampling floors apply when they exceed the command-line values.
    """
    case, concurrency = point.rsplit("-c", 1)
    spec = DECODE_CASES[case]
    metric = getattr(options, "converge_on", None) or spec.converge_on or "tps"
    minimum = max(options.min_samples, spec.min_samples or 0)
    maximum = max(options.max_samples, minimum, spec.max_samples or 0)
    return spec, int(concurrency), metric, minimum, maximum


def suite_decode(bench: Sampler, options: argparse.Namespace, rng: random.Random) -> dict:
    """Temperature-0 serving matrix, sampled until every point is precise.

    Each point reports three primary numbers (step_ms, accepted_per_verified,
    tokens_per_step), the measured throughput (tps and decode_window_tps) and
    derived_tps, the throughput the primaries imply. Only step_ms is stable
    across samples: acceptance follows the generated text, which does not
    repeat. A point keeps sampling until the metric it converges on has a
    confidence interval within --precision.
    """
    points = [f"{case}-c{concurrency}" for case in options.decode_cases for concurrency in options.concurrency]

    output_dir = getattr(options, "output_dir", None)

    def sample(point: str, tag: str) -> dict:
        spec, concurrency, _, _, _ = decode_plan(options, point)
        case = point.rsplit("-c", 1)[0]
        scored = spec.json_schema is not None
        result = bench.run(point, decode_payloads(options.model, case, concurrency, reasoning_body(options)),
                           keep_content=scored)
        if scored:
            for index, request in enumerate(result["requests"]):
                request.update(json_array_progress(request.get("content") or "", json_schema_for(spec, index)))
                save_output(output_dir, point, tag, index, request)
        return result

    print("  warmup: one discarded sample per point", flush=True)
    for point in rng.sample(points, len(points)):
        sample(point, "warmup")
    samples: dict[str, list[dict]] = {point: [] for point in points}
    active = list(points)
    rounds = 0
    while active:
        rounds += 1
        rng.shuffle(active)
        for point in active:
            _, concurrency, _, _, _ = decode_plan(options, point)
            samples[point].append(sample_metrics(sample(point, f"sample{len(samples[point]) + 1:02d}"), concurrency))
        remaining = []
        status = []
        for point in active:
            _, _, metric, minimum, maximum = decode_plan(options, point)
            summary = describe(convergence_values(samples[point], metric))
            interval = summary.get("ci95_pct")
            if summary["n"] >= maximum:
                continue
            if summary["n"] < minimum or interval is None or interval > options.precision:
                remaining.append(point)
                status.append(f"{point} {metric} ±{'-' if interval is None else interval}%")
        print(f"  round {rounds}: {', '.join(sorted(status)) or 'all points converged'}", flush=True)
        active = remaining
    summary = {}
    for point in points:
        spec, _, metric, _, _ = decode_plan(options, point)
        entries = samples[point]
        requests = [request for entry in entries for request in entry["requests"]]
        ok = [request for request in requests if request["ok"]]
        summary[point] = {
            # Primary numbers.
            "step_ms": describe(sample_step_times(entries)),
            "accepted_per_verified": describe(convergence_values(entries, "accepted_per_verified")),
            "tokens_per_step": describe(convergence_values(entries, "tokens_per_step")),
            # Measured throughput, then what the primaries imply.
            "tps": describe([entry["tps"] for entry in entries]),
            "decode_window_tps": describe([entry["decode_window_tps"] for entry in entries if entry.get("decode_window_tps")]),
            "derived_tps": describe(convergence_values(entries, "derived_tps")),
            "per_stream_decode_tps": describe([request["decode_tps"] for request in ok if request["decode_tps"]]),
            "ttft_s": describe([request["ttft_s"] for request in ok]),
            "accepted_per_draft": describe([entry["accepted_per_draft"] for entry in entries if entry["accepted_per_draft"] is not None]),
            "verified_per_draft": describe([entry["verified_per_draft"] for entry in entries if entry.get("verified_per_draft") is not None]),
            "converge_on": metric,
            "max_tokens": spec.max_tokens,
            # Outputs that used the whole budget. A precision case wants every
            # one: a prompt that stops early measures a shorter workload.
            "full_length_outputs": sum(1 for request in ok if used_whole_budget(request, spec.max_tokens)),
            "failed_requests": len(requests) - len(ok),
            "distinct_outputs": len({request["output_sha256"] for request in ok}),
            "reasoning_share": reasoning_share(ok),
        }
        if spec.json_schema is not None:
            scored = [request for request in ok if "json_objects" in request]
            objects = sum(request["json_objects"] for request in scored)
            summary[point]["json_objects"] = describe([request["json_objects"] for request in scored])
            summary[point]["json_prefix_valid"] = (
                round(sum(1 for request in scored if request["json_prefix_valid"]) / len(scored), 3)
                if scored else None
            )
            # Conforming objects over completed objects, across every stream.
            summary[point]["json_schema_valid"] = (
                round(sum(request.get("json_valid_objects", 0) for request in scored) / objects, 3) if objects else None
            )
            summary[point]["json_violation_example"] = next(
                (request["json_violation"] for request in scored if request.get("json_violation")), None
            )
    return {"points": summary, "samples": samples}


def reasoning_share(requests: list[dict]) -> float | None:
    """Fraction of streamed characters that were reasoning, not answer."""
    reasoning = sum(request.get("reasoning_chars") or 0 for request in requests)
    total = reasoning + sum(request.get("content_chars") or 0 for request in requests)
    return round(reasoning / total, 3) if total else None


def suite_sampled(bench: Sampler, options: argparse.Namespace, rng: random.Random) -> dict:
    """Speculative-decoding acceptance on temperature-1.0 traffic, from the engine counters."""
    points = {}
    for case, prompt in SAMPLED_PROMPTS.items():
        for concurrency in (1, 4):
            label = f"{case}-c{concurrency}"
            payloads = [
                chat_payload(options.model, prompt, 256, temperature=1.0, top_p=0.95,
                             seed=1000 + index, **reasoning_body(options))
                for index in range(options.sampled_requests)
            ]
            result = bench.run(label, payloads, concurrency=concurrency)
            requests = result["requests"]
            tokens = sum(request["completion_tokens"] or 0 for request in requests if request["ok"])
            metrics = result["metrics"]
            draft_tokens = metrics["spec_decode_num_draft_tokens"]
            points[label] = {
                "requests": len(requests),
                "failed_requests": sum(not request["ok"] for request in requests),
                "aggregate_tps": round(tokens / result["wall_s"], 2),
                "accepted_per_draft": acceptance(metrics),
                "acceptance_rate": round(metrics["spec_decode_num_accepted_tokens"] / draft_tokens, 4) if draft_tokens else None,
                "drafts": metrics["spec_decode_num_drafts"],
            }
            print(f"  {label}: {points[label]['accepted_per_draft']} accepted per draft", flush=True)
    return {"points": points}


def suite_prefill(bench: Sampler, options: argparse.Namespace, rng: random.Random) -> dict:
    """Cold prefill: unique uncached prompts, one output token."""
    prefill_text = {"filler": filler, "novel": novel_text, "source": source_text}[options.prefill_text]
    sizes = [size for size in options.prefill_sizes if size < options.max_model_len - 64]
    samples: dict[str, list[dict]] = {str(size): [] for size in sizes}
    for _ in range(options.prefill_repeats):
        for size in rng.sample(sizes, len(sizes)):
            payload = {
                "model": options.model,
                "prompt": prefill_text(int(size * 0.97), rng.randrange(1 << 30)) + "\n\nReply with the word ok.",
                "max_tokens": 1,
                "temperature": 0,
                "cache_salt": uuid.uuid4().hex,
            }
            request = bench.run(f"prefill {size}", [("/v1/completions", payload)])["requests"][0]
            samples[str(size)].append(compact([request])[0])
            if request["ok"]:
                print(f"  prefill {size}: {request['prompt_tokens'] / request['ttft_s']:.0f} tok/s", flush=True)
    summary = {}
    for size, requests in samples.items():
        ok = [request for request in requests if request["ok"]]
        summary[size] = {
            "prefill_tps": describe([request["prompt_tokens"] / request["ttft_s"] for request in ok]),
            "ttft_s": describe([request["ttft_s"] for request in ok]),
            "prompt_tokens": ok[0]["prompt_tokens"] if ok else None,
            "failed_requests": len(requests) - len(ok),
        }
    return {"points": summary, "samples": samples}


def suite_prefix(bench: Sampler, options: argparse.Namespace, rng: random.Random) -> dict:
    """Prefix-cache replay: one cold request, then two identical replays."""
    runs = []
    for index in range(3):
        payload = {
            "model": options.model,
            "prompt": filler(int(options.prefix_tokens * 0.97), rng.randrange(1 << 30)) + "\n\nReply with the word ok.",
            "max_tokens": 1,
            "temperature": 0,
            "cache_salt": uuid.uuid4().hex,
        }
        cold = bench.run(f"prefix cold {index + 1}/3", [("/v1/completions", payload)])
        warm = [bench.run(f"prefix warm {index + 1}/3", [("/v1/completions", payload)]) for _ in range(2)]
        queries = sum(sample["metrics"]["prefix_cache_queries"] for sample in warm)
        hits = sum(sample["metrics"]["prefix_cache_hits"] for sample in warm)
        runs.append({
            "cold_ttft_s": cold["requests"][0]["ttft_s"],
            "warm_ttft_s": [sample["requests"][0]["ttft_s"] for sample in warm],
            "warm_hit_rate": round(hits / queries, 4) if queries else None,
            "ok": all(sample["requests"][0]["ok"] for sample in [cold, *warm]),
        })
        print(f"  prefix {index + 1}/3: cold {runs[-1]['cold_ttft_s']} s, warm {runs[-1]['warm_ttft_s']} s", flush=True)
    return {
        "prompt_tokens": options.prefix_tokens,
        "cold_ttft_s": describe([run["cold_ttft_s"] for run in runs if run["ok"]]),
        "warm_ttft_s": describe([value for run in runs if run["ok"] for value in run["warm_ttft_s"]]),
        "warm_hit_rate": describe([run["warm_hit_rate"] for run in runs if run["warm_hit_rate"] is not None]),
        "runs": runs,
    }


def suite_admission(bench: Sampler, options: argparse.Namespace, rng: random.Random) -> dict:
    """Four concurrent long contexts must all be admitted without preemption."""
    tokens = min(options.admission_tokens, options.max_model_len - options.admission_output_tokens - 1024)
    payloads = [
        chat_payload(
            options.model,
            filler(int(tokens * 0.97), rng.randrange(1 << 30))
            + "\n\nWrite a 400-word story that uses words from the document above.",
            options.admission_output_tokens,
            temperature=0,
            seed=42,
            **reasoning_body(options),
        )
        for _ in range(4)
    ]
    for _, payload in payloads:
        payload["cache_salt"] = uuid.uuid4().hex
        if options.admission_force_length:
            payload.update(min_tokens=options.admission_output_tokens, ignore_eos=True)
    result = bench.run("admission 4 x %d" % tokens, payloads)
    requests = result["requests"]
    ok = [request for request in requests if request["ok"]]
    admitted = (
        len(ok) == len(requests)
        and not result["metrics"]["num_preemptions"]
        and result["peak"]["running"] >= len(requests)
    )
    print(f"  admission: {'pass' if admitted else 'FAIL'}; peak KV {result['peak']['kv_cache_usage']:.0%}", flush=True)
    return {
        "context_tokens": tokens,
        "output_tokens": options.admission_output_tokens,
        "force_output_length": options.admission_force_length,
        "ok": admitted,
        "preemptions": result["metrics"]["num_preemptions"],
        "peak_running": result["peak"]["running"],
        "peak_kv_cache_usage": round(result["peak"]["kv_cache_usage"], 4),
        "wall_s": round(result["wall_s"], 2),
        "ttft_s": [request["ttft_s"] for request in requests],
        "per_stream_decode_tps": describe([request["decode_tps"] for request in ok if request["decode_tps"]]),
        "requests": compact(requests),
    }


SUITE_FUNCTIONS = {
    "quality": suite_quality,
    "compliance": suite_compliance,
    "decode": suite_decode,
    "sampled": suite_sampled,
    "prefill": suite_prefill,
    "prefix": suite_prefix,
    "admission": suite_admission,
}

def decode_primary_rows(points: dict) -> list[str]:
    """Primary numbers and both throughputs per decode point, plus warnings for
    outputs that stopped before a precision budget."""
    def cell(stat: dict | None, digits: int) -> str:
        if not stat or not stat.get("n"):
            return "-"
        return f"{stat['mean']:.{digits}f} ±{stat['ci95_pct'] or 0:.1f}%"

    rows = [(point, summary) for point, summary in points.items() if summary.get("tokens_per_step", {}).get("n")]
    if not rows:
        return []
    lines = [
        "decode primaries (mean ±95% CI): step ms | accepted/verified | tokens/step "
        "|| tok/s measured aggregate / decode window | derived"
    ]
    for point, summary in rows:
        lines.append(
            f"  {point:<18} {cell(summary.get('step_ms'), 2):>15} | {cell(summary['accepted_per_verified'], 3):>14} | "
            f"{cell(summary['tokens_per_step'], 3):>14} || {cell(summary['tps'], 1):>14} / "
            f"{cell(summary.get('decode_window_tps'), 1):>14} | {cell(summary.get('derived_tps'), 1):>14}"
        )
        budget = summary.get("max_tokens") or 0
        concurrency = int(point.rsplit("-c", 1)[1])
        completed = summary["tps"]["n"] * concurrency - summary.get("failed_requests", 0)
        full = summary.get("full_length_outputs", completed)
        if budget > DECODE_TOKENS and full < completed:
            lines.append(
                f"  WARN {point}: {completed - full} of {completed} outputs stopped before {budget} tokens; "
                "lengthen the prompt or the point measures a shorter workload"
            )
        if "json_prefix_valid" in summary:
            lines.append(
                f"  {point:<18} JSON: {cell(summary.get('json_objects'), 1)} completed objects per output, "
                f"prefix parses {summary['json_prefix_valid']}, objects conform {summary.get('json_schema_valid')}"
            )
            if summary.get("json_violation_example"):
                lines.append(f"  {point:<18} first violation: {summary['json_violation_example']}")
            if (summary["json_prefix_valid"] or 0) < 1:
                lines.append(f"  WARN {point}: some outputs are not JSON arrays; see outputs/{point}/ in the report directory")
    return lines


def rows(report: dict, reference: dict | None, tolerance: float) -> tuple[list[str], list[str]]:
    """Summary lines and regressions against the reference run, suite by suite."""
    lines: list[str] = []
    regressions: list[str] = []
    suites = report["suites"]
    ref_suites = (reference or {}).get("suites", {})

    def compare(label: str, new: dict, ref: dict | None, unit: str) -> None:
        text = f"  {label:<18} {new['mean']:>9.1f} ±{new['ci95_pct'] or 0:>4.1f}% n={new['n']:<3}"
        change = difference(new, ref) if ref else None
        if change:
            mean, low, high = change
            verdict = "faster" if low > 0 else "slower" if high < 0 else "same"
            text += f" ref {ref['mean']:>8.1f}  {mean:+5.1f}% [{low:+.1f}, {high:+.1f}] {verdict}"
            if verdict == "slower" and mean < -tolerance:
                regressions.append(f"{label} {unit} {mean:+.1f}%")
        lines.append(text)

    if "quality" in suites:
        quality = suites["quality"]
        lines.append(f"quality: LRU {quality['passed']}/{len(quality['runs'])}")
    if "compliance" in suites:
        compliance = suites["compliance"]
        for arm, summary in compliance["arms"].items():
            lines.append(
                f"compliance {arm}: {summary['passed']}/{len(summary['requests'])} complete and conforming, "
                f"accepted/verified {summary.get('accepted_per_verified')}, "
                f"completion p-max {summary['elapsed_s'].get('max')} s"
            )
            for row in summary["requests"]:
                if not row["ok"]:
                    lines.append(f"  {row['subject']}: {row.get('violation') or row.get('error')}")
    if "decode" in suites:
        lines.append("decode (aggregate tok/s, 95% CI):")
        ref_points = ref_suites.get("decode", {}).get("points", {})
        for point, summary in suites["decode"]["points"].items():
            if summary["tps"]["n"]:
                compare(point, summary["tps"], ref_points.get(point, {}).get("tps"), "tok/s")
        lines.extend(decode_primary_rows(suites["decode"]["points"]))
        ref_samples = ref_suites.get("decode", {}).get("samples", {})
        steps = [
            (point, describe(sample_step_times(entries)))
            for point, entries in suites["decode"].get("samples", {}).items()
        ]
        steps = [(point, summary) for point, summary in steps if summary["n"]]
        if steps:
            lines.append("step time (ms, lower is better; independent of acceptance; "
                         "concurrent points are decode-window estimates):")
        for point, summary in steps:
            ref = describe(sample_step_times(ref_samples.get(point, [])))
            text = f"  {point:<18} {summary['mean']:>9.2f} ±{summary['ci95_pct'] or 0:>4.1f}% n={summary['n']:<3}"
            change = difference(summary, ref) if ref["n"] else None
            if change:
                mean, low, high = change
                verdict = "faster" if high < 0 else "slower" if low > 0 else "same"
                text += f" ref {ref['mean']:>8.2f}  {mean:+5.1f}% [{low:+.1f}, {high:+.1f}] {verdict}"
                if verdict == "slower" and mean > tolerance:
                    regressions.append(f"{point} step time {mean:+.1f}%")
            lines.append(text)
    if "sampled" in suites:
        lines.append("sampled (accepted drafts per step, temperature 1.0):")
        ref_points = ref_suites.get("sampled", {}).get("points", {})
        for point, summary in suites["sampled"]["points"].items():
            ref = ref_points.get(point, {}).get("accepted_per_draft")
            suffix = f" ref {ref}" if ref is not None else ""
            lines.append(f"  {point:<18} {summary['accepted_per_draft']}  {summary['aggregate_tps']} tok/s{suffix}")
    if "prefill" in suites:
        lines.append("prefill (tok/s):")
        ref_points = ref_suites.get("prefill", {}).get("points", {})
        for size, summary in suites["prefill"]["points"].items():
            if summary["prefill_tps"]["n"]:
                compare(f"{size} tokens", summary["prefill_tps"], ref_points.get(size, {}).get("prefill_tps"), "prefill tok/s")
    if "prefix" in suites:
        prefix = suites["prefix"]
        lines.append(
            f"prefix replay: cold {prefix['cold_ttft_s'].get('mean')} s, warm "
            f"{prefix['warm_ttft_s'].get('mean')} s, hit rate {prefix['warm_hit_rate'].get('mean')}"
        )
    if "admission" in suites:
        admission = suites["admission"]
        lines.append(
            f"admission 4 x {admission['context_tokens']}: {'pass' if admission['ok'] else 'FAIL'}, "
            f"peak KV {admission['peak_kv_cache_usage']:.0%}, per-stream "
            f"{admission['per_stream_decode_tps'].get('mean')} tok/s"
        )
    return lines, regressions


# Driving the matrix: validation, the workload record, the suite loop and the verdicts.

def int_list(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers: {value!r}") from error


def case_list(value: str) -> list[str]:
    return [case.strip() for case in value.split(",") if case.strip()]


def prepare(options: argparse.Namespace, max_num_seqs: int) -> list[str]:
    """Fill the defaults that depend on --full and the server, check every option, and
    return the suites to run. ``options`` needs model and max_model_len set."""
    if options.suites is None:
        options.suites = ",".join(FULL_SUITES) if options.full else "quality,decode"
    if options.min_samples is None:
        options.min_samples = 6 if options.full else 3
    if options.max_samples is None:
        options.max_samples = 40 if options.full else 4
    suites = [suite.strip() for suite in options.suites.split(",") if suite.strip()]
    unknown = sorted(set(suites) - set(SUITES))
    if unknown:
        raise SystemExit(f"unknown suites: {', '.join(unknown)}")
    if options.concurrency is None:
        options.concurrency = [c for c in (1, 2, 4, 8, 16) if c <= max_num_seqs]
    if not options.concurrency or max(options.concurrency) > max_num_seqs or min(options.concurrency) < 1:
        raise SystemExit(f"--concurrency values must be between 1 and --max-num-seqs ({max_num_seqs})")
    options.decode_cases = list(dict.fromkeys(
        expanded for case in options.decode_cases for expanded in DECODE_CASE_GROUPS.get(case, [case])
    ))
    unknown_cases = sorted(set(options.decode_cases) - set(DECODE_CASES))
    if unknown_cases:
        raise SystemExit(f"unknown decode cases: {', '.join(unknown_cases)}")
    if options.converge_on is not None and options.converge_on not in CONVERGENCE_METRICS:
        raise SystemExit(f"--converge-on must be one of {', '.join(CONVERGENCE_METRICS)}")
    if options.prefill_text not in ("filler", "novel", "source"):
        raise SystemExit("--prefill-text must be filler, novel or source")
    if "compliance" in suites or any(DECODE_CASES[case].json_schema is not None for case in options.decode_cases):
        schema_validator({"type": "object"})
    if options.min_samples < 2 or options.max_samples < options.min_samples:
        raise SystemExit("--min-samples must be at least 2 and no more than --max-samples")
    if options.admission_output_tokens < 1 or options.admission_output_tokens + 1024 >= options.max_model_len:
        raise SystemExit("--admission-output-tokens must be positive and leave room for the prompt")
    return suites


def prompts_digests() -> dict:
    """SHA-256 of the fixed prompt sets, so two reports show they sent the same texts."""
    return {
        "prompts_sha256": hashlib.sha256(
            json.dumps([DECODE_PROMPTS, SAMPLED_PROMPTS, LRU_PROMPT], sort_keys=True).encode()
        ).hexdigest(),
        "precision_prompts_sha256": hashlib.sha256(
            json.dumps(
                [PRECISION_PROSE, PRECISION_JSON_PROMPTS, PRECISION_JSON_SCHEMAS, PRECISION_JSON_SCHEMA],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def workload(options: argparse.Namespace, url: str, suites: list[str], **extra) -> dict:
    """The report's workload record. ``extra`` adds the caller's fields (after the options,
    before the prompt digests)."""
    record = {
        "url": url,
        "suites": suites,
        "seed": options.seed,
        "precision_pct": options.precision,
        "min_samples": options.min_samples,
        "max_samples": options.max_samples,
        "concurrency": options.concurrency,
        "decode_cases": options.decode_cases,
        "converge_on": options.converge_on,
        "prefill_text": options.prefill_text,
        "sampled_requests": options.sampled_requests,
        "prefill_sizes": options.prefill_sizes,
        "prefill_repeats": options.prefill_repeats,
        "prefix_tokens": options.prefix_tokens,
        "admission_tokens": options.admission_tokens,
        "admission_output_tokens": options.admission_output_tokens,
        "admission_force_length": options.admission_force_length,
        **extra,
        **prompts_digests(),
    }
    if reasoning_body(options):
        record["reasoning_off"] = reasoning_body(options)
    return record


def run_suites(sampler: Sampler, options: argparse.Namespace, suites: list[str], report: dict, save) -> None:
    """Every selected suite in the fixed order, saving the report after each. A failed quality
    gate stops the run: the performance numbers after it would be meaningless."""
    rng = random.Random(options.seed)
    for suite in SUITES:
        if suite not in suites:
            continue
        print(f"{suite}:", flush=True)
        report["suites"][suite] = SUITE_FUNCTIONS[suite](sampler, options, rng)
        save()
        if suite == "quality" and not report["suites"][suite]["ok"]:
            raise Abort("the quality gate failed; performance numbers would be meaningless")


def failures(report: dict) -> list[str]:
    """What makes a report fail on its own: an abort, a failed gate, failed requests."""
    out = []
    if report.get("aborted"):
        out.append(f"aborted: {report['aborted']}")
    if "quality" in report["suites"] and not report["suites"]["quality"]["ok"]:
        out.append("quality gate failed")
    if "admission" in report["suites"] and not report["suites"]["admission"]["ok"]:
        out.append("admission failed")
    if "compliance" in report["suites"] and not report["suites"]["compliance"]["ok"]:
        out.append("compliance failed")
    failed_requests = sum(
        point.get("failed_requests", 0)
        for suite in ("decode", "sampled", "prefill")
        for point in report["suites"].get(suite, {}).get("points", {}).values()
    )
    failed_requests += sum(
        arm.get("failed_requests", 0)
        for arm in report["suites"].get("compliance", {}).get("arms", {}).values()
    )
    if failed_requests:
        out.append(f"{failed_requests} requests failed")
    return out


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The matrix's measurement options (``sonar matrix``; a caller may define its own)."""
    parser.add_argument("--full", action="store_true",
                        help="every suite but compliance, each decode point sampled to --precision; "
                             "the default is the quality gate and decode with 3-4 samples")
    parser.add_argument("--suites", help="comma-separated subset of: " + ", ".join(SUITES))
    parser.add_argument("--precision", type=float, default=2.0,
                        help="sample each decode point until its 95%% CI is within this percent of the mean")
    parser.add_argument("--min-samples", type=int, help="default 3, or 6 with --full")
    parser.add_argument("--max-samples", type=int, help="default 4, or 40 with --full")
    parser.add_argument("--concurrency", type=int_list, default=None,
                        help="comma-separated decode concurrencies (default: 1,2,4,8,16 up to --max-num-seqs)")
    parser.add_argument("--decode-cases", type=case_list, default=["prose", "code"],
                        help=f"comma-separated decode cases (default prose,code): {', '.join(DECODE_CASES)}; "
                             f"groups: {', '.join(DECODE_CASE_GROUPS)}")
    parser.add_argument("--converge-on", choices=CONVERGENCE_METRICS,
                        help="the per-sample metric whose 95%% CI must reach --precision before a decode point "
                             "stops sampling; default: the case's choice (step_ms for the precision group), "
                             "otherwise tps")
    parser.add_argument("--prefill-text", choices=("filler", "novel", "source"), default="filler",
                        help="cold-prefill prompt text: the repeating filler (default), random pseudo-words "
                             "(n-grams no cache has seen), or real text (Python standard-library source)")
    parser.add_argument("--sampled-requests", type=int, default=32)
    parser.add_argument("--prefill-sizes", type=int_list, default=[2048, 32768, 65536, 131072])
    parser.add_argument("--prefill-repeats", type=int, default=3)
    parser.add_argument("--prefix-tokens", type=int, default=32768)
    parser.add_argument("--admission-tokens", type=int, default=65536)
    parser.add_argument("--admission-output-tokens", type=int, default=256,
                        help="output budget per admission request; increase for overlapping long prefills")
    parser.add_argument("--admission-force-length", action="store_true",
                        help="force the full admission output budget to keep long contexts active together")
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="fail when a point is significantly slower than the reference by more than this percent")
    parser.add_argument("--seed", type=int, default=0, help="seed for sample order and prompts")
    parser.add_argument("--reasoning-off", type=json.loads, default=None, metavar="JSON",
                        help="what turns the server's reasoning off, merged into the reasoning-off cases' bodies "
                             "(for example '{\"chat_template_kwargs\": {\"thinking\": false}}')")


def add_parser(sub) -> None:
    parser = sub.add_parser("matrix", help="decode points sampled to a precision, with quality, compliance, "
                                           "prefill, prefix and admission checks")
    parser.add_argument("url", help="base URL, e.g. http://127.0.0.1:8000")
    add_arguments(parser)
    parser.add_argument("--model", help="model id (default: the server's first)")
    parser.add_argument("--max-num-seqs", type=int, default=16,
                        help="the server's concurrent request limit (bounds --concurrency)")
    parser.add_argument("--max-model-len", type=int, help="the server's context limit (default: its /v1/models)")
    parser.add_argument("--output", metavar="DIR", help="report directory (matrix.json; JSON cases' outputs)")
    parser.add_argument("--compare", metavar="PATH", help="a reference report to compare with")
    parser.add_argument("--report", metavar="PATH", help="summarize an existing report against --compare instead")
    parser.set_defaults(func=command)


def command(args: argparse.Namespace) -> int:
    from . import __version__, engine
    reference = json.loads(Path(args.compare).read_text(encoding="utf-8")) if args.compare else None
    if args.report:
        lines, regressions = rows(json.loads(Path(args.report).read_text(encoding="utf-8")), reference,
                                  args.tolerance)
        print("\n".join(lines))
        for regression in regressions:
            print(f"FAIL: regression: {regression}")
        return 1 if regressions else 0
    base = args.url.rstrip("/")
    if args.model is None:
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            args.model = json.load(response)["data"][0]["id"]
    args.max_model_len = args.max_model_len or engine.max_model_len(base)
    if not args.max_model_len:
        raise SystemExit("the server reports no max_model_len; pass --max-model-len")
    suites = prepare(args, args.max_num_seqs)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = Path(args.output or f"matrix-{stamp}")
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = output
    report_path = output / "matrix.json"
    report = {"schema": SCHEMA, "tool": {"name": "sonar", "version": __version__}, "started_utc": stamp,
              "workload": workload(args, base, suites), "reference": args.compare, "suites": {}}

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"measuring {base}; writing {report_path}", flush=True)
    sampler = Sampler(base)
    try:
        run_suites(sampler, args, suites, report, save)
    except Abort as abort:
        report["aborted"] = str(abort)
    except OSError as error:
        sampler.halt(str(error))
        report["aborted"] = f"API or telemetry error: {error}"
    except KeyboardInterrupt:
        sampler.halt("interrupted")
        report["aborted"] = "interrupted"
    finally:
        report["discarded_samples"] = sampler.discarded
        report["finished_utc"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        save()
    lines, regressions = rows(report, reference, args.tolerance)
    print("\n".join(lines))
    problems = failures(report) + [f"regression: {item}" for item in regressions]
    for problem in problems:
        print(f"FAIL: {problem}")
    print(f"report: {report_path}")
    return 1 if problems else 0
