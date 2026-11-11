# Sonar

Measurements and verdicts for OpenAI-compatible LLM servers: steady-state load, the
engine's own step cost, what the model computes, long-prompt prefill and the stalls it
causes, and a verdict that says whether a change is real once boot-to-boot noise is
counted.

Sonar is standard-library Python (3.11 or later). It runs from a checkout or a `pip
install`, against any server that speaks the OpenAI chat and completions API; with vLLM it
also reads the server's Prometheus counters, which split speed into engine steps per second
and tokens per step. It knows nothing about the deployment it measures: a caller that does
records the deployment's identity beside the measurements.

```sh
python3 -m sonar run http://127.0.0.1:8000 --mode closed --concurrency 16 --output c16.json
python3 -m sonar suite http://127.0.0.1:8000 --suite standard --out runs/candidate --label candidate
python3 -m sonar table runs/candidate runs/base runs/candidate-again --floors floors.json
```

## Why

A benchmark that starts N requests at once and times them from the first token to the last
measures a ramp (all prefills together) and a tail (fast streams finish first, so the batch
drains toward one): a "16-stream" number is partly an 8-, 4- and 1-stream number. A few
prompts tie speculative decoding's acceptance to a few texts, so a change in the text (a
precision change) reads as a change in speed. And one run of each build is one draw of the
boot: on a multi-node server the same build moves by a few tenths of a percent between
boots, more than many changes being judged.

Sonar measures in a steady state, separates the engine's speed from the text's, pairs
requests across builds so the text cancels, and judges a change against a recorded A/A
floor. The methods follow MLPerf Inference's server scenario, GuideLLM and AIPerf.
[docs/methodology.md](docs/methodology.md) has the details.

## Commands

| Command | What it measures |
| --- | --- |
| `run` | Steady-state load: `closed` (N in flight), `poisson` (open-loop arrivals, with SLO goodput) or `burst`; optionally at depth (`--context`). Throughput over sub-windows with intervals, server step rate and tokens per step, latency percentiles, and a per-request table for pairing. |
| `engine` | The engine cost grid: the server times identical fixed-shape dummy steps (needs a server that exposes `engine_grid` to its dev-mode collective RPC). No text, acceptance or scheduling in it. |
| `model` | What a build computes, with no timing: greedy generation with token ids and speculative acceptance by draft position, then teacher-forced log-probabilities (KL, top-1 agreement, perplexity) on its own or another run's text. |
| `prefill` | Cold prefill rate on real-text prompts of fixed sizes, calibrated with the server's tokenizer. |
| `stall` | What decoding streams wait while one long cold prompt prefills, with optional short probes' time to first token. |
| `suite` | A fixed, named set of the above (data, with a SHA-256), into one directory; `--metadata` records the caller's identity for the run. |
| `compare` | Two results, or two suite runs step by step: Welch intervals, exact paired intervals, tolerances for metrics without one. |
| `table` | The verdict: candidate, reference, candidate again. Headline rows judged against the candidate's own repeat and a recorded A/A floor. |
| `spread` | Boot-to-boot spread of engine grids, one grid a boot. |
| `determinism` | Temperature-0 determinism: the same prompts alone, eight at once, staggered, beside a long prefill, cached, fifteen at once, and again; token and log-probability equality, with digests. |
| `streams`, `single`, `acceptance` | Quick looks: N distinct prompts at once, one stream over many texts, speculative acceptance per prompt. |
| `ttft`, `mixed` | Short-prompt time to first token; latency under mixed traffic (decoding streams, long prompts and short requests on a fixed schedule). |
| `matrix` | Decode points (case x concurrency) sampled until each is precise, with quality, structured-output compliance, sampled acceptance, prefill, prefix-cache and admission checks. |
| `watch` | A window of a server's traffic described from its counters alone (no requests sent). |

Every command takes the server's base URL first and prints a one-line JSON summary last.
`python3 -m sonar COMMAND --help` lists the options.

## Suites

A suite is data: steps, each a command with its arguments. A run writes every step's result
and `suite.json` (the suite, its SHA-256 and the caller's metadata). `compare` and `table`
refuse runs of different suites: a changed suite is a new suite. Built in (`sonar suites`
lists them with their digests):

- `smoke`, about 5 minutes: one and four streams, cold prefill at 8K.
- `standard`, about 40 minutes: model behaviour, closed loops at 1, 4 and 16 streams, prefill
  from 8K to 128K, stalls during a 32K prefill.
- `engine`, about 25 minutes, for kernel and engine changes: model behaviour, the engine grid,
  sixteen streams, prefill at 32K and 128K.

A deployment usually keeps its own suites, sized to its limits (streams, context, arrival
rates) and with its chat template's switches in `extra_body`; pass the file to `--suite`.

## Judging a change

Boot the candidate, run the suite; boot the reference, run it (or reuse a recorded run of the
reference: a released build's numbers do not change); boot the candidate again, run it again.

```sh
python3 -m sonar table runs/candidate runs/reference runs/candidate-again --floors floors.json
```

The verdict block judges reference to candidate against the candidate's repeat. The closed
loops' verdict row, `decode tok/s, paired`, pairs each request with the same prompt on the
other build and takes the median ratio over both candidate boots; its margin is the interval
over requests and the A/A floor in quadrature. The floor (`--floors`, a JSON object of step id
to percent) is the 95% bound of one boot against another for that step, measured on your own
hardware from repeated runs of one build: Sonar cannot know it for you. Without floors the
paired rows are not judged, since their own intervals (a few hundredths of a percent) would
call a build different from itself about half the time. An engine grid step with a floor is
judged the same way. Boot the reference again too (`--reference-repeat`) and every row becomes
the mean over both boots of each side, with the boot term down to x0.707 of one pair's.

## Server features used

- vLLM request extras (`cache_salt`, `ignore_eos`, `min_tokens`, per-chunk usage with
  `continuous_usage_stats`) unless `--no-vllm`;
- `/metrics` counters and gauges (`vllm:*`) for step rates, acceptance, prefix-cache hits,
  foreign-request detection and idle checks;
- `/tokenize` to size documents, and `prompt_logprobs` for teacher-forced scoring;
- for `engine` only, a worker method `engine_grid` reachable through the dev-mode collective
  RPC (`VLLM_SERVER_DEV_MODE=1`), which a production server should not expose.

Sonar sends nothing model-specific unless told: a chat template's reasoning switch, for
example, goes in `--extra-body` (or the matrix's `--reasoning-off`).

## Environment

- `SONAR_STREAM_COLLECTOR`: `current` (default) or `text-chunks`, the earlier rule for which
  streamed chunks count as output; select it to compare with records that rule wrote.
- `SONAR_MODEL_CONTRACT` and `SONAR_MODEL_CONTRACT_SHA256`: an opt-in model barrier for
  `suite` (a frozen contract binding the suite and two saved reference results; the suite
  stops before timing when the model step's result differs). See `sonar/model_evidence.py`.

## Results

Results are JSON (`.gz` to compress): `sonar-run/1`, `sonar-engine/1`, `sonar-model/1`,
`sonar-prefill/1`, `sonar-stall/1` and `sonar-suite/1`. Sonar began as `servebench`, and
reads results recorded under its names (`servebench/1`, `servebench-engine/1`, ...) as the
same formats, so earlier records compare with new runs.

## Tests

```sh
python3 -m unittest discover -s tests -t .
```

Everything runs against fake servers in-process: no GPU and no network. The model barrier's
bounded parser needs Linux; the matrix's JSON cases need `jsonschema`
(`requirements/jsonschema.txt`, hashed).

## License

Apache License 2.0; see [LICENSE](LICENSE).
