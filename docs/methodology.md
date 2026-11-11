# Methodology

How Sonar measures, and why each choice was made. The [README](../README.md) lists the
commands; this page is the reasoning behind them.

## Method

- **Load.** `closed` keeps N requests in flight, starting the next as each ends, so the
  server sees a steady N with new prefills mixed into decode steps. `poisson` sends
  requests at a fixed rate whatever the server does (open loop), so queueing shows as
  latency. `burst` starts N at once, the shape older burst benchmarks use, for comparison.
- **Context.** With `--context TOKENS` (closed loop) each stream owns a long document,
  Python standard-library source sized by the server's tokenizer, and asks every question
  after it under one cache salt. Each document is prefilled once before the clock starts,
  so the window measures decode at that depth plus each question's short prefill, as with
  a long conversation or a coding agent's repository context. The window's prefix-cache
  hit rate shows whether the documents stayed cached. Streams x context must fit the KV
  cache.
- **Window.** A warm-up (default 60 s) runs before the measurement window (default
  300 s); when the window closes no new requests start, and the in-flight ones drain
  outside it.
- **Tokens.** Every streamed chunk is timed with the request's cumulative token count
  (vLLM's per-chunk usage), so the window counts exactly the tokens produced inside it.
- **Prompts.** 256 distinct prompts (sixteen task styles by 32 topics), each request with
  a fresh cache salt so repeats never hit the prefix cache, and by default exactly
  `--max-tokens` output tokens each (`ignore_eos`), so output length is not a variable.
- **Intervals.** The window is cut into equal sub-windows (default 10); each is one
  sample of throughput and of every server rate (batch means), with Student t intervals.
- **Speed and text apart.** With the vLLM adapter the server's counters are read at
  every sub-window boundary: output tok/s = engine steps/s x output tokens per step.
  The step rate is the engine's speed; tokens per step moves with draft acceptance, that
  is with the text. A precision change that alters the text shows in the second, a
  kernel change in the first. The speculative counters by draft position are read too:
  survival (accepted at position k a draft) and coverage (verified at k a draft).
- **Pairs.** Every run keeps a compact table of its requests (`request_table`). The seed
  fixes the prompt order, so two runs of one workload send the same prompt as their i-th
  request. `compare` pairs the requests both runs sent in their windows. It gives the
  median ratio of each pair's decode rate (output tokens after the first, from the first
  chunk to the last) and of its time to first token, with an exact (sign-test) interval.
  A window holds whichever prompts fell in it, and greedy text gives each prompt its own
  acceptance, so windows differ by text more than builds do. On one system, at one stream, the
  window interval was ±3.7% where the same build repeated within 0.13%. Pairs cancel the text.
  A run without a table (one kept before tables existed) is paired window by window
  instead. With a real change the faster build's windows drift onto later prompts, so
  that interval also holds text: about ±2% for a 2% change at one stream.
- **Latency.** Time to first token, time per output token and per-user tok/s
  (percentiles over requests started in the window); with `--slo-ttft` / `--slo-tpot`,
  SLO attainment and goodput (requests per second meeting both limits).
- **Compare.** Batch-mean metrics get a Welch interval; percentile metrics are judged
  against a tolerance (default 3%) and marked "no interval"; paired rows are judged by
  their exact interval. The comparison warns when the two runs' workloads differ.
- **Boots.** A paired interval is a few hundredths of a percent, but a boot moves a build
  by a few tenths. A suite's verdict block (`suite.with_floor`) therefore judges the paired
  decode row, and an engine grid step's geometric mean, against a recorded A/A floor that
  the caller supplies by step: the 95% bound of one boot against another. The run's own
  repeat widens it when larger. With a second candidate boot, the change is the mean over
  both boots and the floor shrinks by sqrt(3/4). The margin is the interval and the floor in
  quadrature. A single repeat is one draw of the boot noise: judged against it alone, a
  build differs from itself half the time.
- **Both boots of each.** Given the reference's own repeat (`--reference-repeat`: the
  reference booted again, against itself and against both candidate boots), every headline
  row is the mean over all four pairings and the boot term counts both boots on each side
  (`suite.boot_term`: sqrt(F_c^2/2b + F_r^2/2r), x0.707 of one pair's for two of each).
  Without it, rows are judged as before.

## Prefill

`python3 -m sonar prefill URL --sizes 32768,262144 --repeats 3` times cold prompts of
fixed sizes. Each prompt is Python standard-library source from a fixed seed, so every
build reads the same text; it is sized by the server's tokenizer and sent with a fresh
cache salt. The rate is prompt tokens over time to first token. Repeats go round the
sizes in turn, and comparisons use Welch intervals.

## Stalls during a long prefill

`python3 -m sonar stall URL --sizes 131072 --streams 8 --repeats 3` measures what
other streams wait while one long cold prompt prefills. The streams decode first (fixed
length, fresh salts, the default prompts); after `--warmup` seconds the prefill step's
document is sent for one output token, and the streams stop `--tail` seconds after its
first token. A stream's gaps between output chunks are its step times: those ending
before the document was sent are the decode baseline, those overlapping its prefill the
stall. It reports the worst overlapping gap (the longest step that carried a prefill
chunk, which the long-prefill budget sets), the overlapping gaps' median and p99, the
baseline median and the prefill rate under decode. A repeat fails unless every stream
decoded across the whole prefill (`--max-tokens` must outlast warm-up and prefill).

Measure more than one depth. With free slots the scheduler can admit work during the
prefill, and policies that reorder admitted work act only then; at `--max-num-seqs` − 1
streams the long prompt takes the last slot and nothing else is admitted until it ends.
At `--max-num-seqs` streams the long prompt waits for a slot instead of stalling anyone:
low decode gaps there are queueing, not interleaving, and the step fails because the
streams stop before it prefills.

`--probes 2,5,8` also sends short cold requests that many seconds after the document
(`--probe-tokens` of real text, default 500; `--probe-max-tokens`, default 64) and reports
the time to first token of those sent before the document's first token: what a new
conversation waits while a long prompt prefills. Leave them free slots: the streams, the
document and the probes in flight must fit `--max-num-seqs`, or the probes measure slot
queueing. Use prompts longer than the unused decode rows ((max-num-seqs − streams) x 6
tokens), which can ride beside a long chunk.

## Engine cost grid

```sh
python3 -m sonar engine http://127.0.0.1:8000 --output grid.json
python3 -m sonar compare base-grid.json candidate-grid.json
```

The server times its own fixed-shape dummy steps, the speculative-decoding cost profile's
steps, at the shapes asked for. The default grid is verification rows (requests x (1 +
depth)):
- 1, 4, 8 and 16 requests at depths 1 and 2, at 4K context. This is what adaptive
  verification runs under load: about two rows a request at 8-16 streams.
- 1-16 requests at the full five drafts, at 4K.
- Depth 2 at 64K and 256K and at the server's longest context.

Every build does identical work with no text, acceptance or scheduling in it, so the step
times are the engine's speed. This is the measurement that judges kernel and engine changes
at decode; tok/s judges what users get. It needs vLLM's dev-mode collective RPC
(`VLLM_SERVER_DEV_MODE=1`) and a worker exposing `engine_grid`; a production server should
not expose that RPC, so suites with a grid step run against a development server. It
refuses a busy server and resets the prefix cache afterwards, since dummy steps write KV
slots. `--max-requests` (the server's `--max-num-seqs`, default 16) leaves out shapes of
more requests.

- **Timing.** Each shape is warmed, then timed in rounds of alternating order (default
  100). Each timed step follows an untimed one of the same shape (`--settle`): switching
  shapes rebuilds the dummy metadata, which once cost 11 ms after a jump between 1M and
  no context. That is the harness's cost, not the engine's.
- **Comparison.**
  - Slow outliers (above the median by five scaled MADs) are dropped first.
  - The Welch intervals hold for all shapes at once (Bonferroni), so a grid of twenty
    shapes does not flag one by chance.
  - The decode shapes' geometric mean gives one overall verdict.
- **Prefill is not in the default grid.** On one system dummy prefill ran 1.7x faster than a real 16K
  prefill's kernels, and called a build 1% slower that real cold prefill found level or
  faster, presumably because identical dummy tokens route the MoE unlike text. `--prefill`
  adds 8K chunks for completeness; judge prefill on real prompts.

### Boot-to-boot spread

`python3 -m sonar spread GRID... [--against GRID...]` takes engine grids of one build,
each from its own boot. It gives every boot's decode geometric mean against the set's mean,
and their standard deviation. With `--against`, it compares two sets with boots as the
samples. On one system grids repeated to about ±0.05% within a boot and moved up to 0.5% between
boots, so changes that small need several boots per build.

## Model behaviour

```sh
python3 -m sonar model http://127.0.0.1:8000 --label base --output base.json.gz
python3 -m sonar model http://127.0.0.1:8000 --label candidate --corpus base.json.gz \
    --output candidate.json.gz
python3 -m sonar compare base.json.gz candidate.json.gz
```

What a build computes, with no timing in it, so a repeat gives the same answer:
- **Greedy generation.** Every prompt once at temperature 0, at a fixed concurrency (default
  16), with each response's token ids. The speculative-decoding counters are read before
  and after: mean acceptance length, draft tokens verified per step, and per draft
  position the acceptance rate and how often that position was verified. On a
  batch-invariant server the text, and with it the acceptance, belongs to the build.
  Compared runs report how many responses are token-identical and where the rest diverge.
- **Teacher-forced scoring.** Token sequences (by default this run's own prompts and
  responses; with `--corpus`, an earlier run's) go back as prompts with
  `prompt_logprobs`, keeping per response token its log-probability and rank and the top K
  (`--top`, vLLM's `--max-logprobs` limit, default 20). Two builds that scored the same
  tokens compare position by position:
  - identical distributions;
  - top-1 agreement;
  - KL divergence, estimated from the top K with the rest of the vocabulary as one bucket;
  - the given token's log-probability change;
  - perplexity.
- **Self-agreement.** A build scoring its own greedy text should rank every given token
  first. Anything less means its prefill arithmetic and the decode arithmetic that wrote the
  text pick different tokens.

The base run is a property of the base build, so it can be kept and reused for every
candidate measured against that build.

## Verdicts across boots

`sonar table CANDIDATE REFERENCE CANDIDATE-AGAIN` turns suite runs into a verdict. Arms run
candidate first (it also heat-soaks the hardware), then the reference, then the candidate
again. The reference may be a recorded run of an earlier build: its numbers do not change,
so it need not be measured again, but it must be a run of the same suite (same digest).

- **Window rows** (output tok/s, the grid's decode geometric mean, prefill rates) count a
  change only beyond its interval plus the candidate's own repeat: one boot moves a build by
  a few tenths of a percent that no within-run interval sees.
- **Paired rows** (`decode tok/s, paired`) have intervals of a few hundredths of a percent, so
  the boot is nearly all of their uncertainty. They, and an engine grid step's geometric mean,
  are judged against a recorded A/A floor,
  `--floors`: by step id, the 95% bound of one boot against another (1.96 x the standard
  deviation of the change between repeated boots of one build, over as many pairs as you
  have). The run's own repeat widens the floor when larger; with both candidate boots the
  change is their mean and the floor shrinks by sqrt(3/4); the margin is the interval and the
  floor in quadrature. Record the floors once per system and keep them with the suites; a
  step without one gets no paired verdict. With the reference's repeat boot as well
  (`--reference-repeat`), each row is the mean over both boots of each side.
- **Rows a step** follow: per arm and closed loop, tokens and verified rows per engine step,
  row yield, wasted rows, and per draft position survival and coverage, so a speed change can
  be read as an engine change (rows cost more or less) or a text change (more or fewer rows
  accepted).

## Determinism

`sonar determinism` serves eight prompts at temperature 0 alone (the reference), all at once,
arriving 100 ms apart, beside a long prompt's prefill, from the prefix cache, fifteen at once
(`--wide`) and alone again, and compares every token and log-probability with the reference.
A batch-invariant server writes the same tokens whatever company a request keeps; equal
tokens with unequal log-probabilities is a numeric difference that has not flipped a token
yet. The summary's digests (with and without log-probabilities) let two builds be compared bit
for bit without keeping the texts: a build that must not change the model's output has the
previous build's digests. `--no-logprobs` compares texts, which on some servers takes a
different verification path.

## The model barrier

A suite run can be bound to a frozen contract (`SONAR_MODEL_CONTRACT`, by exact SHA-256 in
`SONAR_MODEL_CONTRACT_SHA256`): the suite's name and digest, the model step's workload shape,
the corpus and number of scored positions, two saved reference results from distinct boots,
and optionally the run labels allowed and run metadata the candidate must carry (any subset of
what the caller stores with `--metadata`). Before any timed step, the model step's result must
equal both references: the same greedy token ids and the same teacher-forced scores. If it
does not, the suite stops before timing, so a build that computes something else is never
timed as if it were the same build. The references are parsed in a fresh interpreter with
Linux resource limits, and nothing in the barrier sends a request.

## Quick probes

`streams`, `single`, `acceptance`, `ttft` and `mixed` take a few minutes each and print a
JSON line per sample. They are quick looks, not verdicts: a burst holds a ramp and a tail, and
one sample per prompt has no window interval. `streams` reads the server's finished-request
counter around each sample and exits 3 when someone else's requests finished meanwhile.

## The measurement matrix

`sonar matrix` samples decode points (a case at a concurrency) in shuffled rounds until the
metric each converges on (tok/s, step time, or accepted tokens per verified draft) has a 95%
interval within `--precision`, after one discarded warm-up sample a point. Every sample waits
for an idle server and is repeated when other traffic overlapped it. Beside decode it runs a
quality gate (a fixed request whose answer must parse and have the right shape; a failure
stops the run), structured-output compliance with and without the server's JSON-schema
constraint, sampled acceptance, cold prefill on filler, pseudo-words or real text, prefix-cache
replay and long-context admission without preemption. Step time comes from each sample's
decode window and acceptance, (1 + accepted per draft) / per-stream decode rate, which stays
steady when the text, and with it the acceptance, moves.

Its cases send one prompt to N streams (or a fixed set of prompts) in bursts, so on a
batch-invariant server the streams write one text: the matrix shows where a build moves, and
the steady-state suites say by how much. A caller that watches the hardware (memory,
temperatures) passes a guard that can stop the run; Sonar itself watches only the server.

## Reading a server's traffic

`sonar watch` sends nothing: it reads the counters before and after a window (a fixed time or
a command's run), samples the gauges, and reports what the server did: requests by finish
reason, preemptions, prompt tokens and the prefix cache's share, latency quantiles from the
server's own histograms, acceptance by draft position, the share of steps that carried prefill,
and the load gauges. Use it beside production traffic, or around another load generator.

## Records and compatibility

A suite is data with a digest, and a changed suite is a new suite: `compare` and `table` refuse
runs of different digests, so a number never silently changes meaning. A result names its
schema; Sonar reads its predecessor's names (`servebench/1` and the rest) as the same formats.
The stream collector's rule for which chunks count as output can be switched back to the
earlier text-chunk rule (`SONAR_STREAM_COLLECTOR=text-chunks`) to compare with records it wrote.
