# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

The AIME runners below are retained single-family diagnostics. They are not the
current product goodput benchmark and cannot establish a ServeBig contribution.
The current product result is the frozen long-context three-family v2 stream,
with 20 one-in-flight closed-loop users and a full drain under the contract in
the root `RESEARCH_PLAN.md`. Its final best-to-best result is `1.056538249x`;
see
[`results/long_mixed_task_goodput_final_20260901.md`](results/long_mixed_task_goodput_final_20260901.md).

**Hard AIME + LiveCodeBench goodput v1** —
`build_aime_lcb_goodput_workload.py`, `bench_aime_lcb_goodput.py`, and
`score_livecodebench_subset.py` form an independent accuracy/throughput
benchmark. Inspect its complete source and output plan without a download or
filesystem write:

```bash
python benchmarks/build_aime_lcb_goodput_workload.py --plan-only
```

Plan-only validates the same immutable v1 source identifiers, revisions,
schemas, row counts, and stream selections as a real build. A changed spec is
rejected before any source download, cache update, or output write.

Publish once to a fresh output directory. The multi-gigabyte official source
cache belongs on `/data2`; prompts and the small frozen manifest remain on
`/data1`:

```bash
python benchmarks/build_aime_lcb_goodput_workload.py \
  --source-cache /data2/servebig-envs/aime_lcb_sources_v1 \
  --output-dir /data1/lmcache_kv/goodput_campaign/aime25_lcb50_familythink_v1
```

Final is all 30 `math-ai/aime25` problems at revision
`563bb8404243c5f09de6ec262f2db674fe5bce9b` plus 20 problems from the
incremental LiveCodeBench `v6` file at dataset revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`: the first ten medium rows and
first ten hard rows, independently, in frozen source order. Development remains
the disjoint 30
`math-ai/aime24` problems at
`83a7f387baaa524a8bda0022eac0541582297103` plus all 126 medium/hard
incremental `v5` problems (52/74), but it is not used for the formal result.
Warmup is the first `math-ai/amc23` row at
`80815d37005feb82cd7f8fbc6901d5d3eff43057` and the first medium-or-hard
incremental `v4` row in frozen source order. Both warmup rows have
`scored: false` and `reference: {"kind": "unscored"}`. The builder checks the AMC23 source
schema but does not load, parse, validate, or copy its answer values. Measured
AIME24/AIME25 task rows have `scored: true` and retain strict 000–999
integer/boxed-integer answer validation. The builder validates complete source
fields, counts, difficulty distributions, unique ids/text, and pairwise stream
disjointness. It atomically publishes `manifest.json`, `warmup.jsonl`,
`dev.jsonl`, `final.jsonl`, and an answer/test-free
`training_forbidden_texts.jsonl`. LiveCodeBench hidden tests stay only in the
external source cache and never enter a prompt or workload row.

Twenty measured users each keep at most one request in flight and have zero
think time. The two warmup requests must naturally finish and drain. Development admits
new requests only for 300 seconds from the first measured submission, never
wraps the finite queue, then drains every admitted request. Formal final has no
submission window: all 50 tasks must be submitted once and fully drained. Each
of its twenty users receives exactly two or three tasks.

```bash
python benchmarks/bench_aime_lcb_goodput.py \
  --arm-name restored-nowag-dev --stream dev \
  --manifest /data1/lmcache_kv/goodput_campaign/aime25_lcb50_familythink_v1/manifest.json \
  --freetoken-root /path/to/freetoken --python /path/to/server/bin/python \
  --model /path/to/model --gpu 0 \
  --lcb-root /path/to/LiveCodeBench-at-28fef95 \
  --lcb-python /path/to/livecodebench/bin/python \
  --aime-max-tokens 49152 --code-max-tokens 32768 \
  --output /tmp/restored-nowag-dev.json \
  --server-args --max-running-requests 20 \
  --max-seq-len-override 65536
```

`--server-args` must be last and must contain exactly one explicit
`--max-running-requests 20` and one `--max-seq-len-override`, whose value may
be 32768 or 65536. The fixed request enables thinking for AIME and disables it
for code; the manifest records this as
`thinking_by_family: {"aime": true, "code": false}`. Sampling otherwise uses
`n=1`, temperature 0, top-p 1, top-k -1, and a deterministic per-task seed.
The formal defaults are AIME 49152 and code 32768 new tokens with a 65536
server sequence length. The 32768 server setting remains available for
diagnostics when both family caps are explicitly reduced to fit it. Before the
server starts, the runner reads that single server sequence-length value and
requires every exact rendered prompt length plus its selected family cap to fit
it. The same value is recorded in dry-run and result metadata. The request
timeout is at least 3600 seconds and is
only a transport safety ceiling: a timeout, HTTP error, stream error, incomplete SSE,
or judge infrastructure error invalidates the run rather than becoming an
incorrect answer.

The client always reads through `[DONE]` and accepts only natural server
`stop` or `length`; it never stops at the first boxed answer or code block.
AIME uses the final boxed integer in 000–999. Code is scored after online drain
with LiveCodeBench's official hidden-test code-generation evaluator at commit
`28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`; judge wall time is outside the
serving denominator. Run `score_livecodebench_subset.py --help` for its exact
input/output schemas and exit contract.

For measured duration `D`, final reports AIME `C/30`, code pass@1, micro
accuracy, `Q = (A_AIME + A_code) / 2`, raw throughput `T = N/D`, literal
correct throughput `(C_AIME + C_code)/D`, and balanced goodput `T*Q`. It also
reports latency, TTFT, TPOT, average/total prompt and completion tokens, server
token rates, terminal counts, cap hits, and observed concurrency. Per-task
prompts/outputs remain in the JSON result for audit; stdout is summary-only.
Use `--dry-run` to inspect the exact dev/final queue semantics, server command,
caps, and scorer command without starting a server, judge, or GPU and without
writing a file. Workload contract:
[`workloads/aime25_lcb_goodput_v1.json`](workloads/aime25_lcb_goodput_v1.json).

**Historical short-context mixed goodput v1** —
`build_mixed_goodput_workload.py` and `bench_mixed_task_goodput.py` build and
run the immutable official-source bundle:

```bash
python benchmarks/build_mixed_goodput_workload.py \
  --output-dir /data1/lmcache_kv/goodput_campaign/mixed_workload_sources_v1
```

The builder pins gsm-hard revision
`960448f73503112d4226baeb8eb41d3fb5ae2506`, MMLU revision
`c30699e8356da336a370243923dbaf21066bb9fe`, EvalPlus 0.3.1 commit
`e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`, HumanEval+ v0.1.10, and
MBPP+ v0.2.0. It produces 20 warmup tasks, a development stream with 80 tasks
per family, and a disjoint final stream with 140 tasks per family. The
manifest's `training_forbidden_texts` entry points to the non-answer
`training_forbidden_texts.jsonl`: each row has exactly `family`, `task_id`,
`split`, `kind`, and `text`. Corpus preparation applies the declared
`nfkc_casefold_unicode_whitespace_v1` normalization and rejects a sample only
when it contains a complete normalized `problem` or `final_prompt` row.
`contract`, `entry_point`, and `public_test` rows are retained only for audit
counts; their short fragments are never substring filters.

Run one independently tuned arm at a time. The EvalPlus Python may be a small
separate environment; it must contain exactly EvalPlus 0.3.1. The final stream
uses a fixed 600-second submission window; the development stream uses 300
seconds. Both use 20 users, one-second first-entry offsets, 30-second think
time, no wrap, and a complete drain.

```bash
python benchmarks/bench_mixed_task_goodput.py \
  --arm-name upstream-bf16 --stream dev \
  --model /path/to/model --gpu 0 \
  --evalplus-python /path/to/evalplus-0.3.1/bin/python \
  --output /tmp/upstream-mixed-dev.json \
  --server-args --dtype bfloat16 --max-running-requests 20
```

The shared policy is greedy decoding, thinking disabled, and system message
`Follow the requested output format exactly.` Family caps/SLOs are numeric
512/30s, code 768/60s, and knowledge 64/10s. An arm may lower its family token
caps on development but cannot exceed those limits; SLOs and the hard transport
timeout are identical across A/B arms. `--server-args` must be last and freezes
each arm's backend, batching, cache, graph, precision, and NoWAG settings.

The result schema is `freetoken.mixed_task_goodput_result.v1`. `summary.total`
and every `summary.families.<family>` report `submitted`, `judge_correct`, raw
accuracy, SLO successes/rate, goodput, latency, TTFT, TPOT, output tokens, and
terminal reasons. Goodput uses only SLO successes and divides by
`max(window_end,last_terminal,server_idle_ack)-user0_first_submit`; code tests
run after that denominator closes. Every measured row keeps request timing,
usage, finish/error, raw output, parse/verifier verdict, and token observations.
`server_observability` records request/token deltas and peak observed VRAM;
cache-miss and H2D values are explicitly `null` because the current public
`/v1/stats` schema does not expose them. Final stdout contains only the summary,
never per-task output. Use `--dry-run` to inspect the complete frozen command
and policy without starting a server.

The post-drain scorer has a standalone public interface:

```bash
HUMANEVAL_OVERRIDE_PATH=/path/to/HumanEvalPlus-v0.1.10.jsonl \
MBPP_OVERRIDE_PATH=/path/to/MbppPlus-v0.2.0.jsonl \
/path/to/evalplus-0.3.1/bin/python benchmarks/score_evalplus_subset.py \
  --input /tmp/evalplus-input.json --output /tmp/evalplus-output.json
```

Input is
`{"schema":"freetoken.evalplus_subset_input.v1","items":[...]}`. Each item
requires the runner fields `record_id` (non-empty caller-unique string),
`dataset` (`humaneval` or `mbpp`), an official fixed-version `task_id`, and
`solution` (non-empty complete executable Python source, with no Markdown
fence). Output uses schema `freetoken.evalplus_subset_result.v1` and reports
EvalPlus/dataset versions plus an order-preserving `items` list. Each output
item has `record_id`, `task_id`, `dataset`, `base_status`, `plus_status`, and
`passed`; statuses are `pass`, `fail`, or `timeout`, and `passed` is true only
when both statuses are `pass`. Success atomically writes the output and exits
zero. Bad JSON/schema/fields/task ids, a version other than EvalPlus 0.3.1,
data/verifier failures, or an unwritable target exit nonzero without a valid
output contract. Scorer stdout/stderr are diagnostic only. The complete same
contract is printed by `score_evalplus_subset.py --help`.

The historical 2026-09-01 single-4090 final A/B is complete: ServeBig reached
`0.910002x` of the best upstream FreeToken task goodput and did not meet the
`2x+` goal; lower latency and VRAM were outweighed by lower numeric accuracy.
See [`results/mixed_task_goodput_final_20260901.md`](results/mixed_task_goodput_final_20260901.md).

**Long-context mixed goodput v2** —
`build_long_mixed_goodput_workload.py` and
`bench_long_mixed_task_goodput.py` are independent from v1. They freeze three
90–100% Qwen-tokenizer prompt buckets (8192, 16384, 32768 tokens) from real
source text: DocFinQA numeric questions, the fixed SWE-bench Verified/BM25-40K
intersection, and non-code-repository LongBench-v2 multiple choice. Oversize
contexts use question-only Okapi BM25 over contiguous source blocks and restore
selected blocks to source order. Answers, DocFinQA programs, Verified gold/test
patches and evaluator-test lists, hints, and supporting facts never enter
retrieval, prompts, or the training-isolation file. DocFinQA `Program` must be a
string but may be empty,
as it is in an official validation row; the builder neither fills nor infers
it. `Context`, `Question`, and `Answer` remain required non-empty strings. SWE
repository blocks are read from the complete BM25 `text` using strictly paired
`[start of PATH]` / `[end of PATH]` markers with identical paths; HTML
`<code>` tags are ignored because repository source can contain them. Every
path-marked base-repository file is eligible context, including repository test
files; wrapper text outside those markers is excluded.
LongBench-v2 Question values are globally unique after the manifest's
`nfkc_casefold_unicode_whitespace_v1` normalization. Selection reserves them in
warmup, development, then final priority order and fills each bucket from its
unchanged fixed candidate pool without consulting answers.

Inspect the full source/download/traffic plan without downloading anything:

```bash
python benchmarks/build_long_mixed_goodput_workload.py --plan-only
```

Publish once to a fresh directory (an existing output is refused):

```bash
python benchmarks/build_long_mixed_goodput_workload.py \
  --tokenizer /data1/lmcache_kv/models/Qwen3.6-35B-A3B \
  --source-cache /data1/lmcache_kv/goodput_campaign/long_mixed_workload_v2_sources \
  --output-dir /data1/lmcache_kv/goodput_campaign/long_mixed_workload_v2
```

The fixed sources are DocFinQA
`64ebaff62f692495bcc182f45cf9a9606251b19b`, SWE-bench BM25 40K
`c7bc17e54d390580a9d40ecc72948e3eac296b83`, SWE-bench Verified
`78f471bf655a3137b2e8a75af1501690ec009ec3`, and LongBench-v2
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`. DocFinQA is MIT and
LongBench-v2 is Apache-2.0; the two fixed SWE dataset cards do not declare a
license. Numeric normalization is pinned to FinQA commit
`0f16e2867befa6840783e58be38c9efb9229d742` (remove display formatting,
convert percent values by `/100`, then Python `round(..., 5)`). Code scoring is
the official SWE-bench v5.0.1 harness at commit
`87ab1f6ced28f75ba73ca899dc759b019310944a`; it uses each selected Verified
row's literal `swebench/sweb.eval.x86_64.*:latest` image reference.

Warmup is 20 disjoint requests (7 numeric, 7 code, 6 knowledge). Development
contains 240 tasks and a 120-second submission window; final contains 1020
tasks and a 180-second submission window. Twenty users first submit at
`0.5 * user_index` seconds, keep at most one request in flight, and wait two
seconds after each terminal. Queues never wrap. Per family, development uses
32/32/16 tasks and final uses 136/136/68 tasks in the 8K/16K/32K buckets.
Numeric/code/knowledge caps and SLOs are respectively 128/90s, 2048/180s, and
32/60s; the transport timeout is fixed at 210 seconds. The server must use
`--max-seq-len-override 40960` exactly.

Warmup is unscored and only primes the server before the fixed drain. A complete
HTTP 200 SSE terminal with either `finish_reason=stop` or
`finish_reason=length` is a valid warmup completion; a length-limited warmup
does not block online measurement. Transport, HTTP, stream, or incomplete
terminals invalidate the arm. Every success or failure result publishes a
prompt-free `warmup` aggregate with `submitted`, `completed`,
`terminal_reason_counts`, `invalid_terminal_count`, and
`invalid_terminal_reason_counts`; an invalid warmup reason reports the failing
terminal categories and counts.

```bash
python benchmarks/bench_long_mixed_task_goodput.py \
  --arm-name upstream-dev --stream dev \
  --manifest /data1/lmcache_kv/goodput_campaign/long_mixed_workload_v2/manifest.json \
  --freetoken-root /path/to/freetoken --python /path/to/server-venv/bin/python \
  --model /path/to/model --gpu 0 \
  --swebench-python /path/to/swebench-v5.0.1-venv/bin/python \
  --swebench-root /path/to/SWE-bench-v5.0.1 \
  --output /tmp/upstream-long-dev.json \
  --server-args --dtype bfloat16 --max-running-requests 20 \
  --max-seq-len-override 40960
```

Use `--dry-run` with the same required arguments to print the complete command,
fixed traffic, task counts, and theoretical ceilings without starting FreeToken
or Docker. Goodput counts only correct, valid-finish responses within the fixed
family SLO. Its denominator is
`max(window_end,last_terminal,first_server_idle_ack)-user0_first_submit`.
Wrong/format/length/timeout/HTTP/service answers consume their full cycle and
score zero. Official SWE Docker grading begins after the full drain and is not
in the denominator; any verifier used before HTTP terminal would naturally be
online. Result schema `freetoken.long_mixed_task_goodput_result.v2` reports the
total, every family, and every length bucket with raw accuracy, SLO success
rate, goodput, latency, TTFT, TPOT, tokens, terminal reasons, peak observed
VRAM, request token deltas, drain tail, and explicit nulls for unavailable
cache-miss/H2D counters. Final stdout is summary-only and never prints tasks.

The standalone post-drain scorer contract is printed in full by:

```bash
python benchmarks/score_swebench_verified_subset.py --help
```

Its input schema is `freetoken.swebench_verified_subset_input.v2`, with exactly
`record_id`, `instance_id`, and non-empty `model_patch` per unique item. Output
schema `freetoken.swebench_verified_subset_result.v2` preserves item order and
returns `status` (`resolved`, `unresolved`, or `evaluation_error`), boolean
`resolved`, and `report_available`. Invalid contracts, an unknown id, a
non-v5.0.1 checkout, a wrong evaluator commit, invalid fixed Parquet, or a
non-zero harness exit fails the scorer without a valid output contract.
Workload contract:
[`workloads/long_mixed_task_goodput_v2.json`](workloads/long_mixed_task_goodput_v2.json).

The 2026-09-01 long-context single-4090 final A/B is complete. Best ServeBig
reached `199.362545 tasks/h` versus `188.694110 tasks/h` for best upstream
FreeToken, or `1.056538249x`. Both arms had 13 correct, SLO-compliant tasks, so
the result did not meet `2x+`. There is no honest measured `2x` workload
candidate; a new final remains closed pending a production prefill improvement.
See
[`results/long_mixed_task_goodput_final_20260901.md`](results/long_mixed_task_goodput_final_20260901.md).

**`bench_aime_task_goodput.py`** — runs one independently frozen FreeToken arm on
the fixed 20-user/30-task AIME25 closed-loop workload and reports correct boxed
answers per wall-clock hour. Users start ten seconds apart; users 0-9 receive a
second unique problem only after their first task terminates and a fixed 30-second
think time. The first streamed `\boxed{integer}` ends the client task whether it
is right or wrong. The runner immediately closes that stream, verifies the public
disconnect-to-abort behavior before measurement, and requires `/v1/stats` to be
idle after all task terminals before accepting the goodput result.

```bash
python benchmarks/bench_aime_task_goodput.py \
  --arm-name servebig-qwen \
  --freetoken-root /path/to/freetoken-checkout \
  --python /path/to/venv/bin/python \
  --model /path/to/model --gpu 0 \
  --system-prompt "Solve efficiently and stop after the final answer." \
  --answer-instruction "Return the final integer as \\boxed{integer}." \
  --max-tokens 32768 --request-timeout 7200 \
  --output /tmp/servebig-qwen-goodput.json \
  --server-args --dtype bfloat16 --max-running-requests 20
```

`--server-args` must be last and accepts the complete arm-specific serving policy,
including batching, cache, backend, and NoWAG options. Use `--dry-run` to print the
full server command, source commits, request policy, and schedule without starting
the server or downloading AIME25. Run the upstream FreeToken and ServeBig arms as
separate invocations; each result contains its complete frozen configuration.
`--system-prompt` and `--answer-instruction` are one global policy for all 30 tasks;
an empty value disables the corresponding message/suffix, and no per-problem prompt
override exists. Each task keeps server-reported usage when the final usage event is
available and always records model-tokenizer counts for the final prompt and the text
observed before the client terminal, with the counting source and estimation status.
If the server or workload raises, the output is still written atomically as a failed
diagnostic result containing every task row that had already reached a terminal state.
Workload contract: [`workloads/aime25_task_goodput_20user_v1.json`](workloads/aime25_task_goodput_20user_v1.json).

**`bench_aime_steady_goodput.py`** — runs the historical AIME-only steady-state
diagnostic. By default, 20 closed-loop users enter ten seconds apart and wait
30 seconds after each terminal before their next task. Warmup continues until
every user has at least one terminal; the following 1800 seconds are a fixed
submission window. Tasks submitted before that window remain live but do not
count, and tasks already submitted when it closes drain under the same timeout.
The reported denominator is
`max(window_end, last_measured_terminal) - window_start`, so wrong boxes, missing
boxes, timeouts, and HTTP errors contribute zero correct tasks while consuming
their full measured time.

Input is a pre-locked local JSONL stream whose rows contain unique `task_id`,
`problem`, `answer`, and `source` fields. For `N` users, user `u` receives rows
`u`, `N+u`, `2N+u`, and so on; the runner never wraps the stream, and exhaustion
invalidates the arm. This makes per-user task order identical across arms even
when their completion rates differ.

```bash
python benchmarks/bench_aime_steady_goodput.py \
  --arm-name servebig-qwen --freetoken-root /path/to/freetoken-checkout \
  --python /path/to/venv/bin/python --model /path/to/model --gpu 0 \
  --task-jsonl /path/to/locked-aime-stream.jsonl \
  --request-extra-json '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --output /tmp/servebig-qwen-steady-goodput.json \
  --server-args --dtype bfloat16 --max-running-requests 20
```

`--user-count`, `--start-cadence-seconds`, `--think-time-seconds`, and
`--measurement-seconds` may shorten the public workload for local checks; the
defaults above define the product run. Each result separates warmup and measured
tasks, records every request terminal and token observation, freezes both runner
and server commands, and is valid only when post-drain `/v1/stats` reports
`active=0`. Workload contract:
[`workloads/aime_steady_goodput_v1.json`](workloads/aime_steady_goodput_v1.json).

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_lab_agent_policies.py`** — closed-loop public-HTTP comparison of legacy,
mixed, layered, joint, and layered-pipeline batching under a four-user, five-turn
tool-agent burst. It validates exact prompt/output lengths and prefix-cache reuse,
and records TTFT, TPOT, inter-token gaps and makespan. Layered-pipeline freezes a
multi-request wave, packs its uncached prompt rows into static FIFO ragged tiles,
and advances one tile through the current resident group per iteration. It
finishes every tile before moving to the next group. The shared cache reserves
one full expert layer outside the resident group for decode, so it must hold at
least two expert layers. `--max-prefill-length` is the requested per-iteration
tile limit and also estimates admission size; cache geometry may lower the
effective limit, and neither value limits total prompt length.
`--prefill-wave-max-chunks` is an aggregate complete-request admission soft cap,
not a physical tile boundary. Completion logs and JSON expose `reqs`, `groups`,
`group_forwards`, `iterations`, `decode_iterations`, and
`prefill_layer_prepares`. The generated MoE is a directional test, not a
real-model result.

```bash
python benchmarks/bench_lab_agent_policies.py --repetitions 3 --gpu 0 \
  --output /tmp/lab_agent_policies.json
```

Workload contract: [`workloads/lab_agent_burst_v1.json`](workloads/lab_agent_burst_v1.json).
Measured setup and interpretation: [`results/lab_agent_burst_20260825.md`](results/lab_agent_burst_20260825.md).
The focused resident-group probe and raw samples are recorded in
[`results/joint_group_wave_20260825.md`](results/joint_group_wave_20260825.md).
The paper-granularity one-group-per-iteration comparison is in
[`results/layered_pipeline_paper_fair_20260827.md`](results/layered_pipeline_paper_fair_20260827.md).
The sustained-decode experiment with serial periodic long prefills is in
[`results/periodic_long_prefill_lab_20260827.md`](results/periodic_long_prefill_lab_20260827.md).

**`bench_real_conversation_concurrency.py`** — replays real WildChat multi-turn
text under BurstGPT session starts and human think times. It compares the same
manifest-selected sessions across batching policies and records TTFT, TPOT,
request latency, streamed-text event gaps, expert traffic, and peak concurrency.
The current Qwen3.6-MoE crossover and its TTFT, response-length, and Graph0
boundaries are in
[`results/qwen36_real_conversation_crossover_20260829.md`](results/qwen36_real_conversation_crossover_20260829.md).

```bash
python benchmarks/bench_real_conversation_concurrency.py \
  --manifest /tmp/wildchat_burstgpt_qwen35.json \
  --profiles short natural long --user-counts 5 10 20 \
  --output /tmp/real_conversation_concurrency.json
```

**`bench_dsv4_repo_concurrency.py`** — compares legacy and layered-pipeline on
fixed SWE-bench BM25 repository prompts arriving during sustained decode. Prepare
the companion manifest once, then pass its path to the run; results record that
path but do not copy or hash the manifest.

```bash
python benchmarks/bench_dsv4_repo_concurrency.py \
  --prepare-manifest /tmp/dsv4_repo_concurrency_manifest.json

python benchmarks/bench_dsv4_repo_concurrency.py \
  --manifest /tmp/dsv4_repo_concurrency_manifest.json \
  --modes legacy layered-pipeline --gpu 0 \
  --nowag-plugin-src /path/to/nowag/plugin/src \
  --output /tmp/dsv4_repo_concurrency.json
```

The runner has no machine-specific NoWAG plugin default. When the real plugin is
provided as a source checkout, `--nowag-plugin-src` must be passed explicitly.
The current four-user DSV4 result and service-policy boundary are in
[`results/dsv4_repo_concurrency_20260829.md`](results/dsv4_repo_concurrency_20260829.md).
The cache-policy A/B and independent Qwen3.6/DSV4 matrix are in
[`results/resident_next_use_eviction_20260829.md`](results/resident_next_use_eviction_20260829.md).

**`bench_natural_repo_agent.py`** — replays the same ten real SWE-bench BM25
repository prompts on Qwen3.6 and DSV4 using uncompressed BurstGPT arrival times
and per-request source response lengths. It has no artificial driver barrier and
reports paired per-request TTFT, TPOT, latency, and streamed-text gaps. The
natural-load result and its latency/stall trade-off are in
[`results/natural_repo_agent_qwen_dsv4_20260829.md`](results/natural_repo_agent_qwen_dsv4_20260829.md).

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.
