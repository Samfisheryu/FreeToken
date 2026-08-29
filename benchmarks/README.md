# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

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
