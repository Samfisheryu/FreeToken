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
multi-request ragged wave and advances it by one resident group per iteration. It
reserves one full expert layer outside the resident group for decode, so its shared
cache must hold at least two expert layers. `--max-prefill-length` estimates each
request's admission size; `--prefill-wave-max-chunks` is an aggregate complete-request
soft cap and does not create physical forward boundaries. Completion logs and JSON
expose `reqs`, `groups`, `group_forwards`, `iterations`, `decode_iterations`, and
`prefill_layer_prepares`. The generated MoE is a directional test, not a real-model
result.

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
text under BurstGPT session starts and human think times. It compares identical
fixed sessions across batching policies and records TTFT, TPOT, request latency,
inter-token gaps, expert traffic, and peak concurrency. The Qwen3.5-MoE result
and policy boundary are in
[`results/real_conversation_concurrency_20260828.md`](results/real_conversation_concurrency_20260828.md).

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
