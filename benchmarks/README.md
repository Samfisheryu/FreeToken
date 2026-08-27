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
mixed, layered, joint, layered-pipeline, and layered-prefill batching under a four-user, five-turn
tool-agent burst. It validates exact prompt/output lengths and prefix-cache reuse,
and records TTFT, TPOT, inter-token gaps and makespan. The defaults retain the
layered-pipeline G2/CPI1 baseline; `layered-pipeline-cpi2` is an explicit tuning
point. Layered-pipeline reserves one full expert layer outside its resident group
for decode, so its shared cache must hold at least two expert layers. The default
generated MoE is a directional test, not a real-model result. Resident-wave modes
can combine multiple requests by raising `--prefill-wave-max-chunks`; completion
logs and JSON expose request-chunk and frontier-batch counts separately.
The explicit `layered-prefill` mode instead materializes one ragged wave and
reports one `group_forwards` step per resident group; its chunk count is an
admission estimate, not a physical forward boundary.

```bash
python benchmarks/bench_lab_agent_policies.py --repetitions 3 --gpu 0 \
  --output /tmp/lab_agent_policies.json
```

Workload contract: [`workloads/lab_agent_burst_v1.json`](workloads/lab_agent_burst_v1.json).
Measured setup and interpretation: [`results/lab_agent_burst_20260825.md`](results/lab_agent_burst_20260825.md).
The focused resident-group probe and raw samples are recorded in
[`results/joint_group_wave_20260825.md`](results/joint_group_wave_20260825.md).
The group-major layered-pipeline design and final paired comparison are in
[`results/layered_pipeline_group_major_20260826.md`](results/layered_pipeline_group_major_20260826.md).
The paper-granularity one-group-per-iteration comparison for `layered-prefill`
is in [`results/layered_prefill_paper_fair_20260827.md`](results/layered_prefill_paper_fair_20260827.md).

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
