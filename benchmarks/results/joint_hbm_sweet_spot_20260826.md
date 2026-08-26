# Joint batching sweet spot under a 24-slot expert-cache budget

> **Historical result.** This experiment measured the retired design that hard-partitioned
> HBM into protected prefill slots and decode slots. Its `G=2, W=7` result must not be
> used to configure the unified expert pool. See
> [Joint unified expert pool](../../docs/joint-unified-expert-pool.md) for the current
> formulation. The optimized unified `W1` result is available, but selecting a new
> multi-chunk sweet spot still requires a full group/wave sweep.

## Historical conclusion for the partitioned design

For this small-lab workload, the partitioned design selected:

```text
prefill_layer_group_size = 2
prefill_wave_max_chunks = 7
```

This is a real multi-chunk wave. It is not the wave1 execution-only gain.

## Fixed setup

- One RTX 4090; generated 5-layer, 8-expert, top-2 Qwen3-MoE model.
- One 24-slot HBM expert-cache budget shared by protected prefill slots and the
  decode cache.
- Four closed-loop users, five turns each, 512 output tokens per request.
- New prefill is 640 tokens on turn 0 and 832 tokens on later turns; the public
  scheduler token budget is 64.
- TTFT is measured from the client timestamp immediately before HTTP request
  construction/sending to the first non-empty SSE text event. It includes client
  setup, connection, service queueing and scheduling.
- TPOT is the standard client-observed interval from first to last output event,
  divided by `completion_tokens - 1`; pre-first-token queueing is represented in
  TTFT and is not counted twice.

The sweep covered every feasible group size, `G in {1, 2}`, and wave caps 1 through
14. The main validation used one cold repetition followed by five warm repetitions,
or 100 warm requests per policy. All measured requests had the expected prompt,
prefix-cache, new-prefill and output-token counts.

## Formulation

Let:

- `C` be the total expert slots in HBM;
- `E` be the experts per MoE layer;
- `L` be the number of MoE layers;
- `B` be the shared scheduler token budget;
- `P` be the request's new-prefill tokens;
- `d` be the active decode tokens placed in the first mixed batch.

Reserve at least one full expert layer for decode. The feasible group bound is:

```text
G_max = min(L, floor((C - E) / E))
```

Here, `C=24`, `E=8`, and `L=5`, so `G_max=2`. The sweep confirms that `G=2`
beats `G=1` at the useful cross-chunk points: fewer layer-group boundaries matter
more than the eight extra decode-cache slots left by `G=1`.

Because decode consumes the first mixed batch's token budget, the effective chunk
count is:

```text
N(P, d) = ceil((P + d) / B)
```

The 832-token later turn is 13 chunks by itself but 14 chunks whenever at least one
decode token shares its first batch. For a target of `K` waves, use the smallest cap
that reaches that target:

```text
W(K) = ceil(N_max / K)
```

Then choose the smallest `K` whose client-side TPOT P50 and P95 do not exceed the
mixed-policy reference, while TTFT P50 and P95 also improve. In this workload:

```text
K=1 -> W=14: TPOT fails the constraint
K=2 -> W=7:  all four constraints pass
```

This gives `G*=2, W*=7`.

The boundary is visible in the runtime logs. Every wave loads all five MoE layers
once: 5 layer prepares and 31,457,280 H2D bytes.

- W6 executes the long request as `6+6+2`: three expert loads.
- W7 executes it as `7+7`: two balanced expert loads.
- W8 executes it as `8+6`: the same two loads, but a longer first decode stall.
- W14 executes it as `14`: one load, but the longest decode stall.

Thus W7 is the smallest cap that removes the third expert load without making either
of the remaining waves unnecessarily large.

## Validation result

The table uses the 100 warm requests from the focused validation run. Lower is
better; all values are milliseconds.

| Policy | TTFT P50 | TTFT P95 | TPOT P50 | TPOT P95 |
| --- | ---: | ---: | ---: | ---: |
| legacy | 110.102 | 154.757 | 4.702 | 4.923 |
| mixed | 96.134 | 103.173 | 4.757 | 4.881 |
| joint G2/W1 | 80.175 | 88.952 | 4.658 | 4.889 |
| joint G2/W6 | 69.311 | 74.904 | 4.699 | 4.920 |
| **joint G2/W7** | **65.940** | **74.558** | **4.634** | **4.847** |
| joint G2/W8 | 67.152 | 93.041 | 4.716 | 5.406 |

Relative to legacy, G2/W7 reduces TTFT P50/P95 by 40.1%/51.8% and TPOT P50/P95
by 1.45%/1.54%. Relative to mixed, it reduces TTFT P50/P95 by 31.4%/27.7% and
TPOT P50/P95 by 2.57%/0.70%.

The TTFT result is stable: G2/W7 beat mixed in all five warm repetitions at both
P50 and P95. TPOT is a smaller result: it beat mixed in 5/5 repetitions at P50 and
3/5 at P95. The combined TPOT direction also matched the independent full sweep,
but the P95 margin should be described as modest rather than large.

The focused validation had no measurement failures. G2/W7 had no output mismatch
in 120 total requests; mixed had one. Other sweep points also produced four isolated
output mismatches, so exact batch-invariant greedy output is still not established.

## Reproduction

The benchmark and workload are
[`bench_lab_agent_policies.py`](../bench_lab_agent_policies.py) and
[`lab_agent_burst_v1.json`](../workloads/lab_agent_burst_v1.json).

```bash
PYTHONPATH=python:. python benchmarks/bench_lab_agent_policies.py \
  --modes legacy mixed jointG2-wave1 \
  --joint-groups 2 --joint-waves 8 7 6 \
  --repetitions 6 --gpu 0 --output /tmp/lab_agent_joint_validate.json
```

The raw SSE files are intentionally not tracked; the three runs generated about
301 MiB of per-event JSON.
