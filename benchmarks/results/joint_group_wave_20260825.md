# Joint group-wave probe

## Question

When decode is active, does keeping two consecutive MoE layers resident and reusing
them across two prefill chunks improve the current separate-forward `layered` policy?

## Setup

- One RTX 4090; public `ft serve` path with a generated 5-layer, 8-expert Qwen3-MoE
  checkpoint and the real offload MoE kernels.
- Each paired sample runs a 16-token prompt with 192 decode tokens while admitting a
  160-token prompt with one output token. There are 12 distinct prompt pairs.
- FP16, Triton attention, 32-token prefill budget, 24 expert slots, no CUDA graphs and
  no prefix-cache hits for every policy.
- `joint` uses group size 2. `wave1` admits one chunk per wave; `wave2` admits two.

## Result

| Policy | Wall time | Prefill latency | Max decode gap | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| legacy | 0.9141 s | 37.48 ms | 31.07 ms | 210.05 tok/s |
| mixed | 0.8593 s | 43.27 ms | 5.88 ms | 223.45 tok/s |
| layered, G=2 | 0.8707 s | 58.50 ms | 12.06 ms | 220.51 tok/s |
| joint, G=2, wave1 | **0.8469 s** | 35.28 ms | 6.23 ms | **226.71 tok/s** |
| joint, G=2, wave2 | 0.8819 s | **34.17 ms** | 9.94 ms | 217.71 tok/s |

Paired comparisons:

- `joint wave2` cut expert prepares and H2D bytes by 42.9% relative to `wave1`, but
  wall time was 4.45% slower and it lost all 12 paired samples.
- Relative to current `layered`, `joint wave2` made prefill 42.4% faster but wall time
  1.62% slower; it won only 4 of 12 wall-time samples.
- Relative to `mixed`, `joint wave2` made prefill 22.0% faster but wall time 2.50%
  slower and the largest decode gap 65.9% longer; it lost all 12 wall-time samples.
- `joint wave1` beat `layered` by 2.38% in all 12 pairs and beat `mixed` by 1.52% in
  10 of 12 pairs. This is not multi-chunk expert reuse.

The measured request used seven prefill chunks. Current `layered` advances decode while
admitting those chunks and at the two remaining layer-group boundaries, whereas
`joint wave2` produces only one decode token in each of four waves. The saved expert
loads shorten prefill, but the decode work left for later removes the end-to-end gain.

All 60 prefill outputs matched the sequential legacy reference. Decode matched in all
legacy/layered samples and in 10 of 12 mixed and joint samples. A focused long-output
joint test also has a deterministic mismatch. The implementation review found no shared
buffer, KV-range or asynchronous overwrite; exact batch-invariant greedy output is not
established, so this policy is not ready to become the default.

Decision: keep `wave2` as an experimental result, not a serving default. The next
scheduler should preserve decode progress and use multi-chunk reuse only when its saved
expert-transfer time exceeds the decode work it postpones.

Raw samples: [joint_group_wave_rtx4090_synthetic_qwen3moe_20260825.json](joint_group_wave_rtx4090_synthetic_qwen3moe_20260825.json)
