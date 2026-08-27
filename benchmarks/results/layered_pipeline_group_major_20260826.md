# Layered-pipeline group-major scheduling

## Design

`layered-pipeline` keeps one expert group resident while every admitted prompt
chunk advances through that group. It releases the group only after all saved
chunk states reach its end, then replays those states through the next group.
This is group-major order: a wave never returns to a released group.

The public controls are independent:

- `T = --max-prefill-length` limits tokens in one request chunk.
- `P = --layered-pipeline-chunks-per-iteration` limits chunks advanced per
  scheduler iteration. A frontier contains at most one next chunk per request.
- `W = --prefill-wave-max-chunks` is an aggregate soft admission cap. The first
  request remains whole even when it exceeds `W`; later requests join only when
  their complete chunk count fits.

Group 0 can admit newly runnable requests at frontier boundaries. Membership is
frozen before group 1. Saved states are then repacked into canonical frontiers of
at most `P` requests without repeating KV allocation. The first frontier can be
mixed with decode; later frontiers in that iteration are prefill-only. Decode
still traverses the full model and samples at most one token per iteration.

For `C` shared expert slots, `E` experts per layer, requested group size `G`, and
`L` MoE layers, layered-pipeline reserves one full layer for decode outside its
persistent group:

```text
effective_G = min(G, L, floor(C / E) - 1), with C >= 2E
```

Joint does not retain a group across iterations, so its geometry remains
`min(G, L, floor(C / E))`.

## Scaled expert-contention workload

The public scaled benchmark uses a generated FP16 Qwen3-MoE on one RTX 4090:

| Geometry | Value |
| --- | ---: |
| MoE layers / experts / top-k | 8 / 8 / 2 |
| Hidden / MoE intermediate | 512 / 4096 |
| One expert page | 12 MiB |
| All expert weights | 768 MiB |
| Shared cache | 24 pages / 288 MiB |

One 128-token driver begins a 512-token decode. After its first streamed token,
four 2,048-token prefill requests arrive together and each generates one token.
Every repetition therefore has 8,320 prompt tokens and 516 decode tokens.
Serving uses Triton attention, radix caching, `T=128`, CUDA graphs through batch
size 8, and `PYTHONOPTIMIZE=1`.

## Five-policy graph-on comparison

This paired three-repetition run predates the final late-arrival repack described
below. Each policy served 15 requests and the same 24,960 prompt / 1,548 decode
tokens with zero request failures. Client TTFT includes queueing; TPOT uses
streamed token arrival times.

| Policy | Driver TTFT p50 / p95 (ms) | Driver TPOT p50 / p95 (ms) | Prefill TTFT p50 / p95 (ms) | Makespan p50 / p95 (s) | Expert H2D p50 / p95 (GiB) | Output mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 35.711 / 37.591 | 5.469 / 5.633 | 1341.958 / 2095.946 | 2.833 / 2.915 | 56.438 / 58.494 | 0 |
| mixed | 36.617 / 37.738 | 5.702 / 5.901 | 1421.884 / 2175.326 | 2.950 / 3.054 | 60.469 / 60.722 | 3 |
| old layered G8 | 34.600 / 35.926 | 9.599 / 9.607 | 635.270 / 1012.728 | 4.940 / 4.946 | 99.562 / 99.562 | 0 |
| joint G2, W64 | 34.584 / 42.150 | 1.805 / 2.552 | **153.843 / 233.808** | 0.966 / 1.340 | 11.273 / 18.509 | 3 |
| layered-pipeline G1, P16, W64 | **34.255 / 41.748** | **1.787 / 2.261** | 288.353 / 366.397 | **0.956 / 1.198** | **10.172 / 15.519** | 2 |

Joint and layered-pipeline reduce median expert traffic by about 5.0x and 5.5x
relative to legacy. They also reduce median makespan by about 2.9x. Joint reaches
the four long-prefill first tokens sooner; pipeline has slightly lower median
makespan and expert traffic. These are directional results for this scaled model,
not a production-model ranking.

The long greedy generations are not bitwise stable across policy-specific batch
shapes: the table reports every cross-policy mismatch rather than treating it as
a request failure. Fixed-order 16-token Graph0/Graph8 comparisons were exact, and
usage, token counts, KV lengths, and wave accounting remained closed.

## Multi-request accounting and late-arrival repack

In the canonical repetition, both resident policies form two waves: a one-chunk
driver wave and one 64-chunk wave containing all four long requests. The burst
has 16 frontier batches. Joint performs eight group admissions and 16 layer
prepares across the two waves. Pipeline performs 520 chunk-group steps, 136
frontier-group forwards, 40 iterations, and 16 layer prepares. With `G=1`, its
current group, complete next group, and decode reserve exactly fill `C=24`, so all
seven burst group transitions can prefetch the next group.

Arrival timing can make group 0 observe 32 small frontiers before membership is
frozen. The retained repack leaves group 0 unchanged but rebuilds the seven later
groups into 16 canonical frontiers. In the observed late case this reduced total
driver-plus-burst layer-group calls from 264 to 152 without changing chunks,
queries, prepares, KV allocation, or outputs. Non-profile makespan fell from
1.2252 s to 0.9228 s (302.4 ms); the three-run p95 fell 7.1%. A canonical
16-frontier wave is a no-op. Nsight instrumentation amplified the one-time Python
builder cost: its packed device work still fell 23.3 ms while uncovered profile
time increased, so profile and wall-clock effects are reported separately.

## Corrected device bound

The earlier kernel-only calculation was invalid because CUDA Graph decode work
appears in `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`, not as expanded ordinary kernel
rows. The corrected counterfactual uses:

```text
packed = non-copy graph/model base
       + decode_misses * page_copy_time
       + exposed prefill-copy tail
```

At the measured 26.0–26.2 GB/s, one 12 MiB page costs 0.480–0.484 ms. The
identities below reconstruct their actual GPU activity unions before any page
replacement counterfactual:

| Profile envelope | Shape | Decode / prefill misses | GPU union (ms) | Recovered non-copy base (ms) |
| --- | --- | ---: | ---: | ---: |
| joint G2 | 32 burst frontiers | 1529 / 117 | 1246.0–1246.2 | 463.3–470.8 |
| pipeline G1 | 16 burst frontiers | 909 / 106 | 906.6 | 442.5–445.8 |

A separate device route trace recorded one real decode row per token and matched
the cache's final owners, usage epochs, and public miss counters exactly. Joint
and pipeline differed at only one of 4,088 token-layer routes; replaying either
canonical route did not change the miss totals.

| Policy envelope | Recorded current pages (decode + prefill) | Current packed (ms) | Past-only causal packed (ms) | Belady-oracle packed (ms) |
| --- | ---: | ---: | ---: | ---: |
| joint G2, 32-frontier envelope | 659 + 122 | 822.6–834.3 | 773.4–784.7 | 644.0–654.1 |
| pipeline G1, 16-frontier envelope | 757 + 111 | 831.4–837.5 | 795.1–801.0 | 633.7–638.4 |

The causal policy chooses decay and previous-route protection using only the
first 20% of routes. On the held-out 80%, it changes joint decode misses from 448
to 392 (12.5%) and pipeline from 451 to 432 (4.2%). Belady is an oracle bound,
not an implementable eviction policy.

The joint numbers remain a conservative 32-frontier counterfactual. Two final
attempts both reproduced 32 rather than 16 frontiers, so no matching 16-frontier
joint device trace is claimed. Their client spans were 1,435.7–1,463.4 ms, device
spans 1,432.0–1,459.9 ms, and uncovered time 189.7–217.2 ms. The matching
pipeline trace measured 979.1 ms client, 975.4 ms device span, 906.6 ms GPU union,
and 72.5 ms uncovered.

Pipeline's uncovered time splits into 8.29 ms during pure decode, 57.93 ms across
the two active-wave regions, 2.60 ms at the wave transition, and 3.70 ms outside
the device span. Inside active waves, eager/kernel enqueue gaps are 43.10 ms
(4.4% of client span), metadata/memcpy gaps 11.43 ms, and range-replay launch gaps
0.95 ms. No single general, non-shape-specific component exceeds 5%, so the
performance implementation stops here rather than adding another specialized
capture or metadata path.

## Reproduction

Run the public benchmark from the repository root. Omit `--model` to generate the
scaled model, or reuse an existing generated directory across runs.

```bash
PYTHONOPTIMIZE=1 python benchmarks/bench_scaled_expert_contention.py \
  --modes legacy mixed layered-g8 joint-g2-wave64 \
          layered-pipeline-g1-cpi16-wave64 \
  --repetitions 3 --gpu 2 \
  --max-prefill-length 128 --moe-cache-size 24 --cuda-graph-max-bs 8 \
  --output /tmp/scaled_expert_group_major_r3.json
```
