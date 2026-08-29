# Periodic long-prefill lab

> Historical result: this measurement predates the static-tile executor. Its
> one-segment interpretation and performance values do not describe the current
> implementation. See
> [`dsv4_repo_concurrency_20260829.md`](dsv4_repo_concurrency_20260829.md) for a
> current real-model result.

## Workload

This public-HTTP benchmark uses a generated FP16 Qwen3-MoE model with 8
layers, hidden size 512, MoE intermediate size 4,096, 8 experts per layer, and
top-2 routing. Three independent 128-token prompts start together and each
decodes 2,048 tokens. Once all three drivers emit their first non-empty SSE text
event, a fourth lane submits six independent 12,288-token prefills serially;
each produces one token, followed by 100 ms before the next submission.

Both policies use Triton attention, `T=8192`, Graph8, maximum sequence length
16,384, and at most four running requests. The policies are legacy and
layered-pipeline G1/W2. Each current result has three repetitions: 27 requests,
222,336 prompt tokens, 18,450 completion tokens, and zero failures per policy.

Each long request has `K=ceil(12288/8192)=2` for admission accounting. The
pipeline still materializes every admitted request as one ragged segment: each
repetition has 9 requests, 64 groups, 64 group forwards, 64 iterations, 56
decode iterations, and 64 layer prepares. G1 is the finest possible grouping
for this eight-layer model.

## Current results

Lower is better. Values separated by `/` are p50/p95. The prefill-active gap is
the interval between adjacent non-empty driver SSE text events when that
interval overlaps a long-prefill active window; an event can contain multiple
tokens. H2D is expert-weight traffic in decimal GB.

### Shared cache C24

| Metric | Legacy | Layered-pipeline G1/W2 |
| --- | ---: | ---: |
| Driver TTFT (ms) | 72.859 / 74.885 | 115.321 / 118.374 |
| Driver TPOT (ms) | 12.320 / 12.333 | 12.088 / 12.414 |
| Prefill-active SSE gap (ms) | 29.052 / 57.190 | 17.218 / 27.558 |
| Long-prefill TTFT (ms) | 106.028 / 112.273 | 178.008 / 189.696 |
| Makespan (s) | 25.295 / 25.313 | 24.859 / 25.443 |
| Prefill / total H2D p50 (GB) | 11.274 / 604.257 | 6.065 / 590.403 |

Layered-pipeline's p50 makespan is 1.72% below legacy. Its p95 is 0.51%
above legacy.

### Shared cache C40

| Metric | Legacy | Layered-pipeline G1/W2 |
| --- | ---: | ---: |
| Driver TTFT (ms) | 71.958 / 76.375 | 70.980 / 75.969 |
| Driver TPOT (ms) | 4.302 / 4.329 | 4.042 / 4.789 |
| Prefill-active SSE gap (ms) | 6.893 / 50.073 | 10.838 / 17.990 |
| Long-prefill TTFT (ms) | 95.222 / 100.041 | 122.347 / 130.580 |
| Makespan (s) | 8.878 / 8.932 | 8.348 / 9.710 |
| Prefill / total H2D p50 (GB) | 11.274 / 166.082 | 3.674 / 147.069 |

Layered-pipeline's p50 makespan is 5.97% below legacy. It has eight cross-policy
output mismatches at C40 and nine at C24. Token usage and request completion
remain fixed, but later routing differs, so the p95 and H2D spread are not a
same-route causal comparison.

## Shared cache-policy A/B

The final cache replacement applies to ordinary over-capacity decode across
both policies. It is not a layered-pipeline-only optimization.
When a page must be evicted, it keeps experts from layers that will be used
sooner in the fixed forward layer order and evicts the farthest layer first,
then the oldest page. It retains the original LRU when one decode query's full
layer working set fits in cache.

| Cache | Policy | Makespan p50, old -> current (s) | Total H2D p50, old -> current (GB) |
| --- | --- | ---: | ---: |
| C24 | Legacy | 44.688 -> 25.295 (-43.4%) | 1123.579 -> 604.257 (-46.2%) |
| C24 | Layered-pipeline | 43.914 -> 24.859 (-43.4%) | 1099.860 -> 590.403 (-46.3%) |
| C40 | Legacy | 27.438 -> 8.878 (-67.6%) | 662.541 -> 166.082 (-74.9%) |
| C40 | Layered-pipeline | 31.160 -> 8.348 (-73.2%) | 757.303 -> 147.069 (-80.6%) |

A strict four-request, same-workload profile with identical wave structure
isolates the cache change while keeping 22,006 active decode rows and 4,122
layer calls unchanged:

| Metric | Old LRU | Current cache policy |
| --- | ---: | ---: |
| Decode misses | 21,958 | 12,502 |
| Total expert H2D (GB) | 278.561 | 159.514 |
| Client span (s) | 11.088 | 6.638 |
| Device busy union (s) | 11.055 | 6.603 |
| Client span not covered by device union (ms) | 26.8 | 29.3 |

The admission kernel costs about 3.3 microseconds per call. Almost all of the
wall-time reduction therefore comes from fewer expert copies rather than hiding
more host work.

## Boundaries and causal alternatives

- G1 remains the sustained-concurrency point. At C24, G2 increased makespan by
  18.8% and H2D by 38.6%. At C40 on the same-seed long run, G2 increased
  makespan by 6.5%, H2D by 10.3%, and the prefill-active gap from 16.75 to
  27.97 ms.
- On the held-out page trace, layer-distance produced 9,724 misses versus
  17,794 for LRU and 9,062 for hard-pin-aware Belady. It captures 92.4% of the
  oracle's savings. Previous-token protection saved only 12 more pages; a
  training-selected decayed-LFU policy saved 449 pages, about 3.3% of the
  active window. Both causal additions were rejected.
- Layered-pipeline has no fixed-prefill-shape CUDA graph. Graph8 is shared decode
  infrastructure, so these results are not a shape-specific prefill-capture
  gain.

## Reproduction

Run from the repository root. Omitting `--model` creates the public scaled
model.

```bash
PYTHONOPTIMIZE=1 python benchmarks/bench_scaled_periodic_long_prefill.py \
  --repetitions 3 --moe-cache-size 24 \
  --output /tmp/scaled_periodic_long_prefill_layer_distance_c24_r3.json

PYTHONOPTIMIZE=1 python benchmarks/bench_scaled_periodic_long_prefill.py \
  --repetitions 3 --moe-cache-size 40 \
  --output /tmp/scaled_periodic_long_prefill_layer_distance_c40_r3.json
```
