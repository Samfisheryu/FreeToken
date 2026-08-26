# Unified expert pool: one chunk per joint wave

## Result

Under the same small-lab workload and 24-slot HBM budget as the historical
partitioned-cache experiment, optimized unified-pool `joint G2/W1` is faster
than both `legacy` and `mixed` on every client-side metric.

The table uses the 100 warm requests from repetitions 1 through 5. Lower is better;
all values are milliseconds.

| Policy | TTFT P50 | TTFT P95 | TPOT P50 | TPOT P95 |
| --- | ---: | ---: | ---: | ---: |
| legacy | 111.176 | 155.497 | 4.763 | 5.076 |
| mixed | 93.487 | 100.273 | 4.620 | 4.797 |
| joint G2/W1 before fixes | 107.465 | 134.089 | 4.723 | 4.961 |
| joint G2/W1 optimized | **86.170** | **91.518** | **4.563** | **4.747** |

Relative to the initial unified implementation, the optimized path reduces TTFT
P50/P95 by 19.82%/31.75% and TPOT P50/P95 by 3.39%/4.32%. Relative to `mixed`,
it reduces TTFT by 7.83%/8.73% and TPOT by 1.23%/1.04%.

## Implementation finding

Two implementation overheads dominated the gap:

- Assigning a Python scalar through advanced indexing issued an 8-byte pageable
  H2D and a host `cudaStreamSynchronize` for every admitted layer. The 1,000-group
  profile went from 2,000 stream synchronizations to zero.
- The first unified path implemented mapping, group protection, recency release,
  and miss accounting as many generic PyTorch operations. Direct int32 indexing
  reduced the lookup portion of one hot map from 11 generic kernels to one lookup
  kernel before the existing in-place update. Removing redundant repins and recency
  increments, accumulating misses inside the existing LRU kernel, reusing mask
  storage, and using separate LRU query/output tensors reduced a warm G2 lifecycle
  from 19 to 10 device events. Its micro median fell from 894.5 to 607.3 microseconds
  per wave.

The production path still uses the original flashlib LRU and copy kernels;
`python/freetoken/moe/offload_kernels.py` is unchanged.

## Interpretation

`W1` means one scheduler chunk per resident-group wave, not that the complete
prompt fits in one chunk. With a 64-token chunk budget, every chunk traverses the
five MoE layers as three groups. It cannot reuse one admitted group across chunks;
the win here comes from the shared pool retaining expert pages across waves and
from removing avoidable admission overhead.

The remaining group mask, release, mapping, LRU, and conditional-copy launches
are observable in the final trace. The LRU and copy work is required for canonical
admission.

## Validity

- Generated 5-layer, 8-expert, top-2 Qwen3-MoE; 24 HBM expert slots.
- Four closed-loop users, five turns each, 512 requested output tokens.
- Six repetitions per policy; repetition 0 treated as cold, repetitions 1--5 as warm.
- All 100 warm optimized-joint requests had valid measurements and matched the
  reference output. All 120 requests including cold were valid with zero output
  mismatches.
- Runtime logs reported `chunks=1`, `groups=3`, `effective_group_size=2`, and five
  layer prepares per wave, with no fallback or backend error.
- Independent black-box validation covers cold, repeated, overlapping, LRU, stats,
  asynchronous, bank-row, and device-event behavior.

TTFT starts immediately before the client sends the request and includes queueing.
TPOT is measured from the first to last non-empty streamed text event and divided by
`completion_tokens - 1`.

## Reproduction

```bash
PYTHONPATH=python:. python benchmarks/bench_lab_agent_policies.py \
  --modes legacy mixed jointG2-wave1 \
  --repetitions 6 --gpu 1 \
  --output /tmp/unified_joint_single_chunk.json
```
