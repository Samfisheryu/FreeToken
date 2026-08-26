# Unified expert pool: complete prefill in one chunk

> **Cross-run reference.** This experiment predates the final unified-pool hot-path
> fixes. Its within-run policy comparison isolates the complete-prefill case, but its
> absolute latency must not be combined with the final 64-token optimized run.

## Result

The scheduler token budget was raised from 64 to 1024 so each request's complete
640- or 832-token new prefill, plus concurrent decode rows, fits in one scheduler
chunk. All other small-lab workload and 24-slot HBM settings were unchanged.

The table uses the 100 warm requests from repetitions 1 through 5. Lower is better;
all values are milliseconds.

| Policy | TTFT P50 | TTFT P95 | TPOT P50 | TPOT P95 |
| --- | ---: | ---: | ---: | ---: |
| legacy | 19.266 | 23.558 | **4.512** | **4.593** |
| mixed | 18.945 | 23.511 | 4.583 | 4.739 |
| joint G2/W1 | **15.923** | **21.377** | 4.535 | 4.700 |

Relative to `mixed`, joint reduces TTFT P50/P95 by 15.95%/9.08% and TPOT
P50/P95 by 1.06%/0.82%. Relative to `legacy`, joint reduces TTFT P50/P95 by
17.35%/9.26% but increases TPOT P50/P95 by 0.49%/2.32%.

Across the five warm repetitions, joint beat mixed in 5/5 runs at TTFT P50,
4/5 at TTFT P95, 4/5 at TPOT P50, and 4/5 at TPOT P95.

## Interpretation

This result isolates group execution from cross-chunk reuse: one complete prompt
chunk has nothing to reuse across chunks, yet `joint G2` still improves TTFT over
both baselines and improves TPOT over `mixed`. Group-resident execution therefore
has a benefit of its own. It does not dominate `legacy` TPOT at this point.

This is distinct from the final
[one-chunk-per-wave, 64-token result](joint_unified_pool_wave1_20260826.md), where
the request still spans many chunks and optimized `joint G2/W1` beats `mixed` on
all four client-side metrics. The runs answer different questions and their absolute
latencies should not be compared directly.

## Validity

- Generated 5-layer, 8-expert, top-2 Qwen3-MoE; 24 HBM expert slots.
- Four closed-loop users, five turns each, 512 requested output tokens.
- Six repetitions per policy; repetition 0 treated as cold, repetitions 1--5 as warm.
- All 360 requests had valid measurements, expected 640/832 new-prefill counts,
  and reference-matching output.
- Client-side TTFT includes queueing from request send time. TPOT uses the first and
  last non-empty streamed text events and `completion_tokens - 1`.

## Reproduction detail

The benchmark was run with the normal `main` workload and its server configuration
overridden in memory to `max_prefill_length=1024`; no workload file was changed.
The table above is the tracked result record; raw per-event output is not published.
