# Resident next-use eviction for layered-pipeline

## Change

Layered-pipeline resident-group admission now evicts the cached expert page
whose owner layer is used farthest in the next causal layer traversal. Empty
slots remain first choice; hard-pinned pages, query hits, and the active and
target groups remain ineligible. Distance ties use the existing LRU epoch and
then the physical slot number.

The policy is owned by the expert cache. The scheduler supplies only whether
decode is present in the current iteration. There is no new CLI, model branch,
host synchronization, H2D transfer, or allocation. Legacy batching, joint
batching, ordinary prefill, and ordinary decode keep their previous policies.

## Independent black-box coverage

Eight fresh public-service lifecycles passed:

- synthetic candidate and read-only LRU baseline with Graph0 and Graph8;
- Qwen3.6-MoE with Graph0 and Graph8;
- DeepSeek-V4 with `dsv4_sparse`, Graph0 and Graph8.

The matrix covers `G=2/3`, a partial final group, `C=3E/4E`, `T=96/128`,
`W=1/4`, concurrency 1/2/4/8, ragged tiles, an oversized logical wave,
cached follow-up, repeated service use, and decode overlap. Public SSE, usage,
iteration limits, six-field wave accounting, cache statistics, and process
cleanup all passed. Qwen used an explicit 150,000-token KV capacity so the
`C=3E` expert geometry and model activations fit on the 24 GiB test GPU.

## Real-workload A/B

All compared runs used identical requests and token counts, with zero request
failures and zero output mismatches. Negative deltas are improvements.

### Qwen3.6 WildChat/BurstGPT, 20 users x 2 turns

Old LRU is the mean of two fresh-server runs; next-use is one final-code run.
Each run has 40 requests, 5,790 prompt tokens, and 4,085 decode tokens.

| Metric | LRU | Next-use | Delta |
|---|---:|---:|---:|
| Decode missing rows | 913,809 | 899,707 | -1.54% |
| Total expert H2D | 875.77 GB | 864.33 GB | -1.31% |
| Makespan | 48.744 s | 48.413 s | -0.68% |
| TPOT p50 | 87.62 ms | 85.54 ms | -2.38% |
| TPOT p95 | 113.96 ms | 113.04 ms | -0.80% |
| Request latency p50 | 12.737 s | 12.576 s | -1.26% |
| TTFT p50 | 4.308 s | 4.638 s | +7.67% |
| Maximum SSE gap p50 | 0.132 s | 0.166 s | +25.43% |

This point improves cache traffic, sustained decode, request latency, and total
completion time, but it is not an unconditional latency win: first-token time
and the median per-request maximum streamed-text gap regress.

### DeepSeek-V4 SWE-bench 40K repository context

The workload has three 512-token decode drivers and one real 40K SWE-bench BM25
repository prompt generating 128 tokens. Old LRU is the mean of three
fresh-server runs; next-use is one final-code run. Every run has 4 requests,
40,384 prompt tokens, and 1,664 decode tokens.

| Metric | LRU | Next-use | Delta |
|---|---:|---:|---:|
| Decode missing rows | 352,897 | 343,679 | -2.61% |
| Prefill expert H2D | 277.66 GB | 275.80 GB | -0.67% |
| Total expert H2D | 2,522.46 GB | 2,461.98 GB | -2.40% |
| Makespan | 189.949 s | 189.746 s | -0.11% |
| Driver TPOT p50 | 331.49 ms | 332.02 ms | +0.16% |
| Driver latency p50 | 183.139 s | 183.123 s | -0.01% |
| Driver maximum SSE gap p95 | 0.666 s | 0.633 s | -5.01% |
| Repository TTFT | 140.731 s | 141.167 s | +0.31% |
| Repository TPOT | 217.32 ms | 214.06 ms | -1.50% |

At the constrained `C=512` DSV4 point, next-use removes more expert traffic
without a material makespan, foreground TPOT, or request-latency regression.

## Evidence

- Qwen LRU: `resident_eviction_lru_qwen36_short20_gpu1_r1_retry_20260829.json`
  and `resident_eviction_lru_qwen36_short20_gpu1_r2_20260829.json`
- Qwen next-use: `resident_eviction_nextuse_opt_qwen36_short20_gpu1_r1_20260829.json`
- DSV4 LRU: `fair_repo40k_decode128_{ab1,ba2,ab3}.json`
- DSV4 next-use: `resident_eviction_nextuse_repo40k_decode128_gpu1_r1_20260829.json`

The JSON files live under the corresponding `/data1/lmcache_kv/experiments/`
result directories and are not repository artifacts.
