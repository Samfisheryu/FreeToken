# Qwen3.6 long-context mixed online goodput final A/B

## Conclusion

On one RTX 4090, the best upstream FreeToken arm reached `188.694110 tasks/h`
and the best ServeBig arm reached `199.362545 tasks/h`:

```text
ServeBig / upstream FreeToken = 199.362545 / 188.694110 = 1.056538249x
```

ServeBig reduced the online denominator, latency, TTFT, TPOT, and peak VRAM,
but both arms produced exactly 13 correct, SLO-compliant tasks. The result does
not meet the `2x+` goal.

There is no honest next workload candidate with measured evidence of a `2x`
best-to-best separation. No new final should run until a production prefill
optimization demonstrates on development data both mean TTFT at or below
`16.2 s` and capacity for at least 111 submissions under the equivalent
180-second, 20-user traffic contract, without accuracy loss.

Result artifacts: [U1 result](/data1/lmcache_kv/goodput_campaign/long_v2_final_real_20260901_v1/U1-upstream-best-c8-mr090-final.json)
and [S3 result](/data1/lmcache_kv/goodput_campaign/long_v2_final_real_20260901_v1/S3-servebig-best-c8-mr095-final.json).

## Frozen comparison

- The final stream contains real DocFinQA numeric tasks, SWE-bench Verified
  tasks with BM25 repository context, and non-code-repository LongBench-v2
  multiple-choice tasks. Prompt buckets are 8K, 16K, and 32K Qwen tokens.
- Twenty closed-loop users enter at `0.5 * user_index` seconds, keep one request
  in flight, wait two seconds after each terminal, submit for 180 seconds, and
  then fully drain. The queue does not wrap.
- Numeric, code, and knowledge output caps/SLOs are respectively `128/90 s`,
  `2048/180 s`, and `32/60 s`; the hard timeout is 210 seconds.
- Both arms used the same official Qwen3.6-35B-A3B-NVFP4 model entry, BF16
  compute, FlashInfer attention, MoE offload, naive KV, prefill-hit D2D,
  `max-running=8`, CUDA Graph batch 8, 8192-token prefill chunks, and a
  40960-token maximum sequence length.
- Both arms ran on GPU0 and recorded the same device UUID. Under the stated
  target-GPU isolation rule, the measured GPU0 online intervals were isolated.
  An external GPU1 job overlapped only after the online denominator closed and
  did not execute on the measured serving path.

## Final configurations

| Setting | U1: best upstream FreeToken | S3: best ServeBig |
|---|---|---|
| Memory ratio | `0.90` | `0.95` |
| KV capacity | 393277 tokens | 393216 tokens |
| Expert cache | automatic, 6057 slots | 10240 slots |
| Batching | upstream default | mixed |
| Recovery | none | general-mix joint recovered sidecar |

## Aggregate result

Latency, TTFT, and TPOT are client-observed seconds. The denominator includes
the fixed submission window and the complete drain tail.

| Metric | U1 | S3 |
|---|---:|---:|
| Valid task-goodput result | yes | yes |
| Submitted | 64 | 72 |
| Judge correct / SLO success | 13 / 13 | 13 / 13 |
| Accuracy / SLO success rate | 20.3125% / 20.3125% | 18.0556% / 18.0556% |
| Denominator / drain tail | 248.0205 / 68.0205 s | 234.7482 / 54.7482 s |
| Goodput | 188.6941 tasks/h | 199.3625 tasks/h |
| Latency mean / P95 | 64.6329 / 142.8788 | 55.0770 / 126.5206 |
| TTFT mean / P95 | 39.8336 / 52.2100 | 35.5444 / 48.6711 |
| TPOT mean / P95 | 0.58718 / 2.75967 | 0.29525 / 0.91220 |
| Output tokens / tokens per second | 8219 / 33.1384 | 7409 / 31.5615 |
| Peak VRAM | 22.0098 GiB | 19.7637 GiB |
| Request timeout | 0 | 1 |

Relative to U1, S3 submitted `1.125x` as many tasks, completed `1.000x` as many
correct tasks, used a `0.94649x` denominator, and used `0.89795x` peak VRAM.
Output-token throughput is descriptive only because the generated outputs and
submitted task mix differ.

The production `/v1/stats` schema does not expose expert-cache misses or expert
H2D bytes. Both values remain `null`; no estimate replaces them.

## Family and prompt-length result

| Family | U1 accuracy | S3 accuracy | U1 goodput | S3 goodput |
|---|---:|---:|---:|---:|
| Numeric | 13.043% | 11.538% | 43.5448 tasks/h | 46.0067 tasks/h |
| Code | 0% | 0% | 0 | 0 |
| Knowledge | 52.632% | 50.000% | 145.1493 tasks/h | 153.3558 tasks/h |

| Prompt bucket | U1 goodput | S3 goodput |
|---|---:|---:|
| 8K | 72.5747 tasks/h | 76.6779 tasks/h |
| 16K | 72.5747 tasks/h | 61.3423 tasks/h |
| 32K | 43.5448 tasks/h | 61.3423 tasks/h |

Neither a task family nor a prompt bucket establishes a `2x` opportunity.
Selecting the weaker upstream c16 development configuration, narrowing to a
favorable slice after measurement, or changing the correctness/SLO definition
would not be a best-to-best product comparison.
