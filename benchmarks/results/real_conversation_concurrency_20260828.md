# Real-conversation concurrency on Qwen3.5-MoE

## Question and answer

Does `layered-pipeline` beat unchunked `legacy` scheduling on real multi-turn
chat text and real session timing?

Not on the tested Qwen3.5-MoE configuration. `layered-pipeline` consistently
reduced token-to-token latency and visible decode stalls, but it did not produce
a repeatable makespan win. Its first-token latency was much worse because a
prefill wave advances through 40 one-layer resident groups before returning its
first token. Use `legacy` when TTFT or total batch completion time is primary;
the pipeline is useful only when smoother in-flight decode is worth that trade.

## Workload and setup

- Real conversation text comes from
  [AllenAI WildChat](https://huggingface.co/datasets/allenai/WildChat). Real
  session starts and human think times come from
  [BurstGPT v2](https://github.com/HPMLL/BurstGPT/blob/main/README.md).
- Every policy replays the same materialized sessions, prompt text, arrival
  schedule, output limit, and random seeds. Each user has two turns; BurstGPT
  time is compressed by 120x. The direction sweep covers 5, 10, and 20 users.
- The fixed profiles have final-prompt ranges of 36--493 tokens (`short`),
  56--6,291 (`natural`), and 4,107--7,965 (`long`). All fit under `T=8192`, so
  the legacy baseline does not physically chunk these prompts.
- Model: Qwen3.5-MoE 35B-A3B, BF16, 40 layers, 256 experts, Triton attention,
  NoWAG expert offload, radix prefix cache, and `Graph8`.
- Both policies use a 512-expert shared cache. This holds exactly two complete
  expert layers, so the pipeline's only legal resident group is `G=1`; `W=64`.
  Both policies use the same graph setting, and no workload-specific prefill
  shape was captured.
- Every completed comparison below has identical prompt/completion usage and
  zero failed requests. Text can diverge between policies because batch shape
  changes near-tied greedy logits; later prompts are fixed, so performance work
  remains identical.

The public runner is
[`bench_real_conversation_concurrency.py`](../bench_real_conversation_concurrency.py).
Raw request/event JSON stays outside the repository.

## Direction sweep

This single-repetition sweep uses a 128-token response cap. Lower is better.

| Profile / users | Legacy makespan | Pipeline makespan | Change | TTFT p50, legacy -> pipeline | TPOT p50, legacy -> pipeline |
| --- | ---: | ---: | ---: | ---: | ---: |
| short / 5 | 14.584 s | 15.623 s | +7.1% | 399 -> 1,750 ms | 53.02 -> 39.81 ms |
| short / 10 | 27.342 s | 28.157 s | +3.0% | 519 -> 2,989 ms | 85.84 -> 54.75 ms |
| short / 20 | 52.428 s | 50.695 s | -3.3% | 628 -> 4,595 ms | 151.58 -> 93.43 ms |
| natural / 5 | 17.140 s | 22.624 s | +32.0% | 403 -> 2,720 ms | 61.77 -> 57.63 ms |
| natural / 10 | 34.366 s | 46.062 s | +34.0% | 765 -> 6,112 ms | 99.11 -> 104.40 ms |
| natural / 20 | 54.640 s | 71.402 s | +30.7% | 1,551 -> 9,644 ms | 155.96 -> 166.47 ms |
| long / 5 | 30.468 s | 32.715 s | +7.4% | 2,148 -> 4,510 ms | 86.06 -> 70.78 ms |
| long / 10 | 42.042 s | 49.800 s | +18.5% | 3,270 -> 6,750 ms | 113.79 -> 111.92 ms |
| long / 20 | 58.617 s | 74.233 s | +26.6% | 1,422 -> 10,075 ms | 182.78 -> 167.72 ms |

The apparent short/20 makespan win was the only favorable direction point, so
it was repeated independently below.

## Repeated boundary result

The exact short/20, 128-token-cap point was run three times with alternating
policy order. Each repetition contains 40 requests, 5,441 prompt tokens, and
3,621 completion tokens. Makespan summarizes the three repetitions; request
percentiles combine all 120 requests.

| Metric | Legacy p50 / p95 | Pipeline p50 / p95 | Pipeline change |
| --- | ---: | ---: | ---: |
| Makespan | 45.287 / 45.364 s | 45.889 / 46.431 s | +1.33% / +2.35% |
| TTFT | 581.8 / 879.1 ms | 3,822.6 / 6,422.7 ms | +557% / +631% |
| TPOT | 136.96 / 168.54 ms | 81.73 / 108.96 ms | -40.33% / -35.35% |
| Request latency | 11.908 / 20.194 s | 11.392 / 17.751 s | -4.34% / -12.10% |
| Largest SSE gap per request | 0.809 / 1.479 s | 0.161 / 0.805 s | -80.11% / -45.60% |
| Expert H2D | 914.82 / 916.92 GB | 806.07 / 806.15 GB | -11.89% / -12.08% |

All three individual makespan comparisons favored legacy: pipeline was 2.7%,
1.1%, and 1.1% slower. The earlier -3.3% point did not reproduce. The pipeline
does substantially improve decode smoothness and per-request tail latency, but
not aggregate completion time.

## Natural and longer-output checks

The natural/10 profile with a 256-token cap was also repeated three times. Each
repetition has 20 requests, 19,365 prompt tokens, and 3,196 completion tokens.

| Metric | Legacy p50 / p95 | Pipeline p50 / p95 | Pipeline change |
| --- | ---: | ---: | ---: |
| Makespan | 43.735 / 43.836 s | 46.312 / 46.867 s | +5.89% / +6.91% |
| TTFT | 455.2 / 776.6 ms | 3,666.6 / 6,134.6 ms | +706% / +690% |
| TPOT | 92.04 / 110.94 ms | 76.98 / 86.62 ms | -16.36% / -21.92% |
| Expert H2D | 812.61 / 813.13 GB | 779.49 / 779.73 GB | -4.08% / -4.11% |

Increasing the long/10 output cap to 512 tokens did not create a crossover:
legacy completed in 99.770 s and pipeline in 107.095 s (+7.34%). TTFT p50 was
0.685 vs 5.398 s; TPOT p50 was nearly tied at 96.48 vs 95.55 ms. Longer decode
therefore amortizes some pipeline overhead, but not enough to recover TTFT or
makespan on this configuration.

## Interpretation and boundary

The method is doing its intended job: it reduces expert transfers and prevents
long prefill work from blocking every decode token. That is why TPOT and maximum
token gaps improve. The cost is also inherent to this `G=1` configuration: a
new prefill request waits through 40 scheduler iterations before producing its
first token. Legacy instead completes the whole unchunked prefill immediately,
so it wins TTFT and usually makespan.

This result does not prove that all group sizes lose. `G=2/C768` would halve the
number of group iterations, but the tested Qwen3.5 service did not reach serving
under either Graph0 or Graph8 at that geometry; there is no valid performance
number for it. That startup/lifecycle defect must be fixed before testing a less
constrained cache point. Until then, the evidence supports keeping legacy as the
default for real chat traffic and treating layered-pipeline as an explicit
decode-smoothness policy.

## Reproduction

Create a fixed manifest once, then reuse it for every policy:

```bash
python benchmarks/bench_real_conversation_concurrency.py \
  --prepare-manifest /tmp/wildchat_burstgpt_qwen35.json \
  --user-counts 5 10 20
```

Run the direction sweep:

```bash
PYTHONOPTIMIZE=1 python benchmarks/bench_real_conversation_concurrency.py \
  --manifest /tmp/wildchat_burstgpt_qwen35.json \
  --profiles short natural long --user-counts 5 10 20 \
  --modes legacy layered-pipeline-g1-wave64 \
  --response-token-cap 128 --max-prefill-length 8192 \
  --moe-cache-size 512 --num-tokens 180000 \
  --output /tmp/real_conversation_direction.json
```

For the repeated boundary, select only `--profiles short --user-counts 20`, use
`--num-tokens 150000`, and run three fresh servers while alternating mode order.
