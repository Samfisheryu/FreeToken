# Natural repo-agent trace: Qwen3.6 and DSV4

## Result

This natural-time trace does **not** show a general latency win for
`layered-pipeline`. Its repeatable value in this run is narrower: it removes long
streaming stalls when new repository prompts arrive during decode.

- Qwen3.6 was lightly loaded. TPOT and request latency were effectively flat,
  while p95 maximum streamed-text gap fell from 859 ms to 477 ms.
- DSV4 exposed the trade-off clearly. P95 maximum streamed-text gap fell from
  5.41 s to 0.62 s, but p95 TTFT, TPOT, and request latency regressed by 95.8%,
  17.4%, and 15.7%.

This is one paired run per model. It is a useful natural-load boundary, not a
claim about average production traffic or statistical stability.

## Fixed workload

The policies saw the same open-loop trace; no timing was compressed and no
driver or first-token barrier was used.

- Ten real SWE-bench BM25 repository prompts, using the manifest's 4K context
  variant.
- BurstGPT source arrival offsets: `0, 3, 80, 101, 104, 168, 174, 254, 574,
  595` seconds.
- BurstGPT source response lengths: `18, 121, 18, 442, 640, 139, 146, 111,
  132, 21` tokens.
- Requests used `ignore_eos=true` so each policy performed exactly the same
  token work. The length varies per request; there is no shared 128/512-token
  cap.
- Each policy used a fresh server. Qwen ran legacy then layered; DSV4 ran
  layered then legacy.

The prompts and arrival/length traces come from real datasets, but their pairing
is a simulated shared coding-agent service scenario.

Common scheduler configuration was C512, T8192, G1, W64, Graph8, radix cache,
and max-running 16. Qwen used Triton attention and a 150K-token KV pool. DSV4
used `dsv4_sparse` attention and memory ratio 0.7; its public effective tile
limit was 4864 tokens.

## Fairness

| Model | Requests | Prompt tokens / policy | Completion tokens / policy | Failures | Peak inflight, legacy / layered |
|---|---:|---:|---:|---:|---:|
| Qwen3.6 | 10 | 43,951 | 1,788 | 0 | 2 / 2 |
| DSV4 | 10 | 40,000 | 1,788 | 0 | 3 / 4 |

Every task had identical prompt usage and completion usage across policies.
Submission-delay p95 was below 0.2 ms in all four arms. DSV4 layered had more
inflight requests because requests lived longer, not because it received more
input.

Outputs differed across policies for 6/10 tasks on each model. This is reported
but is not a fairness failure: prompts and generated-token counts remained
identical. The result therefore compares serving performance for equal token
work, not semantic output quality.

## Aggregate user metrics

Negative change is better.

| Model | Metric | Legacy p50 / p95 | Layered p50 / p95 | Layered change p50 / p95 |
|---|---|---:|---:|---:|
| Qwen3.6 | TTFT | 1.146 / 1.595 s | 1.032 / 1.972 s | -10.0% / +23.6% |
| Qwen3.6 | TPOT | 17.220 / 25.643 ms | 17.335 / 25.453 ms | +0.7% / -0.7% |
| Qwen3.6 | Request latency | 3.206 / 14.733 s | 3.252 / 15.319 s | +1.4% / +4.0% |
| Qwen3.6 | Max streamed-text gap | 20.016 / 859.212 ms | 20.419 / 476.809 ms | +2.0% / -44.5% |
| DSV4 | TTFT | 5.671 / 10.797 s | 7.700 / 21.144 s | +35.8% / +95.8% |
| DSV4 | TPOT | 102.973 / 255.645 ms | 121.589 / 300.055 ms | +18.1% / +17.4% |
| DSV4 | Request latency | 16.308 / 91.207 s | 20.283 / 105.483 s | +24.4% / +15.7% |
| DSV4 | Max streamed-text gap | 259.072 / 5413.779 ms | 245.035 / 619.134 ms | -5.4% / -88.6% |

Layered won fewer individual requests than the aggregate stall metric might
suggest:

| Model | Lower TTFT | Lower TPOT | Lower request latency | Lower max streamed-text gap |
|---|---:|---:|---:|---:|
| Qwen3.6 | 4/10 | 3/10 | 4/10 | 3/10 |
| DSV4 | 2/10 | 7/10 | 1/10 | 8/10 |

Trace elapsed time was 596.10 vs 596.37 s on Qwen (+0.05%) and 601.78 vs
603.97 s on DSV4 (+0.36%). The 595-second arrival span dominates those values.

## Qwen3.6 requests

`L` is legacy and `P` is layered-pipeline. TPOT is milliseconds; other metrics
are seconds.

| Task | Arrival | Output | Prompt | TTFT L / P | TPOT L / P | Latency L / P |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 18 | 4,464 | 1.590 / 1.029 | 16.376 / 16.423 | 1.869 / 1.309 |
| 1 | 3 | 121 | 4,533 | 1.535 / 1.039 | 17.424 / 17.346 | 3.626 / 3.122 |
| 2 | 80 | 18 | 4,431 | 1.575 / 1.038 | 16.969 / 17.003 | 1.864 / 1.328 |
| 3 | 101 | 442 | 4,066 | 0.731 / 0.982 | 28.146 / 28.134 | 13.144 / 13.390 |
| 4 | 104 | 640 | 4,448 | 1.600 / 2.725 | 22.585 / 22.177 | 16.032 / 16.896 |
| 5 | 168 | 139 | 4,402 | 0.775 / 1.031 | 17.118 / 17.963 | 3.138 / 3.511 |
| 6 | 174 | 146 | 4,448 | 0.761 / 1.030 | 17.321 / 17.324 | 3.273 / 3.542 |
| 7 | 254 | 111 | 4,167 | 0.735 / 1.008 | 16.369 / 16.682 | 2.536 / 2.844 |
| 8 | 574 | 132 | 4,602 | 1.517 / 1.051 | 17.786 / 17.791 | 3.848 / 3.382 |
| 9 | 595 | 21 | 4,390 | 0.764 / 1.032 | 16.676 / 16.802 | 1.098 / 1.369 |

The one meaningful Qwen overlap was tasks 3/4. Task 3's maximum streamed-text
gap fell from 1.536 s to 0.840 s, while task 3/4 latency rose by 0.246/0.864 s.

## DSV4 requests

| Task | Arrival | Output | Prompt | TTFT L / P | TPOT L / P | Latency L / P |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 18 | 4,000 | 10.828 / 7.748 | 143.650 / 320.207 | 13.271 / 13.193 |
| 1 | 3 | 121 | 4,000 | 8.020 / 14.645 | 78.199 / 71.307 | 17.405 / 23.203 |
| 2 | 80 | 18 | 4,000 | 5.434 / 7.650 | 77.001 / 76.806 | 6.744 / 8.957 |
| 3 | 101 | 442 | 4,000 | 10.759 / 7.651 | 125.571 / 162.849 | 66.137 / 79.468 |
| 4 | 104 | 640 | 4,000 | 7.874 / 17.677 | 162.509 / 170.720 | 111.718 / 126.767 |
| 5 | 168 | 139 | 4,000 | 5.544 / 14.969 | 275.864 / 275.426 | 43.614 / 52.979 |
| 6 | 174 | 146 | 4,000 | 5.798 / 23.981 | 230.932 / 204.134 | 39.284 / 53.581 |
| 7 | 254 | 111 | 4,000 | 5.415 / 7.608 | 80.374 / 80.328 | 14.257 / 16.445 |
| 8 | 574 | 132 | 4,000 | 5.435 / 7.592 | 74.621 / 74.589 | 15.211 / 17.363 |
| 9 | 595 | 21 | 4,000 | 5.419 / 7.613 | 67.947 / 67.769 | 6.778 / 8.969 |

DSV4 makes the policy trade-off visible. Tasks 4/5 had maximum streamed-text
gaps of about 5.414 s under legacy; layered reduced them to 0.461/0.625 s. But
tasks 4/5/6 TTFT rose by 9.802/9.425/18.183 s and request latency rose by
15.049/9.365/14.297 s.

## Expert traffic

| Model | Prefill H2D, legacy / layered | Total H2D, legacy / layered | Prefill prepares |
|---|---:|---:|---:|
| Qwen3.6 | 83.36 / 81.16 GB (-2.64%) | 403.50 / 404.91 GB (+0.35%) | 400 / 400 |
| DSV4 | 700.23 / 686.16 GB (-2.01%) | 2625.13 / 2683.50 GB (+2.22%) | 430 / 430 |

Layered reduced prefill expert traffic slightly, but extra decode traffic erased
that saving in this trace.

## Artifacts and reproduction

Raw results:

- `/data1/lmcache_kv/experiments/freetoken_natural_repo_agent_20260829/qwen36_legacy_layered_r1_gpu2.json`
- `/data1/lmcache_kv/experiments/freetoken_natural_repo_agent_20260829/dsv4_layered_legacy_r1_gpu2.json`

The public runner is `benchmarks/bench_natural_repo_agent.py`. Its dry run shows
the complete request plan and server commands. Real NoWAG source checkouts must
be passed explicitly:

```bash
PYTHONPATH=python:. python benchmarks/bench_natural_repo_agent.py \
  --model-profile qwen36 --modes legacy layered-pipeline --gpu 0 \
  --nowag-plugin-src /path/to/nowag/src --output /tmp/qwen-natural.json

PYTHONPATH=python:. python benchmarks/bench_natural_repo_agent.py \
  --model-profile dsv4 --modes layered-pipeline legacy --gpu 0 \
  --nowag-plugin-src /path/to/nowag/src --output /tmp/dsv4-natural.json
```
