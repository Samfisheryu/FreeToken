# Small-lab multi-turn agent burst

## Question

Does a plausible low-concurrency workload make FreeToken's separate prefill/decode
batching visibly worse, and do the mixed or joint policies help?

## Workload

- Four closed-loop agent sessions, five turns each. Users start at 0/50/100/150 ms;
  later turns wait for the prior response plus a 50/100/150/200 ms user-specific
  delay.
- The main profile generates exactly 512 tokens per turn. Prompt lengths are
  640/1472/2304/3136/3968 tokens; radix-prefix reuse leaves 640 tokens of real
  prefill on turn 0 and 832 on each later turn.
- Each repetition therefore contains 15,872 new prefill tokens and 10,240 decode
  tokens. Later prompts use fixed transcripts and tool results, so every policy sees
  identical input even if an output diverges.
- This is a synthetic small-lab tool-agent burst, not an AIME trace or a claim about
  general chat traffic. The optional 192-token profile is exploratory only.

The public HTTP benchmark is
[`bench_lab_agent_policies.py`](../bench_lab_agent_policies.py), driven by
[`lab_agent_burst_v1.json`](../workloads/lab_agent_burst_v1.json).

## Setup

- RTX 4090; generated 5-layer, 8-expert, top-2 Qwen3-MoE; FP16 and the real
  FreeToken offload MoE path.
- The model has 40 layer-local experts while the cache has 24 slots, so all experts
  cannot remain cached at once. The generated expert bank is about 30 MiB.
- Triton attention, 64-token prefill budget, radix cache, no CUDA graphs, group size
  2 for layered/joint.
- Each policy starts a fresh server. Repetition 0 includes full-workload shape/kernel
  cold start; the long-running-serving conclusion uses repetitions 1 and 2. All cold
  measurements remain shown below.

Run from the repository root:

```bash
PYTHONPATH=python:. python benchmarks/bench_lab_agent_policies.py \
  --modes legacy mixed layered_g2_serial joint_g2_wave1 joint_g2_wave2 \
  --repetitions 3 --gpu 0 --output /tmp/lab_agent_policies_r3.json
```

All five modes completed 60/60 measured requests with the exact expected prompt,
cache-hit, new-prefill and output-token counts.

## Main result

Throughput delta is relative to legacy with identical work; positive is better.
Later-turn TTFT is P50. TPOT is P95. Request max-gap statistics first take the
largest inter-token gap within each request, then summarize the 40 requests in warm
repetitions 1 and 2.

| Policy | Makespan r0 / r1 / r2 | Warm throughput delta r1 / r2 | Later TTFT r1 / r2 | TPOT r1 / r2 | Request max gap P50 / P95 | Requests over 50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 13.210 / 13.332 / 13.538 s | baseline | 127.2 / 130.2 ms | 5.157 / 5.130 ms | 110.7 / 225.4 ms | 30 / 40 |
| mixed | 13.885 / 13.201 / 13.160 s | +0.99% / +2.87% | 95.1 / 93.2 ms | 4.803 / 4.724 ms | 9.2 / 11.6 ms | 0 / 40 |
| layered G2 | 14.350 / 13.830 / 13.744 s | -3.60% / -1.50% | 142.7 / 137.8 ms | 4.935 / 4.920 ms | 24.6 / 27.3 ms | 0 / 40 |
| joint G2 wave1 | 13.361 / 13.093 / 13.277 s | +1.83% / +1.96% | 81.2 / 82.8 ms | 4.714 / 4.875 ms | 7.8 / 11.7 ms | 0 / 40 |
| joint G2 wave2 | 13.277 / 13.271 / 13.703 s | +0.46% / -1.21% | 75.7 / 73.3 ms | 4.906 / 5.433 ms | 10.2 / 12.6 ms | 0 / 40 |

`joint wave1` is the only policy that improves throughput, first-turn and later-turn
TTFT, TPOT and maximum token gap in both warm repetitions. Its throughput gain is
small, about 1.8%--2.0%. `mixed` also improves warm throughput, later-turn TTFT, TPOT
and token gaps, but its first-turn TTFT P95 is 10.8%--14.8% worse. Fixed two-chunk
waves do not produce stable throughput or TPOT gains.

The robust serving result is the removal of visible decode stalls: 75% of warm legacy
requests see at least one gap above 50 ms, while mixed and both joint modes see none.
This is a tail-latency result, not a large end-to-end throughput result.

## Short-turn stress

The predeclared exploratory profile uses 192-token outputs and 512 new prefill tokens
on later turns. It makes prefill more frequent relative to decode, but does not enlarge
the throughput gain.

| Policy | Warm makespan r1 / r2 | Warm throughput delta r1 / r2 | Request max gap P95 | Requests over 50 ms |
| --- | ---: | ---: | ---: | ---: |
| legacy | 5.629 / 5.519 s | baseline | 115.2 ms | 9 / 40 |
| mixed | 5.717 / 5.751 s | -1.54% / -4.03% | 10.3 ms | 0 / 40 |
| layered G2 | 6.101 / 6.034 s | -7.73% / -8.54% | 19.6 ms | 0 / 40 |
| joint G2 wave1 | 5.550 / 5.576 s | +1.42% / -1.01% | 11.1 ms | 0 / 40 |
| joint G2 wave2 | 5.712 / 5.713 s | -1.46% / -3.39% | 10.2 ms | 0 / 40 |

Short decode often finishes before the next session's prefill arrives. The harmful
condition is therefore not simply "more prefills": it is repeated long-prefill
arrival while other users still have long decode in flight. The 512-token main
profile exercises that condition more consistently.

## Output consistency and decision

The main three-repetition run had one `joint wave1` mismatch in cold repetition 0;
three additional independent legacy/joint runs had zero mismatches in 60 joint
requests. The main total is therefore 1/120. In the short-turn stress, mixed had 3/60
and joint wave1 had 2/60 mismatches. Fixed-length generation and fixed later inputs
keep the timing work comparable, but strict batch-invariant greedy output is not
established.

Use this workload as a reproducible online tail-latency stress test. Do not claim a
large throughput win or AIME representativeness from the synthetic model. The next
useful validation is the same trace on the real target Qwen/NoWAG checkpoint; only
then decide whether joint wave1 deserves to become a serving default.

Full per-event JSON, including the three focused output-consistency reruns, is stored
under `/data1/lmcache_kv/experiments/freetoken_lab_agent_burst/20260825/`.
