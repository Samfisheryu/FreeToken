# Qwen3.6 real-conversation crossover

## Conclusion

On the tested shared-chat workload, `layered-pipeline` has a repeatable region
where it improves more than one isolated metric. With 10 or 20 concurrent users
and a 128-token response cap, it reduces overall TPOT, overall request latency,
streamed-text stalls, and expert H2D traffic while changing makespan by at most
1.5%. Its cost is substantially worse TTFT.

This is a workload crossover, not a universal replacement for `legacy`. Lower
concurrency and longer input profiles do not meet the same whole-system bar.

## Workload and decision rule

- Prompt text is from WildChat. Session starts and human think durations are
  from BurstGPT v2 and compressed by 120x.
- Each user has two closed-loop turns. The same session identities, text, think
  durations, seeds, and token limits are used for both policies. A second turn
  starts after the first finishes plus its think duration, so its absolute
  submission time legitimately depends on the policy.
- Model: Qwen3.6-35B-A3B, BF16, Triton attention, NoWAG expert offload, radix
  cache, one RTX 4090, `C=512`, `T=8192`, `G=1`, `W=64`, and Graph8.
- A candidate had to improve TPOT p50 and p95 by at least 10%, keep request
  latency p50/p95 and makespan within 5% of legacy, complete identical token
  work with zero failures, and improve at least one additional resource or
  responsiveness metric. Two adjacent concurrency points had to pass.

The 10-user case has 20 requests, 2,492 prompt tokens, and 1,842 completion
tokens per policy. The 20-user case has 40 requests, 5,790 prompt tokens, and
4,085 completion tokens. Every repetition completed with zero failures and
identical per-request usage.

## Graph8 confirmation: AB/BA/AB

The table reports the median of three fresh-server repetitions. Request columns
are p50 / p95; lower is better. H2D uses decimal GB.

| Users | Policy | TTFT (s) | TPOT (ms) | Request latency (s) | Makespan (s) | Expert H2D |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | legacy | 0.517 / 0.848 | 85.95 / 101.51 | 8.698 / 12.152 | 27.407 | 529.67 GB |
| 10 | pipeline | 2.799 / 3.793 | 53.12 / 78.66 | 8.275 / 11.707 | 27.734 | 471.90 GB |
| 20 | legacy | 0.557 / 0.841 | 148.43 / 188.64 | 15.468 / 21.464 | 48.928 | 986.64 GB |
| 20 | pipeline | 4.323 / 6.720 | 87.35 / 114.25 | 12.836 / 20.520 | 48.938 | 876.50 GB |

Across all three repetitions:

- 10 users: TPOT improves 38.2% / 22.4--22.5%, request latency improves
  4.8--4.9% / 3.5--3.7%, H2D improves 10.9--11.1%, and makespan changes
  +1.0--+1.3%.
- 20 users: TPOT improves 41.1% / 35.0--39.9%, request latency improves
  11.2--17.6% / 3.0--4.9%, H2D improves 9.4--11.3%, and makespan changes
  -0.1--+1.5%.
- The maximum streamed-text-event gap improves by 64--89%, but this is
  supporting evidence rather than the selection criterion.

TTFT is the explicit tradeoff: its median rises from 0.52 to 2.80 seconds at
10 users and from 0.56 to 4.32 seconds at 20 users. Use legacy when immediate
first-token service matters more than steady generation and completion latency.

## Boundaries

The benefit is not limited to a 128-token cap. In one fresh-server Graph8 run
with a 256-token cap, TPOT p50/p95 improves 16.8%/15.5% at 10 users and
23.2%/20.0% at 20 users. Request latency does not regress, makespan changes
-1.3%/+0.7%, and H2D improves 8.3%/5.4%.

Graph0 separates scheduling from graph replay. At 20 users, pipeline still
improves TPOT by 40.7%/33.4%, request latency by 21.1%/3.6%, and H2D by 10.3%,
with a 2.2% makespan cost. At 10 users it still improves TPOT and H2D, but
request latency and makespan regress 8--12%. The scheduler benefit is therefore
real at higher concurrency, while CUDA Graphs remain important at the lower
load point.

The predeclared direction sweep also tested 5 users and `natural` inputs.
Short/5 failed the request-latency rule. Natural/5 and natural/10 failed the
latency or makespan rules. Natural/20 exposed a real resource boundary: the
pipeline ran out of activation memory with `T=8192` and a 269K-token KV pool;
the runner stopped, so no `long` result was produced. These cells are not
silently excluded from the conclusion.

Greedy text differs across policies for 7--29 requests depending on case and
run. The JSON `output_mismatch` owner flips with AB/BA reference order; fixed
usage and zero failures are stable, but this benchmark is not an exact-output
equivalence result.

## Reproduction

Use the same fixed manifest for every run, then alternate mode order across
fresh servers:

```bash
PYTHONOPTIMIZE=1 python benchmarks/bench_real_conversation_concurrency.py \
  --manifest /path/to/wildchat_burstgpt_qwen36.json \
  --profiles short --user-counts 10 20 \
  --modes legacy layered-pipeline-g1-wave64 \
  --response-token-cap 128 --max-prefill-length 8192 \
  --moe-cache-size 512 --num-tokens 269000 --cuda-graph-max-bs 8 \
  --output /tmp/qwen36_real_conversation.json
```

Raw request/event results remain outside the repository as
`current_qwen36_short10_20_gpu1_{ab1,ba2,ab3}_20260829.json`, with separate
`cap256` and `graph0` boundary files.
