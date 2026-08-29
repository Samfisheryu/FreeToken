# DSV4 repository-task concurrency

## Conclusion

`layered-pipeline` is useful when a shared service must keep existing coding
agents responsive while another user submits a very long repository prompt. It
is not an unconditional latency win.

In the repeated experiment below, `legacy` stopped the existing users' streamed
text for about 47.5 seconds. `layered-pipeline` capped that gap at about 0.67
seconds, reduced total expert H2D traffic by 17.2%, and changed existing-request
completion latency by only 0.27%. The cost was severe for the new 40K request:
its TTFT increased from 53.8 to 140.7 seconds and its completion latency nearly
doubled.

## Fixed workload and configuration

- Three coding-agent requests use real SWE-bench BM25 prompt text. Each has a
  128-token prompt and generates exactly 512 tokens.
- After all three requests emit their first non-empty SSE text event, one user
  submits the real `django__django-10999` repository task: a 40,000-token prompt
  followed by exactly 128 generated tokens.
- This is a simulated four-user shared-service scenario. The prompt content is
  real dataset material; the first-event release barrier and fixed output limits
  are workload choices, not claims about natural user timing.
- Model and runtime: DeepSeek-V4-Flash-0731, BF16, DSV4 sparse attention, NoWAG
  expert offload, radix cache, one RTX 4090, and `Graph8`.
- Both policies use the same `C=512`, memory ratio 0.7, maximum 16 running
  requests, requested `T=8192`, and identical prompts and output limits.
- The pipeline uses `G=1` and `W=1`. Since one DSV4 layer has 256 experts,
  `C=512` holds one resident prefill layer plus one decode layer; `G=1` is the
  capacity-constrained legal group size, not a tuned winning point.
- Three fresh-server repetitions use AB/BA/AB policy order. Every policy and
  repetition completed 4 requests, 40,384 prompt tokens, and 1,664 completion
  tokens with zero failures.

## Repeated result

The table reports the median of the three repetition-level values. Lower is
better. H2D uses decimal GB.

| Metric | Legacy | Layered-pipeline | Change |
| --- | ---: | ---: | ---: |
| Existing-user largest SSE-event gap, p95 | 47.536 s | 0.666 s | -98.60% |
| Existing-user TPOT, p50 | 345.72 ms | 331.46 ms | -4.13% |
| Existing-user completion latency, p50 | 182.634 s | 183.119 s | +0.27% |
| Existing-user TTFT, p50 | 5.967 s | 11.473 s | +92.27% |
| New 40K task TTFT | 53.817 s | 140.707 s | +161.45% |
| New 40K task TPOT | 250.73 ms | 217.33 ms | -13.32% |
| New 40K task completion latency | 85.663 s | 168.308 s | +96.48% |
| Whole-run makespan | 182.634 s | 189.929 s | +3.99% |
| Prefill expert H2D | 770.251 GB | 277.655 GB | -63.95% |
| Total expert H2D | 3,045.778 GB | 2,522.464 GB | -17.18% |
| Prefill layer prepares | 473 | 172 | -63.64% |

The direction is stable rather than a favorable single run. Pipeline makespan
was 3.95%, 4.07%, and 3.99% slower; existing-request completion latency changed
by +0.22%, +0.34%, and +0.27%. In every repetition, legacy produced six
repo-window SSE gaps longer than five seconds. Pipeline produced none longer
than one second; its largest observed gap was 0.681 seconds.

The policies generated different text for all four requests. Which policy owns
the benchmark's `output_mismatch` field flips with AB/BA order, so this records a
cross-policy numerical-path difference, not a request failure. Fixed token work,
usage, failure count, and per-policy expert traffic are identical across all
three repetitions. These results are performance evidence, not an exact-output
equivalence test.

## Why it happens

At startup, the pipeline reported an effective tile limit of 4,864 tokens after
applying the cache geometry to requested `T=8192`. The 40K prompt therefore used
nine static tiles. DSV4 has 43 layers and `G=1`, so its logical wave performed
`43 * 9 = 387` group forwards but prepared each expert layer only once. Across
all four measured requests, pipeline recorded 172 layer prepares versus 473 for
legacy.

This is not a workload-specific prefill graph result. The DSV4 adapter does not
support resident layer-range graph capture, so the pipeline's group/tile work is
eager. `Graph8` only accelerates the ordinary whole-model decode path shared by
both policies. The measured mechanism is resident expert reuse across static
tiles plus one decode opportunity per pipeline iteration.

## Decision boundary

Use `layered-pipeline` when an already-visible stream must not freeze because
another user opens a large repository context. Use `legacy` when the new long
request's TTFT or completion latency is more important than incumbent-session
smoothness. The scheduler exposes a real service-policy tradeoff; it does not
dominate legacy on every objective.

## Reproduction

Prepare one fixed manifest, then reuse it for every run:

```bash
python benchmarks/bench_dsv4_repo_concurrency.py \
  --prepare-manifest /tmp/dsv4_repo_concurrency_manifest.json
```

Run fresh servers three times, alternating `--modes legacy layered-pipeline`
and `--modes layered-pipeline legacy`:

```bash
PYTHONOPTIMIZE=1 python benchmarks/bench_dsv4_repo_concurrency.py \
  --manifest /tmp/dsv4_repo_concurrency_manifest.json \
  --modes legacy layered-pipeline --cases repo40k_x1 \
  --driver-count 3 --driver-decode-tokens 512 --repo-decode-tokens 128 \
  --gpu 0 --memory-ratio 0.7 --moe-cache-size 512 \
  --max-running-requests 16 --max-prefill-length 8192 \
  --prefill-layer-group-size 1 --prefill-wave-max-chunks 1 \
  --cuda-graph-max-bs 8 --nowag-plugin-src /path/to/nowag/plugin/src \
  --output /tmp/dsv4_repo_concurrency.json
```

The measured raw JSON files remain outside the repository as
`fair_repo40k_decode128_ab1.json`, `fair_repo40k_decode128_ba2.json`, and
`fair_repo40k_decode128_ab3.json`.
