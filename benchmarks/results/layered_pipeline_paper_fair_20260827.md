# Layered-pipeline paper-granularity comparison

## Comparison contract

`layered-pipeline` implements the paper's one-group-per-iteration schedule. A
wave freezes its request membership before group 0, materializes every member's
outstanding prompt range as one ragged batch, and advances that batch through
exactly one resident expert group per scheduler iteration. Planned chunks are
used for admission only; they are not physical forward boundaries.

The public scaled benchmark uses the same generated FP16 Qwen3-MoE, request
materialization, Triton attention, 24-slot expert cache, requested group size 2,
and `T=512` for both policies. One 128-token driver decodes 512 tokens while four
2,048-token prompts arrive after its first streamed token and each generates one
token. Across three repetitions, each policy serves 15 requests with 24,960
prompt tokens and 1,548 completion tokens. All runs completed with zero request
failures.

## Results

Lower is better. H2D is measured expert-weight traffic in decimal GB.

### CUDA Graph enabled (`Graph8`, benchmark default)

| Metric | Legacy | Layered-pipeline G2/W64 | Reduction |
| --- | ---: | ---: | ---: |
| Prefill TTFT p50 / p95 (ms) | 368.383 / 534.960 | 124.093 / 140.395 | 66.31% / 73.76% |
| Driver TPOT p50 / p95 (ms) | 2.427 / 2.582 | 1.994 / 2.208 | 17.84% / 14.48% |
| Makespan p50 / p95 (s) | 1.2785 / 1.3561 | 1.0526 / 1.1695 | 17.67% / 13.76% |
| Expert H2D p50 (GB) | 21.9446 | 16.6724 | 24.03% |

### CUDA Graph disabled (`Graph0`)

| Metric | Legacy | Layered-pipeline G2/W64 | Reduction |
| --- | ---: | ---: | ---: |
| Prefill TTFT p50 / p95 (ms) | 365.608 / 537.778 | 112.887 / 133.152 | 69.12% / 75.24% |
| Driver TPOT p50 / p95 (ms) | 7.407 / 7.485 | 6.701 / 6.758 | 9.53% / 9.71% |
| Makespan p50 / p95 (s) | 3.8281 / 3.9236 | 3.4592 / 3.4877 | 9.64% / 11.11% |
| Expert H2D p50 (GB) | 21.9446 | 12.2683 | 44.09% |

Legacy and layered-pipeline use the same graph setting within each table.
Layered-pipeline wins all four reported client-side and transfer metrics even
with graphs disabled, so the gain comes from resident-group scheduling and
expert reuse rather than a prefill-shape CUDA graph.

This is a directional result for the generated scaled model, not universal TTFT
dominance. On the separate default unchunked short-prompt boundary, legacy still
returns the first token sooner; layered-pipeline targets the paper's looser
token-between-token latency boundary under concurrent long-prefill contention.

## Reproduction

Run from the repository root. Omitting `--model` creates the public scaled model;
an existing generated model directory can be supplied to reuse it.

```bash
PYTHONOPTIMIZE=1 python benchmarks/bench_scaled_expert_contention.py \
  --modes legacy layered-pipeline-g2-wave64 \
  --repetitions 3 --max-prefill-length 512 \
  --output /tmp/layered_pipeline_scaled_paper_fair_r3.json
```

Repeat with CUDA graphs disabled:

```bash
PYTHONOPTIMIZE=1 python benchmarks/bench_scaled_expert_contention.py \
  --modes legacy layered-pipeline-g2-wave64 \
  --repetitions 3 --max-prefill-length 512 \
  --cuda-graph-max-bs 0 \
  --output /tmp/layered_pipeline_scaled_paper_fair_graph0_r3.json
```
