# Joint unified expert pool

## Scope

`joint` uses one canonical HBM expert pool. Prefill and decode share every slot.
`legacy` and `mixed` keep their existing behavior as comparison baselines.

The previous joint design reserved separate prefill and decode regions. Its measured
group/wave optimum does not select parameters for this design.

## Addressing

Let:

- `L` be the number of MoE layers;
- `E` be the number of experts per layer;
- `C` be the number of physical expert slots in HBM;
- `M[layer, expert]` be the physical slot, or `-1` when absent;
- `R[slot]` be the flat logical expert id, or `-1` when empty.

For every resident expert, the maps are inverse:

```text
R[M[layer, expert]] = layer * E + expert
```

Prefill and decode translate routed expert ids through `M` and read the canonical
slot banks directly. A cache hit is never copied into a second contiguous buffer.
Quantization metadata presented to compute uses the same physical-slot indices.

## Group admission

For a group beginning at layer `k`, its effective size is:

```text
G_eff(k) = min(G_requested, L - k, floor(C / E))
```

`joint` therefore requires `C >= E`; it does not reserve another full layer for
decode because all requests in the joint batch traverse the same active group.

The group working set is:

```text
Q(k) = {(layer, expert) |
        k <= layer < k + G_eff(k), 0 <= expert < E}
```

Admission protects `Q(k)` as one working set:

1. Wait for the previous group's release, then protect every resident member of
   `Q(k)` before choosing the first victim.
2. Admit complete layers, filling misses from empty slots and then least-recently-used
   non-group slots. Because `C >= |Q(k)|`, a layer already admitted cannot become a
   victim while the rest of this group enters.
3. Transfer only misses and publish one ready event per layer, allowing the current
   layer's compute to overlap admission of later layers.
4. Give the admitted group a current recency epoch; routed accesses may refine it.

The current group is the protected working set. Physical contiguity is only a
tie-breaker among equivalent empty or eviction slots; existing hits are not moved.

## Release

Group completion records a compute-stream release event. Slots cannot be overwritten
until dependent GPU work has passed that event. Releasing the group ends its protection
but leaves its mappings and weights resident, so later prefill or decode can hit them.

This gives the intended locality hierarchy:

```text
active group protection > in-flight safety > temporal recency > slot-order tie-break
```

## Evaluation boundary

Client-side TTFT starts immediately before request sending and includes queueing.
TPOT covers the interval from the first to last non-empty streamed token and divides
by `completion_tokens - 1`.

The final optimized 64-token, `G2/W1` run used 100 warm requests:

| Policy | TTFT P50 | TTFT P95 | TPOT P50 | TPOT P95 |
| --- | ---: | ---: | ---: | ---: |
| legacy | 111.176 | 155.497 | 4.763 | 5.076 |
| mixed | 93.487 | 100.273 | 4.620 | 4.797 |
| joint G2/W1 | **86.170** | **91.518** | **4.563** | **4.747** |

All values are milliseconds. Relative to `mixed`, joint improved TTFT P50/P95 by
7.83%/8.73% and TPOT P50/P95 by 1.23%/1.04%. `W1` means one scheduler chunk per
resident-group wave; each request still spans many chunks. This run therefore does
not exercise reuse across chunks within one wave. Its gain comes from canonical pages
remaining resident across waves and from group-resident execution. See the
[final W1 result](../benchmarks/results/joint_unified_pool_wave1_20260826.md) for
profiling and validity details.

An earlier [complete-prefill run](../benchmarks/results/joint_unified_pool_full_prefill_chunk_20260826.md)
raised the token budget so each request fit in one chunk. Its within-run comparison
also favored joint over mixed, showing a benefit without chunk reuse, but it predates
the final hot-path fixes and is only a cross-run reference.

A full unified-pool sweep over group size and multi-chunk wave size is still required
before selecting a new sweet spot. The old partitioned-cache `G2/W7` result does not
apply to this design.
