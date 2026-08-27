# Layered Prefill 目标设计与当前实现偏差

本文是 FreeToken 中 layered prefill 的目标设计规范，也是对当前 group-major
实现的审计结论。目标设计来自论文
[From Tokens to Layers](https://arxiv.org/html/2510.08055)，并按 FreeToken 的
expert offload 场景扩展。

## 结论

正确的调度单位是 **layer group**，不是 token chunk，也不是 frontier。

一次 iteration 应完成：

1. decode batch 经过完整模型并产生一个新 token；
2. 当前 pinned group 同时处理 decode 和整个 prefill wave；
3. 其他 group 只处理 decode；
4. prefill wave 的中间状态停在当前 group 之后，下一次 iteration 再进入下一个
   group。

因此，同一个 iteration 内，同一个 active group 必须只执行一次。active group
收到的是一个 ragged batch：decode rows 与 wave 中所有 prefill rows 拼在一起。
不能因为 prompt 在 bookkeeping 上被表示成多个 chunk，就对同一个 group 重复做
多个 forward。

这里的“一次 iteration 一个 forward”指一个完整的模型调度步：每个 group 在这一
步里最多执行一次。实现可以保留分段的 layer-group API，但不允许同一 active group
因为多个 frontier 被重复调用。

## 论文原本的设计

论文的规则非常直接：

- 每次 iteration 只有一个指定 group 同时做 decode 和 prefill；
- 其他 group 都只做 decode，因此 decode 不停顿；
- prefill 每次前进一个 group，经过 `N` 个 groups 后恰好完成；
- prompt 在每一层只经过一次，不沿 token 轴反复重跑模型；
- 多个同时到达的小请求应合并成一个 batch。

论文通过增加 layer group 数控制每轮的工作量，而不是默认把 prompt 切成很多小
chunk。它给出的实验配置是：

```text
N_groups(prompt_length) = max(1, ceil(prompt_length / 512))
```

例如，8192-token prompt 使用16个 groups。每轮计算“8192 tokens × 1/16模型”，
其工作量约等于“512 tokens × 完整模型”。layer 轴的切分本身已经承担了控制 TBT 的
作用。

论文允许 layered prefill 与 chunking 组合，但这是处理超长输入时的补充手段，并且
应使用尽可能大的 chunk。chunking 不是 layered prefill 的默认执行单位。

参考：

- [论文 §4.1–4.4](https://arxiv.org/html/2510.08055#S4)
- [作者仓库的算法说明](https://github.com/scale-snu/layered-prefill#algorithm)

## FreeToken 应实现的调度

### Wave

一个 wave 是一起完成 layered prefill 的一组请求。wave 保存：

- 每个请求的完整 prefill token 范围；
- ragged attention 所需的请求边界和位置；
- 所有 prefill rows 的 hidden/residual state；
- 当前已经通过的 layer group；
- 每个请求的 KV、结束位置和最终输出 row。

wave 可以有多个请求，也可以只有一个长请求。它不应该由一串必须分别 forward 的
frontier 组成。

### 一次 iteration

假设本轮 active group 是 `Gi`：

```text
decode:  G0 -> ... -> Gi-1 -> [Gi] -> Gi+1 -> ... -> Gn -> sample one token
                                   ^
prefill wave:             enter [Gi] once, then park its state
```

执行顺序为：

1. decode rows 先经过 `G0 ... Gi-1`；
2. 把这些 decode rows 与 wave 的全部 prefill rows 拼成一个 ragged batch；
3. `Gi` 对这个 mixed batch 只做一次 forward；
4. 将输出重新分成 decode state 和 prefill state；
5. decode state 继续经过 `Gi+1 ... Gn` 并采样一个 token；
6. prefill state 保存到 wave，下一轮从 `Gi+1` 继续。

当 `Gi` 是最后一个 group 时，wave 的 prefill 完成并产生各请求的首 token。

### 必须保持的约束

- 有 runnable decode 时，每个 iteration 产生一个 decode token；
- 每个 group 每个 iteration 最多执行一次；
- active group 的一次执行同时包含 decode rows 和全部 wave prefill rows；
- 每个 prefill token 在每一层恰好计算一次；
- 请求之间只共享物理 batch，不共享 attention 因果边界；
- KV allocation/accounting 对每个 prefill token只发生一次；
- 只有 wave 过大、显存或 TBT 预算确实容纳不下时才切大 chunk；切分后也不能把小
  chunk 当成默认 forward 边界。

## 当前实现实际在做什么

当前实现已经具备 resident group、partial-layer state 和 mixed decode/prefill 的
基础能力，但正常执行路径仍然是“group 内逐 frontier”。

### 1. Frontier 仍是 token chunk 的执行单位

`scheduler/resident_wave.py` 中：

- `ResidentFrontier` 保存“每个 wave member 最多一个 next chunk”；
- `admit_resident_frontiers()` 每轮给每个请求取一个受 `token_budget` 限制的 chunk；
- 多轮 admission 形成多个 frontier。

这说明 token 轴切分没有只停留在 bookkeeping 层，而是进入了执行计划。

### 2. 一个 iteration 会对同一个 group 做多个 forward

`scheduler/layered_pipeline.py` 中：

- `_select_frontiers()` 一次选择多个 frontier；
- `_begin_iteration_state()` 只把第一个 frontier 与 decode state 合并；
- `_advance_selected_frontiers()` 先执行第一个 mixed frontier，然后在
  `selected[1:]` 循环中逐个调用 `advance_layer_group_prefill()`；
- `advance_step()` 直接以 `len(selected)` 累加 `frontier_group_forwards`。

因此，当前一次 iteration 不是“active group 一次 mixed forward”，而是：

```text
forward 1: decode + frontier 0
forward 2: frontier 1
forward 3: frontier 2
...
```

这是当前实现与目标设计最根本的偏差。

### 3. Prefill 并非每次 iteration 前进一个 group

当前 group 只有在全部 frontier 都执行完后才完成。若一个 group 有多个 frontier，
它会停留多个 scheduler iterations；每个 iteration 又可能包含多个 frontier
forwards。

论文的语义是：一次 iteration 后，整个 prefill wave 前进一个 group。当前实现则是：
先让若干 chunk 在当前 group 前进一步，直到所有 frontier 完成后才换 group。

这实质上是在 layer-major 外面继续保留 token-chunk 调度。

### 4. Decode 只与第一个 frontier 合并

当前 mixed batch 只包含 decode rows 和 `selected[0]`。其余 prefill rows 单独执行。
这既没有形成目标设计中的单个大 ragged batch，也重复产生 forward、metadata 构建和
host dispatch。

### 5. `chunks_per_iteration` 控制的是 forward 数，而非一次 flat batch 的容量

当前 `chunks_per_iteration` 允许一次 iteration 选择多个 chunks，但这些 chunks 并
没有被合成一个 active-group forward。这个参数因此只把多次 forward 放进同一个
scheduler iteration，并没有消除它们。

### 6. Group-boundary repack 只减少碎片，没有修正执行模型

`_repack_frontiers_for_replay()` 会按请求的 chunk ordinal 重排迟到请求，并把多个
请求的同序号 chunk 合在一个 frontier 中。这可以减少 arrival 造成的额外 frontier，
但结果仍是一串 frontier，后续仍逐个 forward。

它修复的是 frontier 数量，不是“frontier 不应该成为 forward 边界”这个根本问题。

### 7. 当前统计里的 iteration 不是论文定义的 iteration

当前 `iterations` 会随着 frontier 数和 `chunks_per_iteration` 改变。同一个 wave 在
同一个 group 可能产生多次 decode、多个 prefill forwards。

按论文定义，没有额外 chunking 时，一个 wave 的 prefill iterations 应等于 group
数量，而不应随 `ceil(prompt_tokens / T)` 增长。

## 当前实现中可以保留的部分

以下能力与目标设计一致，不需要推倒重来：

- active expert group 的 pin、prefetch 和切换；
- `LayerGroupState` 对 hidden/residual 的跨 group 保存；
- decode 在 active group 前后的分段执行；
- decode/prefill state 的 merge 与 split；
- ragged attention 和多请求 batch 基础设施；
- 每请求 KV、abort、terminal output row 的独立记账。

需要重做的是调度和 batch 组织方式，不是底层 kernel，也不是增加特定 shape 的 CUDA
Graph capture。

## 必须进行的结构调整

1. 用一个 wave-level ragged batch 代替可执行 frontier 列表。
2. group0 admission 完成后，把 wave 的全部 prefill rows 一次性准备好。
3. 每个 iteration 只构建一个 mixed batch：`decode rows + all wave prefill rows`。
4. active group 只调用一次 layer-group forward。
5. forward 后立即拆分 decode/prefill state；decode 完整走完模型并采样一次。
6. wave 的 group cursor 每个 iteration 前进一步。
7. 如果必须支持超长输入 chunking，chunk 应是独立的大 wave 单位，而不是在一个
   iteration 内造成多个 group forwards。
8. `frontier` 如果继续存在，只能用于 arrival/admission/输出位置 bookkeeping，不能
   决定物理 forward 次数。

## 验收判据

没有额外 chunking、模型有 `N` 个 layer groups 时，一个 wave 必须满足：

```text
prefill iterations          = N
active-group prefill calls  = N
decode tokens produced      = N    # 始终有 runnable decode 时
```

无论 wave 中有多少请求、每个请求原先可被切成多少个 `T` 大小的 chunk，
`active-group prefill calls` 都不能因此增长。

以4个请求、每个请求16个旧式 chunks、8个 groups为例：

- 当前实现可能形成16个 frontiers、32个 iterations、128次 group forwards；
- 目标实现是一个 flat ragged wave、8个 iterations、8次 active-group mixed forwards。

任何依靠固定 prompt 长度或固定 batch shape 的专属 graph capture 都不属于这项设计
收益。正确的收益应来自调度本身：减少重复 forward、重复 expert I/O、重复 metadata
和 host dispatch，同时保持每轮一个 decode token。
