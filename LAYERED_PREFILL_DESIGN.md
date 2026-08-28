# Layered Pipeline 设计

本文是 FreeToken 中 `layered-pipeline` 的设计规范。调度来自论文
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

## FreeToken 实现

`scheduler/layered_pipeline.py` 实现上述 wave-level 调度：

- wave 在第一个 group forward 前冻结成员；
- 每个成员的当前未缓存 prefill 范围只 materialize 一次，多请求保持
  独立的 ragged attention 边界；
- 每次 iteration 只构建一个 `decode rows + full prefill wave` mixed batch；
- active group 只执行一次 forward，然后拆回 decode 和 prefill state；
- resident expert group 的 pin、prefetch、release 与 decode 保留层由共享 cache
  生命周期管理；
- `LayerGroupState` 保存 wave 在 group 边界的 hidden/residual state；
- KV allocation、abort、terminal output row 和 finish 按请求独立记账。

`--max-prefill-length` (`T`) 只用于计算请求的 planned chunks 和 wave
admission 容量，不会在 wave 内创建额外 forward 边界。
`--prefill-wave-max-chunks` (`W`) 是完整请求的 aggregate soft cap：首个请求
本身若大于 `W`，它仍保持完整并独占 wave。`--prefill-layer-group-size`
(`G`) 控制 resident layer group 的大小，并受 shared expert cache 容量限制。

这条路径复用通用 decode range graph，但不 capture 特定 prompt 长度或 prefill
shape。

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

- 旧 frontier 方案可能形成16个 frontiers、32个 iterations、128次 group forwards；
- 当前实现是一个 flat ragged wave、8个 iterations、8次 active-group mixed forwards。

任何依靠固定 prompt 长度或固定 batch shape 的专属 graph capture 都不属于这项设计
收益。正确的收益应来自调度本身：减少重复 forward、重复 expert I/O、重复 metadata
和 host dispatch，同时保持每轮一个 decode token。
