# Layered-pipeline 纯实现优化与验收

## 范围

- 基线：`a0a519d`，包含 `f5fce33` 的静态 tiling 与 `b676f73` 的 next-use eviction。
- 目标：只修复支持路径上的行为错误、重复准备、无效复制和不随输入规模稳定扩展的内存占用；不改变 admission、wave、tile、group、decode 产出频率或 expert eviction 规则。
- 所有 shape、层数、page、tile、batch 和 backend 分支均由运行时数据决定，没有写死模型尺寸。

## 最终保留的改动

- 合并同一轮的 resident decode 与 wave-final prefill 完成集合，避免 EOS/stop 后投机产生的一个 token 被错误发布。
- active pipeline decode 完成后提前预留下一 KV page，与普通 decode 的分配/计算重叠方式一致。
- 静态 tile 的输入、位置、写入位置、采样参数和 attention metadata 跨 stage 复用；group 执行不再构造四个未使用的 combined mapping tensor。
- Triton attention 直接按 canonical page table 寻址，不再为每个 tile 长期保存递增长的历史索引。
- DeepSeek-V4 的压缩状态可从任意绝对 tile 边界继续，保留对齐块的批量快路径，并正确处理跨块 carry；同时移除构造后立即丢弃的 mixed input。
- FlashInfer 每个 backend 只保留一套大 workspace。tile 只保存轻量描述，当前历史索引按需写入一个几何增长的共享 GPU buffer；内存从近似 `O(N² / tile_size)` 降为 `O(max_prefix + tile_descriptors)`。
- heterogeneous attention 的 decode/prefill 分别使用各自 backend、Batch 和 metadata；`fi,triton` 等公开组合不再把另一 backend 的 metadata 交给 FlashInfer。
- 保留原单套 resident copy plan。next-use eviction、pin、promotion、ready event 和调度顺序不变；prefill-only 容量判断只在确实没有 decode 时取消多余的一层 decode reserve。

## 受控 A/B

基线与候选使用相同模型、请求、生成长度和调度计数。原始结果位于 `/data1/lmcache_kv/experiments/freetoken_layered_pure_opt_ab`。

| 工作负载 | 基线 | 最终候选 | 结论 |
|---|---:|---:|---|
| Qwen long5，cache 512 | makespan 8.71095 s | 8.67280 s | -0.44%；H2D -0.16%，token/wave/iteration 完全一致 |
| Qwen long5，cache 768 | makespan 9.15184 s | 两次均值 9.12174 s | -0.33%；TPOT p50 -1.33%，工作量一致 |
| Qwen short20 | 两次均值 33.54326 s | 33.72207 s | makespan +0.53%、TPOT p50 -0.45%，判定为无实质变化 |
| DeepSeek-V4 repo workload | 在合法非对齐 tile 边界触发断言 | 11/11 请求完成 | 40128 个 prompt token、112 个 decode token、3 waves、473 iterations、129 layer prepares |

最终 Triton 端到端收益是中性到约 0.3–0.5%，没有把内存缩放和正确性修复包装成吞吐大幅提升。

以下尝试经 A/B 否决后已完全撤回：

- 双 resident copy-plan bank：warm 对照约慢 0.25%，没有稳定收益。
- prefill-only next-group hint：8.782 s 对 8.683 s，且 prefill H2D 增加，撤回。
- 任意 batch 的 range-graph padding：慢 2.5–5.6%，恢复 exact-size graph，其余 shape 走 eager。
- Qwen parked-state 复制：8.7822 s 对原 `cat + view` 的 8.7782 s，并增加 `P×hidden` 复制，撤回。

## 独立黑盒验收

测试由未读取生产实现、内部测试、diff 或实现笔记的独立 agent 从公开 HTTP/CLI 契约编写，协调 agent 负责 GPU 执行。

- Qwen3.6-35B-A3B：202/202 通过。
- DeepSeek-V4-Flash-0731：200/200 通过。
- 覆盖 group 2/3、末组 1 层、batch/concurrency 3/5、ragged 与 127/128/129 边界、offload/hybrid、`fi,triton` overlap、stop/EOS/abort 和多次 server lifecycle。
- 受影响仓库回归：132 passed，9 skipped，0 failed；包含真实 Qwen FlashInfer 与 DeepSeek-V4 sparse server。
- `compileall`、变更模块 import、`ft serve --help` 和 `git diff --check` 均通过。

全量非 slow 测试曾得到 1498 passed、30 skipped、11 deselected、65 failed；其中发现的两个本任务相关问题已修复并定向通过。其余失败集中在缺失/不兼容的本地扩展、外部 NoWAG 插件接口和既有 CUDA/数值环境，因此不能声称整个仓库的所有环境测试全绿。

## 代码量

- 生产代码：`+1166 / -271`，净增加 895 行，14 个文件。
- 独立黑盒 Python 测试代码：`+2816 / -0`；测试配置与命令文档另 475 行。
- 生成的扩展、结果 JSON、第三方代码和无关工作树改动均未计入。

本任务范围内的生产代码清理、公开行为、独立矩阵和代码量核算均已完成，达到可交付标准；仓库全量环境测试的既有失败如上单独保留。
