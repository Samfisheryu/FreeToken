# Qwen layered-pipeline 决定性 sweep

## 结论

早期实验参数确实没有充分覆盖该调度的设计点：`C=512/G=1` 无法预取下一组，`T=8192` 的确认 workload 也全部是单 tile wave。但修正这些参数后，仍没有找到一个在用户侧大指标上整体优于 `mixed` 的配置。

最接近胜区的是 `C=1280/G=1/T=8192/W=64`。同一张 RTX 4090 上三轮 fresh-server、LMP/MPL/PLM 换序确认后，layered-pipeline 相对 mixed 的配对变化中位数为：

| 指标 | 变化 | 稳定性 |
| --- | ---: | --- |
| TTFT p50 / p95 | +561.21% / +598.60% | 3/3 更差 |
| TPOT p50 / p95 | -39.50% / -27.65% | 3/3 更好 |
| Request latency p50 / p95 | -2.57% / +1.78% | 方向不稳定，幅度小 |
| Makespan | +5.48% | 3/3 更差 |

因此它不是“完全没用”：它能显著降低正在生成请求的 TPOT，而且相对 legacy 还能稳定降低 request latency。但它不是 mixed 的严格优化；在本场景中，它主要是用约 6–7 倍 TTFT 和约 5.5% 总完成时间损失换 TPOT，而不是把 resident expert 复用转化成整体用户 latency 收益。通用默认仍应是 `mixed`。

## 覆盖与公平性

- 生产代码：`2d315887d9e67feb0ac429f98c74b7ef17ed7839`。
- 模型：Qwen3.6-35B-A3B，BF16、Triton attention、expert offload、radix cache、Graph8；每个 arm 使用单张 RTX 4090，三轮主确认固定在同一张 GPU1。
- 主矩阵共 50 个有效 fresh-server policy arm；环境冲突导致的 4 个启动失败单独保留且不计入结果。
- `T={256,512,8192}`，`W={1,2,4,8,64}`。
- 固定总 expert+KV pool 的几何点：`C/G={512/1,768/1,768/2,1280/1,1280/2,1792/2,1792/3,2304/4}`。`C={512,768,1280,1792,2304}` 对应的 KV token 容量依次为 `{150000,139824,119472,99120,78768}`，保证增加 expert cache 时不偷偷增加总池预算。
- 明确覆盖 next-group prefetch 边界：`C=768/G=2` 不允许，`C=1280/G=2` 与 `C=1792/G=3` 允许。
- conversation workload 每个有效 arm 均为同一 40 请求、5790 prompt tokens、4085 decode tokens、0 失败。
- 确认点实际全为 `S=1`；小 tile screen 实际覆盖 `S=1…6`。另有两个持续 decode 的 repo workload，四个 12K 新请求都实际使用 `S=2`。
- 请求 payload 使用 `temperature=0`、`ignore_eos=True` 和固定逐请求 seed。不同策略仍会因数值执行路径分叉生成文本并改变后续专家路由，因此 5% 内的小差异不作强因果判断；TTFT、TPOT、request latency 和 makespan 仍是实际服务行为。

## 三轮确认

下表是三轮绝对值的中位数，单位为秒。

| 策略 | TTFT p50 / p95 | TPOT p50 / p95 | Request latency p50 / p95 | Makespan |
| --- | ---: | ---: | ---: | ---: |
| legacy | 0.493 / 0.944 | 0.13321 / 0.16253 | 12.914 / 18.832 | 43.988 |
| mixed | 0.545 / 0.959 | 0.12316 / 0.14536 | 11.315 / 17.298 | 42.244 |
| layered-pipeline | 3.634 / 7.073 | 0.07451 / 0.10249 | 11.023 / 17.976 | 45.420 |

相对 legacy，layered-pipeline 的配对变化中位数为 TPOT `-45.17%/-38.35%`、request latency `-15.41%/-9.12%`、makespan `+1.30%`，但 TTFT 为 `+631.24%/+593.12%`。它对 legacy 有明确价值，但 mixed 已取得更平衡的结果。

## 多 tile 验证

为了排除“失败只是因为 wave 都是 `S=1`”，另测了 6 个持续生成 512 tokens 的 driver，并在它们开始输出后加入 4 个互异的 12K repo prompt。两个 cell 都是 `{S1:2, S2:4}`，每个策略均完成 10 请求、48768 prompt tokens、3671 completion tokens，0 失败。

| Cell | 真正 S=2 repo 请求相对 mixed | 整体结果 |
| --- | --- | --- |
| `C1280/G1/W8` | TTFT `+320%/+259%`；TPOT `+63%/+34%`；latency `+138%/+14.7%` | makespan `+4.49%`；driver latency 基本不变 |
| `C1792/G3/W8` | TTFT `+50%/+76%`；TPOT `+25%/+132%`；latency `+39%/+20%` | TTFT、TPOT、latency、makespan 全部更差；makespan `+8.03%` |

这两个单轮模拟 shared coding-agent workload 不能代表所有生产流量，但足以否定“只要持续出现 `S>=2`，layered-pipeline 就会自然胜过 mixed”。

## 参数 sweep 的判断

- 减小 `T` 或增大 `W` 会增加 wave 内 decode 服务频率，TPOT 可明显改善，但 TTFT、tail latency 和 makespan 同时恶化；没有某个 `W` 消除这一取舍。
- 增加 cache 并跨过 next-group prefetch 容量边界确实会改善 layered-pipeline 的绝对时间，但同预算下 mixed 也随更大的 cache 显著变快，优势没有转成整体用户 latency。
- 增大 `G` 减少 group 数并未形成胜区。最大点 `C=2304/G=4` 相对同配置 mixed，TPOT p50 反而约慢 14%，request latency p50/p95 约慢 27%/12%，makespan 约慢 11.5%。
- 当前能够复现的价值来自 decode 服务优先级，而不是已被端到端大指标证明的 cross-tile resident reuse 收益。

## 原始结果

主结果目录：

`/data1/lmcache_kv/experiments/freetoken_layered_decisive_sweep_20260830`

三轮主确认使用：

- `confirm_c1280_g1_t8192_w64_lmp_r1_retry_gpu1.json`
- `confirm_c1280_g1_t8192_w64_mpl_r2_gpu1.json`
- `confirm_c1280_g1_t8192_w64_plm_r3_retry2_gpu1.json`

多 tile 结果使用：

- `repo_multitile_c1280_g1_w8_fixedhbm_gpu1.json`
- `repo_multitile_c1792_g3_w8_fixedhbm_gpu0.json`

旧 GPU0 确认、第一次 GPU1 PLM retry、`C1280/G3` 和原始 `C768` screen 的失败均由外部 GPU 占用或端口冲突造成，未纳入任何统计；成功 retry 文件保留原失败文件而未覆盖。
