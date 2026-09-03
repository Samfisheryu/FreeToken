# Qwen3.6 混合任务 online goodput 最终 A/B

## 结论

单张 RTX 4090 上，最终 U1 原版 FreeToken 为 `1450.547 tasks/h`，S2 ServeBig 为
`1320.000 tasks/h`：

```text
ServeBig / 原版 FreeToken = 1320.000 / 1450.547 = 0.910002x
```

本轮没有达到 `2x+` 目标。S2 将平均延迟从 `0.930 s` 降到 `0.594 s`，TTFT 从
`0.183 s` 降到 `0.117 s`，峰值显存从 `22.188 GiB` 降到 `12.246 GiB`；但总正确率从
`61.42%` 降到 `55.14%`。主要瓶颈是数学与数值推理：正确率从 `24.43%` 降到
`12.78%`，该族 goodput 只有 U1 的 `0.531782x`。更短、更快的错误输出不能增加 task
goodput。

这个冻结流本身还有一个上限：20 个用户、30 秒 think time 和 600 秒提交窗口最多产生约
400 次提交，即使 ServeBig 全部答对也只有 `2400 tasks/h`，相对 U1 的理论上限为
`1.655x`。因此本轮 workload 数学上不能验证 `2x`。该限制在 final 后才被定量暴露，
不能通过事后缩短 think time 改写本轮结论；若下一轮仍以 `2x` 为目标，必须在查看新
final 前冻结一个更接近 serving 饱和、且仍保持闭环用户语义的新任务流。

公开结果：[U1 result](/data1/lmcache_kv/goodput_campaign/mixed_final_20260901/U1-upstream-official-nvfp4-fi-evalv2-pathfix-final.json)、
[S2 result](/data1/lmcache_kv/goodput_campaign/mixed_final_20260901/S2-servebig-nowag-general-recovered-fi-evalv2-pathfix-final.json)。

## 冻结契约

- 20 个闭环用户，用户 `u` 在第 `u` 秒首次进入；每个用户最多一个在途请求，请求终止后
  固定等待 30 秒。
- 20 题 warmup 不计时且完整排空。final 提交窗口固定为 600 秒，最多 420 题，三族各
  140 题；任务流不回绕，窗口结束后完整 drain。
- 数值、代码、知识三族使用冻结且互不重叠的 final stream。公共策略为 greedy、关闭
  thinking、统一 system prompt；`max_tokens/SLO` 分别为 `512/30 s`、`768/60 s`、
  `64/10 s`，硬请求 timeout 为 90 秒。
- goodput 分子只计 judge 正确且满足该族 SLO 的任务；错误、无答案、length、timeout、
  HTTP/服务错误均为 0。分母为首个计分提交到提交窗口和完整 drain 两者较晚者。
- 两个结果均为 `valid_task_goodput=true`，任务队列均未耗尽。U1 分母为 `600.601 s`
  （drain `0.601 s`），S2 为 `600.000 s`（drain `0 s`）。

## 最终配置

| 项目 | U1：最佳原版 FreeToken | S2：最佳 ServeBig |
|---|---|---|
| 生产代码 | upstream `3a20a790` | campaign `27455a41` 加当前生产改动 |
| 模型入口 | 官方 Qwen3.6-35B-A3B-NVFP4 | 同一模型入口，加 NoWAG routed-expert sidecar |
| 公共 serving | BF16 compute；FlashInfer attention；MoE offload；naive KV；`max-running=16`；CUDA Graph batch 16；context 1536；memory ratio 0.95；prefill-hit D2D | 同左 |
| expert/KV | 自动 MoE cache；KV reserve 24576 tokens | expert cache 10240；KV 24576 tokens |
| 调度 | 原版默认调度 | mixed batching |
| 恢复参数 | 无 | general-mix normalizer recovery sidecar |

统一 prompt、关闭 thinking、采样、任务 token 上限、SLO、官方 NVFP4 基座以及双方均可用的
FlashInfer/offload/graph 调优都是公共 baseline，不能归因成 ServeBig 收益。ServeBig 的
相对差异只来自 NoWAG 表示、恢复后的 normalizer、显式 expert cache 和 mixed batching。

## 总量结果

时延列依次为 mean/P50/P95/P99；TTFT 单位为秒，TPOT 单位为毫秒。

| 指标 | U1 | S2 |
|---|---:|---:|
| 提交 / 正确 / SLO 成功 | 394 / 242 / 242 | 399 / 220 / 220 |
| accuracy / SLO success rate | 61.42% / 61.42% | 55.14% / 55.14% |
| goodput | 1450.547 tasks/h | 1320.000 tasks/h |
| latency (s) | 0.930 / 0.337 / 3.585 / 4.570 | 0.594 / 0.198 / 2.366 / 5.912 |
| TTFT (s) | 0.183 / 0.155 / 0.297 / 0.373 | 0.117 / 0.106 / 0.197 / 0.246 |
| TPOT (ms) | 8.10 / 7.16 / 11.64 / 53.80 | 7.67 / 7.20 / 9.64 / 20.14 |
| 输出 token：总量 / 均值 / 每秒 | 39,714 / 100.80 / 66.12 | 23,691 / 59.38 / 39.49 |
| 峰值在途 / 平均在途 | 6 / 0.610 | 4 / 0.395 |
| 峰值显存 | 22.188 GiB | 12.246 GiB |
| 正常 stop / length | 358 / 36 | 377 / 22 |
| timeout / HTTP / 服务错误 | 0 / 0 / 0 | 0 / 0 / 0 |

生产 `/v1/stats` 没有暴露 expert-cache miss 和 H2D byte，因此这两个指标均为 `null`，不以
估算值替代实测。

## 分任务族结果

时延、TTFT 为 mean/P95 秒；TPOT 为 mean/P95 毫秒。输出 token 为总量/均值/每秒，
`length` 单列计数。

| Arm / 任务族 | 提交 / 正确 | accuracy | goodput (tasks/h) | latency | TTFT | TPOT | 输出 token | length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U1 数值 | 131 / 32 | 24.43% | 191.808 | 1.150 / 4.161 | 0.184 / 0.297 | 7.55 / 10.27 | 17,528 / 133.80 / 29.18 | 19 |
| S2 数值 | 133 / 17 | 12.78% | 102.000 | 0.259 / 0.310 | 0.121 / 0.200 | 7.72 / 9.41 | 2,124 / 15.97 / 3.54 | 2 |
| U1 代码 | 133 / 105 | 78.95% | 629.370 | 1.356 / 3.476 | 0.184 / 0.295 | 7.59 / 11.74 | 20,781 / 156.25 / 34.60 | 2 |
| S2 代码 | 134 / 97 | 72.39% | 582.000 | 1.309 / 3.739 | 0.116 / 0.196 | 7.94 / 9.98 | 20,139 / 150.29 / 33.57 | 5 |
| U1 知识 | 130 / 105 | 80.77% | 629.370 | 0.271 / 0.663 | 0.182 / 0.295 | 9.17 / 12.91 | 1,405 / 10.81 / 2.34 | 15 |
| S2 知识 | 132 / 106 | 80.30% | 636.000 | 0.205 / 0.646 | 0.115 / 0.168 | 7.35 / 8.99 | 1,428 / 10.82 / 2.38 | 15 |

分族倍率为：数值 `0.531782x`，代码 `0.924735x`，知识 `1.010535x`。S2 只在知识族
略高于 U1；代码小幅下降，数值质量下降抵消了系统侧的时延和显存收益。

## 通用量化恢复

S2 使用冻结 assignment/codebook、只优化已有 normalizer 的五轮 general-mix 恢复；它
没有新增在线 LoRA side path。离线 diagonal-Hessian 加权重建目标在 train 降低
`1.584%`，在独立 holdout 降低 `1.586%`。这证明恢复目标没有只过拟合 train，但最终
mixed task A/B 表明该幅度不足以恢复数学质量。

产物：[sidecar manifest](/data1/lmcache_kv/goodput_campaign/qwen36_general_mix_norm_recovery_als5_v1/manifest.json)、
[recovery report](/data1/lmcache_kv/goodput_campaign/qwen36_general_mix_norm_recovery_als5_v1/recovery_report.json)。

## 判定与下一步

本轮真实结论是 `0.910002x`，不是 `2x+`。当前瓶颈不是在线吞吐或显存，而是 NoWAG
量化后的任务正确率，尤其是数学与数值推理。下一轮应保持 final stream 封存，先在新的
development stream 上提高 routed-expert 恢复能力；同时若仍评估 `2x`，须预先把新流
设计为可达到该倍率的 serving-bound 负载。只有恢复正确率后，S2 已体现出的 TTFT、
平均延迟和显存优势才可能转化为 task goodput。
