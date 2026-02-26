### 1.3 通信压缩算法：训练数据与通信数据对比

#### 1.3.1 目标（Goal）

- **问题定义**：在分布式训练中引入通信压缩后，训练质量（loss/PPL）与系统开销（通信量/吞吐/显存）如何变化。
- **对比目标**：
  - **质量维度**：训练 `Loss/step` 收敛速度与稳定性；训练结束后的验证集 `PPL`（同源 Wikipedia held-out）。
  - **效率维度**：吞吐（tokens/s）、通信量（bytes/step）、显存占用（allocated/reserved）。
  - **机制解释维度**：压缩误差（相对 L2 误差）与 loss/PPL 变化之间的关联。
- **评价原则**：
  - 所有算法在**同一训练配置**下对比（同数据、同模型、同步数/epoch、同 seed、同并行规模）。
  - 指标以 **baseline（无压缩）** 为参照，重点报告相对提升或下降的幅度。

#### 1.3.2 实验配置与数据采集方法

- **硬件与并行设置**：
  - 单机多卡（默认 2 GPU）；在后续工作中可扩展到两机多卡以放大通信瓶颈。
  - 分布式策略：FSDP；开启混合精度（bfloat16）以降低显存占用。
- **模型与数据**：
  - 模型规模：GPT2 small/medium（按统一配置给出具体层数/隐藏维度）。
  - 序列长度：`max_length = 1024`。
  - 训练集：Wikipedia JSON（如 `wikipedia_en_500mb.json`），通过 `build_tokenized_shards.py` 预构建 tokenized 缓存。
  - 验证集：基于 `config.data_path` 的同源 Wikipedia **5% held-out 子集**（默认取尾部 5% 样本）。
- **训练超参（统一对齐项）**：
  - `batch_size`、`gradient_accumulation_steps`、学习率与调度器、`num_epochs` 与 `max_steps`（短跑实验时固定 `max_steps`）。
  - 压缩开关：`comm_hook ∈ {none, int8, nc, qsgd, onebit_seide, topk, randomk, thresholdv, sketch, ...}`。
  - 误差反馈：是否开启 EF、是否使用本地 EF（`comm_error_feedback` / `comm_ef_local`）。
- **运行脚本与产物**：
  - 运行入口：`scripts/step2_profile.sh`（先完整跑训练，再自动执行指标采集脚本）。
  - 每次运行在 `logs/<run_name>/` 目录下生成：
    - `tensorboard/events.out.tfevents.*`：`Loss/step`、`LearningRate/step`、`Memory/Allocated_GB`、`Memory/Reserved_GB`。
    - `training_metrics.json`：解析后的曲线（loss/显存/lr）、吞吐 `tokens_per_sec`、同源 5% 验证集 `PPL`。
    - `comm_stats_rank0.json`：轻量通信统计（总通信字节数与 `bytes_per_step_per_rank`，按 hook 拆分）。
    - `grad_error_stats_rank0.json`：梯度压缩误差统计（按 hook 聚合的相对 L2 误差均值与最后一次值）。
- **指标与计算口径**：
  - 吞吐：\( \text{tokens/s} = \frac{\text{steps} \times \text{batch\_size} \times \text{max\_length} \times \text{nproc}}{\Delta \text{wall\_time}} \)。
  - 通信量：每 rank 每 step 的 `bytes_per_step_per_rank`，近似自运行时统计器。
  - 压缩误差：在误差反馈更新处记录的 \(\|g - \hat g\|_2 / \|g\|_2\)，跨参数与 step 求平均得到 `rel_l2_mean`。
- **对齐与汇总方法**：
  - 短跑实验按前 `N` 个 step 对齐曲线；完整训练实验按 epoch/step 区间对齐。
  - 对每个 run 汇总：loss/PPL 的最终值或区间平均，吞吐均值，通信量均值，显存峰值与梯度误差均值，形成统一比较表。

#### 1.3.3 实验结果：各压缩算法的统一对比

- **总体对比表**（一行一个配置）：
  - `comm_hook` 及关键超参（是否 EF、本地 EF、bit packing、稀疏比例等）。
  - 训练质量：最终训练 loss、同源 5% 验证 `PPL`。
  - 效率：吞吐 `tokens_per_sec`、显存峰值（`Memory/Reserved_GB`）。
  - 通信：`bytes_per_step_per_rank`（总量与按 hook 分解）。
  - 误差：`rel_l2_mean`（相对 L2 误差越小，说明压缩越“温和”）。
- **曲线对比（示意，下文给出具体图表）**：
  - 训练 loss 曲线：baseline（无压缩）与多种压缩算法叠加对比，观察收敛速度与抖动。
  - 吞吐与通信量：跨 run 的柱状图/折线图，定量展示“带宽节省是否转化为 tokens/s 提升”。
  - 显存曲线：`Memory/Allocated_GB` 与 `Memory/Reserved_GB`，观察压缩对显存峰值的影响（通常较小）。
- **压缩误差与训练质量的关系**：
  - 以 `rel_l2_mean` 为横轴、loss/PPL 为纵轴绘制散点，分析误差大小与最终性能之间的相关性。
  - 重点讨论：哪些方法在保持较小梯度误差的同时显著降低通信量、对 PPL 影响可接受。
- **综合结论（本节小结）**：
  - 哪些通信压缩算法在当前实验设定下形成较优的“通信量–吞吐–训练质量”权衡。
  - 哪些算法在通信并非主要瓶颈时反而引入额外开销，导致吞吐改善有限甚至下降。
  - 给出推荐配置（按“质量优先”和“带宽/性能优先”分两档），为后续章节的深入分析与多机实验提供依据。



下面是按 1.2 一样的文风写好的两小节正文，你可以直接覆盖到 `section_1_3_sparsification.md` 中对应位置。



为了保证不同 run 之间的可比性，短跑实验通常只取前若干个 step 的指标进行对齐（例如前 \(N\) 步的平均 loss 与平均吞吐），完整训练实验则按 epoch 或固定 step 区间（例如最后一个 epoch 或最后若干 step）聚合统计。对每个 run，我们最终汇总出训练 loss/PPL 的代表性数值、吞吐与通信量均值、显存峰值以及梯度误差均值，并组织成统一的对比表格与曲线，为后续 1.3.3 小节中“各压缩算法统一对比”的分析奠定基础。