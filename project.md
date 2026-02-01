# 项目速览：FSDP-Communication-Optimization（/root/llama-7b）

> 面向中期答辩的快速上手文档：帮助新的 agent 在 10 分钟内搞清楚“这个仓库能干什么 / 怎么跑 / 关键实现在哪里 / 当前进度与下一步对比实验怎么做”。

## 1. 项目目标（一句话）

基于 **PyTorch FSDP（Fully Sharded Data Parallel）** 实现分布式训练闭环，并在其中集成/对比 **通信压缩算法**（量化、稀疏化、误差反馈等），实现 **2–4× 通信压缩比**，同时在常用指标（如 PPL）上 **精度下降不超过 1%**。

当前仓库的训练对象并非真正的 LLaMA-7B 微调，而是使用 **GPT-2 架构（~62M 参数）随机初始化**进行预训练式语言建模（Wikipedia 文本续写），用于通信优化实验对比（迭代更快、易复现）。

## 1.1 中期必须完成（里程碑）

下面三步是中期答辩“必须可交付、可复现、可量化对比”的目标（按顺序推进）：

### 步骤 1：用代码证明 FSDP 反向阶段通信是主要瓶颈且可优化

验收要求（需要“证据文件/日志/trace”，不是口头结论）：

- 通过 `torch.profiler`（必要）或 Nsight Systems（可选）采集一次训练迭代的 trace，明确标出：
  - 反向阶段中与 FSDP 相关的通信算子（典型如 `reduce_scatter` / `all_gather`）耗时占比。
  - 在不同 `world_size`（至少 1 卡 vs 2 卡）下，通信占比上升且成为 step time 的主要组成。
- 需要在代码中把采集过程固化下来：
  - 例如在训练入口增加 `--profile` / `--profile_steps` 等参数；
  - 输出 profiler 结果到 `output_dir/logs/<run_name>/profiler/` 并可用 TensorBoard 打开。

输出产物（中期材料可直接引用）：

- profiler trace（TensorBoard 可视化）+ 一份摘要表（每步通信耗时/占比、compute 耗时/占比）。

Linux 环境下推荐用仓库脚本一键跑出证据文件（默认短跑 20 steps，确保 profiler schedule 生效）：

```bash
chmod +x scripts/step1_profile.sh

# 单卡：验证链路与产物
./scripts/step1_profile.sh \
  --data_path /root/llama-7b/datasets/wikipedia_en_10mb.json \
  --output_dir /root/llama-7b/fsdp_output \
  --nproc 1

# 双卡：做 1 vs 2 的通信占比/重叠率对比
./scripts/step1_profile.sh \
  --data_path /root/llama-7b/datasets/wikipedia_en_10mb.json \
  --output_dir /root/llama-7b/fsdp_output \
  --nproc 2 \
  --max_steps 50
```

产物默认写到：

- `output_dir/logs/<run_name>/profiler/summary_rank0.json`（包含通信占比与通信覆盖比/重叠率）
- `output_dir/logs/<run_name>/profiler/comm_op_summary_rank0.csv`
- `output_dir/logs/<run_name>/profiler/*.pt.trace.json`（TensorBoard Profiler 可视化与 overlap 计算输入）

实现说明（本仓库已固化 Step1 取证代码）：

- 训练侧只保留最小接入：`train_epoch_with_monitoring(...)` 调用监控模块（rank0 生效）。
- 耗时/取证逻辑已从训练代码中解耦到 `perf/`：
  - `perf/comm_profiler.py`：统一负责启动/推进/结束 profiler，并写出 `summary_rank0.json` 与 `comm_op_summary_rank0.csv`。
  - `perf/trace_overlap.py`：从 profiler 的 Chrome trace（`*.pt.trace.json`）估算通信-计算重叠（通信覆盖比）。
- 启动脚本：`scripts/step1_profile.sh`（Linux），默认开启：
  - `--profile/--no-profile`：是否启用 profiler（默认启用）
  - `--profile-step-time`：额外统计每 step wall time（默认关闭，脚本中开启）
  - `--max_steps`：短跑步数（用于取证；建议 >= 20）

#### 步骤 1：采集指标说明（字段含义与数据来源）

本仓库 Step1 的“证据文件”主要包含两类：

- `summary_rank0.json`：汇总指标（用于写结论/对比 1 卡 vs 2 卡）
- `comm_op_summary_rank0.csv`：通信相关算子列表（用于截图/定位热点）

`summary_rank0.json` 关键字段：

- `step_time.*`（来源：`time.perf_counter()`，需开启 `--profile-step-time`）
  - `mean_ms/p50_ms/p90_ms/p95_ms`：端到端每 step wall time 分布（包含计算+通信+同步+CPU 开销等）。
  - 作用：作为“真实训练迭代耗时”口径，对比扩展性与抖动。
- `profiler.total_cuda_time_ms / total_cpu_time_ms`（来源：`torch.profiler.profile().key_averages()` 聚合事件总和）
  - 作用：提供被 profiler 观测到的总体事件量级（注意：不是严格意义上的 step wall time）。
- `profiler.comm_cuda_time_ms / comm_cpu_time_ms`（来源：同上，按事件名关键词过滤）
  - 通信事件关键词：`reduce_scatter/all_gather/all_reduce/broadcast/nccl/c10d`（近似口径）。
- `profiler.comm_ratio_cuda / comm_ratio_cpu`
  - 定义：通信事件累计时间 / 全部事件累计时间。
  - 解读：2 卡相对 1 卡显著上升时，可作为“通信占比上升、成为瓶颈”的证据之一。
- `overlap.overall.*`（来源：解析 `*.pt.trace.json` 的 GPU kernel 时间区间并计算交并集；两套口径）
  - `comm_total_ms`：通信相关 GPU kernel 的区间并集总时长（近似）。
  - `compute_total_ms_loose`：非通信 GPU kernel 的区间并集总时长（近似，排除 memcpy/memset；loose 口径，偏宽）。
  - `overlap_ms_loose`：通信区间与 compute(loose) 区间的交集时长。
  - `comm_covered_ratio_loose`（通信覆盖比/重叠率）：$\mathrm{overlap\_ms\_loose} / \mathrm{comm\_total\_ms}$。
  - `comm_exposed_ratio_loose = 1 - comm_covered_ratio_loose`：通信暴露比（loose）。
  - `compute_total_ms_strict`：更严格的 compute 口径（口径A）：仅匹配少量“重计算 kernel”关键词（如 gemm/attention/triton）。
  - `overlap_ms_strict`：通信区间与 compute(strict) 区间的交集时长。
  - `comm_covered_ratio_strict`（口径A 通信覆盖比）：$\mathrm{overlap\_ms\_strict} / \mathrm{comm\_total\_ms}$。
  - `comm_exposed_ratio_strict = 1 - comm_covered_ratio_strict`：通信暴露比（strict）。

`comm_op_summary_rank0.csv` 字段：

- `name/count/cpu_time_total_ms/cuda_time_total_ms`（来源：`prof.key_averages()` 聚合）
  - 作用：列出通信相关热点事件（用于定位 reduce-scatter / all-gather 等是否出现，以及其耗时排序）。

注意事项（避免误解）：

- `comm_ratio_*` 与 `overlap.*` 都是“关键词分类”的近似口径，适合做趋势对比（1 卡 vs 2 卡）与中期证据展示，不等价于严格的算子级因果归因。
- profiler 采用 schedule（默认 wait/warmup/active），因此 trace 只覆盖部分 steps；建议取证时 `--max_steps >= 20`，并用相同设置对比 1 卡与 2 卡。

#### 步骤 1：如何验证（复现流程与预期现象）

1) 单卡先验证“链路与产物”

- 运行：`./scripts/step1_profile.sh --data_path ... --output_dir ... --nproc 1`
- 检查：`output_dir/logs/<run_name>/profiler/` 下存在
  - `summary_rank0.json`
  - `comm_op_summary_rank0.csv`
  - `*.pt.trace.json`

2) 双卡做 1 vs 2 对比验证“通信占比/重叠率趋势”

- 运行：`./scripts/step1_profile.sh --data_path ... --output_dir ... --nproc 2 --max_steps 50`
- 预期（常见趋势，不同硬件/模型会有差异）：
  - `profiler.comm_ratio_cuda`：2 卡通常高于 1 卡（通信事件占比上升）。
  - `overlap.overall.comm_exposed_ratio_strict`：2 卡常见上升（通信更难完全隐藏）。
  - `step_time.mean_ms`：2 卡若扩展效率不理想，会不降反升或下降幅度小。
  - `comm_op_summary_rank0.csv` 中能看到 `reduce_scatter/all_gather/all_reduce/nccl` 相关事件出现在 top 列表。

3) 可视化核对（推荐截图做中期材料）

- `tensorboard --logdir output_dir/logs/<run_name>`
- 打开 “PyTorch Profiler”，对比 1 卡/2 卡 step 时间线中通信相关条块占比与重叠情况。

### 步骤 2：实现“通信压缩热插拔模块”（支持多种压缩算法）

目标：把压缩算法从训练脚本中解耦出来，形成可配置/可扩展的模块，并能以“开关 + 配置”方式在 FSDP 通信路径上启用。

验收要求：

- 训练脚本支持命令行选择压缩算法，例如：`--comm_compress {none,int8,fp16,qsgd,nc,topk,randomk,thresholdv,sketch,gradiveq,signsgd,onebit,...}`（名称可再统一）。
- 压缩模块能在不改训练主逻辑的情况下替换/组合（热插拔）：
  - baseline：不注册 comm hook；
  - 压缩：注册对应 hook，并且保证训练可跑通（至少小数据集/少步数）。

中期范围要求：覆盖开题报告中提到的压缩方法（按报告口径归类），至少包括：

- 稀疏化：Top-k、Random-k、threshold-v、Sketched-SGD（sketch 映射）、GradiVeQ（矩阵分解/低秩/向量量化类）。
- 量化：FP16、8-bit（INT8/FP8 之一先做基线）、1-bit SGD、SignSGD、QSGD、NC（Natural Compression）。
- 误差反馈（Error Feedback）：对量化/稀疏化提供可选残差补偿开关。
- 混合压缩：至少支持“稀疏化 + 量化”的组合配置（哪种先做、是否对索引/值分别编码）。

说明：部分算法在工业实现中会有多种变体（是否无偏、是否需要额外 all-reduce/广播 scale、是否需要稀疏索引通信格式等）。中期要求是“实现可运行版本 + 说明实现口径”，并保证对比实验的公平性（同一训练设置）。

### 步骤 3：设计并落地一套可复现的指标体系，对比各压缩模块

目标：让每个压缩方法的效果可以被同一套指标与同一份结果文件直接对比与画图。

建议的最小指标集合（中期必须落盘）：

- 通信压缩（“压了多少”）：
  - 原始通信字节数 vs 压缩后通信字节数（必须包含索引/元数据/scale 等开销），以及压缩比。
  - 稀疏化密度：非零占比（density）/稀疏率（sparsity），以及索引编码开销占比（index_bytes / bytes_compressed）。
- 通信时间（“省了多少通信时间”）：
  - 每 step 通信耗时（总）与占比。
  - 按算子拆分：`reduce_scatter` / `all_gather` / `all_reduce` 等（至少覆盖反向相关通信）。
- 训练效率（“端到端快了多少”）：
  - step time（均值/中位数 + P90/P95），tokens/s。
  - GPU 显存峰值（allocated / reserved）。
- 训练质量（“精度/收敛有没有受影响”）：
  - 训练 loss 曲线（同等 step 数）+ 验证集 loss/PPL（同等 checkpoint）。

建议的扩展指标（更能体现压缩算法“优越性”，中期强烈推荐至少选 4–6 项落盘）：

- 端到端收益口径（避免只看压缩比“自嗨”）：
  - `time_to_quality`：达到某个目标 PPL/val loss 阈值所需时间（或所需 steps）。
  - `quality_at_time`：固定训练时间预算下的 PPL/val loss（更贴近工程）。
- 重叠与可扩展性（通信优化的核心）：
  - 通信-计算重叠率：通信算子与计算算子重叠的时间比例（profiler 可估）。
  - 有效通信带宽：`effective_bw = bytes_raw / comm_time`（以及压缩后 `bytes_compressed / comm_time`），用于量化“链路利用率”。
  - scaling efficiency：从 1 卡到 2 卡（再到更多卡）时 tokens/s 增长比例与 step time 变化。
- 算法/实现开销（很多压缩方法会把收益吃掉）：
  - 压缩/解压耗时：`compress_time_ms`、`decompress_time_ms`（最好按 GPU kernel/CPU 逻辑区分）。
  - 额外同步开销：为压缩引入的额外 collective 次数（例如额外 `all_reduce(MAX)` 同步 scale），以及其耗时。
  - 额外显存/内存开销：残差（Error Feedback）缓冲、索引缓冲、临时张量峰值。
- 收敛稳定性/数值误差（用于解释“为什么某方法掉点/发散”）：
  - 梯度误差统计（诊断用，可抽样计算）：
    - 相对误差 $\|g-\hat g\|/\|g\|$（L2 或 L∞），以及 cosine 相似度 $\cos(g,\hat g)$。
  - 残差（EF）范数：`residual_norm` 随 step 的变化（判断 EF 是否在“吃掉误差”）。
  - 无效 loss/数值异常计数：NaN/Inf step 次数、梯度裁剪触发频率等。
- 鲁棒性与公平对比（让结果更可信）：
  - 多 seed 重复：报告均值±标准差（至少 3 个 seed）。
  - 丢弃 warmup：统计 step time/吞吐时跳过前 N step（避开编译/缓存抖动）。

输出产物：每次 run 产出一条结构化记录（建议 `results.jsonl` 或 `results.csv`），字段建议分“必填 + 可选扩展”。

- 必填（中期最小闭环）：
  - `run_name, world_size, model, dataset, max_length, batch_size, compress_method, compress_config, seed`
  - `step_time_ms_mean, step_time_ms_p50, step_time_ms_p90, tokens_per_s`
  - `comm_time_ms_total, comm_ratio, comm_time_ms_by_op`（by_op 可用 JSON 字符串）
  - `bytes_raw_total, bytes_compressed_total, compression_ratio`
  - `train_loss, val_loss, val_ppl`
- 可选扩展（体现优越性/解释原因）：
  - `compress_time_ms, decompress_time_ms, extra_collectives, extra_collective_time_ms`
  - `index_bytes, density, residual_norm`
  - `grad_rel_error, grad_cos_sim`
  - `mem_peak_alloc_gb, mem_peak_reserved_gb`
  - `time_to_target_ppl_s, ppl_at_fixed_time`

中期对比建议：优先做 `world_size=2` 的 baseline vs 各压缩方法，固定训练步数与数据，先跑通再扩展规模。

## 2. 运行环境与约定

- OS：Linux
- 分布式：`torchrun`（双卡为主）
- 通信后端：多卡使用 `nccl`
- 训练脚本默认支持单卡（会用 `gloo` 初始化一个单进程组）

### 核心路径约定

- 数据集：`/root/llama-7b/datasets/*.json`
- 模型 tokenizer 词表（GPT2）：`/root/llama-7b/models/gpt2/`
- 训练输出：`/root/llama-7b/fsdp_output/`

> 注意：仓库里存在 `models/` 目录，包含本地权重/配置；仓库也有 `.gitignore/.gitattributes` 用于避免提交大文件。

## 3. 目录结构与关键文件

### 训练与分布式

- `fsdp_train.py`
  - 训练入口：分布式初始化、FSDP 包装、DataLoader、优化器、训练循环、保存。
  - **通信压缩 hook 的实现也在这里**：`fsdp_quantized_comm_hook`（int8 量化 reduce-scatter）。
  - 目前 hook 注册段落在代码中是注释状态（需要打开/参数化，做 baseline vs int8 对比）。

- `train_func.py`
  - `train_epoch_with_monitoring(...)`：带 TensorBoard + torch.profiler 的训练迭代。
  - 约定：只在 rank0 输出 tqdm/写 TensorBoard。

- `run_safe_training.sh`
  - 双卡训练脚本（中期答辩主入口）。
  - 典型参数：`--nproc_per_node=2`，使用 `datasets/wikipedia_en_500mb.json`。
  - **会启动 TensorBoard（6006）并自动释放端口**。

### 数据集与预处理

- `data_base.py`：`WikipediaDataset`
  - 期望输入格式：**JSON 数组**，每条含 `id/title/text`。
  - 关键能力：
    - **流式解析（ijson）**，避免 `json.load` 直接 OOM。
    - **预分词缓存**：按 `max_length` + `shard_size` 生成 shard `.pt` 文件和 `.meta.pt`。
    - **断点续作**：meta 记录 `processed_samples/finalized`，rank0 处理时原子更新。
    - **多卡同步**：non-rank0 轮询等待 `finalized=True` 后再 barrier，避免 NCCL store timeout。

### 模型构建

- `create_model.py`
  - `load_tokenizer()`：从 `./models/gpt2` 读取 GPT2Tokenizer，并补齐特殊 token。
  - `load_model(tokenizer)`：用 `GPT2Config` 构造 `GPT2LMHeadModel(config)`（**随机初始化**）。
  - 文件内也保留了 LLaMA 加载的注释代码，但当前训练主线使用 GPT2。

### 验证与交互

- `validate_model.py`
  - 从训练保存的 `final_model/` 加载权重并在指定数据集上计算 `avg_loss` / `ppl`。
  - 适合中期答辩作为“精度保持性”的客观指标输出。

- `chat_model.py`
  - 极简 REPL：输入 prompt → `model.generate` → 输出。
  - 提示：当前实现里 `build_prompt()` 没被 main 使用（`prompt = user_text`），对话效果偏“续写”，不是指令聊天。

### 数据生成

- `download_wikipedia_en.py`
  - 使用 HuggingFace `datasets` 下载 Wikipedia 并导出为本项目兼容 JSON 数组。
  - 离线环境会给友好的报错与替代建议（缓存/镜像/跳过下载）。

## 4. 数据格式（非常重要）

`WikipediaDataset` 期望：

```json
[
  {"id": "12", "title": "Anarchism", "text": "..."},
  {"id": "25", "title": "Autism", "text": "..."}
]
```

训练时会拼接：`title + "\n" + text` 后 tokenize。

## 5. 训练：如何跑双卡（中期默认）

主要入口是 `run_safe_training.sh`。

```bash
cd /root/llama-7b
./run_safe_training.sh
```

脚本会：
- 打印 GPU 信息
- 后台启动 TensorBoard（端口 6006）
- `torchrun --nproc_per_node=2 fsdp_train.py ...`
- 训练日志保存到：`fsdp_output/training_log_safe_<timestamp>.txt`

### 训练输出产物（预期）

`fsdp_output/` 下会包含：
- 日志 `training_log_safe_*.txt`
- `logs/<run_name>/tensorboard/`：TensorBoard event
- （训练保存逻辑依赖 `fsdp_train.py` 中的 save 实现）通常会有：
  - `final_model/pytorch_model.bin`
  - `final_model/` 下 tokenizer 文件

## 6. 验证（PPL）

```bash
python validate_model.py \
  --model_dir /root/llama-7b/fsdp_output/tmp_save_check/final_model \
  --data_path /root/llama-7b/datasets/wikipedia_en_10mb.json \
  --max_length 128 \
  --batch_size 2
```

输出：`avg_loss` 与 `ppl`。

## 7. 交互式生成（非聊天对齐）

```bash
python chat_model.py \
  --model_dir /root/llama-7b/fsdp_output/final_model \
  --max_new_tokens 128 \
  --temperature 0.8 \
  --top_p 0.95
```

说明：该模型是 Wikipedia 续写式预训练，不保证具备对话能力。

## 8. 通信压缩：当前实现与下一步“中期对比”落地

### 8.1 已实现：INT8 量化 reduce-scatter hook（原型）

位置：`fsdp_train.py` 中

- `GradQuantState`
- `fsdp_quantized_comm_hook(state, full_flat_grad, shard_out)`

关键逻辑：
- 全局 max（`all_reduce(MAX)`) 对齐 scale
- 对称 int8 量化（带 world_size 安全上限 `Qr = 127 // world_size` 避免 sum 溢出）
- `reduce_scatter_tensor` 用 int8 走 `SUM`
- 反量化并除以 world_size 得到平均梯度，写回 `shard_out`

> 注意：当前 `model.register_comm_hook(...)` 在 `fsdp_train.py` 里被注释，默认仍是 baseline。

### 8.2 中期答辩建议的算法对比矩阵（双卡）

建议把实验先聚焦到 world_size=2：

1) Baseline：无压缩（不注册 hook）
2) Quant-INT8：启用 `fsdp_quantized_comm_hook`
3) Sparsify（计划）：Random-k / Top-k（需要设计稀疏格式通信或先做原型）
4) Error Feedback（计划/加分）：对量化/稀疏化增加残差补偿，提高精度保持性

### 8.3 指标（中期必须能落盘/可画图）

- 性能：step_time、tokens/s（建议）
- 通信：通信耗时占比（profiler 或 hook 内计时）
- 精度：val ppl（`validate_model.py`）

输出建议：每次 run 写一个 `results.jsonl`/`results.csv`（后续画图/PPT 直接用）。

## 9. 已知坑与注意事项

1) **大 JSON 文件不能用 `json.load` 全量读**：已通过 `ijson` 流式解析 + shard 缓存解决。
2) **多卡同步**：长预分词阶段不能让非 rank0 过早 barrier（会触发 NCCL store 超时）。现已用“轮询 finalized + barrier”解决。
3) **Profiler**：在非 TTY / tee 场景下 tqdm 输出会乱；已采用 rank0-only。
4) **模型定位**：仓库名称包含 llama7b，但当前主线训练是 GPT2 架构（更利于做通信优化对比）。

## 10. 中期答辩建议交付物（给写 PPT 用）

- 一张系统图：FSDP 前向/反向 + AllGather/ReduceScatter + 压缩插入点
- 一张表：Baseline vs INT8 vs Sparsify（压缩比、吞吐、comm占比、ppl）
- 实验设置页：双卡、数据集、batch/seq、训练步数、seed
- 结果页：
  - tokens/s 或 step_time 对比
  - ppl/loss 对比
  - comm time 占比对比
- 下一步计划：补齐稀疏格式通信 + Error Feedback，扩展到更多卡/更大模型
