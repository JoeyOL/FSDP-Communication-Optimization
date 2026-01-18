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

### 步骤 2：实现“通信压缩热插拔模块”（支持多种压缩算法）

目标：把压缩算法从训练脚本中解耦出来，形成可配置/可扩展的模块，并能以“开关 + 配置”方式在 FSDP 通信路径上启用。

验收要求：

- 训练脚本支持命令行选择压缩算法，例如：`--comm_compress {none,int8,fp16,qsgd,nc,topk,randomk,thresholdv,sketch,gradiveq,signsgd,onebit,...}`（名称可再统一）。
- 压缩模块能在不改训练主逻辑的情况下替换/组合（热插拔）：
  - baseline：不注册 comm hook；
  - 压缩：注册对应 hook，并且保证训练可跑通（至少小数据集/少步数）。

当前仓库实现口径（模块化已落地）：

- 压缩器统一放在 `comm_compress/`，通过注册表按名称创建可注册到 FSDP 的 comm hook。
- 训练入口参数：
  - `--comm_compress <method>`：选择压缩方法（默认 `none`）。
  - `--comm_config_json '{...}'`：方法配置（JSON object 字符串）。

示例：

```bash
# baseline（不压缩）
torchrun --standalone --nproc_per_node=2 fsdp_train.py \
  --data_path /root/llama-7b/datasets/wikipedia_en_10mb.json \
  --output_dir /root/llama-7b/fsdp_output \
  --comm_compress none

# int8：对称 int8 量化 + int8 reduce-scatter
torchrun --standalone --nproc_per_node=2 fsdp_train.py \
  --data_path /root/llama-7b/datasets/wikipedia_en_10mb.json \
  --output_dir /root/llama-7b/fsdp_output \
  --comm_compress int8 \
  --comm_config_json '{"num_bits": 8}'
```

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
  - **通信压缩 hook 通过模块化包接入**：见 `comm_compress/`（训练入口参数 `--comm_compress/--comm_config_json`）。
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

说明：该脚本是“真正的项目启动命令”，默认会启动 TensorBoard 并用 `torchrun` 跑双卡。脚本也支持把额外参数透传给 `fsdp_train.py`，用于做压缩方法/取证参数的对比实验，例如：

```bash
# baseline
./run_safe_training.sh --comm_compress none

# int8 压缩
./run_safe_training.sh --comm_compress int8 --comm_config_json '{"num_bits": 8}'
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

位置：`comm_compress/int8.py`（模块化实现）

启用方式（训练入口参数）：

- `--comm_compress int8`
- `--comm_config_json '{"num_bits": 8}'`

关键逻辑：
- 全局 max（`all_reduce(MAX)`) 对齐 scale
- 对称 int8 量化（带 world_size 安全上限 `Qr = 127 // world_size` 避免 sum 溢出）
- `reduce_scatter_tensor` 用 int8 走 `SUM`
- 反量化并除以 world_size 得到平均梯度，写回 `shard_out`

> 注意：当前 `model.register_comm_hook(...)` 在 `fsdp_train.py` 里被注释，默认仍是 baseline。

> 注意：当前实现通过 `comm_compress/registry.py` 按方法名创建 hook，并在 `fsdp_train.py` 中按参数注册；baseline 为 `--comm_compress none`（默认）。

### 8.2 中期答辩建议的算法对比矩阵（双卡）

建议把实验先聚焦到 world_size=2：

1) Baseline：无压缩（不注册 hook）
2) Quant-INT8：启用 `--comm_compress int8`
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
