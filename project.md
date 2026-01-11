# 项目速览：FSDP-Communication-Optimization（/root/llama-7b）

> 面向中期答辩的快速上手文档：帮助新的 agent 在 10 分钟内搞清楚“这个仓库能干什么 / 怎么跑 / 关键实现在哪里 / 当前进度与下一步对比实验怎么做”。

## 1. 项目目标（一句话）

基于 **PyTorch FSDP（Fully Sharded Data Parallel）** 实现分布式训练闭环，并在其中集成/对比 **通信压缩算法**（量化、稀疏化、误差反馈等），实现 **2–4× 通信压缩比**，同时在常用指标（如 PPL）上 **精度下降不超过 1%**。

当前仓库的训练对象并非真正的 LLaMA-7B 微调，而是使用 **GPT-2 架构（~62M 参数）随机初始化**进行预训练式语言建模（Wikipedia 文本续写），用于通信优化实验对比（迭代更快、易复现）。

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
