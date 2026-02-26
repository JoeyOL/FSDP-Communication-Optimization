## 训练指标采集脚本速查（step2_profile.sh）

默认公共参数示例（单机 **2 卡**，seq_len=1024，可按需改）：

```bash
BASE="./scripts/step2_profile.sh"
```

- 默认 `--nproc 2`，即单机双卡 FSDP 训
- 默认 `--num_epochs 1`  
- 默认开启 **同源 Wikipedia 5% held-out 验证 PPL**（`--eval_wiki_ppl` 打开；可用 `--no_eval_wiki_ppl` 关闭）

step2 的目标是：**先完整跑一次训练，再自动收集本次运行的训练指标**：

- 训练 loss 曲线（`Loss/step`）
- GPU 显存曲线（`Memory/Allocated_GB` / `Memory/Reserved_GB`）
- 学习率曲线（`LearningRate/step`）
- 吞吐（tokens/s）
- 同源 Wikipedia 5% held-out 验证集 PPL（可关）

所有结果会写入：

```text
<output_dir>/logs/<run_name>/training_metrics.json
```

---

### 1. 最简单的使用方式（双卡 + 默认 1 epoch + 默认 PPL）

```bash
$BASE
```

等价于：

```bash
./scripts/step2_profile.sh \
  --data_path datasets/wikipedia_en_300mb.json \
  --output_dir /root/llama-7b/fsdp_output \
  --run_name step2-<timestamp> \
  --num_epochs 1 \
  --nproc 2 \
  --eval_wiki_ppl
```

训练结束后，你会在：

```text
/root/llama-7b/fsdp_output/logs/<run_name>/training_metrics.json
```

看到本次 run 的：

- `loss_curve`：step 级别损失
- `memory_allocated_gb` / `memory_reserved_gb`：显存占用曲线
- `learning_rate_curve`：学习率随 step 变化
- `throughput`：总步数、总 tokens、tokens/s
- `validation`：同源 5% Wikipedia 子集上的 PPL（若未关闭）

---

### 2. 控制训练轮数（num_epochs）

例如训练 **3 个 epoch**：

```bash
$BASE --num_epochs 3
```

等价于传给 `fsdp_train.py`：

```bash
--num_epochs 3
```

`training_metrics.json` 中的 loss/显存/吞吐会覆盖完整 3 个 epoch 的训练过程，PPL 则在训练结束后（用最终 checkpoint）统一计算一次。

---

### 3. 开/关验证集 PPL

#### 3.1 使用同源 Wikipedia 5% held-out（默认行为）

step2 默认已经打开同源 PPL 验证：

```bash
$BASE               # 等价于加了 --eval_wiki_ppl
```

内部行为：

- 从 `config.json` 读取 `data_path`（与你 `--data_path` 相同）
- 构建同一个 `WikipediaDataset`
- 用 **最后 5% 样本** 作为 held-out，用最终模型在这部分数据上计算 token-level PPL
- 结果写入 `training_metrics.json` 的 `validation` 字段

#### 3.2 关闭验证 PPL

如果只关心训练曲线和吞吐，不想跑验证：

```bash
$BASE --no_eval_wiki_ppl
```

这样 `training_metrics.json` 中将不再包含 `validation` 字段。

---

### 4. 调整 batch_size 与梯度累积（建议搭配使用）

step2 中关键训练相关参数：

- `--batch_size`：单卡每个 step 看到的样本序列数（训练脚本里的 `batch_size`）
- `GRADIENT_ACCUMULATION_STEPS`：在 `scripts/step2_profile.sh` 顶部可修改的 Bash 变量（默认 `1`）
- `--nproc`：卡数（默认 2）

一次 **optimizer step** 对应的全局序列数为：

```text
global_batch_seq = batch_size * GRADIENT_ACCUMULATION_STEPS * nproc
```

例如：

- 单卡 `batch_size=1`，`nproc=2`
- 希望 `global_batch_seq ≈ 32`，可以设：

```bash
GRADIENT_ACCUMULATION_STEPS=16   # 在 scripts/step2_profile.sh 顶部修改
./scripts/step2_profile.sh --data_path datasets/wikipedia_en_300mb.json --batch_size 1
```

> **经验建议**：  
> - 做通信/吞吐对比实验时，可以从 `GRADIENT_ACCUMULATION_STEPS=16` 起步  
> - 若 loss 曲线太抖，可以提高到 32，使全局 batch 更大、更平滑  
> - 梯度累积不会增加单步显存，只会让一次优化步覆盖更多 step，训练 wall-clock 会更慢但更稳定

---

### 5. 搭配不同通信压缩配置跑 step2

step2 照样支持所有 `comm_hook` 相关参数（与 `step1_profile.sh`、`run_safe_training.sh` 一致），例如：

#### 5.1 baseline（无压缩）

```bash
$BASE --comm_hook none
```

#### 5.2 Natural Compression（带误差反馈）

```bash
$BASE --comm_hook nc --eval_wiki_ppl
```

如需关闭 9-bit 打包（改用 float32 reduce_scatter）：

```bash
$BASE --comm_hook nc --no_comm_nc_bit_packing
```

#### 5.3 QSGD（整向量 s=4，带误差反馈）

```bash
$BASE --comm_hook qsgd --comm_qsgd_s 4 --comm_qsgd_bucket_size 0 --comm_error_feedback
```

其他压缩方式（Top-k / Random-k / Threshold-v / Sketch / 1-bit / INT8 等）的具体参数含义，可参考 `scripts/comm_usage.md` 中对应小节，这里只需把命令里的 `./scripts/step1_profile.sh` 换成 `./scripts/step2_profile.sh` 即可。

---

### 6. 输出文件速览

一次 step2 运行结束后，主要产物位于：

```text
<output_dir>/logs/<run_name>/
  ├── config.json                  # 本次 run 的关键训练与通信配置（供离线分析）
  ├── tensorboard/                 # events.out.tfevents.*，可用 tensorboard 本地查看曲线
  └── training_metrics.json        # 本脚本收集的训练指标（loss/显存/吞吐/可选 PPL）
```

其中 `training_metrics.json` 的典型字段：

- `loss_curve`：`[{ "step": ..., "value": ..., "wall_time": ... }, ...]`
- `memory_allocated_gb` / `memory_reserved_gb`
- `learning_rate_curve`
- `throughput`：`{ "num_steps", "wall_time_sec", "total_tokens", "tokens_per_sec" }`
- `validation`（可选）：`{ "val_ppl", "val_mean_nll", "val_tokens" }` 或包含 `error` 信息

