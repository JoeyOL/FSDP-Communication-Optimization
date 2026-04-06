## 通信压缩命令速查（step1_profile.sh）

默认公共参数示例（单机 2 卡，可按需改）：

```bash
BASE="./scripts/step1_profile.sh --data_path datasets/wikipedia_en_300mb.json --nproc 2"
```

### 不使用任何压缩

```bash
$BASE
```

等价于：

```bash
./scripts/step1_profile.sh --data_path datasets/wikipedia_en_300mb.json --nproc 2 --max_steps 55 --comm_hook none
```

> **说明**：本课题不实现 FP16、SignSGD；以下仅列出已实现的压缩方式。

### INT8 线性量化（linear）

```bash
$BASE --comm_hook int8 --comm_int8_variant linear
```


### 1-bit Seide

首先要更新
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD
```

```bash
$BASE --comm_hook onebit_seide --comm_onebit_col_size 256 --comm_error_feedback
```


### Natural Compression（±2^k 舍入，可选 9 比特打包）

**比特压缩（默认）**：舍入到 ±2^k 后编码为 9 比特/值（1 符号 + 8 指数），按 shard 打包为字节，`all_to_all` 交换后解码并求和。通信量约为 float32 baseline 的 9/32（约 3.5× 下降）。

```bash
$BASE --comm_hook nc --comm_error_feedback
```

关闭 9 比特打包、改用 `reduce_scatter(float32)`（通信量同 baseline）：


**NC loss 偏大时**：默认已开启 **范数缩放**（先除以 \|\|g\|\|_2 再量化，解码后乘回），小梯度不易被舍入为 0，可明显改善 loss。若需关闭：`$BASE --comm_hook nc --no-comm-nc-norm-scale`（一般不推荐）。

### QSGD（均匀量化，支持分桶）


整向量 QSGD（s=4）+ 误差反馈（推荐）：

```bash
$BASE --comm_hook qsgd --comm_qsgd_s 4 --comm_qsgd_bucket_size 0 --comm_error_feedback
```


按桶 QSGD（例如每 512 元素一桶）：

```bash
$BASE --comm_hook qsgd --comm_qsgd_s 4 --comm_qsgd_bucket_size 512 --comm_error_feedback
```


### Top-k 稀疏

```bash
$BASE --comm_hook topk --comm_topk_ratio 0.01
```

### Random-k 稀疏

```bash
$BASE --comm_hook randomk --comm_topk_ratio 0.01
```

### Threshold-v / Threshold-v（双阈值）

单侧阈值（正侧 v，由 ratio 自动估计）：

```bash
$BASE --comm_hook threshold_v --comm_topk_ratio 0.01 --comm_threshold_v 0
```


### Sketched-SGD（Count Sketch + HEAVYMIX）

单轮（只用 Sketch）：

```bash
$BASE --comm_hook sketch
```

两轮 + HEAVYMIX（按论文）：

```bash
$BASE --comm_hook sketch --comm_sketch_two_round
```


## 安全双 GPU 训练脚本（run_safe_training.sh）示例

`run_safe_training.sh` 直接改变量即可：

- **无压缩（baseline）**：在脚本中设置  
  `COMM_HOOK="none"`, `COMM_ERROR_FEEDBACK=""`

- **1-bit Seide**：

  ```bash
  COMM_HOOK="onebit_seide"
  COMM_ERROR_FEEDBACK=1
  COMM_ONEBIT_COL_SIZE=256
  ```

- **INT8 (linear)**：

  ```bash
  COMM_HOOK="int8"
  COMM_INT8_VARIANT="linear"
  ```

- **INT8 (dynamic_tree)**：

  ```bash
  COMM_HOOK="int8"
  COMM_INT8_VARIANT="dynamic_tree"
  ```

- **QSGD / Top-k / Random-k / Threshold-v / Sketch**：  
  各自对应 `COMM_HOOK` 改为 `qsgd` / `topk` / `randomk` / `threshold_v` / `sketch`，并按上面 `step1_profile.sh` 的参数，把同名 `COMM_*` 变量在脚本里设成对应的值即可。

## 如何计算通信字节数

当前 profiler 的 trace/CSV 只记录 NCCL 核的**调用次数**和 **CUDA 时间**，不直接记录每次调用的字节数。通信量可以用下面两种方式得到。

### 1. 按公式从模型参数推算（推荐）

FSDP 对每个参数的梯度做一次 reduce_scatter（以及若开 EF 则再做一次 all_gather）。设 `world_size = W`，某个参数展平后元素个数为 `numel`（需被 W 整除）。

- **reduce_scatter**：每个 rank 发送的字节数 = `numel * sizeof(dtype)`（每个 rank 把自己的整份梯度按 shard 发给各 rank，等价于发送 `numel` 个元素）。
- **all_gather（仅当开启 EF 且未开 `--comm-ef-local` 时）**：每个 rank 发送的字节数 = `numel * sizeof(dtype)`（每个 rank 发送自己的 shard 给所有 rank，总发送量 = (numel/W) * W = numel 元素）。**开启 `--comm-ef-local` 时不做该 all_gather**，通信量同无 EF。

**每参数字节（单 rank 发送）**（未压缩时 dtype=float32，4 字节）：

- 仅 reduce_scatter：`numel * 4`
- reduce_scatter + EF 的 all_gather：`numel * 4 * 2`

**压缩时**：按实际参与通信的 dtype 或打包方式算，例如：

- float16：`numel * 2`
- QSGD low-bit（scale + float16）：all_reduce 1 个 float + reduce_scatter `numel * 2`
- NC 9-bit 打包：reduce_scatter 改为 all_to_all，每 rank 发送约 `ceil(numel/8)*9` 字节（按 shard 打包后发送）

**总通信字节（单 rank）** = 对所有参数求和：  
`sum( 每参数 reduce_scatter 字节 + 每参数 all_gather 字节（若 EF） )`  
（注意：同一参数在一次 step 里各做一次 reduce_scatter 和一次 all_gather，所以是相加。）

### 2. 用脚本按模型参数求和

在已有模型和 `world_size` 的前提下，可以写一小段脚本遍历 `model.parameters()`，对每个 `p` 用 `p.numel()` 乘以上面的每元素字节数（及是否 EF），再求和。例如无压缩、float32、开 EF 时：  
`total_sent = sum(p.numel() * 4 * 2 for p in model.parameters() if p.numel() % world_size == 0)`（实际还需加上不能整除的边界处理）。  
把这段逻辑放到训练前或单独的分析脚本里即可得到「每 step 单 rank 发送字节数」；再乘 step 数即得整次训练的通信量。


### 混合两阶段压缩（Hybrid: 稀疏 + 量化）

混合压缩先做 Stage-1 稀疏化（Top-k / Threshold-v / Random-k），再对稀疏值做 Stage-2 量化（INT8 / 1-bit），实现乘法级压缩比。

**Top-k + INT8（推荐）**：
```bash
$BASE --comm_hook hybrid_topk_int8_v2 --comm_topk_ratio 0.01 --comm_error_feedback --comm_ef_local
```

**Top-k + 1-bit**：
```bash
$BASE --comm_hook hybrid_topk_1bit --comm_topk_ratio 0.01 --comm_error_feedback --comm_ef_local
```

**Threshold-v + INT8**：
```bash
$BASE --comm_hook hybrid_thresholdv_int8 --comm_topk_ratio 0.01 --comm_error_feedback --comm_ef_local
```

**Threshold-v + 1-bit**：
```bash
$BASE --comm_hook hybrid_thresholdv_1bit --comm_topk_ratio 0.01 --comm_error_feedback --comm_ef_local
```

**Random-k + INT8**：
```bash
$BASE --comm_hook hybrid_randomk_int8 --comm_topk_ratio 0.01 --comm_error_feedback --comm_ef_local
```

> 混合压缩使用已有的 `--comm_topk_ratio` 控制 Stage-1 稀疏率。可通过 `--comm_hybrid_sparse_method` 和 `--comm_hybrid_quant_method` 显式指定方法（通常由 hook 名自动决定）。

### 自适应压缩调度（Adaptive Scheduling）

自适应调度根据训练进度动态调整稀疏比例，在训练早期使用较高压缩率加速通信，后期降低压缩率保证收敛质量。

**Warmup-Decay 策略（推荐）**：
```bash
$BASE --comm_hook adaptive --comm_adaptive_schedule warmup_decay \
  --comm_adaptive_base_hook topk \
  --comm_adaptive_total_steps 200 \
  --comm_adaptive_warmup_fraction 0.1 \
  --comm_adaptive_min_ratio 0.001 --comm_adaptive_max_ratio 0.1 \
  --comm_error_feedback --comm_ef_local
```

**Linear 线性衰减策略**：
```bash
$BASE --comm_hook adaptive --comm_adaptive_schedule step_linear \
  --comm_adaptive_base_hook topk \
  --comm_adaptive_total_steps 200 \
  --comm_adaptive_min_ratio 0.001 --comm_adaptive_max_ratio 0.1 \
  --comm_error_feedback --comm_ef_local
```

**Grad-adaptive 梯度自适应策略**：
```bash
$BASE --comm_hook adaptive --comm_adaptive_schedule grad_adaptive \
  --comm_adaptive_base_hook topk \
  --comm_adaptive_total_steps 200 \
  --comm_adaptive_min_ratio 0.001 --comm_adaptive_max_ratio 0.1 \
  --comm_error_feedback --comm_ef_local
```

> **参数说明**：
> - `--comm_adaptive_base_hook`：底层压缩算法（topk / randomk / thresholdv / hybrid）
> - `--comm_adaptive_schedule`：调度策略（warmup_decay / step_linear / grad_adaptive）
> - `--comm_adaptive_total_steps`：总训练步数，0 表示自动使用 max_steps
> - `--comm_adaptive_warmup_fraction`：warmup_decay 策略的预热阶段占比
> - `--comm_adaptive_min_ratio / --comm_adaptive_max_ratio`：稀疏率动态范围

