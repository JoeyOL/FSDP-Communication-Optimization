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

### INT8 Dynamic Tree 量化（dynamic_tree）

```bash
$BASE --comm_hook int8 --comm_int8_variant dynamic_tree
```

### 1-bit Seide

```bash
$BASE --comm_hook onebit_seide --comm_onebit_col_size 256 --comm_error_feedback
```

如需关闭误差反馈（不推荐）：

```bash
$BASE --comm_hook onebit_seide --comm_onebit_col_size 256
```

### Natural Compression（±2^k 舍入，可选 9 比特打包）

**比特压缩（默认）**：舍入到 ±2^k 后编码为 9 比特/值（1 符号 + 8 指数），按 shard 打包为字节，`all_to_all` 交换后解码并求和。通信量约为 float32 baseline 的 9/32（约 3.5× 下降）。

```bash
$BASE --comm_hook nc --comm_error_feedback
```

关闭 9 比特打包、改用 `reduce_scatter(float32)`（通信量同 baseline）：

```bash
$BASE --comm_hook nc --no_comm_nc_bit_packing
```

**NC loss 偏大时**：默认已开启 **范数缩放**（先除以 \|\|g\|\|_2 再量化，解码后乘回），小梯度不易被舍入为 0，可明显改善 loss。若需关闭：`$BASE --comm_hook nc --no-comm-nc-norm-scale`（一般不推荐）。

### QSGD（均匀量化，支持分桶）

**论文里为何能省通信（Alistarh et al., NIPS 2017）**：  
QSGD 在论文里是「量化 + 编码」两步：量化后不传 float，而是把量化结果编码成比特串（范数 + 级别/符号的打包或 Elias 编码），使每轮通信从 32n 比特降到约 2.8n+32 等，实现约 5.7× 带宽节省。

**本实现**：  
- **low_bit_comm=True（默认）**：**all_reduce(1 float)** 同步全局 scale + **reduce_scatter(int16)**，与 baseline 同一集体、数据量减半，**不再使用 all_to_all**，利于缩短通信时间。  
- **low_bit_comm=False**（`--no-comm-qsgd-low-bit`）：**reduce_scatter(float16)**，同样集体、数据量减半。  
建议整向量/按桶均可加 `--comm_error_feedback` 以减轻量化噪声。

整向量 QSGD（s=4）+ 误差反馈（推荐）：

```bash
$BASE --comm_hook qsgd --comm_qsgd_s 4 --comm_qsgd_bucket_size 0 --comm_error_feedback
```

默认已用 **reduce_scatter(int16)** 替代 all_to_all。若需 float16 可关低比特：

```bash
$BASE --comm_hook qsgd --comm_qsgd_s 4 --comm_qsgd_bucket_size 0 --no-comm-qsgd-low-bit
```

整向量 QSGD（s=4，无误差反馈，易劣于 baseline）：

```bash
$BASE --comm_hook qsgd --comm_qsgd_s 4 --comm_qsgd_bucket_size 0
```

按桶 QSGD（例如每 512 元素一桶）：

```bash
$BASE --comm_hook qsgd --comm_qsgd_s 4 --comm_qsgd_bucket_size 512 --comm_error_feedback
```

**为何 QSGD 有时不能缩短通信时间（基于 step1-20260221-123618 vs step1-20260221-135840）**  
- **baseline（comm_hook=none）**：AllGather 2450 次、约 9.49s，ReduceScatter(f32) 1250 次、约 5.76s，**总通信约 15.24s**。  
- **QSGD（bucket_size=512 + error_feedback）**：ReduceScatter 改为 **f16**，1250 次约 **2.65s**（约减半，符合数据量减半）；但 AllGather 变为 **3700 次、约 14.30s**。  
- **原因**：误差反馈在每次 hook 后要 **all_gather** 完整 shard 以更新 residual（`_apply_error_feedback` → `_all_gather_shard`），所以 AllGather 次数与时间显著增加，**总通信约 16.97s**，比 baseline 略差。  
- **结论**：在单机 2 卡、当前实现下，**error_feedback 引入的 AllGather 开销抵消了 ReduceScatter 减半的收益**；若更关注通信时间，可尝试关闭 error_feedback，或使用**本地误差反馈**（见下）。

**误差反馈优化：本地 EF（`--comm-ef-local`）**  
- 标准 EF：每参数在通信后用 **all_gather** 拼回完整梯度以更新 residual，导致 AllGather 次数与时间增加。  
- **本地 EF**：仅对本 rank 的 **shard** 做残差更新（residual = 本 shard 的 to_compress - shard_out），**不做 all_gather**，通信量与无 EF 时一致，仅多一次本地减法。  
- 使用方式：在开启 `--comm_error_feedback` 时同时加 `--comm_ef_local`，例如：  
  `$BASE --comm_hook qsgd --comm_error_feedback --comm_ef_local` 或  
  `$BASE --comm_hook nc --comm_error_feedback --comm_ef_local`。  
- 注意：本地 EF 的残差只作用于本 rank 的 shard，与论文里「全量 residual」略有差异，通常仍能明显减轻量化/稀疏噪声，且不增加通信时间。

**开/关 EF 平均损失差不多？**  
- **本地 EF（默认）**：只对本 shard 做残差，对 loss 的改善通常弱于全量 EF；在步数少、压缩较轻（如 NC 9-bit）或 batch 较小时，有无 EF 的 loss 差异可能不明显。  
- 若想确认「EF 本身有没有用」：用**全量 EF** 跑同样配置（`--comm_error_feedback --no_comm_ef_local`），看 loss 是否比无 EF 或本地 EF 更稳/更低；全量 EF 更贴近论文，但会多一次 AllGather。  
- 若更在意通信时间：保持默认本地 EF 即可；若更在意收敛/loss：可尝试全量 EF（牺牲一点通信）。

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

显式双阈值（正负不对称）：

```bash
$BASE --comm_hook threshold_v --comm_threshold_v 0.05 --comm_threshold_v_neg -0.02
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

