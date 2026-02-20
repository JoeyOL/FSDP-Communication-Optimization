## 通信压缩命令速查（step1_profile.sh）

默认公共参数示例（单机 2 卡，可按需改）：

```bash
BASE="./scripts/step1_profile.sh --data_path datasets/wikipedia_en_300mb.json --nproc 2 --max_steps 55"
```

### 不使用任何压缩

```bash
$BASE
```

等价于：

```bash
./scripts/step1_profile.sh --data_path datasets/wikipedia_en_300mb.json --nproc 2 --max_steps 55 --comm_hook none
```

### FP16（仅作为对照：不改变通信，只改算子精度）

> 通过 `fsdp_train.py` 里的 `--dtype`/混合精度参数控制，和 `--comm-hook` 无关，这里不单列命令。

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

### SignSGD

标准 SignSGD（带尺度）：

```bash
$BASE --comm_hook signsgd
```

论文式 “仅方向 + lr 当步长”：

```bash
$BASE --comm_hook signsgd --comm_signsgd_use_delta
```

### QSGD（均匀量化，支持分桶）

整向量 QSGD（s=4）：

```bash
$BASE --comm_hook qsgd --comm_qsgd_s 4 --comm_qsgd_bucket_size 0
```

按桶 QSGD（例如每 512 元素一桶）：

```bash
$BASE --comm_hook qsgd --comm_qsgd_s 4 --comm_qsgd_bucket_size 512
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

- **QSGD / Top-k / Random-k / Threshold-v / SignSGD / Sketch**：  
  各自对应 `COMM_HOOK` 改为 `qsgd` / `topk` / `randomk` / `threshold_v` / `signsgd` / `sketch`，并按上面 `step1_profile.sh` 的参数，把同名 `COMM_*` 变量在脚本里设成对应的值即可。

