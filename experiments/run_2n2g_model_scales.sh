#!/usr/bin/env bash
# 两机两卡：依次跑 small / medium / large，通信 hook 为 none（不注册压缩 hook）。
#
# 用法（与 scripts/step1_profile.sh 文档一致）：
#   主节点（node_rank=0）：
#     MASTER_ADDR=10.0.0.3 NODE_RANK=0 bash experiments/run_2n2g_model_scales.sh
#   另一台（node_rank=1）：
#     MASTER_ADDR=10.0.0.3 NODE_RANK=1 bash experiments/run_2n2g_model_scales.sh
#
# 可选环境变量：
#   DATA_PATH、OUTPUT_DIR、MASTER_PORT、MAX_STEPS、BATCH_SIZE、MAX_LENGTH 等会传给 step1_profile.sh
#   RUN_NAME_PREFIX  run_name 前缀（默认 2n2g-scale），完整名为 ${RUN_NAME_PREFIX}-<固定后缀>

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "请设置 MASTER_ADDR 为主节点 IP，例如: MASTER_ADDR=10.0.0.3 $0" >&2
  exit 2
fi

NODE_RANK="${NODE_RANK:-0}"

common=(
  --data_path "${DATA_PATH:-datasets/wikipedia_en_500mb.json}"
  --output_dir "${OUTPUT_DIR:-fsdp_output}"
  --nnodes 2
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "${MASTER_PORT:-29500}"
  --nproc 1
  --max_steps "${MAX_STEPS:-51}"
  --batch_size "${BATCH_SIZE:-8}"
  --max_length "${MAX_LENGTH:-1024}"
  --comm_hook none
)

RNPF="${RUN_NAME_PREFIX:-2n2g-scale}"

for size in small medium large; do
  # run_name：场景 + 无压缩 + 模型规模（与 logs 子目录一一对应）
  run_name="${RNPF}-nohook-comm-none-ms-${size}"
  echo "========== ${run_name} | NODE_RANK=${NODE_RANK} =========="
  ./scripts/step1_profile.sh "${common[@]}" \
    --model_size "$size" \
    --run_name "$run_name"
done

echo "[done] NODE_RANK=${NODE_RANK} 全部规模跑完。离线汇总请在 rank0 上: python tools/collect_training_metrics.py --log_dir ..."
