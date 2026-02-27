#!/usr/bin/env bash
# 循环跑 step1_profile.sh 和 step2_profile.sh 的小工具脚本
# 用于在一组通信压缩配置上批量收集 profile 与训练指标。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 脚本名
SCRIPT_NAME="step1_profile.sh"

# 配置列表：每一行是「实验标签|step1 附加参数|step2 附加参数」
# 可以仿照下面两行自行扩展更多配置。
CONFIGS=(
  # "baseline|--comm_hook none"
  # "int8_linear|--comm_hook int8 --comm_int8_variant linear"
  # "onebit_seide|--comm_hook onebit_seide --comm_onebit_col_size 256 --comm_error_feedback"
  # "nc|--comm_hook nc --comm_error_feedback"
  # "qsgd|--comm_hook qsgd --comm_qsgd_s 4 --comm_qsgd_bucket_size 0 --comm_error_feedback"
  # "topk|--comm_hook topk --comm_topk_ratio 0.01"
  "randomk|--comm_hook randomk"
  "threshold_v|--comm_hook threshold_v --comm_threshold_v 0"
  "sketch|--comm_hook sketch"
)

usage() {
  cat <<EOF
用法示例：
  ./scripts/run_step_profile_grid.sh
EOF
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "${ROOT_DIR}"

for cfg in "${CONFIGS[@]}"; do
  IFS="|" read -r LABEL ARGS <<< "${cfg}"

  TS="$(date +%Y%m%d-%H%M%S)"
  RUN_NAME="${SCRIPT_NAME}-${LABEL}-${TS}"

  echo "============================================================"
  echo "[GRID] Running config: ${LABEL}"
  echo "  run_name = ${RUN_NAME}"
  echo "  args = ${ARGS}"
  echo "  script = ${SCRIPT_NAME}"
  echo "============================================================"

  # ---- Phase 1: step1_profile（耗时取证）----
  echo "[GRID][${LABEL}]: running ${SCRIPT_NAME}"
  ./scripts/${SCRIPT_NAME} \
    --run_name "${RUN_NAME}" \
    ${ARGS}

  echo "[GRID][${LABEL}] Done."
  echo
done

