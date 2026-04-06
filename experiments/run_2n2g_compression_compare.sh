#!/usr/bin/env bash
# 两机两卡：单算法基线 + run_chapter5.sh 中的混合/自适应/消融对比（均走 step1_profile 取证流程）。
#
# 基线含：int8 linear、onebit_seide、NC、整向量 QSGD(s=4)、topk、randomk、threshold_v、
#         单轮 Sketched-SGD（不传 --comm_sketch_two_round）。
# 第五章：组合压缩（hybrid_*）与自适应调度（adaptive）归为一大类；其下含固定 topk 稀疏率对照、消融（无 EF / 稀疏率 / 预热比例）等。
#
# 用法：
#   MASTER_ADDR=10.0.0.3 NODE_RANK=0 bash experiments/run_2n2g_compression_compare.sh
#   MASTER_ADDR=10.0.0.3 NODE_RANK=1 bash experiments/run_2n2g_compression_compare.sh
#
# 可选环境变量：
#   DATA_PATH、OUTPUT_DIR、MASTER_PORT、MAX_STEPS、MODEL_SIZE、BATCH_SIZE、MAX_LENGTH
#   RUN_NAME_PREFIX  run_name 前缀（默认 2n2g-cmp），完整名为 ${RUN_NAME_PREFIX}-<实验 id>
#   默认 MAX_STEPS=200（与第五章脚本步数一致，便于自适应调度）；快速试跑可设 MAX_STEPS=50

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "请设置 MASTER_ADDR 为主节点 IP，例如: MASTER_ADDR=10.0.0.3 $0" >&2
  exit 2
fi

NODE_RANK="${NODE_RANK:-0}"
MAX_STEPS="${MAX_STEPS:-51}"
MODEL_SIZE="${MODEL_SIZE:-medium}"

# 与 experiments/run_chapter5.sh 对齐：误差反馈 + 本地 EF
EF=(--comm_error_feedback --comm_ef_local)

common=(
  --data_path "${DATA_PATH:-datasets/wikipedia_en_500mb.json}"
  --output_dir "${OUTPUT_DIR:-fsdp_output}"
  --nnodes 2
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "${MASTER_PORT:-29500}"
  --nproc 1
  --max_steps "$MAX_STEPS"
  --model_size "$MODEL_SIZE"
  --batch_size "${BATCH_SIZE:-8}"
  --max_length "${MAX_LENGTH:-1024}"
)

RNPF="${RUN_NAME_PREFIX:-2n2g-cmp}"

# 第一个参数为唯一 run id（接在 RUN_NAME_PREFIX 后），须与实验配置一致，便于对照 logs/<run_name>
run_exp() {
  local run_id="$1"
  shift
  local run_name="${RNPF}-${run_id}"
  echo "========== ${run_name} | NODE_RANK=${NODE_RANK} =========="
  ./scripts/step1_profile.sh "${common[@]}" --run_name "$run_name" "$@"
}

# ---------- A. 单算法基线 ----------
# run_exp A-int8-linear-ms"${MODEL_SIZE}" --comm_hook int8 --comm_int8_variant linear "${EF[@]}"

# run_exp A-onebit-seide-ms"${MODEL_SIZE}" --comm_hook onebit_seide "${EF[@]}"

# run_exp A-nc-ms"${MODEL_SIZE}" --comm_hook nc "${EF[@]}"

# run_exp A-qsgd-s4-bucket0-ms"${MODEL_SIZE}" --comm_hook qsgd --comm_qsgd_s 4 --comm_qsgd_bucket_size 0 "${EF[@]}"

# run_exp A-topk-r001-ef-ms"${MODEL_SIZE}" --comm_hook topk --comm_topk_ratio 0.01 "${EF[@]}"

# run_exp A-randomk-r001-ef-ms"${MODEL_SIZE}" --comm_hook randomk --comm_topk_ratio 0.01 "${EF[@]}"

# run_exp A-thresholdv-r001-ef-ms"${MODEL_SIZE}" --comm_hook threshold_v --comm_topk_ratio 0.01 "${EF[@]}"

# 单轮 Sketch（不显式开启 comm_sketch_two_round）
# run_exp A-sketch-1round-r001-ef-ms"${MODEL_SIZE}" --comm_hook sketch --comm_topk_ratio 0.01 "${EF[@]}"

# ---------- B. 第五章：组合压缩与自适应调度（同一类「高级策略」）----------
# B.1 混合两阶段（稀疏 + 量化/1bit）
# run_exp B1-hybrid-topk-int8-v2-r001-ef-ms"${MODEL_SIZE}" --comm_hook hybrid_topk_int8_v2 --comm_topk_ratio 0.01 "${EF[@]}"

# run_exp B1-hybrid-topk-1bit-r001-ef-ms"${MODEL_SIZE}" --comm_hook hybrid_topk_1bit --comm_topk_ratio 0.01 "${EF[@]}"

# run_exp B1-hybrid-thresholdv-int8-r001-ef-ms"${MODEL_SIZE}" --comm_hook hybrid_thresholdv_int8 --comm_topk_ratio 0.01 "${EF[@]}"

# run_exp B1-hybrid-thresholdv-1bit-r001-ef-ms"${MODEL_SIZE}" --comm_hook hybrid_thresholdv_1bit --comm_topk_ratio 0.01 "${EF[@]}"

# run_exp B1-hybrid-randomk-int8-r001-ef-ms"${MODEL_SIZE}" --comm_hook hybrid_randomk_int8 --comm_topk_ratio 0.01 "${EF[@]}"

# B.2 自适应调度：在训练过程中调节压缩强度（透传 fsdp 参数）
ADPT_BASE=(
  -- --comm-adaptive-total-steps "$MAX_STEPS"
  --comm-adaptive-min-ratio 0.001
  --comm-adaptive-max-ratio 0.1
  --comm-adaptive-base-hook topk
)

# run_exp B2-adaptive-base-topk-warmupdecay-wf01-steps"${MAX_STEPS}"-ms"${MODEL_SIZE}" --comm_hook adaptive "${EF[@]}" "${ADPT_BASE[@]}" \
#   --comm-adaptive-schedule warmup_decay --comm-adaptive-warmup-fraction 0.1

# run_exp B2-adaptive-base-topk-steplinear-steps"${MAX_STEPS}"-ms"${MODEL_SIZE}" --comm_hook adaptive "${EF[@]}" "${ADPT_BASE[@]}" \
#   --comm-adaptive-schedule step_linear

run_exp B2-adaptive-base-topk-grad-steps"${MAX_STEPS}"-ms"${MODEL_SIZE}" --comm_hook adaptive "${EF[@]}" "${ADPT_BASE[@]}" \
  --comm-adaptive-schedule grad_adaptive

# B.3 固定 topk 稀疏率（对照：自适应内部也以 topk 为基，这里用固定 ratio 做基线）
run_exp B3-topk-fixed-r005-ef-ms"${MODEL_SIZE}" --comm_hook topk --comm_topk_ratio 0.05 "${EF[@]}"

run_exp B3-topk-fixed-r01-ef-ms"${MODEL_SIZE}" --comm_hook topk --comm_topk_ratio 0.1 "${EF[@]}"

# ---------- C. 第五章：消融 ----------
run_exp C-hybrid-topk-int8-v2-r001-no-ef-ms"${MODEL_SIZE}" --comm_hook hybrid_topk_int8_v2 --comm_topk_ratio 0.01 \
  -- --no-comm-error-feedback

for ratio in 0.001 0.005 0.01 0.05 0.1; do
  case "$ratio" in
    0.001) rid=r0001 ;;
    0.005) rid=r0005 ;;
    0.01) rid=r001 ;;
    0.05) rid=r005 ;;
    0.1) rid=r01 ;;
    *) rid="${ratio//./p}" ;;
  esac
  run_exp "C-hybrid-topk-int8-v2-sens-${rid}-ef-ms${MODEL_SIZE}" --comm_hook hybrid_topk_int8_v2 --comm_topk_ratio "$ratio" "${EF[@]}"
done

for wf in 0.0 0.1 0.3; do
  case "$wf" in
    0.0) wfid=wf0 ;;
    0.1) wfid=wf01 ;;
    0.3) wfid=wf03 ;;
    *) wfid="wf${wf//./p}" ;;
  esac
  run_exp "C-adaptive-warmupdecay-${wfid}-steps${MAX_STEPS}-ms${MODEL_SIZE}" --comm_hook adaptive "${EF[@]}" \
    -- --comm-adaptive-schedule warmup_decay \
      --comm-adaptive-total-steps "$MAX_STEPS" \
      --comm-adaptive-warmup-fraction "$wf" \
      --comm-adaptive-min-ratio 0.001 \
      --comm-adaptive-max-ratio 0.1 \
      --comm-adaptive-base-hook topk
done

echo "[done] NODE_RANK=${NODE_RANK} 对比实验序列结束。trace 后处理仅在 rank0 执行。"
