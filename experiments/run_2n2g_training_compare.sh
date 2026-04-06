#!/usr/bin/env bash
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
EVAL_WIKI_PPL="${EVAL_WIKI_PPL:-0}"

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

if [[ "$EVAL_WIKI_PPL" == "1" ]]; then
  common+=(--eval_wiki_ppl)
else
  common+=(--no_eval_wiki_ppl)
fi

RNPF="${RUN_NAME_PREFIX:-2n2g-cmp-s2}"

run_exp() {
  local run_id="$1"
  shift
  local run_name="${RNPF}-${run_id}"
  echo "========== ${run_name} | NODE_RANK=${NODE_RANK} =========="
  ./scripts/step2_profile.sh "${common[@]}" --run_name "$run_name" "$@"
}

ADPT_BASE=(
  -- --comm-adaptive-total-steps "$MAX_STEPS"
  --comm-adaptive-min-ratio 0.001
  --comm-adaptive-max-ratio 0.1
  --comm-adaptive-base-hook topk
)

run_exp B2-adaptive-base-topk-grad-steps"${MAX_STEPS}"-ms"${MODEL_SIZE}" --comm_hook adaptive "${EF[@]}" "${ADPT_BASE[@]}" \
  --comm-adaptive-schedule grad_adaptive

run_exp B3-topk-fixed-r005-ef-ms"${MODEL_SIZE}" --comm_hook topk --comm_topk_ratio 0.05 "${EF[@]}"
run_exp B3-topk-fixed-r01-ef-ms"${MODEL_SIZE}" --comm_hook topk --comm_topk_ratio 0.1 "${EF[@]}"

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

echo "[done] NODE_RANK=${NODE_RANK} 指标采集对比实验结束。请查看 <output_dir>/logs/<run_name>/training_metrics.json"#!/usr/bin/env bash
# 两机两卡：用于训练指标采集（step2_profile）版的压缩算法对比脚本。
#
# 与 run_2n2g_compression_compare.sh（step1）区别：
# - 这里走 scripts/step2_profile.sh
# - 重点产物是 <output_dir>/logs/<run_name>/training_metrics.json
#   （loss/显存曲线、学习率曲线、吞吐、可选验证 PPL）
#
# 用法：
#   MASTER_ADDR=10.0.0.3 NODE_RANK=0 bash experiments/run_2n2g_compression_compare_step2.sh
#   MASTER_ADDR=10.0.0.3 NODE_RANK=1 bash experiments/run_2n2g_compression_compare_step2.sh
#
# 可选环境变量：
#   DATA_PATH、OUTPUT_DIR、MASTER_PORT、MAX_STEPS、MODEL_SIZE、BATCH_SIZE、MAX_LENGTH
#   RUN_NAME_PREFIX  run_name 前缀（默认 2n2g-cmp-s2），完整名为 ${RUN_NAME_PREFIX}-<实验 id>
#   EVAL_WIKI_PPL   是否做同源验证 PPL（1=开启，0=关闭，默认 0）

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
EVAL_WIKI_PPL="${EVAL_WIKI_PPL:-0}"

# 与 chapter5 / step1 对齐：误差反馈 + 本地 EF
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

if [[ "$EVAL_WIKI_PPL" == "1" ]]; then
  common+=(--eval_wiki_ppl)
else
  common+=(--no_eval_wiki_ppl)
fi

RNPF="${RUN_NAME_PREFIX:-2n2g-cmp-s2}"

# 第一个参数为唯一 run id（接在 RUN_NAME_PREFIX 后），须与实验配置一致
run_exp() {
  local run_id="$1"
  shift
  local run_name="${RNPF}-${run_id}"
  echo "========== ${run_name} | NODE_RANK=${NODE_RANK} =========="
  ./scripts/step2_profile.sh "${common[@]}" --run_name "$run_name" "$@"
}

# ---------- A. 单算法基线 ----------
run_exp A-int8-linear-ms"${MODEL_SIZE}" --comm_hook int8 --comm_int8_variant linear "${EF[@]}"
run_exp A-onebit-seide-ms"${MODEL_SIZE}" --comm_hook onebit_seide "${EF[@]}"
run_exp A-nc-ms"${MODEL_SIZE}" --comm_hook nc "${EF[@]}"
run_exp A-qsgd-s4-bucket0-ms"${MODEL_SIZE}" --comm_hook qsgd --comm_qsgd_s 4 --comm_qsgd_bucket_size 0 "${EF[@]}"
run_exp A-topk-r001-ef-ms"${MODEL_SIZE}" --comm_hook topk --comm_topk_ratio 0.01 "${EF[@]}"
run_exp A-randomk-r001-ef-ms"${MODEL_SIZE}" --comm_hook randomk --comm_topk_ratio 0.01 "${EF[@]}"
run_exp A-thresholdv-r001-ef-ms"${MODEL_SIZE}" --comm_hook threshold_v --comm_topk_ratio 0.01 "${EF[@]}"
run_exp A-sketch-1round-r001-ef-ms"${MODEL_SIZE}" --comm_hook sketch --comm_topk_ratio 0.01 "${EF[@]}"

# ---------- B. 组合压缩与自适应调度 ----------
# B.1 混合两阶段（稀疏 + 量化/1bit）
run_exp B1-hybrid-topk-int8-v2-r001-ef-ms"${MODEL_SIZE}" --comm_hook hybrid_topk_int8_v2 --comm_topk_ratio 0.01 "${EF[@]}"
run_exp B1-hybrid-topk-1bit-r001-ef-ms"${MODEL_SIZE}" --comm_hook hybrid_topk_1bit --comm_topk_ratio 0.01 "${EF[@]}"
run_exp B1-hybrid-thresholdv-int8-r001-ef-ms"${MODEL_SIZE}" --comm_hook hybrid_thresholdv_int8 --comm_topk_ratio 0.01 "${EF[@]}"
run_exp B1-hybrid-thresholdv-1bit-r001-ef-ms"${MODEL_SIZE}" --comm_hook hybrid_thresholdv_1bit --comm_topk_ratio 0.01 "${EF[@]}"
run_exp B1-hybrid-randomk-int8-r001-ef-ms"${MODEL_SIZE}" --comm_hook hybrid_randomk_int8 --comm_topk_ratio 0.01 "${EF[@]}"

# B.2 自适应调度
ADPT_BASE=(
  -- --comm-adaptive-total-steps "$MAX_STEPS"
  --comm-adaptive-min-ratio 0.001
  --comm-adaptive-max-ratio 0.1
  --comm-adaptive-base-hook topk
)

run_exp B2-adaptive-base-topk-warmupdecay-wf01-steps"${MAX_STEPS}"-ms"${MODEL_SIZE}" --comm_hook adaptive "${EF[@]}" "${ADPT_BASE[@]}" \
  --comm-adaptive-schedule warmup_decay

run_exp B2-adaptive-base-topk-steplinear-steps"${MAX_STEPS}"-ms"${MODEL_SIZE}" --comm_hook adaptive "${EF[@]}" "${ADPT_BASE[@]}" \
  --comm-adaptive-schedule step_linear

run_exp B2-adaptive-base-topk-grad-steps"${MAX_STEPS}"-ms"${MODEL_SIZE}" --comm_hook adaptive "${EF[@]}" "${ADPT_BASE[@]}" \
  --comm-adaptive-schedule grad_adaptive

# B.3 固定 topk 稀疏率（对照）
run_exp B3-topk-fixed-r005-ef-ms"${MODEL_SIZE}" --comm_hook topk --comm_topk_ratio 0.05 "${EF[@]}"
run_exp B3-topk-fixed-r01-ef-ms"${MODEL_SIZE}" --comm_hook topk --comm_topk_ratio 0.1 "${EF[@]}"

# ---------- C. 消融 ----------
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

echo "[done] NODE_RANK=${NODE_RANK} 指标采集对比实验结束。请查看 <output_dir>/logs/<run_name>/training_metrics.json"