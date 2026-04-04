#!/bin/bash
# 第五章实验一键运行脚本（对齐 step2_profile.sh）
# 总计 19 组实验，每组 200 steps
# 用法: bash experiments/run_chapter5.sh [--nproc N] [--data_path PATH]
#
# 依赖: scripts/step2_profile.sh（训练 + 指标采集）

set -e
cd "$(dirname "$0")/.."

# ====== 可覆盖的全局默认值 ======
NPROC=2
DATA_PATH="datasets/wikipedia_en_500mb.json"
MAX_STEPS=200
BATCH_SIZE=8
MAX_LENGTH=1024
MODEL_SIZE="medium"
OUTPUT_DIR="/root/llama-7b/fsdp_output"

# 解析命令行覆盖
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc)        NPROC="$2"; shift 2 ;;
    --data_path)    DATA_PATH="$2"; shift 2 ;;
    --max_steps)    MAX_STEPS="$2"; shift 2 ;;
    --batch_size)   BATCH_SIZE="$2"; shift 2 ;;
    --max_length)   MAX_LENGTH="$2"; shift 2 ;;
    --model_size)   MODEL_SIZE="$2"; shift 2 ;;
    --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown: $1" >&2; exit 2 ;;
  esac
done

# 公共参数
BASE="./scripts/step2_profile.sh \
  --data_path $DATA_PATH \
  --nproc $NPROC \
  --max_steps $MAX_STEPS \
  --batch_size $BATCH_SIZE \
  --max_length $MAX_LENGTH \
  --model_size $MODEL_SIZE \
  --output_dir $OUTPUT_DIR"

echo "=========================================="
echo "  第五章实验 - 混合压缩与自适应调度"
echo "  公共配置: nproc=$NPROC, steps=$MAX_STEPS, batch=$BATCH_SIZE"
echo "  开始时间: $(date)"
echo "=========================================="

# ===== 实验一：混合两阶段压缩 (5组) =====
echo ""
echo "===== 实验一：混合两阶段压缩 ====="

echo "[1/19] hybrid_topk_int8"
$BASE --run_name ch5-hybrid_topk_int8 \
  --comm_hook hybrid_topk_int8_v2 \
  --comm_topk_ratio 0.01 \
  --comm_hybrid_sparse_method topk \
  --comm_hybrid_quant_method int8 \
  --comm_error_feedback --comm_ef_local

echo "[2/19] hybrid_topk_1bit"
$BASE --run_name ch5-hybrid_topk_1bit \
  --comm_hook hybrid_topk_1bit \
  --comm_topk_ratio 0.01 \
  --comm_hybrid_sparse_method topk \
  --comm_hybrid_quant_method 1bit \
  --comm_error_feedback --comm_ef_local

echo "[3/19] hybrid_thresholdv_int8"
$BASE --run_name ch5-hybrid_thresholdv_int8 \
  --comm_hook hybrid_thresholdv_int8 \
  --comm_topk_ratio 0.01 \
  --comm_hybrid_sparse_method thresholdv \
  --comm_hybrid_quant_method int8 \
  --comm_error_feedback --comm_ef_local

echo "[4/19] hybrid_thresholdv_1bit"
$BASE --run_name ch5-hybrid_thresholdv_1bit \
  --comm_hook hybrid_thresholdv_1bit \
  --comm_topk_ratio 0.01 \
  --comm_hybrid_sparse_method thresholdv \
  --comm_hybrid_quant_method 1bit \
  --comm_error_feedback --comm_ef_local

echo "[5/19] hybrid_randomk_int8"
$BASE --run_name ch5-hybrid_randomk_int8 \
  --comm_hook hybrid_randomk_int8 \
  --comm_topk_ratio 0.01 \
  --comm_hybrid_sparse_method randomk \
  --comm_hybrid_quant_method int8 \
  --comm_error_feedback --comm_ef_local

# ===== 实验二：自适应调度 (3组 + 2组固定 ratio 基线) =====
echo ""
echo "===== 实验二：自适应压缩调度 ====="

echo "[6/19] adaptive_warmup_decay"
$BASE --run_name ch5-adaptive_warmup_decay \
  --comm_hook adaptive \
  --comm_adaptive_schedule warmup_decay \
  --comm_adaptive_base_hook topk \
  --comm_adaptive_total_steps $MAX_STEPS \
  --comm_adaptive_warmup_fraction 0.1 \
  --comm_adaptive_min_ratio 0.001 \
  --comm_adaptive_max_ratio 0.1 \
  --comm_error_feedback --comm_ef_local

echo "[7/19] adaptive_step_linear"
$BASE --run_name ch5-adaptive_linear \
  --comm_hook adaptive \
  --comm_adaptive_schedule step_linear \
  --comm_adaptive_base_hook topk \
  --comm_adaptive_total_steps $MAX_STEPS \
  --comm_adaptive_min_ratio 0.001 \
  --comm_adaptive_max_ratio 0.1 \
  --comm_error_feedback --comm_ef_local

echo "[8/19] adaptive_grad_adaptive"
$BASE --run_name ch5-adaptive_grad \
  --comm_hook adaptive \
  --comm_adaptive_schedule grad_adaptive \
  --comm_adaptive_base_hook topk \
  --comm_adaptive_total_steps $MAX_STEPS \
  --comm_adaptive_min_ratio 0.001 \
  --comm_adaptive_max_ratio 0.1 \
  --comm_error_feedback --comm_ef_local

echo "[9/19] topk_ratio_0.05 (固定基线)"
$BASE --run_name ch5-topk_005 \
  --comm_hook topk \
  --comm_topk_ratio 0.05 \
  --comm_error_feedback --comm_ef_local

echo "[10/19] topk_ratio_0.1 (固定基线)"
$BASE --run_name ch5-topk_01 \
  --comm_hook topk \
  --comm_topk_ratio 0.1 \
  --comm_error_feedback --comm_ef_local

# ===== 实验三：消融实验 =====
echo ""
echo "===== 实验三：消融实验 ====="

echo "[11/19] hybrid_topk_int8 无误差反馈"
$BASE --run_name ch5-hybrid_topk_int8_no_ef \
  --comm_hook hybrid_topk_int8_v2 \
  --comm_topk_ratio 0.01 \
  --comm_hybrid_sparse_method topk \
  --comm_hybrid_quant_method int8

echo "[12-16/19] 稀疏率敏感性"
for ratio in 0.001 0.005 0.01 0.05 0.1; do
  echo "  hybrid_topk_int8 ratio=${ratio}"
  $BASE --run_name ch5-hybrid_topk_int8_r${ratio} \
    --comm_hook hybrid_topk_int8_v2 \
    --comm_topk_ratio ${ratio} \
    --comm_hybrid_sparse_method topk \
    --comm_hybrid_quant_method int8 \
    --comm_error_feedback --comm_ef_local
done

echo "[17-19/19] 预热比例敏感性"
for wf in 0.0 0.1 0.3; do
  echo "  warmup_fraction=${wf}"
  $BASE --run_name ch5-adaptive_wf${wf} \
    --comm_hook adaptive \
    --comm_adaptive_schedule warmup_decay \
    --comm_adaptive_base_hook topk \
    --comm_adaptive_total_steps $MAX_STEPS \
    --comm_adaptive_warmup_fraction ${wf} \
    --comm_adaptive_min_ratio 0.001 \
    --comm_adaptive_max_ratio 0.1 \
    --comm_error_feedback --comm_ef_local
done

echo ""
echo "=========================================="
echo "  全部实验完成! 结束时间: $(date)"
echo "  输出目录: $OUTPUT_DIR/logs/ch5-*"
echo "  每组实验的指标: <log_dir>/training_metrics.json"
echo "=========================================="
