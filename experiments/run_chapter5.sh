#!/bin/bash
# 第五章实验一键运行脚本
# 总计 19 组实验，每组 200 steps，预计总耗时 2-4 小时
# 用法: bash experiments/run_chapter5.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p logs/chapter5

COMMON="--model_path models/gpt2 --model_size small \
  --data_path datasets/wikipedia_en_10mb.json \
  --batch_size 8 --max_length 1024 --max_steps 200 --num_epochs 1 \
  --learning_rate 6e-5 --warmup_steps 20 --seed 42"

EF="--comm-error-feedback --comm-ef-local"

echo "=========================================="
echo "  第五章实验 - 混合压缩与自适应调度"
echo "  总计 19 组，开始时间: $(date)"
echo "=========================================="

# ===== 实验一：混合压缩 (5组) =====
echo ""
echo "===== 实验一：混合两阶段压缩 ====="

echo "[1/19] hybrid_topk_int8"
torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
  --output_dir output/hybrid_topk_int8 \
  --comm-hook hybrid_topk_int8_v2 --comm-topk-ratio 0.01 $EF \
  2>&1 | tee logs/chapter5/hybrid_topk_int8.log

echo "[2/19] hybrid_topk_1bit"
torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
  --output_dir output/hybrid_topk_1bit \
  --comm-hook hybrid_topk_1bit --comm-topk-ratio 0.01 $EF \
  2>&1 | tee logs/chapter5/hybrid_topk_1bit.log

echo "[3/19] hybrid_thresholdv_int8"
torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
  --output_dir output/hybrid_thresholdv_int8 \
  --comm-hook hybrid_thresholdv_int8 --comm-topk-ratio 0.01 $EF \
  2>&1 | tee logs/chapter5/hybrid_thresholdv_int8.log

echo "[4/19] hybrid_thresholdv_1bit"
torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
  --output_dir output/hybrid_thresholdv_1bit \
  --comm-hook hybrid_thresholdv_1bit --comm-topk-ratio 0.01 $EF \
  2>&1 | tee logs/chapter5/hybrid_thresholdv_1bit.log

echo "[5/19] hybrid_randomk_int8"
torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
  --output_dir output/hybrid_randomk_int8 \
  --comm-hook hybrid_randomk_int8 --comm-topk-ratio 0.01 $EF \
  2>&1 | tee logs/chapter5/hybrid_randomk_int8.log

# ===== 实验二：自适应调度 (3组 + 2组基线) =====
echo ""
echo "===== 实验二：自适应压缩调度 ====="

ADPT="--comm-adaptive-total-steps 200 \
  --comm-adaptive-min-ratio 0.001 --comm-adaptive-max-ratio 0.1 \
  --comm-adaptive-base-hook topk $EF"

echo "[6/19] adaptive_warmup_decay"
torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
  --output_dir output/adaptive_warmup_decay \
  --comm-hook adaptive --comm-adaptive-schedule warmup_decay \
  --comm-adaptive-warmup-fraction 0.1 $ADPT \
  2>&1 | tee logs/chapter5/adaptive_warmup_decay.log

echo "[7/19] adaptive_linear"
torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
  --output_dir output/adaptive_linear \
  --comm-hook adaptive --comm-adaptive-schedule step_linear $ADPT \
  2>&1 | tee logs/chapter5/adaptive_linear.log

echo "[8/19] adaptive_grad"
torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
  --output_dir output/adaptive_grad \
  --comm-hook adaptive --comm-adaptive-schedule grad_adaptive $ADPT \
  2>&1 | tee logs/chapter5/adaptive_grad.log

echo "[9/19] topk_ratio_0.05 (基线)"
torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
  --output_dir output/topk_005 \
  --comm-hook topk --comm-topk-ratio 0.05 $EF \
  2>&1 | tee logs/chapter5/topk_005.log

echo "[10/19] topk_ratio_0.1 (基线)"
torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
  --output_dir output/topk_01 \
  --comm-hook topk --comm-topk-ratio 0.1 $EF \
  2>&1 | tee logs/chapter5/topk_01.log

# ===== 实验三：消融实验 =====
echo ""
echo "===== 实验三：消融实验 ====="

echo "[11/19] hybrid_topk_int8 无误差反馈"
torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
  --output_dir output/hybrid_topk_int8_no_ef \
  --comm-hook hybrid_topk_int8_v2 --comm-topk-ratio 0.01 \
  --no-comm-error-feedback \
  2>&1 | tee logs/chapter5/hybrid_topk_int8_no_ef.log

echo "[12-16/19] 稀疏率敏感性"
for ratio in 0.001 0.005 0.01 0.05 0.1; do
  echo "  hybrid_topk_int8 ratio=${ratio}"
  torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
    --output_dir output/hybrid_topk_int8_r${ratio} \
    --comm-hook hybrid_topk_int8_v2 --comm-topk-ratio ${ratio} $EF \
    2>&1 | tee logs/chapter5/hybrid_topk_int8_r${ratio}.log
done

echo "[17-19/19] 预热比例敏感性"
for wf in 0.0 0.1 0.3; do
  echo "  warmup_fraction=${wf}"
  torchrun --nproc_per_node=2 fsdp_train.py $COMMON \
    --output_dir output/adaptive_wf${wf} \
    --comm-hook adaptive --comm-adaptive-schedule warmup_decay \
    --comm-adaptive-total-steps 200 --comm-adaptive-warmup-fraction ${wf} \
    --comm-adaptive-min-ratio 0.001 --comm-adaptive-max-ratio 0.1 \
    --comm-adaptive-base-hook topk $EF \
    2>&1 | tee logs/chapter5/adaptive_wf${wf}.log
done

echo ""
echo "=========================================="
echo "  全部实验完成! 结束时间: $(date)"
echo "  日志目录: logs/chapter5/"
echo "=========================================="
