#!/bin/bash

# 安全的双GPU FSDP训练脚本
# 使用更保守的参数避免梯度爆炸

echo "🚀 启动LLaMA-7B 安全双GPU FSDP训练"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 配置参数
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GPU_COUNT=2
MODEL_PATH="/root/llama-7b/models"
DATA_PATH="datasets/wikipedia_en_300mb.json"
OUTPUT_DIR="/root/llama-7b/fsdp_output"
LOG_FILE="$OUTPUT_DIR/training_log_safe_${TIMESTAMP}.txt"

# 训练超参（确保与 fsdp_train.py 透传一致）
MASTER_PORT=29501
BATCH_SIZE=16
GRAD_ACC_STEPS=4
LEARNING_RATE=6e-5
NUM_EPOCHS=2
WARMUP_STEPS=0
WEIGHT_DECAY=0.01
MAX_LENGTH=512
DATALOADER_NUM_WORKERS=2
SEED=42
MODEL_SIZE="small"

# 通信压缩（与 fsdp_train.py 一致）
COMM_HOOK="none"
COMM_ERROR_FEEDBACK=""           # 设为 1 启用误差反馈
COMM_SPARSE_COMM="yes"          # 稀疏算法用 indices+values，空则 --no-comm-sparse-comm
COMM_QSGD_S=4
COMM_QSGD_BUCKET_SIZE=0
COMM_TOPK_RATIO=0.01
COMM_THRESHOLD_V=""
COMM_THRESHOLD_V_NEG=""
COMM_INT8_VARIANT="linear"
COMM_ONEBIT_COL_SIZE=256
COMM_ONEBIT_USE_CPP=1        # 1 = 默认启用 C++/CUDA 补丁；空 = 关闭
COMM_SIGNSGD_USE_DELTA=""
COMM_SKETCH_TWO_ROUND=""

# 是否启用性能分析
PROFILE=false

# profile flag for BooleanOptionalAction
if [[ "$PROFILE" == "true" ]]; then
    PROFILE_FLAG="--profile"
else
    PROFILE_FLAG="--no-profile"
fi

echo "📊 训练配置:"
echo "   • GPU数量: $GPU_COUNT"
echo "   • 模型路径: $MODEL_PATH"
echo "   • 数据路径: $DATA_PATH"
echo "   • 输出目录: $OUTPUT_DIR"
echo "   • 时间戳: $TIMESTAMP"
echo "   • master_port: $MASTER_PORT"
echo "   • batch_size: $BATCH_SIZE"
echo "   • grad_accum_steps: $GRAD_ACC_STEPS"
echo "   • learning_rate: $LEARNING_RATE"
echo "   • num_epochs: $NUM_EPOCHS"
echo "   • warmup_steps: $WARMUP_STEPS"
echo "   • weight_decay: $WEIGHT_DECAY"
echo "   • max_length: $MAX_LENGTH"
echo "   • dataloader_num_workers: $DATALOADER_NUM_WORKERS"
echo "   • seed: $SEED"
echo "   • comm_hook: $COMM_HOOK"
echo "   • comm_error_feedback: ${COMM_ERROR_FEEDBACK:-off}"
echo "   • comm_sparse_comm: ${COMM_SPARSE_COMM:-off}"
echo "   • comm_qsgd_s: $COMM_QSGD_S, comm_topk_ratio: $COMM_TOPK_RATIO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查GPU
echo "💾 GPU信息:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

export CUDA_VISIBLE_DEVICES=0,1

RUN_NAME="llama7b-fsdp-$(date +%Y-%m-%d-%H-%M)"
LOG_DIR="/root/llama-7b/fsdp_output/logs/${RUN_NAME}/tensorboard"

echo "🚀 启动带监控的LLaMA-7B训练..."
echo "Run Name: ${RUN_NAME}"
echo "TensorBoard 日志目录: ${LOG_DIR}"

# 清理之前的日志 (可选)
# rm -rf /root/llama-7b/fsdp_output/logs/*
# 检查端口是否被占用
PORT=6006
if lsof -i :$PORT | grep LISTEN; then
    echo "⚠️ 端口 $PORT 已被占用，正在释放..."
    lsof -ti :$PORT | xargs -r kill -9
    sleep 2
    echo "✅ 端口 $PORT 已释放。"
fi

# 在后台启动 TensorBoard
tensorboard --logdir /root/llama-7b/fsdp_output/logs --host 0.0.0.0 --port $PORT &
TENSORBOARD_PID=$!
echo "📊 TensorBoard 已在后台启动 (PID: ${TENSORBOARD_PID})。访问 http://<your-ip>:${PORT}"

# 等待 TensorBoard 启动
sleep 5

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo ""
echo "⏳ 3秒后开始训练... (Ctrl+C 取消)"
sleep 3

echo "🚀 开始安全双GPU训练..."
echo "📝 日志文件: $LOG_FILE"

# 压缩参数（透传）
COMM_EXTRA=()
[[ -n "$COMM_ERROR_FEEDBACK" ]] && COMM_EXTRA+=(--comm-error-feedback)
[[ -n "$COMM_SPARSE_COMM" ]] && COMM_EXTRA+=(--comm-sparse-comm) || COMM_EXTRA+=(--no-comm-sparse-comm)
COMM_EXTRA+=(--comm-qsgd-s "$COMM_QSGD_S" --comm-qsgd-bucket-size "$COMM_QSGD_BUCKET_SIZE")
COMM_EXTRA+=(--comm-topk-ratio "$COMM_TOPK_RATIO" --comm-int8-variant "$COMM_INT8_VARIANT")
COMM_EXTRA+=(--comm-onebit-col-size "$COMM_ONEBIT_COL_SIZE")
[[ -z "$COMM_ONEBIT_USE_CPP" ]] && COMM_EXTRA+=(--no-comm-onebit-use-cpp)
[[ -n "$COMM_THRESHOLD_V" ]] && COMM_EXTRA+=(--comm-threshold-v "$COMM_THRESHOLD_V")
[[ -n "$COMM_THRESHOLD_V_NEG" ]] && COMM_EXTRA+=(--comm-threshold-v-neg "$COMM_THRESHOLD_V_NEG")
[[ -n "$COMM_SIGNSGD_USE_DELTA" ]] && COMM_EXTRA+=(--comm-signsgd-use-delta)
[[ -n "$COMM_SKETCH_TWO_ROUND" ]] && COMM_EXTRA+=(--comm-sketch-two-round)

# 启动训练并记录日志
torchrun \
    --nproc_per_node=$GPU_COUNT \
    --master_port=$MASTER_PORT \
    fsdp_train.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --max_length $MAX_LENGTH \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --model_size "$MODEL_SIZE" \
    --comm-hook "$COMM_HOOK" \
    "${COMM_EXTRA[@]}" \
    $PROFILE_FLAG \
    --seed $SEED \
    --run_name "llama7b-safe-${TIMESTAMP}" 2>&1 | tee "$LOG_FILE"

training_exit_code=${PIPESTATUS[0]}

echo ""
if [ $training_exit_code -eq 0 ]; then
    echo "✅ 训练成功完成!"
else
    echo "❌ 训练失败，退出码: $training_exit_code"
fi

echo "📁 输出目录: $OUTPUT_DIR"
echo "📝 日志文件: $LOG_FILE"

echo ""
echo "📋 输出文件:"
ls -la "$OUTPUT_DIR" | tail -10

echo ""
echo "🎉 安全双GPU训练任务完成! 时间戳: $(date)"
