#!/usr/bin/env bash
# Step2: 先执行训练，训练结束后自动采集本次运行的数据（loss/显存曲线、吞吐、可选验证集 PPL）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Step2: 训练 + 训练数据采集（先跑训练，再从 TensorBoard 与 config 采集 loss/显存/吞吐，可选验证集 PPL；不采集通信/trace）

用法：
  ./scripts/step2_profile.sh --data_path /path/to/wiki.json [options] [-- <passthrough args to fsdp_train.py>]

常用参数：
  --output_dir DIR            输出目录（默认 /root/llama-7b/fsdp_output）
  --run_name NAME             运行名（默认 step2-<timestamp>）
  --num_epochs N              训练轮数（默认 1）
  --max_steps N               最多训练多少个 step（跨 epoch 计数；0 表示不限制，默认 55）
  --eval_wiki_ppl             开启同源 Wikipedia 5% held-out 验证 PPL（基于 data_path 自动划分，无需单独 val.json，默认已开启）
  --no_eval_wiki_ppl          关闭同源 Wikipedia 验证 PPL
  --nproc N                   进程/卡数（默认 2；>1 使用 torchrun）

其他参数与 step1_profile.sh 一致（batch_size、max_length、comm_hook 等）。

产物：
  tensorboard、config.json，此外：
  <output_dir>/logs/<run_name>/training_metrics.json   （loss 曲线、显存曲线、吞吐 tokens/s、可选 val_ppl）

示例：
  ./scripts/step2_profile.sh --data_path datasets/wikipedia_en_300mb.json
  ./scripts/step2_profile.sh --data_path datasets/wikipedia_en_300mb.json --no_eval_wiki_ppl
EOF
}

DATA_PATH="datasets/wikipedia_en_500mb.json"
OUTPUT_DIR="/root/llama-7b/fsdp_output"
RUN_NAME=""
EVAL_WIKI_PPL=1
NPROC=2
NUM_EPOCHS=1
MAX_STEPS=200
NNODES=1
NODE_RANK=0
MASTER_ADDR=""
MASTER_PORT=29500
DATASET_MAX_SAMPLES=0
BATCH_SIZE=16
MAX_LENGTH=1024
GRADIENT_ACCUMULATION_STEPS=1
MODEL_SIZE="medium"
COMM_HOOK="none"
COMM_ERROR_FEEDBACK=""
COMM_EF_LOCAL=""
COMM_SPARSE_COMM="yes"
COMM_QSGD_S=4
COMM_QSGD_BUCKET_SIZE=0
COMM_TOPK_RATIO=0.01
COMM_THRESHOLD_V=""
COMM_THRESHOLD_V_NEG=""
COMM_INT8_VARIANT="linear"
COMM_ONEBIT_COL_SIZE=256
COMM_ONEBIT_USE_CPP=1
COMM_SIGNSGD_USE_DELTA=""
COMM_SKETCH_TWO_ROUND=""
COMM_NC_BIT_PACKING=1

PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --data_path)
      DATA_PATH="$2"; shift 2 ;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --run_name)
      RUN_NAME="$2"; shift 2 ;;
    --num_epochs)
      NUM_EPOCHS="$2"; shift 2 ;;
    --eval_wiki_ppl)
      EVAL_WIKI_PPL=1; shift 1 ;;
    --no_eval_wiki_ppl)
      EVAL_WIKI_PPL=0; shift 1 ;;
    --nproc)
      NPROC="$2"; shift 2 ;;
    --nnodes)
      NNODES="$2"; shift 2 ;;
    --node_rank)
      NODE_RANK="$2"; shift 2 ;;
    --master_addr)
      MASTER_ADDR="$2"; shift 2 ;;
    --master_port)
      MASTER_PORT="$2"; shift 2 ;;
    --max_steps)
      MAX_STEPS="$2"; shift 2 ;;
    --dataset_max_samples)
      DATASET_MAX_SAMPLES="$2"; shift 2 ;;
    --batch_size)
      BATCH_SIZE="$2"; shift 2 ;;
    --max_length)
      MAX_LENGTH="$2"; shift 2 ;;
    --model_size)
      MODEL_SIZE="$2"; shift 2 ;;
    --comm_hook)
      COMM_HOOK="$2"; shift 2 ;;
    --comm_error_feedback)
      COMM_ERROR_FEEDBACK=1; shift 1 ;;
    --comm_ef_local)
      COMM_EF_LOCAL=1; shift 1 ;;
    --no_comm_ef_local)
      COMM_EF_LOCAL=0; shift 1 ;;
    --no_comm_sparse_comm)
      COMM_SPARSE_COMM=""; shift 1 ;;
    --comm_qsgd_s)
      COMM_QSGD_S="$2"; shift 2 ;;
    --comm_qsgd_bucket_size)
      COMM_QSGD_BUCKET_SIZE="$2"; shift 2 ;;
    --comm_topk_ratio)
      COMM_TOPK_RATIO="$2"; shift 2 ;;
    --comm_threshold_v)
      COMM_THRESHOLD_V="$2"; shift 2 ;;
    --comm_threshold_v_neg)
      COMM_THRESHOLD_V_NEG="$2"; shift 2 ;;
    --comm_int8_variant)
      COMM_INT8_VARIANT="$2"; shift 2 ;;
    --comm_onebit_col_size)
      COMM_ONEBIT_COL_SIZE="$2"; shift 2 ;;
    --no_comm_onebit_use_cpp)
      COMM_ONEBIT_USE_CPP=""; shift 1 ;;
    --comm_signsgd_use_delta)
      COMM_SIGNSGD_USE_DELTA=1; shift 1 ;;
    --comm_sketch_two_round)
      COMM_SKETCH_TWO_ROUND=1; shift 1 ;;
    --no_comm_nc_bit_packing)
      COMM_NC_BIT_PACKING=""; shift 1 ;;
    --)
      shift
      PASSTHROUGH+=("$@")
      break
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$DATA_PATH" ]]; then
  echo "--data_path is required" >&2
  usage
  exit 2
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="step2-$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "$OUTPUT_DIR"

LOG_DIR="${OUTPUT_DIR}/logs/${RUN_NAME}"
mkdir -p "$LOG_DIR"

cat > "$LOG_DIR/config.json" << EOF
{
  "run_name": "${RUN_NAME}",
  "data_path": "${DATA_PATH}",
  "output_dir": "${OUTPUT_DIR}",
  "num_epochs": ${NUM_EPOCHS},
  "batch_size": ${BATCH_SIZE},
  "max_length": ${MAX_LENGTH},
  "dataset_max_samples": ${DATASET_MAX_SAMPLES},
  "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS},
  "model_size": "${MODEL_SIZE}",
  "warmup_steps": 0,
  "nproc": ${NPROC},
  "nnodes": ${NNODES},
  "comm_hook": "${COMM_HOOK}",
  "comm_error_feedback": $([ -n "$COMM_ERROR_FEEDBACK" ] && echo true || echo false),
  "comm_ef_local": $([ "$COMM_EF_LOCAL" = "0" ] && echo false || echo true),
  "comm_sparse_comm": $([ -n "$COMM_SPARSE_COMM" ] && echo true || echo false),
  "comm_qsgd_s": ${COMM_QSGD_S},
  "comm_qsgd_bucket_size": ${COMM_QSGD_BUCKET_SIZE},
  "comm_topk_ratio": ${COMM_TOPK_RATIO},
  "comm_int8_variant": "${COMM_INT8_VARIANT}",
  "comm_onebit_col_size": ${COMM_ONEBIT_COL_SIZE},
  "comm_onebit_use_cpp": $([ -n "$COMM_ONEBIT_USE_CPP" ] && echo true || echo false),
  "comm_nc_bit_packing": $([ -n "$COMM_NC_BIT_PACKING" ] && echo true || echo false),
  "timestamp": "$(date -Iseconds)"
}
EOF
echo "[CONFIG] Saved config to ${LOG_DIR}/config.json"

COMM_EXTRA=()
[[ -n "$COMM_ERROR_FEEDBACK" ]] && COMM_EXTRA+=(--comm-error-feedback)
[[ "$COMM_EF_LOCAL" = "1" ]] && COMM_EXTRA+=(--comm-ef-local)
[[ "$COMM_EF_LOCAL" = "0" ]] && COMM_EXTRA+=(--no-comm-ef-local)
[[ -n "$COMM_SPARSE_COMM" ]] && COMM_EXTRA+=(--comm-sparse-comm) || COMM_EXTRA+=(--no-comm-sparse-comm)
COMM_EXTRA+=(--comm-qsgd-s "$COMM_QSGD_S" --comm-qsgd-bucket-size "$COMM_QSGD_BUCKET_SIZE")
COMM_EXTRA+=(--comm-topk-ratio "$COMM_TOPK_RATIO" --comm-int8-variant "$COMM_INT8_VARIANT")
COMM_EXTRA+=(--comm-onebit-col-size "$COMM_ONEBIT_COL_SIZE")
[[ -z "$COMM_ONEBIT_USE_CPP" ]] && COMM_EXTRA+=(--no-comm-onebit-use-cpp)
[[ -n "$COMM_THRESHOLD_V" ]] && COMM_EXTRA+=(--comm-threshold-v "$COMM_THRESHOLD_V")
[[ -n "$COMM_THRESHOLD_V_NEG" ]] && COMM_EXTRA+=(--comm-threshold-v-neg "$COMM_THRESHOLD_V_NEG")
[[ -n "$COMM_SIGNSGD_USE_DELTA" ]] && COMM_EXTRA+=(--comm-signsgd-use-delta)
[[ -n "$COMM_SKETCH_TWO_ROUND" ]] && COMM_EXTRA+=(--comm-sketch-two-round)
[[ -n "$COMM_NC_BIT_PACKING" ]] && COMM_EXTRA+=(--comm-nc-bit-packing) || COMM_EXTRA+=(--no-comm-nc-bit-packing)

BASE_ARGS=(
  fsdp_train.py
  --data_path "$DATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --run_name "$RUN_NAME"
  --num_epochs "$NUM_EPOCHS"
  --batch_size "$BATCH_SIZE"
  --max_length "$MAX_LENGTH"
  --dataset_max_samples "$DATASET_MAX_SAMPLES"
  --max_steps "$MAX_STEPS"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --model_size "$MODEL_SIZE"
  --comm-hook "$COMM_HOOK"
  "${COMM_EXTRA[@]}"
  --warmup_steps 0
  --no-profile
  --tensorboard
)

if [[ "$NNODES" -le 1 ]]; then
  if [[ "$NPROC" -le 1 ]]; then
    CMD=(python "${BASE_ARGS[@]}" "${PASSTHROUGH[@]}")
  else
    CMD=(torchrun --standalone --nproc_per_node="$NPROC" "${BASE_ARGS[@]}" "${PASSTHROUGH[@]}")
  fi
else
  if [[ -z "$MASTER_ADDR" ]]; then
    echo "--master_addr is required when --nnodes > 1" >&2
    exit 2
  fi
  CMD=(torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --nproc_per_node="$NPROC" "${BASE_ARGS[@]}" "${PASSTHROUGH[@]}")
fi

echo "[STEP2] Phase 1: 训练"
echo "[RUN] ${CMD[*]}"
cd "$ROOT"
"${CMD[@]}"

# ---------- Phase 2: 训练数据采集（loss/显存曲线、吞吐、可选 PPL）----------
if [[ "$NNODES" -gt 1 && "$NODE_RANK" -ne 0 ]]; then
  echo "[POST] skip training_metrics on node_rank=$NODE_RANK (only run on node_rank=0)"
else
  echo "[STEP2] Phase 2: 训练数据采集 -> ${LOG_DIR}/training_metrics.json"
  COLLECT_ARGS=(python tools/collect_training_metrics.py --log_dir "$LOG_DIR")
  [[ "$EVAL_WIKI_PPL" = "1" ]] && COLLECT_ARGS+=(--eval_wiki_5pct)
  "${COLLECT_ARGS[@]}"
fi

echo "[OUT] log_dir=$LOG_DIR"
echo "[OUT] training_metrics=$LOG_DIR/training_metrics.json"
