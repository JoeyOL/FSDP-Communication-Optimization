#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Step1: 一键启动耗时取证（torch.profiler + step wall time + overlap）

用法：
  ./scripts/step1_profile.sh --data_path /path/to/wiki.json [options] [-- <passthrough args to fsdp_train.py>]

常用参数：
  --output_dir DIR            输出目录（默认 out）
  --run_name NAME             运行名（默认 step1-<timestamp>）
  --nproc N                   进程/卡数（默认 1；>1 使用 torchrun）
  --max_steps N               短跑步数（默认 20；建议 >= 20）

多机参数（torchrun）：
  --nnodes N                  机器数（默认 1；>1 启用多机 torchrun）
  --node_rank R               本机 rank（默认 0；两机时分别为 0/1）
  --master_addr HOST          主节点地址（nnodes>1 时必填）
  --master_port PORT          主节点端口（默认 29500）
  --dataset_max_samples N     加载样本上限（默认 200）
  --batch_size N              batch size（默认 2）
  --max_length N              序列长度（默认 128）
  --comm_hook NAME            通信压缩 hook（默认 none）
  --comm_error_feedback       启用误差反馈（EF）
  --no_comm_sparse_comm       关闭稀疏通信（indices+values）
  --comm_qsgd_s N             QSGD 水平数 s（默认 4）
  --comm_qsgd_bucket_size N   QSGD 桶大小（0=整向量，如 512）
  --comm_topk_ratio R         Top-k/Random-k/Threshold 稀疏比例（默认 0.01）
  --comm_threshold_v V        Threshold-v 正侧阈值（0=用 ratio 分位数）
  --comm_threshold_v_neg V    Threshold-v 负侧阈值（缺省与 v 对称）
  --comm_int8_variant V       INT8 变体：linear 或 dynamic_tree
  --comm_onebit_col_size N    1-bit Seide 列大小（默认 256）
  --comm_signsgd_use_delta    SignSGD 仅传方向、步长用 lr
  --comm_sketch_two_round     Sketched-SGD 两轮 + HEAVYMIX
  其他压缩参数可通过 -- 透传，例如: -- --comm-error-feedback

产物：
  <output_dir>/logs/<run_name>/profiler/*.pt.trace.json
  <output_dir>/logs/<run_name>/profiler/summary_rank0.json （离线脚本生成）
  <output_dir>/logs/<run_name>/profiler/comm_op_summary_rank0.csv （离线脚本生成）

示例：
  # 单机 1 卡
  ./scripts/step1_profile.sh --data_path datasets/wikipedia_en_300mb.json --nproc 1

  # 单机 2 卡
  ./scripts/step1_profile.sh --data_path datasets/wikipedia_en_300mb.json --nproc 2 --max_steps 50

  # 两机两卡（两机各 1 卡，总 2 卡）
  # 机器0（node_rank=0，同时也是 master）：
  ./scripts/step1_profile.sh --data_path datasets/wikipedia_en_300mb.json     --nnodes 2 --node_rank 0 --master_addr 10.0.0.1 --master_port 29500     --nproc 1 --max_steps 50 --run_name step1-2node-2gpu

  # 机器1（node_rank=1）：
  ./scripts/step1_profile.sh --data_path datasets/wikipedia_en_300mb.json     --nnodes 2 --node_rank 1 --master_addr 10.0.0.1 --master_port 29500     --nproc 1 --max_steps 50 --run_name step1-2node-2gpu

  # 注意：离线统计（summary/csv）只在 node_rank=0 上生成。
EOF
}

DATA_PATH=""
OUTPUT_DIR="/root/llama-7b/fsdp_output"
RUN_NAME=""
NPROC=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=""
MASTER_PORT=29500
MAX_STEPS=55
DATASET_MAX_SAMPLES=0
BATCH_SIZE=1
MAX_LENGTH=1024
GRADIENT_ACCUMULATION_STEPS=1
MODEL_SIZE="medium"
COMM_HOOK="none"
COMM_ERROR_FEEDBACK=""
COMM_SPARSE_COMM="yes"
COMM_QSGD_S=4
COMM_QSGD_BUCKET_SIZE=0
COMM_TOPK_RATIO=0.01
COMM_THRESHOLD_V=""
COMM_THRESHOLD_V_NEG=""
COMM_INT8_VARIANT="linear"
COMM_ONEBIT_COL_SIZE=256
COMM_SIGNSGD_USE_DELTA=""
COMM_SKETCH_TWO_ROUND=""

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
    --comm_signsgd_use_delta)
      COMM_SIGNSGD_USE_DELTA=1; shift 1 ;;
    --comm_sketch_two_round)
      COMM_SKETCH_TWO_ROUND=1; shift 1 ;;
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
  RUN_NAME="step1-$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "$OUTPUT_DIR"


LOG_DIR="${OUTPUT_DIR}/logs/${RUN_NAME}"
mkdir -p "$LOG_DIR"

cat > "$LOG_DIR/config.json" << EOF
{
  "run_name": "${RUN_NAME}",
  "data_path": "${DATA_PATH}",
  "output_dir": "${OUTPUT_DIR}",
  "num_epochs": 1,
  "batch_size": ${BATCH_SIZE},
  "max_length": ${MAX_LENGTH},
  "dataset_max_samples": ${DATASET_MAX_SAMPLES},
  "max_steps": ${MAX_STEPS},
  "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS},
  "model_size": "${MODEL_SIZE}",
  "warmup_steps": 0,
  "nproc": ${NPROC},
  "nnodes": ${NNODES},
  "comm_hook": "${COMM_HOOK}",
  "comm_error_feedback": $([ -n "$COMM_ERROR_FEEDBACK" ] && echo true || echo false),
  "comm_sparse_comm": $([ -n "$COMM_SPARSE_COMM" ] && echo true || echo false),
  "comm_qsgd_s": ${COMM_QSGD_S},
  "comm_qsgd_bucket_size": ${COMM_QSGD_BUCKET_SIZE},
  "comm_topk_ratio": ${COMM_TOPK_RATIO},
  "comm_int8_variant": "${COMM_INT8_VARIANT}",
  "comm_onebit_col_size": ${COMM_ONEBIT_COL_SIZE},
  "timestamp": "$(date -Iseconds)"
}
EOF
echo "[CONFIG] Saved config to ${LOG_DIR}/config.json"

# 压缩相关参数（透传到 fsdp_train.py）
COMM_EXTRA=()
[[ -n "$COMM_ERROR_FEEDBACK" ]] && COMM_EXTRA+=(--comm-error-feedback)
[[ -n "$COMM_SPARSE_COMM" ]] && COMM_EXTRA+=(--comm-sparse-comm) || COMM_EXTRA+=(--no-comm-sparse-comm)
COMM_EXTRA+=(--comm-qsgd-s "$COMM_QSGD_S" --comm-qsgd-bucket-size "$COMM_QSGD_BUCKET_SIZE")
COMM_EXTRA+=(--comm-topk-ratio "$COMM_TOPK_RATIO" --comm-int8-variant "$COMM_INT8_VARIANT")
COMM_EXTRA+=(--comm-onebit-col-size "$COMM_ONEBIT_COL_SIZE")
[[ -n "$COMM_THRESHOLD_V" ]] && COMM_EXTRA+=(--comm-threshold-v "$COMM_THRESHOLD_V")
[[ -n "$COMM_THRESHOLD_V_NEG" ]] && COMM_EXTRA+=(--comm-threshold-v-neg "$COMM_THRESHOLD_V_NEG")
[[ -n "$COMM_SIGNSGD_USE_DELTA" ]] && COMM_EXTRA+=(--comm-signsgd-use-delta)
[[ -n "$COMM_SKETCH_TWO_ROUND" ]] && COMM_EXTRA+=(--comm-sketch-two-round)

# NOTE: torchrun expects the training script/module directly (e.g. fsdp_train.py),
# not a nested "python fsdp_train.py" command.
BASE_ARGS=(
  fsdp_train.py
  --data_path "$DATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --run_name "$RUN_NAME"
  --num_epochs 1
  --batch_size "$BATCH_SIZE"
  --max_length "$MAX_LENGTH"
  --dataset_max_samples "$DATASET_MAX_SAMPLES"
  --max_steps "$MAX_STEPS"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --model_size "$MODEL_SIZE"
  --comm-hook "$COMM_HOOK"
  "${COMM_EXTRA[@]}"
  --warmup_steps 0
  --profile
  --profile_step_time
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

echo "[RUN] ${CMD[*]}"
"${CMD[@]}"

profiler_dir="${OUTPUT_DIR}/logs/${RUN_NAME}/profiler"

if [[ "$NNODES" -gt 1 && "$NODE_RANK" -ne 0 ]]; then
  echo "[POST] skip trace stats on node_rank=$NODE_RANK (only run on node_rank=0)"
  trace_file=""
else
  trace_file=""
  if [[ -d "$profiler_dir" ]]; then
    trace_file=$(ls -t "$profiler_dir"/*.pt.trace.json 2>/dev/null | head -n 1 || true)
  fi

  if [[ -z "$trace_file" ]]; then
    echo "[WARN] no trace file found in $profiler_dir"
  else
    echo "[POST] compute trace stats: $trace_file"
    python tools/compute_trace_stats.py "$trace_file" --out_dir "$profiler_dir" --compat_rank0_names --csv_all_kernels
  fi
fi

echo "[OUT] profiler_dir=$profiler_dir"
echo "[OUT] trace=$trace_file"
echo "[OUT] summary=$profiler_dir/summary_rank0.json"
echo "[OUT] comm_csv=$profiler_dir/comm_op_summary_rank0.csv"
