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
  --dataset_max_samples N     加载样本上限（默认 200）
  --batch_size N              batch size（默认 2）
  --max_length N              序列长度（默认 128）

产物：
  <output_dir>/logs/<run_name>/profiler/summary_rank0.json
  <output_dir>/logs/<run_name>/profiler/comm_op_summary_rank0.csv
  <output_dir>/logs/<run_name>/profiler/*.pt.trace.json

示例：
  ./scripts/step1_profile.sh --data_path datasets/wikipedia_en_1k.json --nproc 1
  ./scripts/step1_profile.sh --data_path datasets/wikipedia_en_10mb.json --nproc 2 --max_steps 50
EOF
}

DATA_PATH=""
OUTPUT_DIR="out"
RUN_NAME=""
NPROC=1
MAX_STEPS=200
DATASET_MAX_SAMPLES=200
BATCH_SIZE=2
MAX_LENGTH=128

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
    --max_steps)
      MAX_STEPS="$2"; shift 2 ;;
    --dataset_max_samples)
      DATASET_MAX_SAMPLES="$2"; shift 2 ;;
    --batch_size)
      BATCH_SIZE="$2"; shift 2 ;;
    --max_length)
      MAX_LENGTH="$2"; shift 2 ;;
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
  --profile
  --profile_step_time
)

if [[ "$NPROC" -le 1 ]]; then
  CMD=(python "${BASE_ARGS[@]}" "${PASSTHROUGH[@]}")
else
  CMD=(torchrun --standalone --nproc_per_node="$NPROC" "${BASE_ARGS[@]}" "${PASSTHROUGH[@]}")
fi

echo "[RUN] ${CMD[*]}"
"${CMD[@]}"

echo "[OUT] profiler_dir=${OUTPUT_DIR}/logs/${RUN_NAME}/profiler"
echo "[OUT] summary=${OUTPUT_DIR}/logs/${RUN_NAME}/profiler/summary_rank0.json"
echo "[OUT] comm_csv=${OUTPUT_DIR}/logs/${RUN_NAME}/profiler/comm_op_summary_rank0.csv"
