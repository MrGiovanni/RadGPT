#!/usr/bin/env bash
# Usage:
#   bash LaunchMultiGPUFlex.sh [DATA_PATH] [SAVE_NAME] [STEP] [LLM_SIZE] [NUM_GPUS] [INST_PER_GPU] [GPU_PER_INST] [BASE_GPU] [TOP_GPU_USAGE] [HF_CACHE]
#
# Examples:
#   1) Use defaults except for data/save paths:
#      bash LaunchMultiGPUFlex.sh /my_data.feather /my_save.csv
#
#   2) Override everything:
#      bash LaunchMultiGPUFlex.sh \
#         reports_concat.feather \
#         reports_concat_size_and_type.csv \
#         "type and size multi-organ" \
#         large \
#         4 \
#         1 \
#         2 \
#         0 \
#         0.75 \
#         /mnt/sdh/pedro/HFCache
#
#   3) Only override HF_CACHE:
#      bash LaunchMultiGPUFlex.sh /my_data.feather /my_save.csv "" "" "" "" "" "" /mnt/sdh/pedro/HFCache
#
# Notes:
#   - If an argument is "", its default is used.
#   - The 9th argument (TOP_GPU_USAGE) defaults to 0.95.
#   - GPU_PER_INST: if >0, ignore INST_PER_GPU and assign that many GPUs per VLLM instance.
#   - Raise an error if both GPU_PER_INST and INST_PER_GPU are >1.
#
# Before running, activate your conda environment:
#   source /path/to/anaconda3/etc/profile.d/conda.sh
#   conda activate vllm2

##############################################################################
# 1) Exit on any error
##############################################################################
set -euxo pipefail

##############################################################################
# 2) Parse arguments with defaults
##############################################################################
DATA_PATH="${1:-/path/to/data.feather}"   # 1) Path to your dataset
SAVE_NAME="${2:-/path/to/save.csv}"       # 2) Path/filename to save results
STEP="${3:-diagnoses}"                    # 3) A string for the Python code's --step
LLM_SIZE="${4:-small}"                    # 4) "small" or "large" model
NUM_GPUS="${5:-8}"                        # 5) Number of GPUs to use
INST_PER_GPU="${6:-1}"                    # 6) Number of VLLM instances per GPU
GPU_PER_INST="${7:-0}"                    # 7) Number of GPUs per VLLM instance (if >0, ignore INST_PER_GPU)
BASE_GPU="${8:-0}"                        # 8) Base GPU index
TOP_GPU_USAGE="${9:-0.95}"                # 9) Top fraction of GPU memory usage
HF_CACHE="${10:-./HFCache}"                # 10) HF cache directory

##############################################################################
# 3) Check for conflicts and compute total instances
##############################################################################
if [ "$GPU_PER_INST" -gt 1 ] && [ "$INST_PER_GPU" -gt 1 ]; then
  echo "Error: cannot have both GPU_PER_INST>1 and INST_PER_GPU>1."
  exit 1
fi

if [ "$GPU_PER_INST" -gt 0 ]; then
  # Using multiple GPUs per instance: ignore INST_PER_GPU.
  remainder=$(( NUM_GPUS % GPU_PER_INST ))
  if [ "$remainder" -ne 0 ]; then
    echo "Error: NUM_GPUS ($NUM_GPUS) is not divisible by GPU_PER_INST ($GPU_PER_INST)."
    exit 1
  fi
  TOTAL_INSTANCES=$(( NUM_GPUS / GPU_PER_INST ))
  echo "GPU_PER_INST=$GPU_PER_INST => total VLLM instances=$TOTAL_INSTANCES (ignoring INST_PER_GPU)"
else
  TOTAL_INSTANCES=$(( NUM_GPUS * INST_PER_GPU ))
  echo "INST_PER_GPU=$INST_PER_GPU => total VLLM instances=$TOTAL_INSTANCES"
fi

##############################################################################
# 4) Randomly pick a BASE_PORT, ensuring subsequent ports are free
##############################################################################
randomize_base_port() {
  local start_range=1000
  local end_range=9999
  while true; do
    try_port=$((start_range + RANDOM % (end_range - start_range + 1)))
    all_free=true
    for offset in $(seq 0 $((TOTAL_INSTANCES - 1))); do
      p=$((try_port + offset))
      if lsof -i :"$p" -sTCP:LISTEN > /dev/null 2>&1; then
        all_free=false
        break
      fi
    done
    if $all_free; then
      echo "$try_port"
      return 0
    fi
  done
}
BASE_PORT=$(randomize_base_port)
echo "Selected BASE_PORT=$BASE_PORT"

##############################################################################
# 5) Decide which model/arguments to use (small vs. large)
##############################################################################
case "$LLM_SIZE" in
  small)
    MODEL="iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8"
    MODEL_OPTS="--max-model-len 12000 --dtype float16"
    ;;
  large)
    MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    MODEL_OPTS="--dtype half --max-model-len 60000 --tensor-parallel-size 1"
    ;;
  *)
    echo "Unknown LLM_SIZE: '$LLM_SIZE'. Must be 'small' or 'large'."
    exit 1
    ;;
esac

##############################################################################
# 6) Adjust tensor-parallel-size if GPU_PER_INST > 0
##############################################################################
if [ "$GPU_PER_INST" -gt 0 ]; then
  MODEL_OPTS="${MODEL_OPTS/--tensor-parallel-size 1/}"
  MODEL_OPTS="$MODEL_OPTS --tensor-parallel-size $GPU_PER_INST"
fi

##############################################################################
# 7) GPU Memory Utilization
##############################################################################
calc_gpu_mem_per_instance() {
  local usage="$1"
  local inst="$2"
  if [ "$inst" -le 1 ]; then
    echo "$usage"
  else
    python -c "print(round($usage / $inst, 2))"
  fi
}

if [ "$GPU_PER_INST" -gt 0 ]; then
  GPU_MEM="$TOP_GPU_USAGE"
else
  case "$INST_PER_GPU" in
    1)
      GPU_MEM="$TOP_GPU_USAGE"
      ;;
    2)
      GPU_MEM="$(python -c "print(round($TOP_GPU_USAGE / 2, 2))")"
      ;;
    3)
      GPU_MEM="$(python -c "print(round($TOP_GPU_USAGE / 3, 2))")"
      ;;
    *)
      GPU_MEM="$(python -c "print(round($TOP_GPU_USAGE / $INST_PER_GPU, 2))")"
      ;;
  esac
fi

##############################################################################
# 8) Launch VLLM Instances
##############################################################################
#clean logs
rm -f API_*.log
rm -f LLM_part_*.log

instance_id=0
if [ "$GPU_PER_INST" -gt 0 ]; then
  # One instance per group of GPU_PER_INST GPUs
  for i in $(seq 0 $((TOTAL_INSTANCES - 1))); do
    start_gpu=$((BASE_GPU + i * GPU_PER_INST))
    GPU_LIST=""
    for g in $(seq 0 $((GPU_PER_INST - 1))); do
      current_gpu=$((start_gpu + g))
      if [ -z "$GPU_LIST" ]; then
        GPU_LIST="$current_gpu"
      else
        GPU_LIST="$GPU_LIST,$current_gpu"
      fi
    done
    PORT=$((BASE_PORT + i))
    echo "Launching VLLM instance #$i on GPUs $GPU_LIST (port $PORT)"
    echo "Memory per instance: $GPU_MEM"
    echo "HF_CACHE: $HF_CACHE"
    TRANSFORMERS_CACHE="$HF_CACHE" HF_HOME="$HF_CACHE" CUDA_VISIBLE_DEVICES="$GPU_LIST" \
      vllm serve "$MODEL" \
                 $MODEL_OPTS \
                 --port "$PORT" \
                 --gpu_memory_utilization "$GPU_MEM" \
                 --enforce-eager \
                 > "API_MULTI_GPU_${GPU_LIST}_INS${i}.log" 2>&1 &
  done
else
  # Old approach: multiple instances per GPU
  instance_id=0
  for gpu_index in $(seq 0 $((NUM_GPUS - 1))); do
    for ins_index in $(seq 0 $((INST_PER_GPU - 1))); do
      PORT=$((BASE_PORT + instance_id))
      GPU=$((BASE_GPU + gpu_index))
      GPU_MEM="$(calc_gpu_mem_per_instance "$TOP_GPU_USAGE" "$INST_PER_GPU")"
      echo "Launching VLLM instance #$instance_id on GPU $GPU (port $PORT)"
      echo "Memory per instance: $GPU_MEM"
      echo "HF_CACHE: $HF_CACHE"
      TRANSFORMERS_CACHE="$HF_CACHE" HF_HOME="$HF_CACHE" CUDA_VISIBLE_DEVICES="$GPU" \
        vllm serve "$MODEL" \
                   $MODEL_OPTS \
                   --port "$PORT" \
                   --gpu_memory_utilization "$GPU_MEM" \
                   --enforce-eager \
                   > "API_GPU${GPU}_INS${ins_index}.log" 2>&1 &
      instance_id=$((instance_id + 1))
    done
  done
fi

##############################################################################
# 9) Wait for All VLLM Instances to be Ready
##############################################################################
for i in $(seq 0 $((TOTAL_INSTANCES - 1))); do
  PORT=$((BASE_PORT + i))
  while ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null; do
    echo "Waiting for API on port $PORT..."
    sleep 5
  done
done

echo "All vllm APIs are ready. Running python scripts..." >> FastDiseases.log

##############################################################################
# 10) Launch Python Jobs (One per VLLM Instance)
##############################################################################
for i in $(seq 0 $((TOTAL_INSTANCES - 1))); do
  PORT=$((BASE_PORT + i))
  echo "Launching Python script for instance #$i on port $PORT"
  python RunRadGPT.py \
    --port "$PORT" \
    --data_path "$DATA_PATH" \
    --institution "UCSF" \
    --step "$STEP" \
    --save_name "$SAVE_NAME" \
    --parts "$TOTAL_INSTANCES" \
    --part "$i" \
    >> "LLM_part_${i}.log" 2>&1 &
done

##############################################################################
# 11) Wait for All Background Jobs
##############################################################################
wait