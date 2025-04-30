#!/bin/sh
# Usage: ./run_vllm.sh DATA_PATH SAVE_NAME
# Example: ./run_vllm.sh /path/to/cleaned_400k_reports.feather /path/to/400k_diseases_fast3.csv

# Exit script on any error (including piped commands)
set -euxo pipefail

# 1) Set the HF cache path variable
HF_CACHE="/mnt/sdh/pedro/HFCache"

# 2) Set the base port for the APIs (this will become 8000, 8001, 8002, 8003)
BASE_PORT=8700

# 3) Set the base GPU index
BASE_GPU=4

# 4) Parse arguments for data path and save name
DATA_PATH="$1"
SAVE_NAME="$2"

# (Optional) Activate conda environment
# source /mnt/sdc/pedro/anaconda3/etc/profile.d/conda.sh
# conda activate vllm2

# Launch 4 vllm instances (one per GPU, each on a different port)
TRANSFORMERS_CACHE="$HF_CACHE" HF_HOME="$HF_CACHE" CUDA_VISIBLE_DEVICES=$((BASE_GPU+0)) \
  vllm serve iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8 \
             --max-model-len 12000 \
             --dtype float16 \
             --port $((BASE_PORT+0)) \
             --gpu_memory_utilization 0.95 \
             --enforce-eager \
             > "API_$((BASE_GPU+0)).log" 2>&1 &

TRANSFORMERS_CACHE="$HF_CACHE" HF_HOME="$HF_CACHE" CUDA_VISIBLE_DEVICES=$((BASE_GPU+1)) \
  vllm serve iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8 \
             --max-model-len 12000 \
             --dtype float16 \
             --port $((BASE_PORT+1)) \
             --gpu_memory_utilization 0.95 \
             --enforce-eager \
             > "API_$((BASE_GPU+1)).log" 2>&1 &

TRANSFORMERS_CACHE="$HF_CACHE" HF_HOME="$HF_CACHE" CUDA_VISIBLE_DEVICES=$((BASE_GPU+2)) \
  vllm serve iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8 \
             --max-model-len 12000 \
             --dtype float16 \
             --port $((BASE_PORT+2)) \
             --gpu_memory_utilization 0.95 \
             --enforce-eager \
             > "API_$((BASE_GPU+2)).log" 2>&1 &

TRANSFORMERS_CACHE="$HF_CACHE" HF_HOME="$HF_CACHE" CUDA_VISIBLE_DEVICES=$((BASE_GPU+3)) \
  vllm serve iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8 \
             --max-model-len 12000 \
             --dtype float16 \
             --port $((BASE_PORT+3)) \
             --gpu_memory_utilization 0.95 \
             --enforce-eager \
             > "API_$((BASE_GPU+3)).log" 2>&1 &

# Wait until each vllm API is ready
while ! curl -s http://localhost:$((BASE_PORT+0))/v1/models; do
    echo "Waiting for API on port $((BASE_PORT+0))..."
    sleep 5
done

while ! curl -s http://localhost:$((BASE_PORT+1))/v1/models; do
    echo "Waiting for API on port $((BASE_PORT+1))..."
    sleep 5
done

while ! curl -s http://localhost:$((BASE_PORT+2))/v1/models; do
    echo "Waiting for API on port $((BASE_PORT+2))..."
    sleep 5
done

while ! curl -s http://localhost:$((BASE_PORT+3))/v1/models; do
    echo "Waiting for API on port $((BASE_PORT+3))..."
    sleep 5
done

echo "All vllm APIs are ready. Running python scripts..." >> FastDiseases.log

# Run the Python scripts (one per port), splitting the data into 4 parts
python RunRadGPT.py --port $((BASE_PORT+0)) \
                    --data_path "$DATA_PATH" \
                    --institution "UCSF" \
                    --step "diagnoses" \
                    --save_name "$SAVE_NAME" \
                    --parts 4 --part 0 >> 400k_part_0_24_fast.log 2>&1 &

python RunRadGPT.py --port $((BASE_PORT+1)) \
                    --data_path "$DATA_PATH" \
                    --institution "UCSF" \
                    --step "diagnoses" \
                    --save_name "$SAVE_NAME" \
                    --parts 4 --part 1 >> 400k_part_1_24_fast.log 2>&1 &

python RunRadGPT.py --port $((BASE_PORT+2)) \
                    --data_path "$DATA_PATH" \
                    --institution "UCSF" \
                    --step "diagnoses" \
                    --save_name "$SAVE_NAME" \
                    --parts 4 --part 2 >> 400k_part_2_24_fast.log 2>&1 &

python RunRadGPT.py --port $((BASE_PORT+3)) \
                    --data_path "$DATA_PATH" \
                    --institution "UCSF" \
                    --step "diagnoses" \
                    --save_name "$SAVE_NAME" \
                    --parts 4 --part 3 >> 400k_part_3_24_fast.log 2>&1 &

# Wait for all background jobs (vllm + python scripts) to finish
wait