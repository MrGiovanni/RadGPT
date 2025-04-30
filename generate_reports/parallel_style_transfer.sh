#!/bin/bash

# -----------------------------------------------------------------------------
# Automated Script to Launch VLLM Serve Instances and Corresponding Python Scripts
# -----------------------------------------------------------------------------
#
# **Inputs:**
#   1. GPUS: Comma-separated list of GPU IDs (e.g., "0,1,2,3,4,5,6,7")
#   2. NUM_INSTANCES: Number of VLLM serve instances to launch (e.g., 2)
#   3. STRUCTURED_REPORTS: Path to the structured reports CSV file
#   4. NARRATIVE_REPORTS: Path to the narrative reports CSV file
#   5. OUTPUT: Path to the output CSV file
#   6. CACHE_DIR (Optional): Path to the cache directory (default: "../HFCache")
#
# **Usage:**
#   bash script_name.sh "0,1,2,3,4,5,6,7" 2 /path/to/structured_reports.csv /path/to/narrative_reports.csv /path/to/output.csv /custom/cache/dir
#
# -----------------------------------------------------------------------------

# Exit immediately if a command exits with a non-zero status
set -e
export NCCL_P2P_DISABLE=1

# ----------------------------
# Function Definitions
# ----------------------------

# Function to display usage instructions
usage() {
    echo "Usage: $0 <gpus> <num_instances> <structured_reports> <narrative_reports> <output> [cache_dir]"
    echo "  <gpus>: Comma-separated list of GPU IDs (e.g., \"0,1,2,3\")"
    echo "  <num_instances>: Number of VLLM serve instances to launch (e.g., 2)"
    echo "  <structured_reports>: Path to the structured reports CSV file"
    echo "  <narrative_reports>: Path to the narrative reports CSV file"
    echo "  <output>: Path to the output CSV file"
    echo "  [cache_dir]: (Optional) Path to the cache directory (default: ../HFCache)"
    exit 1
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check API readiness
check_api_ready() {
    local port=$1
    while ! curl -s "http://localhost:$port/v1/models" > /dev/null; do
        sleep 5
    done
}

# Function to join array elements with a delimiter
join_by() {
    local IFS="$1"
    shift
    echo "$*"
}

# ----------------------------
# Input Validation
# ----------------------------

# Assign input arguments to variables
GPUS=$1                   # e.g., "0,1,2,3,4,5,6,7"
NUM_INSTANCES=$2          # e.g., 2
STRUCTURED_REPORTS=$3     # Path to structured reports file
NARRATIVE_REPORTS=$4      # Path to narrative reports file
OUTPUT=$5                 # Path to output file
CACHE_DIR=${6:-../HFCache} # Path to TRANSFORMERS_CACHE and HF_HOME (default: ../HFCache)

# Check if required arguments are provided
if [ -z "$GPUS" ] || [ -z "$NUM_INSTANCES" ] || [ -z "$STRUCTURED_REPORTS" ] || [ -z "$NARRATIVE_REPORTS" ] || [ -z "$OUTPUT" ]; then
    echo "Error: Missing required arguments."
    usage
fi

# Check if vllm command exists
if ! command_exists vllm; then
    echo "Error: 'vllm' command not found. Please ensure VLLM is installed and accessible in PATH."
    exit 1
fi

# Check if Python is available
if ! command_exists python3; then
    echo "Error: 'python3' command not found. Please ensure Python 3 is installed and accessible in PATH."
    exit 1
fi

# ----------------------------
# GPU Grouping and Validation
# ----------------------------

# Split GPUs into an array
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
TOTAL_GPUS=${#GPU_ARRAY[@]}
GPUS_PER_INSTANCE=$((TOTAL_GPUS / NUM_INSTANCES))

# Validate that GPUs can be evenly split across instances
if [ $((TOTAL_GPUS % NUM_INSTANCES)) -ne 0 ]; then
    echo "Error: Number of GPUs ($TOTAL_GPUS) cannot be evenly split across $NUM_INSTANCES instances."
    exit 1
fi

# ----------------------------
# Port Assignment
# ----------------------------

# Generate a random PORT_BASE between 8000 and 8900
PORT_BASE=$((8000 + RANDOM % 901))
echo "Selected PORT_BASE: $PORT_BASE"

# ----------------------------
# Launch VLLM Serve Instances
# ----------------------------

VLLM_PIDS=()

echo "Launching VLLM serve instances..."

for ((i = 0; i < NUM_INSTANCES; i++)); do
    # Calculate the starting index for GPU assignment
    START_INDEX=$((i * GPUS_PER_INSTANCE))
    
    # Extract the GPU group for this instance and join with commas
    GPU_GROUP=$(join_by ',' "${GPU_ARRAY[@]:START_INDEX:GPUS_PER_INSTANCE}")
    
    PORT=$((PORT_BASE + i))
    LOG_FILE="API${i}.log"
    
    echo "Instance $i: GPUs=$GPU_GROUP, Port=$PORT, Log=$LOG_FILE"
    
    # Launch vllm serve with the assigned GPUs and port
    TRANSFORMERS_CACHE="$CACHE_DIR" HF_HOME="$CACHE_DIR" CUDA_VISIBLE_DEVICES="$GPU_GROUP" \
    vllm serve "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" \
        --dtype=half \
        --tensor-parallel-size="$GPUS_PER_INSTANCE" \
        --gpu_memory_utilization=0.9 \
        --port="$PORT" \
        --max_model_len=100000 \
        --enforce-eager > "$LOG_FILE" 2>&1 &
    
    # Save the PID of the vllm serve process
    VLLM_PIDS+=($!)
    
    echo "Launched VLLM serve instance $i with PID ${VLLM_PIDS[$i]}"
done

# ----------------------------
# Wait for APIs to be Ready
# ----------------------------

echo "Waiting for all VLLM APIs to be ready, this can take a few minutes"

for ((i = 0; i < NUM_INSTANCES; i++)); do
    PORT=$((PORT_BASE + i))
    LOG_FILE="API${i}.log"
    
    echo "Checking API on port $PORT (log: $LOG_FILE)..."
    
    # Print new lines in the log file as they appear
    tail --lines=0 -f "$LOG_FILE" &
    TAIL_PID=$!
    
    # Wait for the API to be ready
    check_api_ready "$PORT"
    
    # Kill the tail process after the API is ready
    kill "$TAIL_PID" 2>/dev/null
    
    echo "API on port $PORT is ready."
done

echo "All VLLM APIs are ready."

# ----------------------------
# Launch Python Scripts
# ----------------------------

PYTHON_PIDS=()

echo "Launching Python scripts..."

for ((i = 0; i < NUM_INSTANCES; i++)); do
    PORT=$((PORT_BASE + i))
    LOG_FILE="StyleTransferPart${i}.log"
    
    echo "Running StyleTransferAA.py for part $i on port $PORT, Log=$LOG_FILE"
    
    python3 StyleTransferAA.py \
        --port="$PORT" \
        --structured_reports="$STRUCTURED_REPORTS" \
        --narrative_reports="$NARRATIVE_REPORTS" \
        --output="$OUTPUT" \
        --parts="$NUM_INSTANCES" \
        --current_part="$i" >> "$LOG_FILE" 2>&1 &
    
    PYTHON_PIDS+=($!)
    
    echo "Launched Python script for part $i with PID ${PYTHON_PIDS[$i]}"
done

# ----------------------------
# Wait for Python Scripts to Finish
# ----------------------------

echo "Waiting for all Python scripts to finish..."

for ((i = 0; i < NUM_INSTANCES; i++)); do
    PID=${PYTHON_PIDS[$i]}
    echo "Waiting for Python script with PID $PID..."
    wait "$PID"
    echo "Python script with PID $PID has finished."
done

echo "All Python scripts have completed."

# ----------------------------
# Terminate VLLM Serve Instances
# ----------------------------

echo "Terminating all VLLM serve instances..."

for ((i = 0; i < NUM_INSTANCES; i++)); do
    PID=${VLLM_PIDS[$i]}
    echo "Terminating VLLM serve instance $i with PID $PID..."
    kill "$PID" || echo "Failed to terminate VLLM serve instance $i with PID $PID."
done

echo "All VLLM serve instances have been terminated."

# -----------------------------------------------------------------------------
# End of Script
# -----------------------------------------------------------------------------
