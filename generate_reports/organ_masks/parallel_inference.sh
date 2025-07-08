#!/usr/bin/env bash
#
# Usage:
#   bash parallel_inference.sh --pth <path/to/dataset> --outdir <path/to/output> --checkpoint <path/to/model> --gpus <gpu0,gpu1,...>
#
# Example:
#   bash parallel_inference.sh \
#     --pth /data/dataset \
#     --outdir /data/output \
#     --checkpoint /models/model.ckpt \
#     --gpus 0,1,2,3
#
# Description:
#   This script launches one instance of PredictSubOrgansnUnet.py per specified GPU.
#   It divides the workload into num_parts equal to the number of GPUs and assigns
#   each part an ID (part_id from 0 to num_parts-1). Logs for each process are
#   saved to ./logs/part<part_id>_gpu<gpu>.log and output is also printed
#   to the terminal via tee.

set -euo pipefail

#——— Parse arguments —————————————————————————————————————————————
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pth)        pth="$2"; shift 2 ;;
    --outdir)     outdir="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --gpus)       IFS=',' read -r -a gpus <<< "$2"; shift 2 ;;
    *) echo "Usage: $0 --pth <path> --outdir <outdir> --checkpoint <ckpt> --gpus gpu0,gpu1,…"; exit 1 ;;
  esac
done

#——— Validate ————————————————————————————————————————————————
if [[ -z "${pth:-}" || -z "${outdir:-}" || -z "${checkpoint:-}" || "${#gpus[@]}" -eq 0 ]]; then
  echo "Missing required argument." >&2
  exit 1
fi

num_parts=${#gpus[@]}

#——— Prepare logs directory ——————————————————————————————————————
mkdir -p "logs"

#——— Launch one process per GPU ——————————————————————————————————
for part_id in "${!gpus[@]}"; do
  gpu=${gpus[$part_id]}
  logf="logs/part${part_id}_gpu${gpu}.log"
  echo "Part $part_id of $num_parts on GPU $gpu (log: $logf)"
  python PredictSubOrgansnUnet.py \
    --num_parts  "$num_parts" \
    --part_id    "$part_id" \
    --gpu        "$gpu" \
    --pth        "$pth" \
    --outdir     "$outdir" \
    --checkpoint "$checkpoint" \
    2>&1 | tee -a "$logf" &
done

wait
 echo "All jobs completed."