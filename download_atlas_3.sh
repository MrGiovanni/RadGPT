#!/usr/bin/env bash
set -euo pipefail

# Where to put everything
ROOT="data"
IM_DIR="${ROOT}/image_only"
MSK_DIR="${ROOT}/mask_only"
IDS_DIR="${ROOT}/TrainTestIDS"

# Hugging Face "resolve" base
BASE="https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini/resolve/main"

mkdir -p "$ROOT" "$IM_DIR" "$MSK_DIR" "$IDS_DIR"

echo "Downloading metadata CSV..."
wget --show-progress -c -O "${ROOT}/AbdomenAtlas3.0MiniWithMeta.csv" \
  "${BASE}/AbdomenAtlas3.0MiniWithMeta.csv?download=true"

echo "Downloading Train/Test ID CSVs..."
for f in IID_train.csv IID_test.csv OOD_train.csv OOD_test.csv; do
  wget --show-progress -c -O "${IDS_DIR}/${f}" \
    "${BASE}/TrainTestIDS/${f}?download=true"
done

# Helper to fetch and extract one tarball
fetch_and_extract () {
  local file="$1" dest="$2"
  local url="${BASE}/${file}?download=true"

  echo "Downloading ${file} ..."
  wget --show-progress -c -O "${ROOT}/${file}" "${url}"

  echo "Extracting ${file} -> ${dest}"
  tar -xzf "${ROOT}/${file}" -C "${dest}"
  rm -f "${ROOT}/${file}"
}

# The dataset uses contiguous BDMAP ranges in chunks of 232 cases:
# e.g., 00000001_00000232, 00000233_00000464, ..., up to 00009262.
START=1
END_MAX=9262
STEP=232

echo "Downloading image_only tarballs..."
i=$START
while [[ $i -le $END_MAX ]]; do
  start=$(printf "%08d" "$i")
  end_raw=$(( i + STEP - 1 ))
  # clamp the last chunk to END_MAX
  if [[ $end_raw -gt $END_MAX ]]; then end_raw=$END_MAX; fi
  end=$(printf "%08d" "$end_raw")

  file="image_only/AbdomenAtlas3_images_BDMAP_BDMAP_${start}_BDMAP_${end}.tar.gz"
  fetch_and_extract "$file" "$IM_DIR"

  i=$(( end_raw + 1 ))
done

echo "Downloading mask_only tarballs..."
i=$START
while [[ $i -le $END_MAX ]]; do
  start=$(printf "%08d" "$i")
  end_raw=$(( i + STEP - 1 ))
  if [[ $end_raw -gt $END_MAX ]]; then end_raw=$END_MAX; fi
  end=$(printf "%08d" "$end_raw")

  file="mask_only/AbdomenAtlas3_masks_BDMAP_BDMAP_${start}_BDMAP_${end}.tar.gz"
  fetch_and_extract "$file" "$MSK_DIR"

  i=$(( end_raw + 1 ))
done

echo "All done."