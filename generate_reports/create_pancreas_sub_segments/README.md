# Pancreas Sub-Segmentation Script

This deterministic algorithm splits a pancreas segmentation mask into head, body, and tail regions using a corresponding SMA (Superior Mesenteric Artery) segmentation. It processes NIfTI (`.nii.gz`) files, reorients/downsamples them as needed, and saves three separate segmentations (`head`, `body`, `tail`).

## Dataset Format

Assemble the dataset in this format:
```
AbdomenAtlas
├── BDMAP_A0000001
|    ├── ct.nii.gz
│    └── predictions
│          ├── liver_tumor.nii.gz
│          ├── kidney_tumor.nii.gz
│          ├── pancreas_tumor.nii.gz
│          ├── aorta.nii.gz
│          ├── gall_bladder.nii.gz
│          ├── kidney_left.nii.gz
│          ├── kidney_right.nii.gz
│          ├── liver.nii.gz
│          ├── pancreas.nii.gz
│          └──...
├── BDMAP_A0000002
|    ├── ct.nii.gz
│    └── predictions
│          ├── liver_tumor.nii.gz
│          ├── kidney_tumor.nii.gz
│          ├── pancreas_tumor.nii.gz
│          ├── aorta.nii.gz
│          ├── gall_bladder.nii.gz
│          ├── kidney_left.nii.gz
│          ├── kidney_right.nii.gz
│          ├── liver.nii.gz
│          ├── pancreas.nii.gz
│          └──...
...
```

## 1. Installation

<details>
<summary style="margin-left: 25px;">[Optional] Install Anaconda on Linux</summary>
<div style="margin-left: 25px;">
    
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p ./anaconda3
./anaconda3/bin/conda init
source ~/.bashrc
```
</div>
</details>

```bash
conda create -n pancreas python=3.12 -y
conda activate pancreas
conda install -y ipykernel
conda install -y pip
pip install -r requirements.txt
```

## 2. Usage

```bash
python SegmentPancreas.py \
  --source_dir /path/to/source \
  --sma_dir /path/to/sma \
  --destination_dir /path/to/destination \
  --restart \
  --parts 1 \
  --current_part 0 \
  --num_processes 1

  •	--source_dir: Directory containing subfolders/cases with pancreas.nii.gz.
	•	--sma_dir: Directory containing subfolders/cases with superior_mesenteric_artery.nii.gz.
	•	--destination_dir: Path to where you want the output masks saved.
	•	--restart: If included, reprocess even if outputs already exist.
	•	--parts / --current_part: Useful for splitting the workload if you have many cases and want to run them in chunks.
	•	--num_processes: Number of processes to run in parallel.
