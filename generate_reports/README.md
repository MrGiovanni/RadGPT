# Generate Reports from segmentation

### Dataset Format

Assemble the dataset in this format:

<details>
<summary style="margin-left: 25px;">Dataset format.</summary>
<div style="margin-left: 25px;">

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
</div>
</details>

Place the folder AbdomenAtlas inside this folder (RadGPTReportGeneration). You may use your own dataset, just rename and organize it as above.

## Generate structured reports from segmentation
RadGPT uses deterministic algorithms to extract information from CTs and segmentation masks (tumor size, attenuation, location, interaction with blood vessels and organs, organ size, steatosis,...), and create structured radiology reports.

```bash
python3 CreateAAReports.py --th 10 --csv_file /path/to/output/file.csv --num_workers 10 --dataset AA
```

## Generate narrative reports (style adaptation w/ LLM)
RadGPT adapts the style of the structured reports, mimicking the style of human-made narrative reports. It uses LLMs and in-context learning. This step requires examples of human-made reports, we provide a few in free_text_reports.csv. Please add more reports in the file to make style adaptation effective (you can use AbdomenAtlas 3.0 narrative reports here).
<details>
<summary style="margin-left: 25px;">[Optional] Match the writing style of any hospital.</summary>
<div style="margin-left: 25px;">
    
If you substitute free_text_reports.csv with the reports from any institution, RadGPT will create narrative reports in the style of that institution. You will need to include in the CSV which are the tumor types found in each of these reports. You can extract this information using LLMs (see [evaluate_reports/README.md](evaluate_reports/README.md)).
</div>
</details>


```bash
bash parallel_style_transfer.sh "0,1,2,3,4,5,6,7" 2 /path/to/structured_reports.csv free_text_reports.csv /path/to/output.csv
```

Arguments:
```
1. GPUS: Comma-separated list of GPU IDs (e.g., "0,1,2,3,4,5,6,7" for 8 gpus)
2. NUM_INSTANCES: Number of LLMs you will create. Each one takes about 70GB. Calculate the total memory of all gpus in parameter 1, and check how many LLMs they fit. For 8 gpus of 25 GB each, we have a total of 200GB and can fit 2 LLMs (2*70=140). Thus, we set NUM_INSTANCES to 2 in this case.
3. STRUCTURED_REPORTS: Path to the structured reports CSV file
4. HUMAN_MADE_REPORTS: Path to the human-made reports CSV file
5. OUTPUT: Path to the output CSV file
```


## Generate Enhanced Human Reports (w/ LLM)
RadGPT can fuse the structured/narrative reports it creates with human-made reports or notes. This combines, in a single enhanced human report, the broad diagnostic spectrum of human-made reports, with the precise, quantitative and comprehensive tumor information in the RadGPT structured and narrative reports.


1- Deploy Llama API (takes about 10 minutes)

About 70GB of VRAM should be enough for this model. Select the number of GPUs below according to this requirement (e.g., we used 4 x 24GB GPUs below). To modify the number of GPUs, change CUDA_VISIBLE_DEVICES and tensor-parallel-size, the number must be powers of 2 (1,2,4,8,...)
```bash
export NCCL_P2P_DISABLE=1
TRANSFORMERS_CACHE=../HFCache HF_HOME=../HFCache CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --dtype=half --tensor-parallel-size 4 --gpu_memory_utilization 0.9 --port 8000 --max_model_len 120000 --enforce-eager > API.log 2>&1 &
# Check if the API is up
while ! curl -s http://localhost:8000/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done
```

2- Fuse reports

```bash
python3 fuse.py --port 8000 --reports /path/to/narrative/reports.csv --radiology_notes AbdomenAtlasRadiologistsNotes.csv --output fusion_reports.csv
```

3- Stop the LLM API

```bash
pkill -f "vllm serve"
```
