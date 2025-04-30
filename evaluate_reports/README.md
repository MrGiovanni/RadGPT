Code to extract quantitative information from radiology reports, such as presence/absence of liver, pancreas and kidney tumor, using a Large Language Model (LLM).

## Installation

Skip installation if you already installed the conda environment.

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
git clone https://github.com/PedroRASB/RadGPT
cd RadGPT/LabelerLLM
conda create -n vllm python=3.12 -y
conda activate vllm
conda install -y ipykernel
conda install -y pip
pip install vllm==0.6.1.post2
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
pip install -r requirements.txt
mkdir HFCache
```


## Use the LLM to analyze radiology reports for cancer


0- Organize the data

Organize the data in a CSV, like in the example combined_data.csv, using the same headers for 'Anon Acc #' and 'Anon Report Text'. This data file will be called /path/to/data/csv later in this readme.

1- Deploy Llama API

About 70GB of VRAM should be enough for this model. Select the number of GPUs below according to this requirement (e.g., we used 4 x 24GB GPUs below). To modify number of GPUs, change CUDA_VISIBLE_DEVICES and tensor-parallel-size, number must be powers of 2 (1,2,4,8,...)
```bash
cd ..
TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --dtype=half --tensor-parallel-size 4 --gpu_memory_utilization 0.9 --port 8000 --max_model_len 120000 --enforce-eager > API.log 2>&1 &
# Check if the API is up
while ! curl -s http://localhost:8000/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done
cd LabelerLLM
```

2- Tumor detection 

Quickly runs through many reports and classifies tumor or no tumor
```bash
python RunRadGPT.py --port 8000 --data_path '/path/to/data/csv' --institution 'UCSF' --step 'tumor detection' --save_name '/path/to/step1/results/csv'
```
3- Malignancy detection

Check tumor positive reports (last step) reports for malignancy
```bash
python RunRadGPT.py --port 8000 --data_path '/path/to/data/csv' --institution 'UCSF' --step 'malignancy detection' --save_name '/path/to/step2/results/csv' --last_step_csv '/path/to/step1/results/csv' --fast '0'
```
4- Size and location measurement

Get size and location of all malignant tumors found in the last step
```bash
python RunRadGPT.py --port 8000 --data_path '/path/to/data/csv' --institution 'UCSF' --step 'malignant size' --save_name '/path/to/step2/results/csv' --last_step_csv '/path/to/step2/results/csv' --fast '0'
```


<details>
  <summary>Improving parallelization (click to expand)</summary>
    
About 60 GB of video memory is needed to run one Llama API. So, with many GPUs, you can run many Llamas, placing each API in one port, and letting each Llama analyze part of your dataset (CSV).
First, launch the APIs. Place each of them in the appropriate GPUs and ports. Here we place one API in GPUs 0-1 (port 8000), and other in 3-4 (port 8001). --tensor-parallel-size is set to 2, using 2 GPUs per API. 

```bash
TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=0,1 vllm serve "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --dtype=half --tensor-parallel-size 2 --gpu_memory_utilization 0.9 --port 8000 --max_model_len 120000 --enforce-eager > API.log 2>&1 &
TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=2,3 vllm serve "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --dtype=half --tensor-parallel-size 2 --gpu_memory_utilization 0.9 --port 8001 --max_model_len 120000 --enforce-eager > API.log 2>&1 &
# Check if the API is up
while ! curl -s http://localhost:8000/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done
# Check if the API is up
while ! curl -s http://localhost:8001/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done
```

Then, run the code as instrcuted before, just add the --parts and --part parameter to each command. --parts should match the number of Llamas you launched, it indicates in how many parts your csv will be split. --part indicates which part each process will analyze. Each process should have a different --part, ranging from 0 to parts-1. Here is an example for tumor size measurement, notice each process uses a different port and part.

```bash
python RunRadGPT.py --port 8000 --data_path '/path/to/data/csv' --institution 'UCSF' --step 'malignant size' --save_name '/path/to/step2/results/csv' --last_step_csv '/path/to/step2/results/csv' --fast '0' --part 0 --parts 2 &
python RunRadGPT.py --port 8001 --data_path '/path/to/data/csv' --institution 'UCSF' --step 'malignant size' --save_name '/path/to/step2/results/csv' --last_step_csv '/path/to/step2/results/csv' --fast '0' --part 1 --parts 2 &
wait
```
    
</details>

#### Example output

See outputs_size_malignant_size.csv, it is the output produced by the step 4 above, and it contains the information about all steps (tumor presence, malignancy, size and location).

