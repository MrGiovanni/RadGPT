<h1 align="center">RadGPT & AbdomenAtlas 3.0</h1>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=MrGiovanni/RadGPT&left_color=%2363C7E6&right_color=%23CEE75F)
[![GitHub Repo stars](https://img.shields.io/github/stars/MrGiovanni/RadGPT?style=social)](https://github.com/MrGiovanni/RadGPT/stargazers)
<a href="https://twitter.com/bodymaps317">
        <img src="https://img.shields.io/twitter/follow/BodyMaps?style=social" alt="Follow on Twitter" />
</a><br/>
**Subscribe us: https://groups.google.com/u/2/g/bodymaps**  

</div>

<div align="center">
 
![logo](document/fig_teaser.png)
</div>

> [!NOTE]
> **We have publicly released our dataset!**  
> Please check the download instructions below.


AbdomenAtlas 3.0 is the first public dataset with high quality abdominal CTs and paired radiology reports. The database includes more than 9,000 CT scans with radiology reports and per-voxel annotations of liver, kidney and pancreatic tumors.

Moreover, we present RadGPT, a segmentation-based report generation model which significantly surpasses the current state of the art in report generation for abdominal CTs.

Our “superhuman” reports are more accurate, detailed, standardized, and generated faster than traditional human-made reports.


## Paper

<b>RadGPT: Constructing 3D Image-Text Tumor Datasets</b> <br/>
[Pedro R. A. S. Bassi](https://scholar.google.com/citations?user=NftgL6gAAAAJ&hl=en), Mehmet Yavuz, Kang Wang, Sezgin Er, Ibrahim E. Hamamci, [Wenxuan Li](https://scholar.google.com/citations?hl=en&user=tpNZM2YAAAAJ), Xiaoxi Chen, Sergio Decherchi, Andrea Cavalli, [Yang Yang](https://scholar.google.com/citations?hl=en&user=6XsJUBIAAAAJ), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Zongwei Zhou](https://www.zongweiz.com/)* <br/>
*Johns Hopkins University* <br/>
ICCV, 2025 <br/>
<a href='https://www.zongweiz.com/dataset'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://www.cs.jhu.edu/~zongwei/publication/bassi2025radgpt.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> <a href='document/bassi2024rsna_radgpt.pdf'><img src='https://img.shields.io/badge/Slides-RSNA-orange'></a> [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/WxgyHNi2tLc)

## Download the AbdomenAtlas 3.0 Dataset

You can directly download AbdomenAtlas 3.0 from [HuggingFace](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini/) using your preferred HuggingFace dowload method (e.g., huggingface-cli), *or you can just use the simple download code below*:

```bash
git clone https://github.com/MrGiovanni/RadGPT.git
cd RadGPT
bash download_atlas_3.sh
```

### Train/Test Splits

You can find csv files with the training and testing IDs inside the downloaded dataset (folder TrainTestIDS).

- IID train/test split: the dataset was randomly partitioned into traing (90%) and testing (10%). These train/test partitions were used in our paper (IID results).
- OOD train/test split: the training and testing patitions come from different hospitals. This is useful for external evaluation, but the training set here is smaller (50% of the whole dataset). These train/test partitions were not used in our paper.

## Installation

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
git clone https://github.com/MrGiovanni/RadGPT.git
cd RadGPT
conda create -n vllm python=3.12 -y
conda activate vllm
conda install -y ipykernel
conda install -y pip
pip install vllm==0.6.1.post2
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
pip install -r requirements.txt
mkdir HFCache
```


## Generate Structured, Narrative and Enhanced Human Reports

Use RadGPT to generate reports from organ and tumor per-voxel segmentations and to enhance human-made reports.

[generate_reports/README.md](generate_reports/README.md)

## Evaluate the Diagnoses in the Reports with LLM

LLM (labeler) extracts binary labels indicating if reports indicate the presence or absence of liver, kidney and pancreatic cancers (or any cancer). These labels can be used to compare AI-made reports to human-made reports (ground-truth) and evaluate cancer detection specificity and sensitivity.

[evaluate_reports/README.md](evaluate_reports/README.md)


## Citation

```
@InProceedings{bassi2025radgpt,
    author    = {Bassi, Pedro R.A.S. and Yavuz, Mehmet Can and Hamamci, Ibrahim Ethem and Er, Sezgin and Chen, Xiaoxi and Li, Wenxuan and Menze, Bjoern and Decherchi, Sergio and Cavalli, Andrea and Wang, Kang and Yang, Yang and Yuille, Alan and Zhou, Zongwei},
    title     = {RadGPT: Constructing 3D Image-Text Tumor Datasets},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {23720-23730},
    url={https://github.com/MrGiovanni/PanTS}
}
```

## Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research, the Patrick J. McGovern Foundation Award, and the National Institutes of Health (NIH) under Award Number R01EB037669. We would like to thank the Johns Hopkins Research IT team in [IT@JH](https://researchit.jhu.edu/) for their support and infrastructure resources where some of these analyses were conducted; especially [DISCOVERY HPC](https://researchit.jhu.edu/research-hpc/). Paper content is covered by patents pending.




