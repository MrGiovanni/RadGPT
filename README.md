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

AbdomenAtlas 3.0 is the first public dataset with high quality abdominal CTs and paired radiology reports. The database includes more than 9,000 CT scans with radiology reports and per-voxel annotations of liver, kidney and pancreatic tumors.

Moreover, we present RadGPT, a segmentation-based report generation model which significantly surpasses the current state of the art in report generation for abdominal CTs.

Our “superhuman” reports are more accurate, detailed, standardized, and generated faster than traditional human-made reports. **Email zzhou82@jh.edu to get early access to this dataset.**


## Paper

<b>RadGPT: Constructing 3D Image-Text Tumor Datasets</b> <br/>
[Pedro R. A. S. Bassi](https://scholar.google.com/citations?user=NftgL6gAAAAJ&hl=en), Mehmet Yavuz, Kang Wang, Sezgin Er, Ibrahim E. Hamamci, [Wenxuan Li](https://scholar.google.com/citations?hl=en&user=tpNZM2YAAAAJ), Xiaoxi Chen, Sergio Decherchi, Andrea Cavalli, [Yang Yang](https://scholar.google.com/citations?hl=en&user=6XsJUBIAAAAJ), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Zongwei Zhou](https://www.zongweiz.com/)* <br/>
*Johns Hopkins University* <br/>
<a href='https://www.zongweiz.com/dataset'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://www.cs.jhu.edu/~zongwei/publication/bassi2025radgpt.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> <a href='document/bassi2024rsna_radgpt.pdf'><img src='https://img.shields.io/badge/Slides-RSNA-orange'></a> [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/WxgyHNi2tLc)

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
git clone https://github.com/PedroRASB/RadGPT
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
@article{bassi2025radgpt,
  title={RadGPT: Constructing 3D Image-Text Tumor Datasets},
  author={Bassi, Pedro RAS and Yavuz, Mehmet Can and Wang, Kang and Chen, Xiaoxi and Li, Wenxuan and Decherchi, Sergio and Cavalli, Andrea and Yang, Yang and Yuille, Alan and Zhou, Zongwei},
  journal={arXiv preprint arXiv:2501.04678},
  year={2025},
  url={https://github.com/MrGiovanni/RadGPT}
}
```

## Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the McGovern Foundation. Paper content is covered by patents pending.




