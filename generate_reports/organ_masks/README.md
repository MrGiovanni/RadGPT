# Create organ and sub-segment masks

This code creates the organ and organ sub-segment (e.g. pancreatic head, body and tail).



## Download the nnU-Net

1. Download our organ segmenter from huggingface.co/prasb/OrganSubSegmenter/: [click here](https://huggingface.co/prasb/OrganSubSegmenter/resolve/main/nnUNetTrainer__nnUNetPlannerResEncL_torchres_isotropic__3d_fullres.zip?download=true) or use the command below
```bash
cd RadGPT/generate_reports/organ_masks/
conda create -n hf python=3.12 -y
conda activate hf pip
pip install huggingface-hub
huggingface-cli download prasb/OrganSubSegmenter --local-dir .
```
2. Unzip the nnU-Net: 
```bash
unzip nnUNetTrainer__nnUNetPlannerResEncL_torchres_isotropic__3d_fullres.zip
```

## Create Organ Masks: Inference the nnU-Net

1- Data format

Ensure all your CT scans are in the nifti format (.nii.gz), and stored in a single folder. The file names do not matter.

```
/path/to/dataset/
├── image1.nii.gz
├── image2.nii.gz
├── image3.nii.gz
...
```


<details>
<summary style="margin-left: 25px;">Alternative format: AbdomenAtlas</summary>
<div style="margin-left: 25px;">

Our code also accepts data in the AbdomenAtlas format, shown below.

```
/path/to/dataset/
├── BDMAP_00000001
|    └── ct.nii.gz
├── BDMAP_00000002
|    └── ct.nii.gz
...
```
</div>
</details>

2- Install nnU-Net (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)
```bash
conda create -n organ_masks python=3.12 -y
conda activate organ_masks
conda install -y pip
pip install torch torchvision torchaudio
pip install nnunetv2
```

For a custom dataset, you may use symlinks to change its format but save disk space.


3- Inference the nnU-Net:

```bash
bash parallel_inference.sh --pth /path/to/dataset/merlinabdominalctdataset/merlin_data/ --outdir /path/to/nnunet/output --checkpoint nnUNetTrainer__nnUNetPlannerResEncL_torchres_isotropic__3d_fullres/ --gpus 0
```

Arguments:
- --pth: path to the dataset folder containing the nifti files (/path/to/dataset/ above)
- --BDMAP_format: add this argument if your dataset is in the alternative format (AbdomenAtlas format)
- --gpus: list of gpus you want to use. Set to 0 for a single GPU computer. For 4 gpus, set --gpus 0,1,2,3

> [!NOTE]
> --pth points to the dataset root. This is the folder that contains folders called BDMAP_... inside it. See "Alternative format: AbdomenAtlas" above. In that example, you would use --pth /path/to/dataset/, not --pth /path/to/dataset/BDMAP_00000001.

<details>
  <summary>Other dataset formats</summary>
If your data is not in the formats explained in 1, you need to change files_input inside PredictSubOrgansnUnet.py. files_input should be a list of lists. Each of these lists should contain the path to one nii.gz file you want to inference. The variable files_output is a list of strings. It has the output locations for each of the input files. See https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/readme.md for more information. It may be easier to just change the dataset to the format specified in 1, you can use symlinks to save disk space.

```python
files_input = [['path/to/first/ct.nii.gz'],['path/to/second/ct.nii.gz'],...,['path/to/last/ct.nii.gz']]
files_output = ['path/to/output/first/ct.nii.gz','path/to/output/second/ct.nii.gz',...,'path/to/output/last/ct.nii.gz']
```

</details>


## Split labels

The nnU-Net outputs combined labels (multiple organs in one file). We split them, creating one file per organ.

```bash
python split_labels.py --input_dir /path/to/nnunet/output/ --output_dir /path/to/split/labels/otuput/
```

<details>
  <summary>(Optional) Train nnU-Net to sub-segment organs</summary>


### (I) prepare dataset

**This code will convert a dataset from the BDMAP format to the nnU-Net format.**

0. Define nnunet paths
```bash
export nnUNet_raw=/path/to/nnUNet_raw/
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed/
export nnUNet_results=/path/to/nnUNet_results/
```

1. Combine labels. The script merges the BDMAP labels (one per organ) into combined labels. To change the labels used, edit the label map in combine_labels.py. The output are combined label in the BDMAP structure.

```bash
python3 combine_labels.py --dataset /path/to/dataset/in/BDMAP/format --destination /path/to/output/of/step1/ --cases /path/to/csv/with/BDMAP/ids --num_workers 10
```

2. Copy dataset to nnUNet raw folder, chaning file names to the nnUNet standard. Change paths in the beginning of the copy_dataset.py script. Target path must be in the nnunet_raw folder, and include the a dataset_id (use any number above 300) and name. E.g.: Dataset300_smallAtlas has id 300 and name smallAtlas.

```bash
python3 copy_dataset.py
```

3. Verify if mask and CT shapes match. Remove/solve unmatching cases.

```bash
python verify_data.py --dataset_dir /path/to/nnUNet_raw/dataset_with_id_and_name/imagesTr
```

4. Create a dataset json. Change the Dataset300_smallAtlas.py, change target_dataset_id, target_dataset_name and raw_dir (nnUNet raw directory). For id, put any number above 300. You will use this dataset_id in the other steps. Change the ids variable: the label map here should **match the one in step 1**. If you have label superposition, you may need to change superposing_groups too (see script)

```bash
python Dataset300_smallAtlas.py
```

5. Extract fingerprint. NP is just the number of processes.

```bash
nnUNetv2_extract_fingerprint -d dataset_id -np 15
```

6. Create plans for the nnUNet training. Here, we use ResEncL with isotropic spacing.

```bash
nnUNetv2_plan_experiment -d dataset_id -overwrite_target_spacing 1 1 1 -overwrite_plans_name nnUNetPlannerResEncL_torchres_isotropic -pl nnUNetPlannerResEncL_torchres
```

7. Preprocess the dataset. This takes a long time.

```bash
nnUNetv2_preprocess -d dataset_id -npfp 64 -np 64 -c 3d_fullres -pl nnUNetPlannerResEncL_torchres_isotropic --npz
```

### (II) Train

```bash
nnUNetv2_train dataset_id 3d_fullres all -p nnUNetPlannerResEncL_torchres_isotropic --npz
```

</details>
