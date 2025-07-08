#Example usage: python3 combine_labels.py --dataset /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas3.0Mini/AbdomenAtlas3.0Mini/  --destination /mnt/ccvl15/pedro/atlas_300_labels --cases /mnt/sdc/pedro/UCSF/foundational/data_code/atlas_ids_300.csv --num_workers 10


import os
import argparse
import nibabel as nib
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import SeparateTumors as st
import SimpleITK as sitk


# Define the label mapping. We need one label for each possible class overlap.
#https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
#this is an nnunet label mapping. In the integer map, each label is assigned a unique integer value (the first value in the list). 
labels=['kidney_lesion','liver_lesion','pancreatic_lesion']



def merge_lesions(file_path):
    found=False
    read_error=False
    volume=None
    for tpe in ['lesion','tumor','cyst']:
        pth=file_path.replace('lesion',tpe)
        if not os.path.exists(pth):
            continue
        try:
            nii = nib.load(pth)
            v = nii.get_fdata()
            affine = nii.affine
            header = nii.header
            if volume is None:
                volume=binarize_volume(v)
            else:
                volume+=binarize_volume(v)
            found=True
            #print(f"Found {pth}")
        except:
            read_error=True
    if read_error:
        return 'break', None, None
    if not found:
        return 'continue', None, None
    return volume, affine, header


def process_tumor_annotation(
    input_nifti_path,
    output_nifti_path=None,
    denoise=False,
    erosion_iterations=1,
    dilation_iterations=4,
    max_ch=10,
    affine=None,header=None
):
    if isinstance(input_nifti_path, str):
        nii = nib.load(input_nifti_path)
        volume = nii.get_fdata()
        affine = nii.affine
        header = nii.header
    else:
        volume = input_nifti_path
        affine = affine
        header = header

    # 1. Threshold to get a binary mask
    volume_bool = volume > 0

    # Convert the binary mask to a SimpleITK image for morphological operations
    sitk_image = sitk.GetImageFromArray(volume_bool.astype(np.uint8))

    # 2. (Optional) Morphological denoising
    if denoise:
        # Erode
        for _ in range(erosion_iterations):
            erode_filter = sitk.BinaryErodeImageFilter()
            erode_filter.SetForegroundValue(1)
            erode_filter.SetKernelRadius(1)
            erode_filter.SetKernelType(sitk.sitkBall)
            sitk_image = erode_filter.Execute(sitk_image)

        # Dilate
        for _ in range(dilation_iterations):
            dilate_filter = sitk.BinaryDilateImageFilter()
            dilate_filter.SetForegroundValue(1)
            dilate_filter.SetKernelRadius(1)
            dilate_filter.SetKernelType(sitk.sitkBall)
            sitk_image = dilate_filter.Execute(sitk_image)

        morph_np = sitk.GetArrayFromImage(sitk_image).astype(bool)
        final_mask = np.logical_and(volume_bool, morph_np)
    else:
        final_mask = volume_bool

    # 3. Connected component labeling
    sitk_final = sitk.GetImageFromArray(final_mask.astype(np.uint8))
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)  # 26-connectivity in 3D
    labeled_sitk = cc_filter.Execute(sitk_final)

    labeled_volume = sitk.GetArrayFromImage(labeled_sitk)

    # Identify how many labels
    unique_labels = np.unique(labeled_volume)
    num_labels = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)

    # 4. Handle the case of no labels (no tumors)
    if num_labels == 0:
        # Provide a single channel of all False
        final_4d = final_mask[np.newaxis, ...]  # shape (1, X, Y, Z)
        if output_nifti_path:
            out_nii = nib.Nifti1Image(final_4d.astype(np.uint8), affine, header)
            nib.save(out_nii, output_nifti_path)
        return final_4d.astype(bool), affine, header

    # 5. Sort the labels by volume (descending)
    labeled_volumes = []
    for label_idx in unique_labels:
        if label_idx == 0:
            continue  # skip background
        mask_component = (labeled_volume == label_idx)
        labeled_volumes.append((label_idx, np.sum(mask_component)))
    labeled_volumes.sort(key=lambda x: x[1], reverse=True)

    # 6. Create output 4D array with channels first: (N, X, Y, Z)
    num_channels = min(num_labels, max_ch)
    final_4d = np.zeros((num_channels,) + final_mask.shape, dtype=bool)

    for out_idx, (comp_label, _) in enumerate(labeled_volumes[:num_channels]):
        final_4d[out_idx] = (labeled_volume == comp_label)

    # 7. (Optional) Save the result
    #    Be aware that many standard tools/viewers expect shape (X, Y, Z, N)
    #    and may interpret shape (N, X, Y, Z) in unexpected ways.
    if output_nifti_path:
        out_nii = nib.Nifti1Image(final_4d.astype(np.uint8), affine, header)
        nib.save(out_nii, output_nifti_path)

    return final_4d, affine, header


def binarize_volume(volume, threshold=0.5):
    """Binarize the volume using the given threshold."""
    return (volume > threshold).astype(np.uint8)

def merge_labels(input_dir, output_filename):
    """Combine segmentations into a single label file.
    Important: missing  labels are not written. Thus, they are negatives for that label."""
    combined_volume = None
    affine = None  # To store the affine transformation of the first volume
    header = None  # To store the NIfTI header of the first volume
    save = True
    
    for label_name in labels:
        #print('label_value', label_value)
        #print('label_name', label_name)
        #raise ValueError
        file_path = os.path.join(input_dir, f"{label_name}.nii.gz")
        
        volume, affine, header=merge_lesions(file_path)
        if isinstance(volume, str):
            if volume=='continue':
                print(f"Label '{label_name}' not found in {input_dir}. Ignoring this label.")
                continue
            if volume=='break':
                print(f"Error reading {file_path}. Skipping this CT.")
                save=False
                break
        
        #separate lesion
        binarized_volume, affine, header = process_tumor_annotation(volume, output_nifti_path=os.path.join(output_filename, f"{label_name}.nii.gz"),
                                                                    affine=affine,header=header)

def binarize_volume(volume, threshold=0.5):
    """Binarize the volume using the given threshold."""
    return (volume > threshold).astype(np.uint8)

def process_case(args):
    """Process a single case in the dataset."""
    case, dataset_dir, output_dir = args
    case_dir = os.path.join(dataset_dir, case)
    output_dir = os.path.join(output_dir, case)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    predictions_dir = os.path.join(case_dir, "segmentations")
    # Check if the predictions directory exists
    if not os.path.exists(predictions_dir):
        predictions_dir = os.path.join(case_dir, "predictions")
    
    # Output filename
    merge_labels(predictions_dir, output_dir)
    
    return case, True



def process_dataset(dataset_dir, destination, cases, num_workers):
    """Process the dataset with parallel workers."""
    cases = pd.read_csv(cases)["BDMAP_ID"].tolist()
    saved = os.listdir(dataset_dir)
    #print('cases', cases)
    ##print('saved', saved)
    #print('dataset_dir', dataset_dir)
    print('saved:',len(os.listdir(dataset_dir)))
    print('fisrt 10:',os.listdir(dataset_dir)[:10])
    print('cases:',len(cases))

    cases = [case for case in cases if case in saved]

    print(cases)
    
    # Use a pool of workers to process cases in parallel
    with Pool(num_workers) as pool:
        # Wrap the function with tqdm for progress tracking
        results = list(tqdm(
            pool.imap_unordered(process_case, [(case, dataset_dir, destination) for case in cases]),
            total=len(cases),
            desc="Processing dataset"
        ))
    
    # Print summary of results
    successful_cases = [case for case, success in results if success]
    failed_cases = [case for case, success in results if not success]
    
    print(f"\nProcessing complete. {len(successful_cases)} cases processed successfully.")
    if failed_cases:
        print(f"{len(failed_cases)} cases failed: {failed_cases}")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Combine segmentation labels into a single NIfTI file with parallel processing.")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("--destination", type=str, help="Path to the output directory.")
    parser.add_argument("--cases", type=str, help="Path to cases csv.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers to use.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the dataset
    process_dataset(args.dataset_dir, args.destination, args.cases, args.num_workers)