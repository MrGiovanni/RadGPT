#Example usage: python3 combine_labels.py --dataset /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas3.0Mini/AbdomenAtlas3.0Mini/  --destination /mnt/ccvl15/pedro/atlas_300_labels --cases /mnt/sdc/pedro/UCSF/foundational/data_code/atlas_ids_300.csv --num_workers 10


import os
import argparse
import nibabel as nib
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd



# Define the label mapping. We need one label for each possible class overlap.
#https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
#this is an nnunet label mapping. In the integer map, each label is assigned a unique integer value (the first value in the list). 
labels={'background': 0,
            'kidney_right': 1,
            'kidney_left': 2,
            'kidney_lesion': 3,
            'kidney_lesion_kidney_right': 4,
            'kidney_lesion_kidney_left': 5,
            'pancreas': 6,
            'pancreas_head': 7,
            'pancreas_body': 8,
            'pancreas_tail': 9,
            'pancreatic_lesion': 10,
            'pancreatic_lesion_pancreas_head': 11,
            'pancreatic_lesion_pancreas_body': 12,
            'pancreatic_lesion_pancreas_tail': 13,
            'liver': 14,
            'liver_segment_1': 15,
            'liver_segment_2': 16,
            'liver_segment_3': 17,
            'liver_segment_4': 18,
            'liver_segment_5': 19,
            'liver_segment_6': 20,
            'liver_segment_7': 21,
            'liver_segment_8': 22,
            'liver_lesion': 23,
            'liver_lesion_liver_segment_1': 24,
            'liver_lesion_liver_segment_2': 25,
            'liver_lesion_liver_segment_3': 26,
            'liver_lesion_liver_segment_4': 27,
            'liver_lesion_liver_segment_5': 28,
            'liver_lesion_liver_segment_6': 29,
            'liver_lesion_liver_segment_7': 30,
            'liver_lesion_liver_segment_8': 31,
            'spleen': 32,
            'colon': 33,
            'stomach': 34,
            'duodenum': 35,
            'common_bile_duct': 36,
            'intestine': 37,
            'aorta': 38,
            'postcava': 39,
            'adrenal_gland_left': 40,
            'adrenal_gland_right': 41,
            'gall_bladder': 42,
            'bladder': 43,
            'celiac_trunk': 44,
            'esophagus': 45,
            'hepatic_vessel': 46,
            'portal_vein_and_splenic_vein': 47,
            'lung_left': 48,
            'lung_right': 49,
            'prostate': 50,
            'rectum': 51,
            'femur_left': 52,
            'femur_right': 53,
            'superior_mesenteric_artery': 54,
            'veins': 55}

region_class_order=list(range(1,1+len(labels)))

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
            if volume is None:
                volume=binarize_volume(v)
            else:
                volume+=binarize_volume(v)
            found=True
            #print(f"Found {pth}")
        except:
            read_error=True
    if read_error:
        return 'break'
    if not found:
        return 'continue'
    return volume

def binarize_volume(volume, threshold=0.5):
    """Binarize the volume using the given threshold."""
    return (volume > threshold).astype(np.uint8)

def combine_labels(input_dir, output_filename):
    """Combine segmentations into a single label file.
    Important: missing  labels are not written. Thus, they are negatives for that label."""
    combined_volume = None
    affine = None  # To store the affine transformation of the first volume
    header = None  # To store the NIfTI header of the first volume
    save = True
    
    for label_name, label_value in labels.items():
        #print('label_value', label_value)
        #print('label_name', label_name)
        #raise ValueError
        if isinstance(label_value, list):
            label_value = label_value[0]
        if label_name == 'background':
            continue
        file_path = os.path.join(input_dir, f"{label_name}.nii.gz")
        if 'lesion' not in label_name and not label_name.endswith('lesion'):
            if not os.path.exists(file_path):
                print(f"Label '{label_name}' not found in {input_dir}. Ignoring this label.")
                continue
        
        # Read the NIfTI file
        if 'lesion' not in label_name:
            try:
                nii = nib.load(file_path)
                volume = nii.get_fdata()
            except:
                print(f"Error reading {file_path}. Skipping this CT.")
                save = False
                break
        elif label_name.endswith('lesion'):
            volume=merge_lesions(file_path)
            if isinstance(volume, str):
                if volume=='continue':
                    print(f"Label '{label_name}' not found in {input_dir}. Ignoring this label.")
                    continue
                if volume=='break':
                    print(f"Error reading {file_path}. Skipping this CT.")
                    save=False
                    break
        else:
            #lesion_subseg
            lesion_pth=file_path[:file_path.rfind('lesion')]+'lesion'+'.nii.gz'
            lesion=merge_lesions(lesion_pth)
            #print('lesion pth:',lesion_pth)
            if isinstance(volume, str):
                if volume=='continue':
                    #print(f"Label '{label_name}' not found in {input_dir}. Ignoring this label.")
                    continue
                if volume=='break':
                    #print(f"Error reading {file_path}. Skipping this CT.")
                    save=False
                    break
            
            segment=file_path[file_path.rfind('lesion_')+len('lesion_'):]
            segment_pth = os.path.join(input_dir, segment)
            try:
                segment = nib.load(segment_pth).get_fdata()
            except:
                print(f"Error reading {segment_pth}. Continuing.")
                continue
            segment = binarize_volume(segment)
            lesion = binarize_volume(lesion)
            #get overlap between lesion and subsegment
            volume = lesion*segment
        
        # Binarize the volume
        binarized_volume = binarize_volume(volume)
        
        # Initialize the combined volume if not already done
        if combined_volume is None:
            combined_volume = np.zeros_like(binarized_volume, dtype=np.uint8)
            affine = nii.affine
            header = nii.header  # Save the header of the first file
        
        # Overwrite the combined volume with the current label value
        combined_volume[binarized_volume > 0] = label_value
    
    if save:
        # Save the combined volume with the original header
        combined_nii = nib.Nifti1Image(combined_volume, affine, header=header)
        nib.save(combined_nii, output_filename)

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
    output_filename = os.path.join(output_dir, "combined_labels.nii.gz")
    
    # Combine labels
    combine_labels(predictions_dir, output_filename)
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