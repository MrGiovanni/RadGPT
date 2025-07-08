#Example usage: python /projects/bodymaps/Pedro/foundational/data_code/combine_labels_multi_tumor.py --dataset_dir /projects/bodymaps/Pedro/data/Radiologist_annotated_300_UCSF/ --destination /projects/bodymaps/Pedro/data/Radiologist_annotated_300_UCSF_nnunet/ --num_workers 3


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
labels_n=[   'background',
            'kidney_right',
            'kidney_left',
            'kidney_lesion',
            'kidney_lesion_kidney_right',
            'kidney_lesion_kidney_left',
            'pancreas',
            'pancreas_head',
            'pancreas_body',
            'pancreas_tail',
            'pancreatic_lesion',
            'pancreatic_lesion_pancreas_head',
            'pancreatic_lesion_pancreas_body',
            'pancreatic_lesion_pancreas_tail',
            'liver',
            'liver_segment_1',
            'liver_segment_2',
            'liver_segment_3',
            'liver_segment_4',
            'liver_segment_5',
            'liver_segment_6',
            'liver_segment_7',
            'liver_segment_8',
            'liver_lesion',
            'liver_lesion_liver_segment_1',
            'liver_lesion_liver_segment_2',
            'liver_lesion_liver_segment_3',
            'liver_lesion_liver_segment_4',
            'liver_lesion_liver_segment_5',
            'liver_lesion_liver_segment_6',
            'liver_lesion_liver_segment_7',
            'liver_lesion_liver_segment_8',
            'spleen',
            'spleen_lesion',
            'colon',
            'colon_lesion',
            'stomach',
            'stomach_lesion',
            'duodenum',
            'duodenum_lesion',
            'common_bile_duct',
            'intestine',
            'aorta',
            'postcava',
            'adrenal_gland_left',
            'adrenal_gland_right',
            'adrenal_lesion',
            'adrenal_lesion_adrenal_gland_left',
            'adrenal_lesion_adrenal_gland_right',
            'gall_bladder',
            'gallbladder_lesion',
            'gallbladder_lesion_gall_bladder',
            'bladder',
            'bladder_lesion',
            'celiac_trunk',
            'esophagus',
            'esophagus_lesion',
            'hepatic_vessel',
            'portal_vein_and_splenic_vein',
            'lung_left',
            'lung_right',
            'lung_lesion',
            'lung_lesion_lung_left',
            'lung_lesion_lung_right',
            'prostate',
            'prostate_lesion',
            'uterus_lesion',
            'uterus_lesion_prostate',#this is a bug, our prostate annotation should actually be called prostate/uterus
            'rectum',
            'femur_left',
            'femur_right',
            'superior_mesenteric_artery',
            'veins',
            'bone_lesion',
            'breast_lesion',
            ]

labels={}
for i,name in enumerate(labels_n,0):
    labels[name]=i

region_class_order=list(range(1,1+len(labels)))

def merge_lesions(file_path):
    """
    Ensures lesion encompasses tumor and cyst
    """
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
            if isinstance(lesion, str):
                if lesion=='continue':
                    #print(f"Label '{label_name}' not found in {input_dir}. Ignoring this label.")
                    continue
                if lesion=='break':
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
            nii = nib.load(file_path)
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
    if cases is None:
        cases = [f for f in os.listdir(dataset_dir) if 'BDMAP_' in f]
    else:
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
    parser.add_argument("--cases", type=str, default=None,  help="Path to cases csv.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers to use.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the dataset
    process_dataset(args.dataset_dir, args.destination, args.cases, args.num_workers)