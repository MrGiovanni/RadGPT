"""
Usage example: 
python split_labels.py --input_dir /projects/bodymaps/Pedro/data/JHH_liver_segments/ --output_dir /projects/bodymaps/Pedro/data/JHH_liver_segments_sep_labels/ --parts 3 --part 0
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor
import concurrent
from tqdm import tqdm

labels={
    'background': 0,
    'aorta': 1,
    'gall_bladder': 2,
    'kidney_left': 3,
    'kidney_right': 4,
    'postcava': 5,
    'spleen': 6,
    'stomach': 7,
    'adrenal_gland_left': 8,
    'adrenal_gland_right': 9,
    'bladder': 10,
    'celiac_trunk': 11,
    'colon': 12,
    'duodenum': 13,
    'esophagus': 14,
    'femur_left': 15,
    'femur_right': 16,
    'hepatic_vessel': 17,
    'intestine': 18,
    'lung_left': 19,
    'lung_right': 20,
    'portal_vein_and_splenic_vein': 21,
    'prostate': 22,
    'rectum': 23,
    'liver_segment_1': 24,
    'liver_segment_2': 25,
    'liver_segment_3': 26,
    'liver_segment_4': 27,
    'liver_segment_5': 28,
    'liver_segment_6': 29,
    'liver_segment_7': 30,
    'liver_segment_8': 31,
    'pancreas_head': 32,
    'pancreas_body': 33,
    'pancreas_tail': 34,
}

# Classes we want to output explicitly
#out_labels = [
#    "liver",
#    "pancreas",
#    "kidney_right",
#    "kidney_left",
#    "liver_lesion",
#    "kidney_lesion",
#    "pancreatic_lesion",
#    "pancreas_head",
#    "pancreas_body",
#    "pancreas_tail"
#]
# Add "liver_segment_1" through "liver_segment_8"
#for i in range(1, 9):
#    out_labels.append("liver_segment_" + str(i))
out_labels = list(labels.keys())
out_labels+=['liver','pancreas']

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split segmentation labels into multiple .nii.gz files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input .nii.gz files to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the splitted .nii.gz files will be saved.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="Number of parallel processes to use for splitting.",
    )
    parser.add_argument(
        "--restart",
        action='store_true',
        help="Overwrites files already saved.",
    )

    parser.add_argument(
        "--parts",
        type=int,
        default=1,
        help="Total number of parts to split the files into.",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=0,
        help="Index of the part to process (0-indexed).",
    )
    parser.add_argument(
        "--JHH",
        action='store_true',
        help="Processes only JHH files.",
    )
    return parser.parse_args()


def build_mapping():
    """
    Build the 'mapping' dictionary:
    Each key is an output class name,
    each value is a list of label IDs that should be merged into that class.
    """

    mapping = {}
    for key in out_labels:
        ids = []
        # Find all label IDs whose name contains `key`
        for name, val in labels.items():
            if key in name:
                ids.append(val)

        # Special case: "pancreas" also includes "pancreatic"
        if key == "pancreas":
            for name, val in labels.items():
                if "pancreatic" in name:
                    ids.append(val)

        mapping[key] = sorted(set(ids))

    return mapping



def split_segmentation(input_nifti_path, output_folder, classes_dict):
    """
    Reads a NIfTI segmentation, then saves separate .nii.gz masks
    for each class (merging labels if necessary).
    """
    print('Splitting', input_nifti_path)
    seg_img = nib.load(input_nifti_path)
    # *** Fix: load as float, then cast to int16 (or directly from dataobj) ***
    seg_data = seg_img.get_fdata().astype(np.int16)
    # Or use: seg_data = np.asanyarray(seg_img.dataobj).astype(np.int16)

    # Ensure output_folder exists
    os.makedirs(output_folder, exist_ok=True)

    # We'll name output files using:
    #   <base of input file>_<class_name>.nii.gz
    base_name = os.path.basename(input_nifti_path)
    if base_name.endswith(".nii.gz"):
        base_name = base_name[:-7]
    else:
        base_name = os.path.splitext(base_name)[0]

    # Optionally create a subfolder for each input file
    this_out_folder = os.path.join(output_folder, base_name, 'predictions')
    os.makedirs(this_out_folder, exist_ok=True)

    for class_name, label_list in classes_dict.items():
        # Create binary mask: 1 if label in label_list, else 0
        mask = np.isin(seg_data, label_list).astype(np.uint8)

        mask_nifti = nib.Nifti1Image(mask, seg_img.affine, seg_img.header)
        out_file = os.path.join(this_out_folder, f"{class_name}.nii.gz")
        nib.save(mask_nifti, out_file)

    print(f"Finished splitting {input_nifti_path}!")

def process_file(input_path, output_dir, classes_dict):
    """Wrapper to process a single file."""
    split_segmentation(input_path, output_dir, classes_dict)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Build the label mapping once
    classes_dict = build_mapping()
    print(classes_dict)

    # Gather all .nii.gz files in the input directory
    nifti_files = glob.glob(os.path.join(args.input_dir, "*.nii.gz"))
    if len(nifti_files) == 0:
        print(f"No NIfTI files found in {args.input_dir}!")
        return

    if args.JHH:
        nifti_files = [f for f in nifti_files if (('BDMAP_A' in f) or ('BDMAP_V' in f))]

    # remove from the list all files aleady saved
    if not args.restart:
        new_nifti_files = []
        for fpath in nifti_files:
            base_name = os.path.basename(fpath)
            if base_name.endswith(".nii.gz"):
                base_name = base_name[:-7]
            else:
                base_name = os.path.splitext(base_name)[0]
            
            # The same logic used in split_segmentation to build the output folder
            this_out_folder = os.path.join(args.output_dir, base_name, 'predictions')
            
            # Check if every class_name has a corresponding <class_name>.nii.gz
            all_exist = True
            for class_name in classes_dict.keys():
                out_file = os.path.join(this_out_folder, f"{class_name}.nii.gz")
                if not os.path.exists(out_file):
                    all_exist = False
                    break
            
            # Keep this file if not all masks are present
            if not all_exist:
                new_nifti_files.append(fpath)
        
        nifti_files = new_nifti_files

    # Split the nifti_files list into parts and process only the selected part.
    total_files = len(nifti_files)
    if args.parts > 1:
        files_per_part = (total_files + args.parts - 1) // args.parts  # round up
        start_idx = args.part * files_per_part
        end_idx = min(start_idx + files_per_part, total_files)
        nifti_files = nifti_files[start_idx:end_idx]
        print(f"Processing part {args.part+1}/{args.parts}: files {start_idx} to {end_idx-1}")
    else:
        print("Processing all files (only one part).")
    print(f"Total files to process: {len(nifti_files)}")

    # Use parallel processing
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for fpath in nifti_files:
            futures.append(
                executor.submit(process_file, fpath, args.output_dir, classes_dict)
            )
        #for future in futures:
        #    future.result()
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"):
            future.result()

    print(f"All files processed. Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
