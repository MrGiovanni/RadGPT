import pandas as pd
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load cases from CSV files
cases = pd.read_csv('/dev/shm/pedro/foundational/data_code/atlas_ids_300.csv')["BDMAP_ID"].tolist()
#cases += pd.read_csv('/dev/shm/pedro/foundational/data_code/UCSF_ids_1234.csv')["BDMAP_ID"].tolist()

# Directories
source_ct_dir = '/mnt/realccvl15/zzhou82/data/AbdomenAtlasPro'
source_ct_dir = '/mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_only/AbdomenAtlas1.1Mini/AbdomenAtlas1.1Mini/'
source_mask_dir = '/ccvl/net/ccvl15/pedro/vista3d_labels'
dest_ct_dir = '/ccvl/net/ccvl15/pedro/nnUNet_raw/Dataset300_smallAtlas/imagesTr'
dest_mask_dir = '/ccvl/net/ccvl15/pedro/nnUNet_raw/Dataset300_smallAtlas/labelsTr'

# Create destination directories if they don't exist
os.makedirs(dest_ct_dir, exist_ok=True)
os.makedirs(dest_mask_dir, exist_ok=True)

def copy_case(case):
    """
    Copies the CT and mask for a given case.
    Skips cases that are already copied.
    Returns a status message.
    """
    source_ct = os.path.join(source_ct_dir, case, 'ct.nii.gz')
    source_mask = os.path.join(source_mask_dir, case, 'combined_labels.nii.gz')
    dest_ct = os.path.join(dest_ct_dir, f'{case}_0000.nii.gz')
    dest_mask = os.path.join(dest_mask_dir, f'{case}.nii.gz')
    
    # Skip if both files are already copied
    #if os.path.exists(dest_ct) and os.path.exists(dest_mask):
    #    return f'Skipped {case} (already copied)'

    # Check if source files exist
    if not os.path.exists(source_ct):
        return f'CT not found for {case}'
    if not os.path.exists(source_mask):
        return f'Mask not found for {case}'

    try:
        # Copy files
        shutil.copy(source_ct, dest_ct)
        shutil.copy(source_mask, dest_mask)
        return f'Copied {case}'
    except Exception as e:
        return f'Error copying {case}: {e}'

def parallel_copy(cases, max_workers=8):
    """
    Copies files for all cases in parallel with a progress bar.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(copy_case, case): case for case in cases}
        
        # Wrap with tqdm for a progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Copying cases"):
            print(future.result())

# Run the parallel copying
if __name__ == "__main__":
    parallel_copy(cases, max_workers=16)