import os
import nibabel as nib
from multiprocessing import Pool
from tqdm import tqdm

def verify_shape(ct_path):
    """
    Verifies if the shape of the CT image matches the label shape for a given case.
    """
    # Replace 'imagesTr' with 'labelsTr' to locate the corresponding label file
    label_path = ct_path.replace('imagesTr', 'labelsTr').replace('_0000.nii.gz', '.nii.gz')
    try:
        # Load the CT and label files
        ct_image = nib.load(ct_path)
        label_image = nib.load(label_path)
        
        # Compare shapes
        if ct_image.shape != label_image.shape:
            return ct_path, ct_image.shape, label_image.shape
    except Exception as e:
        # Return the error for debugging
        return ct_path, str(e), None

    return None

def check_shapes(dataset_dir, num_workers=4):
    """
    Checks all cases in the dataset for shape mismatches.
    """
    # Get the list of CT files in the dataset directory
    case_paths = [os.path.join(dataset_dir, case) for case in os.listdir(dataset_dir) if case.endswith(".nii.gz")]
    
    # Use multiprocessing to verify shapes in parallel
    mismatches = []
    with Pool(num_workers) as pool:
        # Wrap pool.imap_unordered with tqdm for progress tracking
        for result in tqdm(pool.imap_unordered(verify_shape, case_paths), total=len(case_paths), desc="Verifying shapes"):
            if result is not None:
                mismatches.append(result)
    
    # Print summary
    total_mismatches = len(mismatches)
    print(f"\nTotal cases verified: {len(case_paths)}")
    
    if total_mismatches > 0:
        print(f"Shape mismatches found: {total_mismatches}")
        for case, ct_shape, label_shape in mismatches:
            print(f"Case: {os.path.basename(case)}")  # Ensure case name is displayed
            print(f"  CT Shape: {ct_shape}")
            print(f"  Label Shape: {label_shape}")
    else:
        print("All cases have matching shapes.")

    # Ensure the total mismatches are explicitly printed
    print(f"\nSummary: {total_mismatches} shape mismatch(es) found.")

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Verify shape consistency between CT images and labels.")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()

    # Run shape verification
    check_shapes(args.dataset_dir, args.num_workers)