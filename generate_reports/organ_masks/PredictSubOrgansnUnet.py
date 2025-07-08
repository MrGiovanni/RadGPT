"""
Use this to run the nnU-Net inference. You can run multiple instances of this code in parallel!
Just change the `part_id` and `gpu` arguments to split the task into multiple parts.
Example of splitting it in 4 parts:
    python PredictSubOrgansnUnet.py --num_parts 4 --part_id 0 --gpu 0
    python PredictSubOrgansnUnet.py --num_parts 4 --part_id 1 --gpu 1
    python PredictSubOrgansnUnet.py --num_parts 4 --part_id 2 --gpu 2
    python PredictSubOrgansnUnet.py --num_parts 4 --part_id 3 --gpu 3
"""


import argparse
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os
import pandas as pd

def split_files(files_input, files_output, num_parts, part_id):
    """
    Splits the files_input and files_output into num_parts and selects the partition for the given part_id.

    Args:
        files_input (list): List of input files.
        files_output (list): List of output files.
        num_parts (int): Number of partitions.
        part_id (int): Partition ID to select (starting from 0).

    Returns:
        tuple: (files_input_partition, files_output_partition) for the given part_id.
    """
    assert len(files_input) == len(files_output), "files_input and files_output must have the same length"
    assert 0 <= part_id < num_parts, "Invalid part_id. It must be in the range [0, num_parts-1]"

    # Split the files based on num_parts
    total_files = len(files_input)
    files_per_part = (total_files + num_parts - 1) // num_parts  # Ensure rounding up
    start_idx = part_id * files_per_part
    end_idx = min(start_idx + files_per_part, total_files)

    # Return the partition
    return files_input[start_idx:end_idx], files_output[start_idx:end_idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Run nnUNet prediction with custom arguments.")
    parser.add_argument('--pth', type=str, default='/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/', 
                        help="Path to input images folder")
    parser.add_argument('--outdir', type=str, default='/projects/bodymaps/Pedro/data/JHH_liver_segments/', 
                        help="Path to output folder for predictions")
    parser.add_argument('--checkpoint', type=str, 
                        default='./nnUNetOrgansAndSubSegments/', 
                        help="Path to model checkpoint folder")
    parser.add_argument('--num_parts', type=int, default=1, 
                        help="Number of parts to split the task into")
    parser.add_argument('--part_id', type=int, default=0, 
                        help="ID of the current part (0-indexed)")
    parser.add_argument('--gpu', type=int, default=0, 
                        help="ID of the gpu to be used")
    parser.add_argument('--workers', type=int, default=2, 
                        help="Number of worker processes for preprocessing and segmentation export")
    parser.add_argument('--BDMAP_format', action='store_true', 
                    help="Enable BDMAP format")
    parser.add_argument('--ids', default=None, help='Path to csv with a BDMAP ID column and the cases you want to inference')            
    parser.add_argument('--reset', action='store_true', 
                    help="Overwrites old cases")
    return parser.parse_args()


def filter_existing_outputs(files_input, files_output):
    """
    Filter out cases where the output file already exists.
    
    Parameters:
        files_input (list): A list of input file paths (or lists of file paths).
        files_output (list): A list of output file paths.
        
    Returns:
        tuple: Two lists, (filtered_files_input, filtered_files_output), where
               each pair corresponds to an output that does not already exist.
    """
    filtered_input = []
    filtered_output = []
    
    for inp, out in zip(files_input, files_output):
        if not os.path.exists(out+'.nii.gz'):
            filtered_input.append(inp)
            filtered_output.append(out)
        else:
            print(f"Skipping already saved output: {out}")
    
    return filtered_input, filtered_output

def main():
    args = parse_args()

    # Instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', args.gpu),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    if args.ids is not None:
        ids = pd.read_csv(args.ids)
        ids = sorted(ids['BDMAP ID'].to_list())
    else:
        ids = sorted(os.listdir(args.pth))

    # Collect input and output files
    if args.BDMAP_format:
        print('Path:',args.pth)
        print('Files in path:',len(ids))

        files_input = [[os.path.join(args.pth, folder, 'ct.nii.gz')] 
                        for folder in ids 
                        if ('BDMAP' in folder)]
        print('Input files before split:',len(files_input))
        files_output = [os.path.join(args.outdir, folder) 
                        for folder in ids 
                        if ('BDMAP' in folder)]
        files_input, files_output = split_files(files_input, files_output, args.num_parts, args.part_id)
    else:
        files_input = [[os.path.join(args.pth, file)] for file in ids if (file.endswith('.nii.gz') and file[0]!='.')]
        files_output = [os.path.join(args.outdir, file.replace('.nii.gz','')) for file in ids if (file.endswith('.nii.gz') and file[0]!='.')]
        files_input, files_output = split_files(files_input, files_output, args.num_parts, args.part_id)
        
    files_input, files_output = filter_existing_outputs(files_input, files_output)

    print('Input:', files_input[:10])
    print('Output:', files_output[:10])
    print(f'Cases to predict in part {args.part_id}: {len(files_input)}')


    # Initialize the network architecture and load the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(args.checkpoint),
        use_folds=('all',),
        checkpoint_name='checkpoint_final.pth',
    )

    # Run the prediction
    predictor.predict_from_files(
        files_input, files_output,
        save_probabilities=False,
        overwrite=args.reset,
        num_processes_preprocessing=args.workers,
        num_processes_segmentation_export=args.workers,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )


if __name__ == '__main__':
    main()