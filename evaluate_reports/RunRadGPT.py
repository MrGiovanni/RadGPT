"""
This code runs the LLM on radiology or pathology reports. 
Example:
python RunRadGPT.py --port 8000 --data_path '/path/to/data/csv' --institution 'UCSF' --step 'type and size pathology' --save_name '/path/to/results/csv' --fast '0'
Inputs: see add_argument below.
"""


import argparse
import pandas as pd
import RadGPT as rgpt
import csv

def update_csv_header_if_wrong(csv_file, correct_header, output_file=None):
    """
    Check if the header of a CSV file is correct, and update it if wrong.
    
    Parameters:
    csv_file (str): Path to the input CSV file.
    correct_header (list): List containing the correct header values.
    output_file (str): Path to save the updated CSV file. If None, overwrites the original file.
    
    Returns:
    None
    """
    # Read the file and check the header
    with open(csv_file, 'r') as infile:
        reader = list(csv.reader(infile))
        current_header = reader[0]  # Get the current header

        # Check if the current header matches the correct header
        if current_header != correct_header:
            print("Header is incorrect, updating...")
            # Replace the incorrect header with the correct one
            reader[0] = correct_header
            
            # If output_file is None, overwrite the original file
            if output_file is None:
                output_file = csv_file
            
            # Write the updated content back to the file
            with open(output_file, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(reader)
            print(f"Header has been updated and saved to {output_file}.")
        else:
            print("Header is correct, no changes made.")

# Example usage
correct_header = ["Accession Number", "Liver Tumor", "Kidney Tumor", "Pancreas Tumor", 
                  "DNN answer", "Malignant Tumor in pancreas", "DNN answer 2"]

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run inference loop with RadGPT on an Excel file.')
    
    # Define arguments
    parser.add_argument('--port', required=True, help='Port for the RadGPT service (use the number you used in the vllm serve command, see readme)')
    parser.add_argument('--data_path', required=True, help='Path to csv file containing data (radiology or pathology reports)')
    parser.add_argument('--institution', required=True, help='Name of the institution (UCSF or COH)')
    parser.add_argument('--step', required=True, help='Step to execute in the inference loop. See readme. Can be: tumor detection/malignancy detection/malignant size/type and size pathology')
    parser.add_argument('--save_name', required=True, help='Path to save the output CSV file')
    parser.add_argument('--last_step_csv', default=None, help='Path to the last step CSV file')
    parser.add_argument('--list_to_run', default=None, help='Path to list of samples to evaluate (optional, if none is provided, it reads from the CSV)')
    parser.add_argument('--fast', default='1', help='Send 0 for slow mode, 1 for fast mode. Fast mode uses a smaller prompt, and may reduce accuracy.')
    parser.add_argument('--restart', default=False, action='store_true', help='Restart the inference loop, overwirtes the file in save_name')
    parser.add_argument('--parts', default='1', help='Optional. Use this to run multiple instances of RunRadGPT.py in parallel. Parts should be the number of instances.')
    parser.add_argument('--part', default='0', help='Optional. If running in parallel, each RunRadGPT.py should have a different part.')



    # Parse the arguments
    args = parser.parse_args()

    #load csv as dataframe
    if args.last_step_csv is not None:
        update_csv_header_if_wrong(args.last_step_csv, correct_header, args.last_step_csv)
        last_step_csv = pd.read_csv(args.last_step_csv)
    else:
        if args.step=='malignancy detection' or args.step=='malignant size':
            raise ValueError('last_step_csv must be provided for malignancy detection')
        last_step_csv = {}

    if args.list_to_run is not None:
        list_to_run = pd.read_csv(args.list_to_run)
        list_to_run=list_to_run['Accession Number'].tolist()
        if args.last_step_csv is not None:
            last_step_csv = last_step_csv[last_step_csv['Accession Number'].isin(list_to_run)]
        else:
            last_step_csv = list_to_run
    else:
        list_to_run = None


    # Construct the base URL using the provided port
    base_url = f'http://0.0.0.0:{args.port}/v1'

    if 'xlsx' in args.data_path:
        try:
            data = pd.read_excel(args.data_path, sheet_name=1)
        except:
            data = pd.read_excel(args.data_path)
    elif 'csv' in args.data_path:
        data = pd.read_csv(args.data_path)
    elif '.feather' in args.data_path:
        data = pd.read_feather(args.data_path)
    else:
        raise ValueError('Data file must be in .xlsx, .csv, or .feather format')

    # check if Anon Report Text is in columns
    if 'Anon Report Text' in data.columns:
        #drop na values in Anon Report Text
        data = data.dropna(subset=['Anon Report Text'])
    if 'Report' in data.columns:
        #drop na values in Report
        data = data.dropna(subset=['Report'])
    if ' Report' in data.columns:
        #drop na values in Report
        data = data.dropna(subset=[' Report'])

    #partition data
    length = len(data)
    parts = int(args.parts)
    if parts>1:
        part = int(args.part)
        part_length = length//parts
        start = part*part_length
        end = (part+1)*part_length
        if part==parts-1:
            data = data.iloc[start:]
        else:
            data = data.iloc[start:end]

    # Run the inference loop with the provided parameters
    outputs = rgpt.inference_loop(data, base_url=base_url, step=args.step, 
                                  institution=args.institution, 
                                  save_name=args.save_name,
                                  outputs=last_step_csv,
                                  fast=(args.fast=='1'), restart=args.restart,
                                  item_list=list_to_run)

    # Optionally print or save the result if needed
    print(f"Inference completed. Results saved to {args.save_name}")

if __name__ == "__main__":
    main()
