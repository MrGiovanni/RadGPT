import argparse
import os
import pandas as pd
import RadGPT_Style as rgpt

def process_reports(port, structured_report_path, narrative_report_path, output_path, parts=1, current_part=0):
    # Load structured and narrative reports
    #load structured reports with just 2 columns

    #load and clean structured reports
    try:
        structured_reports = pd.read_csv(structured_report_path, usecols=['Case', 'Report'])
    except:
        structured_reports = pd.read_csv(structured_report_path, usecols=['Case', ' Report'])
        structured_reports.rename(columns={' Report':'Report'}, inplace=True)
    structured_reports.drop_duplicates(subset=['Case'], inplace=True)
    #drop nan
    structured_reports.dropna(subset=['Case'], inplace=True)
    structured_reports.dropna(subset=['Report'], inplace=True)
    #remove any column where case does not start with BDMAP_ followed by 6 numbers
    structured_reports = structured_reports[structured_reports['Case'].str.match('BDMAP_\d{6}')]
    #order by case
    structured_reports.sort_values(by='Case', inplace=True)

    
    narrative_reports = pd.read_csv(narrative_report_path)

    # Define new column for processed reports
    new_col_name = "Generated_Report"

    # Check if the output file exists
    if not os.path.exists(output_path):
        # If not, create it and add the new column if necessary
        structured_reports[new_col_name] = None
        structured_reports.head(0).to_csv(output_path, index=False)  # Write headers only

    base_url = f'http://0.0.0.0:{port}/v1'

    ids=structured_reports['Case'].tolist()
    if parts>1:
        print('part split:', current_part, parts)
        #split the ids into parts (parts is the number of parts)
        ids=get_part(ids, parts, current_part)

    #remove from cases the cases already processes
    processed=pd.read_csv(output_path)
    processed_ids=processed['Case'].tolist()
    ids=[x for x in ids if x not in processed_ids]

    print('Already processed:', len(processed_ids))
    print(f"Processing {len(ids)} reports...")

    for idx, row in structured_reports.iterrows():
        id=row['Case']
        if id not in ids:
            continue
        # Skip rows where the new report already exists
        if pd.notnull(row.get(new_col_name, None)):
            print(f"Row {id} already processed. Skipping...")
            continue

        # Extract the structured report
        try:
            report = row['Report']  # Adjust the column name if necessary
        except:
            report = row[' Report']

        # Perform the style transfer
        try:
            print(f"Processing row {idx}, {id}")
            _, _, new_report = rgpt.style_transfer(narrative_reports, 10, report, max_tokens=None, base_url=base_url)

            # Append the new row with the generated report
            row[new_col_name] = new_report
            pd.DataFrame([row]).to_csv(output_path, mode='a', index=False, header=False)
            print(f"Row {idx} processed and saved.")

        except Exception as e:
            print(f"Error processing row {idx}: {e}")

    print(f"All reports processed. Output saved to {output_path}")

def get_part(lst, parts, current_part):
    # Calculate the size of each chunk
    parts_size = len(lst) // parts
    # Calculate the start and end index of the current part
    start = current_part * parts_size
    end = start + parts_size
    # If it is the last part, include the remaining elements
    if end>len(lst):
        end=len(lst)
    print('Start:', start, 'End:', end)
    return lst[start:end]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process structured radiology reports and create styled reports.")
    parser.add_argument("--port", type=int, required=True, help="Port number for the RadGPT server (e.g., 8000).")
    parser.add_argument("--structured_reports", type=str, required=True, help="Path to the structured reports CSV file.")
    parser.add_argument("--narrative_reports", type=str, required=True, help="Path to the narrative reports CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the processed output CSV file.")
    parser.add_argument("--parts", type=int, default=1, help="Number of parts to split the reports.")
    parser.add_argument("--current_part", type=int, default=0, help="Current part to process.")
    #restart arg
    parser.add_argument("--restart", type=bool, default=False, help="Restart the process from the beginning.")

    args = parser.parse_args()

    #if restart is True, delete the output file
    if args.restart:
        if os.path.exists(args.output):
            os.remove(args.output)

    process_reports(args.port, args.structured_reports, args.narrative_reports, args.output, args.parts, args.current_part)