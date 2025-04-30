"""
This code provides a function that goes through a folder with many word files, and organizes their content into a CSV.
"""

import os
import pandas as pd
from docx import Document
import re

def extract_text_to_csv(folder_path, output_csv):
    """
    Extracts text from .docx and .rtf files in a folder and saves it to a CSV file.
    Skips empty or corrupted files.
    
    Args:
        folder_path (str): Path to the folder containing the files.
        output_csv (str): Path to save the output CSV file.
    """
    data = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.docx') or file.endswith('.rtf'):
                file_path = os.path.join(root, file)
                
                # Extract text content
                text = ""
                try:
                    if file.endswith('.docx'):
                        # Try to open the .docx file
                        doc = Document(file_path)
                        text = "\n".join([paragraph.text for paragraph in doc.paragraphs]).strip()
                    elif file.endswith('.rtf'):
                        # Handle .rtf files
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                except Exception as e:
                    # Log the issue and skip the file
                    print(f"Skipping file due to error ({e}): {file_path}")
                    continue

                # Skip empty files
                if not text:
                    print(f"Skipping empty file: {file_path}")
                    continue

                # Process file name
                base_name = os.path.basename(file)
                cleaned_name = re.sub(r'.*_', '', base_name)  # Remove anything before and including '_'
                cleaned_name = os.path.splitext(cleaned_name)[0]  # Remove file extension

                # Append to data
                data.append({"Accession Number": cleaned_name, "Report Text": text})

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"Data successfully saved to {output_csv}")
    else:
        print("No valid files found. CSV not created.")
