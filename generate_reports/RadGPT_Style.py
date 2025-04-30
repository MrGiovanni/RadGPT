import transformers
import torch
import os
import pandas as pd
import numpy as np
import math
import re
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import random
from openai import OpenAI
import copy
import csv
from string import Template

instructions = Template("""
You are provided with a **structured radiology report** and ${n} other radiology reports that have different writing styles compared to the structured report.

**Task:**
Please **paraphrase** the structured report to match the writing style of the other reports. 

**Important Guidelines:**
1. **Do Not Alter Medical Information:** 
   - Do not change, add, or remove any medical details such as tumor measurements, types, or locations. You may remove HU values.
2. **Maintain Original Meaning:**
   - Ensure that the rephrased report conveys the same information as the original structured report.
3. **Match Writing Style:**
   - Adapt the language, tone, and structure to align with the provided example reports.
4. **Begin your report text with "#start" and finish it with "#end"**
5. **Provide justification**, go thorugh all medical findings in your rephrased report (e.g., tumor size, no evidence of metastasis) and show where the information comes from in the structured report. Justification should come after "#end".
6. **Pay attention to the Example Reports**: your writing style must be consistent with the examples. 
7. **Organization must match:** if the example have an Impressions and Results seciton, you must add them. If the example reports talks about all abdominal organs in a single paragraph, you must do it too. You may skip sections you cannot fill due to lack of information, like patient history. 
8. **Do not add new findings:** If the structured report does not mention the presence or absence of a medical condition (e.g., metastases), you must NOT include it in your rephrased report.
9. **Keep coherence:** avoid going back and forth between medical findings or organs. For example, do not talk about the size of a pancreatic tumor, then mention the liver, and then go back to pancreatic findings. Keep the information about each organ together.
10. **Always include an impressions section with the mpost important findings.**
                        
**Example of Rephrasing:**
- **Structured Report:** 
  "PDAC 1: Pancreatic body/tail. Hypoattenuating pancreas PDAC measuring 6.0 x 3.4 cm (centered on slice 356). Its mean HU value is 39.17 +/- 29.65, and its volume is 27.519 cm^3."
  
- **Paraphrased Report:** 
  "#start
  
  The patient has a pancreatic adenocarcinoma located in the body and tail of the pancreas, measuring 6.0 x 3.4 centimeters (image slice 356). The lesion is hypoattenuating and has a volume of 27.519 cm³.
  
  #end
  
  Justification:
  a. **Tumor Type:** Maintained as "pancreatic adenocarcinoma", originally "PDAC".
  b. **Location:** Preserved as "body and tail of the pancreas", originally "Pancreatic body/tail".
  c. **Measurements:** Kept as "6.0 x 3.4 centimeters", originally "measuring 6.0 x 3.4 cm".
  d. **Imaging Slice:** Retained as "image slice 356", originally "centered on slice 356".
  e. **Attenuation:** Maintained as "hypoattenuating", originally "Hypoattenuating pancreas PDAC".
  f. **Volume:** Kept as "27.519 cm³", originally "volume is 27.519 cm^3".
                        
  - **Note:** Removed mean HU value as per guidelines."

**Example Reports (Target Style):**
${examples}

**Structured Report to Paraphrase:**
${structured_report}
""")


system = ("You are a knowledgeable, efficient, and direct AI assistant, and an expert in radiology and radiology reports.")

def get_examples(df, n, tumor_locations):
    filtered_df = df.copy()  # Make a copy to avoid modifying the original dataframe

    # Define a mapping from tumor locations to column names
    tumor_column_map = {
        'liver': 'Liver Tumor',
        'kidney': 'Kidney Tumor',
        'pancreas': 'Pancreas Tumor'
    }

    for tumor_location in tumor_locations:
        if tumor_location == 'other':
            break
        if tumor_location == 'healthy':
            filtered_df = filtered_df[
                (filtered_df['Liver Tumor'] == 0.0) &
                (filtered_df['Pancreas Tumor'] == 0.0) &
                (filtered_df['Kidney Tumor'] == 0.0)
            ]
            break
        if tumor_location not in ['healthy', 'liver', 'kidney', 'pancreas', 'other']:
            raise ValueError('Invalid tumor location, must be one of "healthy", "liver", "kidney", "pancreas", or "other"')

        # Filter based on the current tumor location
        filtered_df = filtered_df[filtered_df[tumor_column_map[tumor_location]] == 1.0]

    #get n random examples
    examples = filtered_df.sample(n)
    examples =  examples['Report Text'].tolist()
    return examples

def create_prompt(df, n, structured_report):
    examples = get_examples(df, n, get_labels_structured(structured_report))
    prompt = instructions.substitute(n=n, examples='\n'.join([f'- Example {i}:\n{ex}' for i,ex in enumerate(examples,1)]), structured_report=structured_report)
    prompt= [{"role": "system", "content": system},
              {"role": "user", "content": prompt}]
    return prompt


def get_labels_structured(report):
    labels=[]
    if 'kidney lesion' in report or 'kidney tumor' in report:
        labels.append('kidney')
    if 'liver lesion' in report or 'liver tumor' in report:
        labels.append('liver')
    if 'pancreas lesion' in report or 'pancreas tumor' in report \
        or 'PDAC' in report or 'PNET' in report:
        labels.append('pancreas')
    return labels

def style_transfer(df, n, structured_report,max_tokens=None,base_url='http://0.0.0.0:8000/v1'):
    client,model_name=InitializeOpenAIClient(base_url)
    conver=create_prompt(df, n, structured_report)
    response=request_API(conver,model_name,client,max_tokens)
    answer = response.choices[0].message.content
    conver.append({"role": "assistant","content": [{"type": "text", "text": response.choices[0].message.content}]})
    report = answer.split("#start")[1].split("#end")[0]
    return conver, answer, report



merge_instructions = Template("""
You are provided with a CT scan **structured radiology report** and notes written by a radiologist, about the same CT scan.

Your task is to identify any information in the notes that is not already included in the structured report and add it to the appropriate sections of the report. Please follow these guidelines:

1. **Do not remove** any existing information from the structured report. However, you may improve the report's details using **only** relevant information from the notes.
2. **Avoid adding any new findings** not already mentioned in either the notes or the structured report.
3. **Maintain the report's structure**. Carefully place new information in the correct sections inside "FINDINGS", considering which organ the information mentions. For instance, if the notes mention "cirrhosis," add it to the **"Liver"** section under **"FINDINGS"**.
4. **Add new sections if necessary**. If the notes refer to an organ not covered in the structured report, create a new section for it. If the notes mention patient metadata (e.g., sex and age), you may add it to the beginning of the report.
5. **Update the IMPRESSION section if needed**. Besides the FINDINGS, include any critical information from the notes in the report's **IMPRESSION** section, summarizing or rephrasing it. Do not add new sections if the notes do not provide concrete information for them.
6. **Use consistent termonology**. If possible, make the terminology in the stentences you add to the report match the terminology in the original structured report.
7. **Begin your report text with "#start" and finish it with "#end"**
8. **Provide justification**, explain where in the report you added each piece of information from the notes. Also, explain why other information in the report was not removed or altered.
9. **Do not** write non-informative sentences such as "Patient metadata: Not available in the provided notes" or Sex: Not specified."
                              
The notes are as follows:
${cinical_info}

The current structured report is:
${structured_report}
""")

def create_prompt_add_info(structured_report, cinical_info):
    prompt = merge_instructions.substitute(cinical_info=cinical_info, structured_report=structured_report)
    prompt= [{"role": "system", "content": system},
              {"role": "user", "content": prompt}]
    return prompt

clt=None
mdl=None
def InitializeOpenAIClient(base_url='http://0.0.0.0:8000/v1'):
    global clt, mdl
    if clt is not None:
        return clt,mdl
    else:
        # Initialize the client with the API key and base URL
        clt = OpenAI(api_key='YOUR_API_KEY', base_url=base_url)

        # Define the model name and the image path
        mdl = clt.models.list().data[0].id# Update this with the actual path to your PNG image
        print('Initialized model and client.')
        return clt,mdl
    
def request_API(cv,model_name,client,max_tokens):
    print('Requesting API')

    if max_tokens is None:
        return client.chat.completions.create(
            model=model_name,
            messages=cv,
            temperature=0,
            top_p=1,
            timeout=6000)
    else:
        return client.chat.completions.create(
            model=model_name,
            messages=cv,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1,
            timeout=6000)
    
def add_info(structured_report, cinical_info, max_tokens=None,base_url='http://0.0.0.0:8000/v1'):
    client,model_name=InitializeOpenAIClient(base_url)
    conver=create_prompt_add_info(structured_report, cinical_info)
    response=request_API(conver,model_name,client,max_tokens)
    answer = response.choices[0].message.content
    conver.append({"role": "assistant","content": [{"type": "text", "text": response.choices[0].message.content}]})
    report = answer.split("#start")[1].split("#end")[0]
    return conver, answer, report

def iterate_add_info(structured_reports, clinical_notes, max_tokens=None,base_url='http://0.0.0.0:8000/v1',
                     prt=False,save_path='ImprovedReports.csv',id_col='BDMAP ID',report_col='structured report',notes_col='Notes'):
    #check if id_col is in the columns
    if id_col not in structured_reports.columns:
        #if Case in in the columns, rename it to id_col
        if 'Case' in structured_reports.columns:
            structured_reports=structured_reports.rename(columns={'Case': id_col})
        else:
            raise ValueError(f'ID column not found: {id_col}')
        
    #check if report_col is in the columns
    if 'narrative' not in report_col and report_col not in structured_reports.columns:
        #if ' Report' in the columns, rename it to report_col
        if ' Report' in structured_reports.columns:
            structured_reports=structured_reports.rename(columns={' Report': report_col})
        elif 'Report' in structured_reports.columns:
            structured_reports=structured_reports.rename(columns={'Report': report_col})
        else:
            raise ValueError(f'Report column not found: {report_col}')
        
    if 'narrative' in report_col and report_col not in structured_reports.columns:
        structured_reports=structured_reports.rename(columns={'Generated_Report': report_col})
        #print column names
        print(structured_reports.columns)

    #create empty col
    structured_reports['fusion '+report_col]=np.nan
    structured_reports['radiologist notes']=np.nan

    

    reports={}
    clinical_notes=clinical_notes.dropna(subset=[notes_col])
    for index, row in clinical_notes.iterrows():
        id=row[id_col]
        report = structured_reports.loc[structured_reports[id_col]==row[id_col]][report_col].values
        if len(report)==0:
            continue
        report=report[0]
        if not isinstance(report, str) or report=='nan' or report=='':
            continue
        #print(report)
        #[' Report']
        note=row[notes_col]
        if print: 
            print()  
            print('ID:', id)
            print('Note:', note)
            print('Original Report:', report)
        conver, answer, new_report = add_info(report, note, max_tokens,base_url)
        reports[id]={'original':report,
                     'improved':new_report}
        if print:
            print('Improved report:', new_report) 
        #save
        structured_reports.loc[structured_reports[id_col]==row[id_col],'fusion '+report_col]=new_report
        structured_reports.loc[structured_reports[id_col]==row[id_col],'radiologist notes']=note

    if save_path:
        structured_reports.to_csv(save_path)

    return structured_reports