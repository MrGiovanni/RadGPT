"""
This code includes many functions to run the LLM on radiology or pathology reports.

inference_loop is the main function here. It will take as input the reports, call the LLM multiple times and write its outputs to a csv.
"""

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
from concurrent.futures import ThreadPoolExecutor
import csv
import ast
import time
import ast
from itertools import chain
import matplotlib.pyplot as plt
import tqdm

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

def CreateConversation(text, conver,role='user'):
    #if no previous conversation, send conver=[]. Do not automatically define conver above.
    cnv=copy.deepcopy(conver)
    
    cnv.append({
            'role': role,
            'content': [{
                'type': 'text',
                'text': text,
            }],
        })
    
    return cnv

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

def SendMessageAPI(text, conver, base_url='http://0.0.0.0:8000/v1',  
                    prt=True,max_tokens=None,
                    batch=1,
                    labels=None, id=None):
    """
    Sends a message to the LM deploy API.

    Args:
        text (str): The text message to send.
        conver (list): A list of conversation objects.
        base_url (str, optional): The base URL of the LM deploy API. Defaults to 'http://0.0.0.0:8000/v1'.
        size (int, optional): The size to resize the images to. Defaults to None.
        prt (bool, optional): Whether to print the images and conversation. Defaults to True.
        print_conversation (bool, optional): Whether to print the conversation. Defaults to False.
        max_tokens (int, optional): The maximum number of tokens in the completion response. Defaults to None.

    Returns:
        tuple: A tuple containing the updated conversation and the answer from the LM deploy API.
    """
    #if no previous conversation, send conver=[]. Do not automatically define conver above.
    client,model_name=InitializeOpenAIClient(base_url)

    if text is not None:
        if batch>1:
            for i in range(batch):
                #print('Batch:',i)
                #print('img_file_list:',img_file_list[i])
                #print('text:',text[i])
                #print('conver:',conver[i])
                conver[i]=CreateConversation(text=text[i], conver=conver[i])
        else:
            conver=CreateConversation(text=text, conver=conver)

    response=[]
    for i in range(batch):
        if batch==1:
            response=request_API(conver,model_name,client,max_tokens)
        else:
            # Use ThreadPoolExecutor to send both requests concurrently
            with ThreadPoolExecutor() as executor:
                # Map both the conversation and model name to each thread
                response = list(executor.map(request_API, conver, [model_name] * len(conver),[client] * len(conver),[max_tokens] * len(conver)))
        

    if batch==1:
        # Print the response
        answer = response.choices[0].message.content
        if prt:
            print('Conversation:')
            for item in conver:
                print(item['content'])
            if id is not None:
                print('ID:',id)
            print('Answer:',answer)
            if labels is not None:
                print('Labels:',labels)
        conver.append({"role": "assistant","content": [{"type": "text", "text": response.choices[0].message.content}]})
    else:
        answer=[]
        for i in range(batch):
            answer.append(response[i].choices[0].message.content)
            if prt:
                print('Conversation:')
                for item in conver[i]:
                    print(item['content'])
                
                if id is not None:
                    print('ID:',id[i])
                print('Answer:',answer[i])
                if labels is not None:
                    print('Labels:',labels[i])
            conver[i].append({"role": "assistant","content": [{"type": "text", "text": response[i].choices[0].message.content}]})

    return conver, answer











systemFastV0 = ("You are a knowledgeable, efficient, and direct AI assistant, and an expert in radiology reports."
    "Your answer should follow this template, "
    "substituting _ by 0 (indicating tumor absence), 1 (indicating tumor presence), or U (uncertain presence of tumor):"
    " liver tumor=_; kidney tumor=_; pancreas tumor=_")

system = ("You are a knowledgeable, efficient, and direct AI assistant, and an expert in radiology and radiology reports.")

instructions0ShotFastV0=("Instructions: Discover if a CT scan radiology report indicates the presence "
                   "of liver tumors, pancreas tumors or kidney tumors. Output binary labels for "
                   "each of these categories, where 1 indicates tumor presence, 0 tumor absence, and U uncertain presence of tumor. "
                   "Example: liver tumor presence=1; kidney tumor presence=U; pancreas tumor presence=0. "
                   "Answer with only the labels, do not repeat this prompt. ")

instructions0ShotFastLiverV1="""Instructions: Analyze a radiology report and answer the following questions. I want you to provide answers by filling a template, and not answering anything beyond this template.
1- Is any liver tumor present? 
Template--substitute _ by yes, no or uncertain: liver lesion presence=_;
Consider that: (a) 'unremarkable' means that an organ has no tumor; (b) organs not mentioned in the report have no tumor; (c) tumors may be described with many words, such as metastasis, tumor, lesion, mass, cyst, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding; (d) consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. Examples of lesions that are not tumors: ulcers, wounds, infections, inflammations, postinflammatory calcification, scars, renal calculi, nephrolithiasis, renal stones, or other diseases that are not tumors.
(e) Uncertany: You should answer uncertain if all findings if the report mentions abnormalities in the liver (e.g., hyperdensities or hypodensities), but says that they could not be characterized. Common words for uncertainty are: ill-defined, too small to characterize, and uncertain.
2- Is any of these types of liver tumor explicitly mentioned as present: Hepatic Hemangioma (HH), Focal Nodular Hyperplasia (FNH), Bile Duct Adenoma, Simple Liver Cyst (SLC), Hepatocellular Carcinoma (HCC), Cholangiocarcinoma (CCA), Hepatic Adenoma (HA), Mucinous Cystic Neoplasm (MCN)?
Template--substitute _ by yes, no, or uncertain (meaning the report explicitly indicates suspition for this type of lesion): HH=_; FNH=_; Bile Duct Adenoma=_; SLC=_; HCC=_; CCA=_; HA=_; MCN=_;
3- Is any malignant liver lesion present?
Template--substitute _ by yes, no or uncertain: malignant liver lesion presence=_;
Consider that: tumors are malignant if the report explictly mentions it, if it is growing (or reducing with cancer tratement), if it is cancer, or if it is an index lesion in an oncologic patient. HCC, CCA or matastasis are always malignant, HA, MCN are sometimes malignant, and HH, FNH, Bile Duct Adenoma, and SLC are benign. Remember that a patient may have both benign and malignant tumors.
4- What is the size of the largest malignant liver lesion in mm? 
Template: the template depends if the largest lesion is reported in 1D measurements (e.g., 15 mm), 2D measurements (e.g., 15 x 10 mm) or 3D measurements (e.g., 40 x 30 x 30 mm). ou may need to convert from cm to mm.
1D Template--substitute _ by the correct number (which should be 0 if there is no malignant liver tumor): largest malignant liver lesion size=_ mm;
2D Template--substitute _ by the correct number: largest malignant liver lesion size=_ x _ mm;
3D Template--substitute _ by the correct number: largest malignant liver lesion size=_ x _ x _ mm;
Consider that: (a) you may need to convert form cm to mm; (b) you must pay attention on which measurement refers to which lesion; (c) if tumor sizes are not informed for any malignant liver tumor, write "largest malignant liver lesion size=uncertain mm; (d) you should consider that the largest measured malignant lesion is the largest the patient has, unless the report explicitly says otherwise.
5- How many malignant tumors are present in the liver, if any?
Template--substitute _ by the correct number: number of malignant liver lesions=_
Consider that: (a) the number should be 0 if there is no malignant liver tumor; (b) write 'number of malignant lesions=uncertain' if the report mentions multiple lesions but does not count them.
6- How many tumors (benign and malignant) are present in the liver, if any?
Template--substitute _ by the correct number: number of liver lesions=_
Consider that: (a) the number should be 0 if there is no liver tumor; (b) write 'number of liver lesions=uncertain' if the report mentions multiple lesions but does not count them.
"""

instructions0ShotFastV2="""Instructions: Analyze a radiology report and answer the following questions. I want you to provide answers by filling a template, and not answering anything beyond this template.
1- Is any liver tumor present? 
Template--substitute _ by yes, no or uncertain: liver lesion presence=_;
Consider that: (a) 'unremarkable' means that an organ has no tumor; (b) organs not mentioned in the report have no tumor; (c) tumors may be described with many words, such as metastasis, tumor, lesion, mass, cyst, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding; (d) consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. Examples of lesions that are not tumors: ulcers, wounds, infections, inflammations, scars, renal calculi, nephrolithiasis, renal stones, or other diseases that are not tumors.
(e) Uncertainty: You should answer uncertain if the report mentions abnormalities in the liver (e.g., hyperdensities or hypodensities), but says that they could not be characterized. Common words for uncertainty are: ill-defined, too small to characterize, and uncertain.
If you answered no, skip the next liver questions (2-7) and go to the pancreas questions (do not include skipped questions in the template).
2- Is any of these types of liver tumor explicitly mentioned as present: Hepatic Hemangioma (HH), Focal Nodular Hyperplasia (FNH), Bile Duct Adenoma, Simple Liver Cyst (SLC), Hepatocellular Carcinoma (HCC), Cholangiocarcinoma (CCA), Hepatic Adenoma (HA), Mucinous Cystic Neoplasm (MCN)?
Template--substitute _ by yes, no, or uncertain (meaning the report explicitly indicates suspicion for this type of lesion): HH=_; FNH=_; Bile Duct Adenoma=_; SLC=_; HCC=_; CCA=_; HA=_; MCN=_;
3- Is any malignant liver lesion present?
Template--substitute _ by yes, no or uncertain: malignant liver lesion presence=_;
Consider that: tumors are malignant if the report explicitly mentions it, if it is growing (or reducing with cancer treatment), if it is cancer, or if it is an index lesion in an oncologic patient. HCC, CCA or metastasis are always malignant, HA, MCN are sometimes malignant, and HH, FNH, Bile Duct Adenoma, and SLC are benign. Remember that a patient may have both benign and malignant tumors.
4- What is the size of the largest malignant liver lesion, if any? 
Template: the template depends if the largest lesion is reported in 1D measurements (e.g., 15 mm), 2D measurements (e.g., 15 x 10 mm) or 3D measurements (e.g., 40 x 30 x 30 mm). You may write in cm or mm, as written in the report.
1D Template--substitute _ by the correct number (which should be 0 if there is no malignant liver tumor) and substitute unit by cm or mm: largest malignant liver lesion size=_ unit;
2D Template--substitute _ by the correct number and substitute unit by cm or mm: largest malignant liver lesion size=_ x _ unit;
3D Template--substitute _ by the correct number and substitute unit by cm or mm: largest malignant liver lesion size=_ x _ x _ unit;
Consider that: (a) you must pay attention to which measurement refers to which lesion; (b) you must check if the measurement if current or previous; (c) if tumor sizes are not informed for any malignant liver tumor, write "largest malignant liver lesion size=uncertain"; (d) you should consider that the largest measured malignant lesion is the largest the patient has, unless the report explicitly says otherwise.
5- What is the previous size of the lesion you mentioned in the last question, if any? 
Template--use the same structure as before (answer uncertain if this is not informed): previous size of the liver malignant lesion=_ unit; _ x _ unit; _ x _ x _ unit;
Consider that: Many reports have references to previous tumor sizes. For the largest current lesion, give me its previous size, if the report informs it. Pay close attention to past tenses, and adverbs like "previously" or "before", or references to dates. Analyze the syntax of the sentence to understand if the sizes are current or previous.
6- What is the size of the largest benign liver lesion, if any? 
Template--use the same structure as above: largest malignant liver lesion size=_ unit; _ x _ unit; _ x _ x _ unit;
7- How many malignant tumors are present in the liver, if any?
Template--substitute _ by the correct number: number of malignant liver lesions=_
Consider that: (a) the number should be 0 if there is no malignant liver tumor; (b) write 'number of malignant lesions=uncertain' if the report mentions multiple lesions but does not count them.
8- How many tumors (benign and malignant) are present in the liver, if any?
Template--substitute _ by the correct number: number of liver lesions=_
Consider that: (a) the number should be 0 if there is no liver tumor; (b) write 'number of liver lesions=uncertain' if the report mentions multiple lesions but does not count them.


### Pancreas Questions:
1- Is any pancreatic tumor present?
Template--substitute _ by yes, no or uncertain: pancreatic lesion presence=_;
Consider that: apply the same rules for identifying lesions in the pancreas. Use the list of tumors: Serous Cystadenoma (SCA), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET), Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC).
If you answered no, skip the next pancreas questions (2-7) and go to the kidney questions (do not include skipped questions in the template).
2- Is any of these types of pancreatic tumors explicitly mentioned as present: Serous Cystadenoma (SCA), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET), Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC)?
Template--substitute _ by yes, no, or uncertain: SCA=_; MCA=_; IPMN=_; SPN=_; PNET=_; PDAC=_; MCC=_;
3- Is any malignant pancreatic lesion present?
Template--substitute _ by yes, no or uncertain: malignant pancreatic lesion presence=_;
Remember: PDAC, MCC are always malignant, MCA, IPMN, SPN, and PNET may be malignant, and SCA is benign.
4- What is the size of the largest malignant pancreatic lesion? 
Template--use the same structure as for liver tumors: largest malignant pancreatic lesion size=_ unit; _ x _ unit; _ x _ x _ unit;
5- What is the previous size of the lesion you mentioned in the last question, if any? Many reports have references to previous tumor sizes. For the largest current lesion, give me its previous size, if the report informs it.
Template--use the same structure as before (answer uncertain if this is not informed): previous size of the malignant pancreas lesion=_ unit; _ x _ unit; _ x _ x _ unit;
6- What is the size of the largest benign pancreatic lesiont?
Template--use the same structure as for malignant lesions: largest benign pancreatic lesion size=_ unit; _ x _ unit; _ x _ x _ unit;
7- How many malignant tumors are present in the pancreas?
Template--substitute _ by the correct number: number of malignant pancreatic lesions=_
8- How many tumors (benign and malignant) are present in the pancreas?
Template--substitute _ by the correct number: number of pancreatic lesions=_

### Kidney Questions:
1- Is any kidney tumor present?
Template--substitute _ by yes, no or uncertain: kidney lesion presence=_;
Use the list of kidney tumors: Renal Oncocytoma (RO), Angiomyolipoma (AML), Simple Renal Cyst, Renal Cell Carcinoma (RCC), Transitional Cell Carcinoma (TCC), Wilms Tumor, Cystic Nephroma (CN), Multilocular Cystic Renal Neoplasm of Low Malignant Potential (MCRNLMP).
If you answered no, skip the next liver questions (2-7) and stop answering here  (do not include skipped questions in the template).
2- Is any of these types of kidney tumors explicitly mentioned as present: Renal Oncocytoma (RO), Angiomyolipoma (AML), Simple Renal Cyst, Renal Cell Carcinoma (RCC), Transitional Cell Carcinoma (TCC), Wilms Tumor, Cystic Nephroma (CN), Multilocular Cystic Renal Neoplasm of Low Malignant Potential (MCRNLMP)?
Template--substitute _ by yes, no, or uncertain: RO=_; AML=_; Simple Renal Cyst=_; RCC=_; TCC=_; Wilms Tumor=_; CN=_; MCRNLMP=_;
3- Is any malignant kidney lesion present?
Template--substitute _ by yes, no or uncertain: malignant kidney lesion presence=_;
Remember: RCC, TCC, Wilms Tumor are always malignant. RO, AML, Simple Renal Cyst are benign, and CN, MCRNLMP may be malignant.
4- What is the size of the largest malignant kidney lesion? 
Template--use the same structure as for liver tumors: largest malignant kidney lesion size=_ unit; _ x _ unit; _ x _ x _ unit;
5- What is the previous size of the lesion you mentioned in the last question, if any? Many reports have references to previous tumor sizes. For the largest current lesion, give me its previous size, if the report informs it.
Template--use the same structure as before (answer uncertain if this is not informed): previous size of the malignant kidney lesion=_ unit; _ x _ unit; _ x _ x _ unit;
6- What is the size of the largest benign kidney lesion?
Template--use the same structure as for malignant lesions: largest benign kidney lesion size=_ unit; _ x _ unit; _ x _ x _ unit;
7- How many malignant tumors are present in the kidneys?
Template--substitute _ by the correct number: number of malignant kidney lesions=_
8- How many tumors (benign and malignant) are present in the kidneys?
Template--substitute _ by the correct number: number of kidney lesions=_
"""


instructions0ShotFastCompact="""Instructions: Analyze a radiology report and answer the following questions for the liver, pancreas, and kidneys. I want you to provide answers by filling the template provided below for each organ (liver, pancreas, and kidneys), without answering anything beyond the template.
1- Is any tumor present in the liver, pancreas, or kidneys? 
Template--substitute _ by yes, no, or uncertain for each organ:
liver lesion presence=_; pancreatic lesion presence=_; kidney lesion presence=_;
Consider that: (a) 'unremarkable' means that an organ has no tumor; (b) organs not mentioned in the report have no tumor; (c) tumors may be described with many words, such as metastasis, tumor, lesion, mass, cyst, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic findings; (d) consider any lesion, hyperdensity, or hypodensity a tumor unless the report explicitly says it is something else (e.g., ulcers, infections, scars, renal calculi, nephrolithiasis, or other diseases).
(e) Uncertainty: Answer "uncertain" if abnormalities are present but could not be characterized (e.g., ill-defined, too small to characterize, or uncertain).
Stopping: If you answered no for all organs, stop answering here.

2- Are any of these specific types of tumors explicitly mentioned as present in the liver, pancreas, or kidneys? 
Template--substitute _ by yes, no, or uncertain for each tumor type:
Liver: HH=_; FNH=_; Bile Duct Adenoma=_; SLC=_; HCC=_; CCA=_; HA=_; MCN=_;  
Pancreas: SCA=_; MCA=_; IPMN=_; SPN=_; PNET=_; PDAC=_; MCC=_;  
Kidneys: RO=_; AML=_; Simple Renal Cyst=_; RCC=_; TCC=_; Wilms Tumor=_; CN=_; MCRNLMP=_;  

3- Is any malignant lesion present in the liver, pancreas, or kidneys?
Template--substitute _ by yes, no, or uncertain for each organ:
malignant liver lesion presence=_; malignant pancreatic lesion presence=_; malignant kidney lesion presence=_;
Consider that: (a) tumors are malignant if the report explicitly mentions it, if it is growing (or shrinking with cancer treatment), or if it is an index lesion in an oncologic patient; (b) specific tumors are always malignant (e.g., HCC, CCA, PDAC, MCC, RCC, TCC, Wilms Tumor, metastasis); (c) some tumors may be malignant (e.g., HA, MCN, MCA, IPMN, SPN, PNET, CN, MCRNLMP); and (d) benign tumors include HH, FNH, Bile Duct Adenoma, SLC, SCA, RO, AML, Simple Renal Cyst.

4- What is the size of the largest malignant lesion in each organ, in mm? 
Template--use the appropriate template based on whether the measurement is 1D, 2D, or 3D for each organ:
liver: largest malignant liver lesion size=_ mm / _ x _ mm / _ x _ x _ mm;
pancreas: largest malignant pancreatic lesion size=_ mm / _ x _ mm / _ x _ x _ mm;
kidneys: largest malignant kidney lesion size=_ mm / _ x _ mm / _ x _ x _ mm;
Consider that: (a) you may need to convert from cm to mm; (b) pay attention to which measurement refers to which lesion; (c) if sizes are not informed, write "uncertain"; (c) important: many reports have references to previous tumor sizes, you MUST ignore any previous measurements and consider only the largest lesion currently.

5- How many malignant tumors are present in the liver, pancreas, and kidneys?
Template--substitute _ by the correct number or uncertain for each organ:
number of malignant liver lesions=_; number of malignant pancreatic lesions=_; number of malignant kidney lesions=_;

6- How many tumors (benign and malignant) are present in the liver, pancreas, and kidneys?
Template--substitute _ by the correct number or uncertain for each organ:
number of liver lesions=_; number of pancreatic lesions=_; number of kidney lesions=_.
"""



instructions0Shot="""Carefully analyze the radiology report below, looking carefully at the findings and impressions sections (if available). Your task is answering the following questions:
1- Does the report indicate the presence of a liver tumor? Answer yes, no or it is uncertain. Justify your answer.
2- Does the report indicate the presence of a pancreas tumor? Answer yes, no or it is uncertain. Justify your answer.
3- Does the report indicate the presence of a kidney tumor? Answer yes, no or it is uncertain. Justify your answer.
After answering each of the 3 quesitons, fill in the following template, substituting _ by 'yes', 'no' or 'uncertain'. Do not change the template structure (e.g., keep using ; to separate the answers):
liver tumor presence=_; kidney tumor presence=_; pancreas tumor presence=_"""



instructions0ShotFast=("Instructions: Discover if the CT scan radiology report below indicates the presence "
                   "of liver tumors, pancreas tumors or kidney tumors. "
                   "Output labels for each of these categories, yes should indicate tumor presence, no tumor absence, and U uncertain presence of tumor. "
                   "Example: liver tumor presence=yes; kidney tumor presence=U; pancreas tumor presence=no. "
                   "Answer with only the labels, do not repeat this prompt. "
                   " Follow these rules for interpreting radiology reports: \n "
              "1- 'unremarkable' means that an organ has no tumor. \n "
              "2- Multiple words can be used to describe tumors, and you may check both the findings and impressions sections of the report (if present) to understand if an organ has tumors. Some words are: as metastasis, tumor, lesion, mass, cyst, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding"
              "3- Consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. "
              "Many conditions are not tumors, and should not be interpreted as so, unless a tumor is also reported along with the diease. Examples of liver conditions that are not tumors: Hepatitis, Cirrhosis, Fatty Liver Disease (FLD), Liver Fibrosis, Hemochromatosis, Primary Biliary Cholangitis (PBC), Primary Sclerosing Cholangitis (PSC), Wilson's Disease, Liver Abscess, Alpha-1 Antitrypsin Deficiency (A1ATD), steatosis, granulomas, Cholestasis, Budd-Chiari Syndrome (BCS), transplant, Gilbert's Syndromeulcers, wounds, infections, inflammations, and scars."
              "For the kidneys, some common conditions that are not tumors are: stents, inflammation, postinflammatory calcification, transplant, Chronic Kidney Disease (CKD), Acute Kidney Injury (AKI), Glomerulonephritis, Nephrotic Syndrome, Polycystic Kidney Disease (PKD), Pyelonephritis, Hydronephrosis, Renal Artery Stenosis (RAS), Diabetic Nephropathy, Hypertensive Nephrosclerosis, Interstitial Nephritis, Renal Tubular Acidosis (RTA), Goodpasture Syndrome, and Alport Syndrome. "
              "For the pancreas: Pancreatitis, Pancreatic Insufficiency, Cystic Fibrosis (CF), Diabetes Mellitus (DM), Exocrine Pancreatic Insufficiency (EPI), pancreatectomy, and Pancreatic Pseudocyst. \n"
              "Now some exmples of specific tumor names are: "
              "Liver: Hepatic Hemangioma (HH), Focal Nodular Hyperplasia (FNH), Bile Duct Adenoma, Simple Liver Cyst (SLC), Hepatocellular Carcinoma (HCC), Cholangiocarcinoma (CCA), Hepatic Adenoma (HA), Mucinous Cystic Neoplasm (MCN). \n "
                "Pancreas: Serous Cystadenoma (SCA), Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET). \n "
                "Kidney: Renal Oncocytoma (RO), Angiomyolipoma (AML), Simple Renal Cyst, Bosniak IIF cystic lesion, Renal Cell Carcinoma (RCC), Transitional Cell Carcinoma (TCC), Wilms Tumor, Cystic Nephroma (CN), Multilocular Cystic Renal Neoplasm of Low Malignant Potential (MCRNLMP), hydronephrosis, allograft. \n "
              "4- Consider any benign (e.g., cyst) or malingnat tumor a tumor. Thus, any type of cyst is a tumor. \n "
              "5- Organs never mentioned in the report have no tumors. \n "
                "6- Do not assume a lesion is uncertain unless it is explictly reported as uncertain. Many words can be used to describe uncertain lesions, such as: ill-defined, too small to characterize, nonspecific, and uncertain. Reports may express uncertainty about the tumor type (e.g., cyst or hemangioma), but certainty it is a tumor--in this case, you must consider the lesion a tumor. \n "
                "7- Organs with no tumor but other pathologies should be reported as 0.")
#100% accuracy!! but quite a few nans

instructionsLongitudinalPancreasAll=("Instructions: Below, I am providing you a sequence of radiology reports for CT scans of the same patient. They are numbered, as report 1, report 2, and so on. "
                         "Your task is to carefully read the reports, looking for pancreatic tumors (benign, malignant or cysts). "
                         "First, discover what is the first report that mentions a pancreatic tumor, if any. "
                         "Second, carefully read the reports before the first report that mentions a pancreatic tumor. From these reports, which ones do not mention any suspicious findings in the pancreas? Those are pre-diagnosis reports. \n"
                         "In your answer, fill out the following template: \n"
                         "first diagnosis report=_; pre-diagnosis reports=_;\n"
                         "An example is: first diagnosis report=3; pre-diagnosis reports=1,2;\n"
                         "If you find no pancreatic tumor in any report, answer: \n"
                         "An example is: first diagnosis report=none; pre-diagnosis reports=none;\n"
                         "If you find no pancreatic pre-diagnostic report (i.e., report 1 shows sign of a pancreatic tumor), answer pre-diagnosis reports=none\n"
                  " Follow these rules for interpreting radiology reports: \n "
             "1- 'unremarkable' means that an organ has no tumor. \n "
             "2- Multiple words can be used to describe tumors, and you may check both the findings and impressions sections of the report (if present) to understand if an organ has tumors. Some words are: as metastasis, tumor, lesion, mass, cyst, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding"
             "3- Consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. "
             "Many conditions are not tumors, and should not be interpreted as so, unless a tumor is also reported along with the disease. "
             "Example of conditions that are not tumors: Pancreatitis, Pancreatic Insufficiency, Cystic Fibrosis (CF), Diabetes Mellitus (DM), Exocrine Pancreatic Insufficiency (EPI), pancreatectomy, Pancreatic Pseudocyst, and fat infiltration. \n"
             "Now some examples of specific tumor names are: Serous Cystadenoma (SCA), Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET). \n "
             "4- Consider any benign (e.g., cyst) or malignant tumor a tumor. Thus, any type of cyst is a tumor. \n "
             "5- If the pancreas is not mentioned in a report, the report indicates no pancreatic tumors.\n"
             "6- No report AFTER the first diagnosis can be considered pre-diagnostic, even if they show no pancreatic tumor. Pre-diagnostic reports are only those before the first report showing a pancreas tumor. \n"
             "7- If no report mentions a pancreatic tumor, YOU MUST ANSWER 'none' for both first diagnosis and pre-diagnosis reports. \n"
             "8- If a report mentions a Whipple procedure or any pancreatic surgery, you CANNOT consider it a pre-diagnostic report. \n"
             "Justify your answer, explaining why the first diagnosis report has a pancreatic tumor, and why the pre-diagnosis reports do not have suspicion of pancreatic tumors. Cite sentences from the reports to support your justification.")


instructionsLongitudinalPancreas=("Instructions: Below, I am providing you a sequence of radiology reports for CT scans of the same patient. They are numbered, as report 1, report 2, and so on. "
                         "Your task is to carefully read the reports, looking for pancreatic tumors. "
                         "First, discover what is the first report that mentions a pancreatic tumor, if any. "
                         "Second, carefully read the reports before the first report that mentions a pancreatic tumor. Among these reports, which ones do not mention any suspicious findings in the pancreas? Those are pre-diagnosis reports. \n"
                         "In your answer, fill out the following template: \n"
                         "first diagnosis report=_; pre-diagnosis reports=_;\n"
                         "An example is: first diagnosis report=3; pre-diagnosis reports=1,2;\n"
                         "If you find no pancreatic tumor in any report, answer: \n"
                         "An example is: first diagnosis report=none; pre-diagnosis reports=none;\n"
                         "If you find no pancreatic pre-diagnostic report (i.e., report 1 shows sign of a pancreatic tumor), answer pre-diagnosis reports=none\n"
                  "Closely follow these rules, pay great attention to them: \n "
             "1- 'unremarkable' means that an organ has no tumor. If the pancreas is not mentioned in a report, the report indicates no pancreatic tumors. \n "
             "2- Multiple words can be used to describe tumors, and you may check both the findings and impressions sections of the report (if present) to understand if an organ has tumors. Some words are: as tumor, lesion, mass, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding"
             "3- Consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. "
             "Many conditions are not tumors, and should not be interpreted as so, unless a tumor is also reported along with the disease. "
             "Example of conditions that are not tumors: Pancreatitis, Pancreatic Insufficiency, Cystic Fibrosis (CF), Diabetes Mellitus (DM), Exocrine Pancreatic Insufficiency (EPI), pancreatectomy, Pancreatic Pseudocyst, and fat infiltration. \n"
             "4- No report AFTER the first diagnosis can be considered pre-diagnostic, even if they show no pancreatic tumor. Pre-diagnostic reports are only those before the first report showing a pancreas tumor. \n"
             "5- If no report mentions a pancreatic tumor, YOU MUST ANSWER 'none' for both first diagnosis and pre-diagnosis reports. \n"
             "6- If a report mentions a Whipple procedure, pancreatectomy, or any pancreatic surgery, you CANNOT consider it a pre-diagnostic report. \n"
             "7- If a report mentions pancreatic duct dilation, suspicion of a pancreatic tumor or focal pancreatic atrophy, you CANNOT consider it a pre-diagnostic report. \n\n"
             "Justify your answer, explaining why the first diagnosis report has a pancreatic tumor, and why the pre-diagnosis reports do not have suspicion of pancreatic tumors. Cite sentences from the reports to support your justification.")



instructionsLongitudinalPancreasDiagnosis=("Instructions: Below, I am providing you a sequence of radiology reports for CT scans of the same patient. They are numbered, as report 1, report 2, and so on. "
                         "Your task is to carefully read the reports, looking for pancreatic tumors, and discover the types of pancreatic tumors the patient had. "
                         "Use the tumor locations (pancreas head, body and tail) to track the tumors through time. This information can help you discover if a tumor type was unknown in a previous report, but clarified in a later one. \n"
                         "You must list all types of pancreatic tumors the patient had in the reports. \n"
                         "Answer by filling out the following template: \n"
                         "tumor types: _; _; _; \n"
                         "You should list all tumor types you find, separating them by semicolons. You will not necessairly find 3 types, ajust the number of semicolons to the number of tumor types you find. \n"
                         "Example: \n"
                         "tumor types: PDAC; Cyst; Unknown; \n"
                         "Only say Unknown if a the type of one or more tumors are not mentioned in any of the reports. \n"
                         "An example is: first diagnosis report=3; pre-diagnosis reports=1,2;\n"
                         "If you find no pancreatic tumor in any report, answer: \n"
                         "tumor types: none;\n"
                  "Closely follow these rules for interpreting report, pay great attention to them: \n "
             "1- In your answer, try using the abbreviations in the list below:"
             "Unknown, Cyst, Metastasis, Serous Cystadenoma (SCA), Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET)."
             "1- If a report says the pancreas is 'unremarkable', it means that it has no tumor. If the pancreas is not mentioned in a report, the report indicates no pancreatic tumors. \n "
             "2- Multiple words can be used to describe tumors, and you may check both the findings and impressions sections of the report (if present) to understand if an organ has tumors. Some words are: as tumor, lesion, mass, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding"
             "3- Consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. "
             "4- NEVER guess. If a tumor type is not explicitly mentioned in any report, you must say Unknown. \n"
             "5- Remember all reports belong to the same parient and appear in chronological order. Try your best to track the tumors through time, to discover their types if possible. \n"
             "Many conditions are not tumors, and should not be interpreted as so, unless a tumor is also reported along with the disease. "
             "Example of conditions that are not tumors: Pancreatitis, Pancreatic Insufficiency, Cystic Fibrosis (CF), Diabetes Mellitus (DM), Exocrine Pancreatic Insufficiency (EPI), pancreatectomy, Pancreatic Pseudocyst, and fat infiltration. \n"
             "ALWAYS CAREFULLY JUSTIFY your answer, explain from which sentences in the report you deduced the tumor type. Also, explain how do you trace each pancreatic tumor across diverse reports, and what were their types in each report. Explain which pancreatic tumors have unknown types (across all reports), if any. Cite sentences from the reports to support your justification.")


preDiagnositcConfirmation=("Instructions: Below, I am providing you a radiology report for a CT scan. "
                         "Your task is to carefully read the report and identify if it presents any tumor suspicion in the pancreas, or a pancreas surgery. "
                         "In your answer, fill out the following template: \n"
                         "pancreatic tumor suspicion=_; pancreas surgery=_;\n"
                         "Replace _ by yes or no. \n"
                  "Closely follow these rules, pay great attention to them: \n "
             "1- 'unremarkable' means that an organ has no tumor suspicion. \n" 
             "2- If the pancreas is not mentioned in a report, the report indicates no pancreatic tumor suspicion. \n "
             "3- If the report mentions pancreatic duct dilation or focal pancreatic atrophy, answer pancreatic tumor suspicion=yes; ... \n"
             "4- Multiple words can be used to describe tumors, and you may check both the findings and impressions sections of the report (if present) to understand if the pancreas has tumor suspicions. Some words are: as tumor, lesion, mass, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding"
             "5- Consider any cyst, lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. In such cases, answer pancreatic tumor suspicion=yes; ... \n"
             "Many conditions are not tumors, and should not be interpreted as so, unless a tumor is also reported along with the disease. "
             "Example of conditions that are not tumors: Pancreatitis, Pancreatic Insufficiency, Cystic Fibrosis (CF), Diabetes Mellitus (DM), Exocrine Pancreatic Insufficiency (EPI), pancreatectomy, Pancreatic Pseudocyst, and fat infiltration. \n"
             "6- Examples of pancreas surgery are Whipple procedure, pancreatectomy and Pancreaticoduodenectomy. \n"
             "Justify your answer explaining why the report shows suspicion of pancreatic tumors or not. Cite sentences from the reports to support your justification.")


instructions0ShotMalignancyFast=("Instructions: The radiology report below mentions a %(organ)s tumor (or tumors). Read it carefully, paying special attention to the findings, clinical history and impressions sections (if available). \n"
                     "Does the report mention any malignant tumor in the %(organ)s? \n"
                   "Answer me by just filling out the template below, substituting _ by yes (malignancy present), no (malignancy absence), or U (uncertain presence of malignancy): \n "
                   "malignant tumor in %(organ)s=_ \n"
                   "Answer with only the filled template, do not repeat this prompt. "
                   " Follow these rules for interpreting radiology reports: \n "
                    "1- Some words are only used for describing malignant tumors, for example: metastasis, cancer, growing, or any oncologic lesion and index lesion in cancer patients. \n"
                    "Reports may sometimes mention the specific tumor type. In the %(organ)s, benign tumors are: %(benign_tumors)s. \n "
                    "Malignant tumors are: %(malignant_tumors)s. \n "
                    "Tumors that may be both benign or malignant are: %(both_tumors)s. \n "
                    "2- If the report does not mention that the tumor is benign or does not specify lesion type, but the tumor is growing in relation to a past measurement (with no benign explanation--e.g., cysts can grow and are benign), consider it malignant. \n"
                    "3- If the report impressions explicitly state no abnormality in the %(organ)s or abdomen, assume no malignancy. \n"
                    "4- It the report does not mention the tumor type, but you read that the patient has cancer in the %(organ)s, or has history of malignant tumors in the %(organ)s (analyze the clinical history or finginds sections if they are present), consider the tumor malignant. \n")
#if they do not say it is malignant but it is growing, it is malignant, or it is a cancer patient or it is an index lesion and not specifically benign, it is malignant


instructions0ShotMalignancy=("Instructions: The radiology report below mentions a %(organ)s tumor (or tumors). Read it carefully, paying special attention to the findings, clinical history and impressions sections (if available). \n"
                     "Does the report mention any malignant tumor in the %(organ)s? \n"
                   "Answer me by just filling out the template below, substituting _ by yes (malignancy present), no (malignancy absence), or U (uncertain presence of malignancy): \n "
                   "malignant tumor in %(organ)s=_ \n"
                   "Besides filling the template, justify your answer, carefully mentioning each section of the report if present: clinical history, findings and impressions. "
                   " Follow these rules for interpreting radiology reports: \n "
                    "1- Some words are only used for describing malignant tumors, for example: metastasis, cancer, growing, or any oncologic lesion and index lesion in cancer patients. \n"
                    "Reports may sometimes mention the specific tumor type. In the %(organ)s, benign tumors are: %(benign_tumors)s. \n "
                    "Malignant tumors are: %(malignant_tumors)s. \n "
                    "Tumors that may be both benign or malignant are: %(both_tumors)s. \n "
                    "2- If the report does not mention that the tumor is benign or does not specify lesion type, but the tumor is growing in relation to a past measurement (with no benign explanation--e.g., cysts can grow and are benign), consider it malignant. \n"
                    "3- If the report impressions explicitly state no abnormality in the %(organ)s or abdomen, assume no malignancy. \n"
                    "4- It the report does not mention the tumor type, but you read that the patient has cancer in the %(organ)s, or has history of malignant tumors in the %(organ)s (analyze the clinical history or finginds sections if they are present), consider the tumor malignant. \n")

instructions0ShotMalignantSize=("Instructions: The radiology report below mentions a malignant tumor (or tumors) in the %(organ)s. "
                    "Read it carefully, paying special attention to the findings, clinical history and impressions sections (if available). \n"
                    "Your task is to list the sizes and locations of all malignant tumors in the %(organ)s. \n"
                    "Fill out the template below, using one line per malignant tumor in the %(organ)s (you may add or remove lines from the template). Substitute the first _ in each line by the the tumor size, and the second by its location: \n "
                    "%(organ)s malignant tumor size = _; location = _;\n"
                    "%(organ)s malignant tumor size = _; location = _;\n"
                    "... \n"
                    "Reports can write the size of the tumor in 1D, 2D or 3D measurements, and you should use the same standards used in the report."
                    "Write 1D measurements as: 15 mm; 2D measurements as: 15 x 10 mm; and 3D measurements as: 40 x 30 x 30 mm. You may use either cm or mm, but you MUST WRITE in each line of the filled template the unit you are using (cm or mm). If a report does not specify the unit, assume it is mm. \n"
                    "For location, chose one of these options for each tumor: %(organ_locations)s \n"
                    "You can use location=U if the report does not specify the location of the tumor. \n"
                    "Besides filling the template, justify your answer, carefully mentioning each section of the report if present: clinical history, findings and impressions.\n"
                    "Some report may refer to past measurements (using words like previously, before, or giving dates). Ignore previous measuremtns. Provide me a synthatic analysis of the report sentences mentioning %(organ)s tumor sizes. In this analysis, explain which measurement refers to which tumor, if the measurement is current or past, and if the corresponding tumor is malignant or benign.\n"
                    "Follow these rules for interpreting radiology reports: \n "
                    "1- Some words are only used for describing malignant tumors, for example: metastasis, cancer, growing, or any oncologic lesion and index lesion in cancer patients. \n"
                    "Reports may sometimes mention the specific tumor type. In the %(organ)s, benign tumors are: %(benign_tumors)s. \n "
                    "Malignant tumors are: %(malignant_tumors)s. \n "
                    "Tumors that may be both benign or malignant are: %(both_tumors)s. \n "
                    "2- If the report does not mention that the tumor is benign or does not specify lesion type, but the tumor is growing in relation to a past measurement (with no benign explanation--e.g., cysts can grow and are benign), consider it malignant. \n"
                    "3- If the report impressions explicitly state no abnormality in the %(organ)s or abdomen, assume no malignancy. \n"
                    "4- If the report does not mention the tumor type, but you read that the patient has cancer in the %(organ)s, or has history of malignant tumors in the %(organ)s (analyze the clinical history or finginds sections if they are present), consider the tumor malignant. \n"
                    "5- If the report mentions multiple malignant tumors in the %(organ)s, list the sizes of all of them. \n"
                    "6- If the report does not mention a certain tumor size, write 'U' to indicate uncertain (e.g., malignant tumor 2 = U). \n")

instructions0ShotSizenType = (
    "Instructions: The radiology report below mentions one or more tumors in the %(organ)s. "
    "Read it carefully, paying special attention to the findings, clinical history, and impressions sections (if available). \n"
    "Your task is to list the types, certainty of tumor type, sizes, and locations of all tumors in the %(organ)s. \n"
    "Fill out the template below, using one line per tumor in the %(organ)s (you may add or remove lines from the template): \n"
    "%(organ)s tumor 1: type = _; certainty = _; size = _; location = _;\n"
    "%(organ)s tumor 2: type = _; certainty = _; size = _; location = _;\n"
    "... \n"
    "If the report mentions no tumor, use the following template: \n"
    "no tumor found. \n"
    "\nSize: "
    "Reports can write the size of the tumor in 1D, 2D, or 3D measurements, and you should use the same standards used in the report. "
    "Write 1D measurements as: 15 mm; 2D measurements as: 15 x 10 mm; and 3D measurements as: 40 x 30 x 30 mm. You may use either cm or mm, but you MUST WRITE in each line of the filled template the unit you are using (cm or mm). If a report does not specify the unit, assume it is mm. \n"
    "Say size = U if the report does not specify the size of a tumor. \n"
    "\nLocation: "
    "For location, choose one of these options for each tumor: %(organ_locations)s \n"
    "Say location = U if the report does not specify the location of a tumor. \n"
    "\nType: "
    "If the report informs tumor type, inform it. Tumor type list: %(benign_tumors)s, %(malignant_tumors)s, %(both_tumors)s. \n"
    "Otherwise, say type = U if the report does not specify the type of the tumor. \n"
    "Follow these rules:"
    "1- If the tumor type is not specified in findings, you may deduce it from the clinical history or impressions sections. \n"
    "2- Cyst, or cystic lesion, is a common type of tumor. For any cysts, you say type = cyst. \n"
    "3- Assign type = metastasis if the %(organ)s tumor is described as a metastasis originating from a cancer in another organ. If the report mentions metastatic cancer in **another organ** along with lesions in the %(organ)s, classify the %(organ)s lesions as type = metastasis unless explicitly stated otherwise. Determine certainty based on the level of confidence expressed in the report. \n"
    "4- If the report does not mention the tumor type, but you read that the patient has cancer in the %(organ)s, or has a history of malignant tumors in the %(organ)s, you may say type = cancer. However, do not say type = cancer if a more specific tumor type is given (like cyst, PDAC and PNET in pancreas, RCC in kidney, HCC in liver,...). \n"
    "5- Try using terms from the tumor type list. E.g., if the list says 'Pancreatic Ductal Adenocarcinoma (PDAC)' and the report mentions 'adenocarcinoma in the pancreas', say type = Pancreatic Ductal Adenocarcinoma (PDAC). \n"
    "%(extra_info)s"
    "\nCertainty: "
    "Certainty of the tumor type, according to the report. If a report mentions a tumor type in the findings, history, or impressions, without demonstrating uncertainty, say certainty = certain. "
    "If the report expresses strong confidence in tumor type, say certainty = high. "
    "If the report mentions a tumor type but expresses significant uncertainty about it, say certainty = low. "
    "If the report does not mention the tumor type, say certainty = U. \n"
    "\nJustification: "
    "Besides filling the template, justify your answer, carefully mentioning each section of the report if present: history, findings, and impressions. "
    "Explain from which sentences you got each size, location, and type.\n"
    "Some reports may refer to past measurements (using words like previously, before, or giving dates). Ignore previous measurements. "
    "Provide me a syntactic analysis of the report sentences mentioning %(organ)s tumor sizes. In this analysis, explain which measurement refers to which tumor, if the measurement is current or past, and if the corresponding tumor is malignant or benign.\n"
    "I will provide an example of a report and a correct answer for a %(organ)s tumor:\n"
    "Example report: \n"
    "%(example_report)s \n"
    "Example answer: \n"
    "%(example_answer)s \n"
    "\n"
    "End of the example. \n"
    "\n"
)


instructionsHCC = (
    "Instructions: The radiology report below may mention one or more tumors in the liver. "
    "Read it carefully, paying special attention to the findings, clinical history, and impressions sections (if available). \n"
    "Your task is to list the types, certainty of tumor type, sizes, locations, arterial enhancement, washout, capsule, threshold growth, and LI-RADS score of all tumors in the liver. \n"
    "Fill out the template below, using one line per tumor in the liver (you may add or remove lines from the template): \n"
    "liver tumor 1: type = _; certainty = _; size = _; location = _; arterial enhancement = _; washout = _; capsule = _; threshold growth = _; LI-RADS = _;\n"
    "liver tumor 2: type = _; certainty = _; size = _; location = _; arterial enhancement = _; washout = _; capsule = _; threshold growth = _; LI-RADS = _;\n"
    "... \n"
    "Disregard tumors only if none of the following details are provided: size, location, or type. \n"
    "\nSize: "
    "Reports can write the size of the tumor in 1D, 2D, or 3D measurements, and you should use the same standards used in the report. "
    "Write 1D measurements as: 15 mm; 2D measurements as: 15 x 10 mm; and 3D measurements as: 40 x 30 x 30 mm. You may use either cm or mm, but you MUST WRITE in each line of the filled template the unit you are using (cm or mm). If a report does not specify the unit, assume it is mm. \n"
    "Say size = U if the report does not specify the size of a tumor. \n"
    "\nLocation: "
    "For location, choose one of these options for each tumor: %(organ_locations)s \n"
    "Say location = U if the report does not specify the location of a tumor. \n"
    "\nType: "
    "If using the LI-RADS classification, the rpeort is evaluating for HCC, so HCC is a probable type. \n"
    "If the report informs tumor type, inform it. Tumor type list: %(benign_tumors)s, %(malignant_tumors)s, %(both_tumors)s. \n"
    "Otherwise, say type = U if the report does not specify the type of the tumor. \n"
    "Follow these rules:"
    "1- If the tumor type is not specified in findings, you may deduce it from the clinical history or impressions sections. \n"
    "2- Cyst, or cystic lesion, is a common type of tumor. For any cysts, you say type = cyst. \n"
    "3- Assign type = metastasis if the liver tumor is described as a metastasis originating from a cancer in another organ. If the report mentions metastatic cancer in **another organ** along with lesions in the liver, classify the liver lesions as type = metastasis unless explicitly stated otherwise. Determine certainty based on the level of confidence expressed in the report. \n"
    "4- If the report does not mention the tumor type, but you read that the patient has cancer in the liver, or has a history of malignant tumors in the liver, you may say type = cancer. However, do not say type = cancer if a more specific tumor type is given (like cyst, PDAC and PNET in pancreas, RCC in kidney, HCC in liver,...). \n"
    "5- Try using terms from the tumor type list. E.g., if the list says 'Pancreatic Ductal Adenocarcinoma (PDAC)' and the report mentions 'adenocarcinoma in the pancreas', say type = Pancreatic Ductal Adenocarcinoma (PDAC). \n"
    "%(extra_info)s"
    "\nCertainty: "
    "Certainty of the tumor type, according to the report. If a report mentions a tumor type in the findings, history, or impressions, without demonstrating uncertainty, say certainty = certain. "
    "If the report expresses strong confidence in tumor type, say certainty = high. "
    "If the report mentions a tumor type but expresses significant uncertainty about it, say certainty = low. "
    "If the report does not mention the tumor type, say certainty = U. \n"
    "\nArterial enhancement: "
    "Arterial phase hyperenhancement (APHE) is non-rim arterial hyperenhancement of a lesion which is greater than the enhancement of the surrounding liver. \n"
    "It is a main consideration in the LI-RADS classification, and should me mentioned explicitly. \n"
    "If the report does not mention the arterial enhancement, say arterial enhancement = U. \n"
    "\nWashout: "
    "Non-peripheral washout is a decrease in attenuation or intensity from earlier to later phase, resulting in hypoenhancement in the portal venous or delayed phase. \n"
    "It is a main consideration in the LI-RADS classification, and should me mentioned explicitly. \n"
    "If the report does not mention the washout, say washout = U. \n"
    "\nCapsule: "
    "Capsule is a smooth, uniform border surrounding all or most of an observation. \n"
    "It is a main consideration in the LI-RADS classification, and should me mentioned explicitly. \n"
    "If the report does not mention the capsule, say capsule = U. \n"
    "\nThreshold growth: "
    "Threshold growth is increase in lesion size of 50 or more within 6 months time during follow-up imaging. \n"
    "It is a main consideration in the LI-RADS classification, and should me mentioned explicitly. \n"
    "If the report does not mention the threshold growth, say threshold growth = U. \n"
    "\nLI-RADS score: "
    "LI-RADS score is a classification of the likelihood of malignancy of a HCC tumor, based on the size, location, and enhancement of the tumor. \n"
    "It should be mentioned explicitly. \n"
    "If the report does not mention the LI-RADS score, say LI-RADS = U. \n"
    "\nJustification: "
    "Besides filling the template, justify your answer, carefully mentioning each section of the report if present: history, findings, and impressions. "
    "Explain from which sentences you got each size, location, type, arterial enhancement, washout, capsule, threshold growth, and LI-RADS score.\n"
    "Some reports may refer to past measurements (using words like previously, before, or giving dates). Ignore previous measurements. "
    "Provide me a syntactic analysis of the report sentences mentioning liver tumor sizes, types, arterial enhancement, washout, capsule, threshold growth, and LI-RADS score. In this analysis, explain which measurement refers to which tumor, if the measurement is current or past, and if the corresponding tumor is malignant or benign.\n"
    "I will provide an example of a report and a correct answer for a liver tumor:\n"
    "Example report: \n"
    "%(example_report)s \n"
    "Example answer: \n"
    "%(example_answer)s \n"
    "\n"
    "End of the example. \n"
    "\n"
    "Now the report you should analyze: \n"
)

reportHCC = """
#Private.
"""

answerHCC = """
#Private.
"""

instructions0ShotSizenTypePathology = (
    "Instructions: The pathology report below mentions one or more tumors in the %(organ)s. "
    "Read it carefully, paying special attention to the Final Diagnosis section, which is usually in the beginning of the report. \n"
    "Your task is to list the types, certainty of tumor type, sizes, and locations of all tumors in the %(organ)s. \n"
    "Fill out the template below, using one line per tumor in the %(organ)s (you may add or remove lines from the template): \n"
    "%(organ)s tumor 1: type = _; certainty = _; size = _; location = _;\n"
    "%(organ)s tumor 2: type = _; certainty = _; size = _; location = _;\n"
    "... \n"
    "If the report mentions no tumor, use the following template: \n"
    "no tumor found. \n"
    "\nSize: "
    "Reports can write the size of the tumor in 1D, 2D, or 3D measurements, and you should use the same standards used in the report. "
    "Write 1D measurements as: 15 mm; 2D measurements as: 15 x 10 mm; and 3D measurements as: 40 x 30 x 30 mm. You may use either cm or mm, but you MUST WRITE in each line of the filled template the unit you are using (cm or mm). If a report does not specify the unit, assume it is mm. \n"
    "Say size = U if the report does not specify the size of a tumor. \n"
    "\nLocation: "
    "For location, choose one of these options for each tumor: %(organ_locations)s \n"
    "Say location = U if the report does not specify the location of a tumor. \n"
    "\nType: "
    "If the report informs tumor type, inform it. Tumor type list: %(benign_tumors)s, %(malignant_tumors)s, %(both_tumors)s. \n"
    "Otherwise, say type = U if the report does not specify the type of the tumor. \n"
    "Follow these rules:"
    "1- Cyst, or cystic lesion, is a common type of tumor. For any cysts, you say type = cyst. \n"
    "2- Assign type = metastasis if the %(organ)s tumor is described as a metastasis originating from a cancer in another organ. If the report mentions metastatic cancer in **another organ** along with lesions in the %(organ)s, classify the %(organ)s lesions as type = metastasis unless explicitly stated otherwise. Determine certainty based on the level of confidence expressed in the report. \n"
    "3- Try using terms from the tumor type list. E.g., if the list says 'Pancreatic Ductal Adenocarcinoma (PDAC)' and the report mentions 'adenocarcinoma in the pancreas', say type = Pancreatic Ductal Adenocarcinoma (PDAC). \n"
    "%(extra_info)s"
    "\nCertainty: "
    "Certainty of the tumor type, according to the report. If a report mentions a tumor type without demonstrating uncertainty, say certainty = certain. "
    "If the report expresses strong confidence in tumor type, say certainty = high. "
    "If the report mentions a tumor type but expresses significant uncertainty about it, say certainty = low. "
    "If the report does not mention the tumor type, say certainty = U. \n"
    "\nJustification: "
    "Besides filling the template, justify your answer, carefully mentioning sentences of the report. "
    "Explain from which sentences you got each size, location, and type.\n"
    "Provide me a syntactic analysis of the report sentences mentioning %(organ)s tumor sizes. In this analysis, explain which measurement refers to which tumor, if the measurement is current or past, and if the corresponding tumor is malignant or benign.\n"
    "I will provide an example of a report and a correct answer for a %(organ)s tumor:\n"
    "Example report: \n"
    "%(example_report)s \n"
    "Example answer: \n"
    "%(example_answer)s \n"
    "\n"
    "End of the example. \n"
    "\n"
)



pathologyReportPancreas="""
#Private.
"""

pathologyReportPancreasAnswer="""
#Private.
"""


instructions0ShotSizenTypeMultiOrgan = (
    "Instructions: The radiology report below possibly mentions one or more tumors or cysts. "
    "Read it carefully, paying special attention to the findings, clinical history, and impressions sections (if available). \n"
    "Your task is to list the types, certainty of tumor type, sizes, organ, locations and attenuation of all tumors in the report. \n"
    "Fill out the template below, using one line per tumor (you may add or remove lines from the template): \n"
    "tumor 1: type = _; certainty = _; size = _; organ = _; location = _; attenuation = _; \n"
    "tumor 2: type = _; certainty = _; size = _; organ = _; location = _; attenuation = _;\n"
    "... \n"
    "If the report mentions no tumor, use the following template: \n"
    "no tumor found. \n"
    "Consider the following instructions: \n"
    "\nSize: "
    "Reports can write the size of the tumor in 1D, 2D, or 3D measurements, and you should use the same standards used in the report. "
    "Write 1D measurements as: 15 mm; 2D measurements as: 15 x 10 mm; and 3D measurements as: 40 x 30 x 30 mm. You may use either cm or mm, but you MUST WRITE in each line of the filled template the unit you are using (cm or mm). If a report does not specify the unit, assume it is mm. \n"
    "Say size = U if the report does not specify the size of a tumor. \n"
    "\nOrgan:"
    "Organ is the organ where the tumor is located. Use standard organ names, like: liver, pancreas, kidney, spleen, colon, pelvis, adrenal gland, bladder, gallbladder, breast, stomach, lung, esophagus, uterus, bone, prostate, and duodenum. \n"
    "\nLocation: "
    "For location, check if the report specify the sub-segment or part of the organ where the tumor is. For the liver, consider the 8 couinaud sub-segments. The pancreas, consider head, body, neck, tail and uncinate process. For the kidney, consider left and right. For the lungs, consider the lobes. For other organs, just check if the report mentions some type of organ region or sub-segment. \n"
    "Say location = U if the report does not specify the intra-organ location of a tumor. \n"
    "\nType: "
    "If the report provides tumor type, inform it. \n"
    "Otherwise, say type = U if the report does not specify the type of the tumor. \n"
    "Follow these rules:"
    "1- If the tumor type is not specified in findings, you may deduce it from the clinical history or impressions sections. \n"
    "2- Cyst, or cystic lesion, is a common type of tumor. For any cysts, you say type = cyst. \n"
    "3- Assign type = metastasis if the tumor is described as a metastasis originating from a cancer in another organ, unless the report explicitly states otherwise. Determine certainty based on the level of confidence expressed in the report. \n"
    "4- If the report does not mention the tumor type, but you read that the patient has cancer in the organ where the tumor is, or has a history of malignant tumors in the organ, you may say type = malignant. However, do not say type = malignant if a more specific tumor type is given (like cyst, PDAC and PNET in pancreas, RCC in kidney, HCC in liver,...). \n"
    "5- Try reporting types using their most standard name, followed by their acronym. For example, if the report mentions 'adenocarcinoma in the pancreas' or 'PDAC', say type = Pancreatic Ductal Adenocarcinoma (PDAC). \n"
    "\nCertainty: "
    "Certainty of the tumor type, according to the report. If a report mentions a tumor type in the findings, history, or impressions, without demonstrating uncertainty, say certainty = certain. "
    "If the report expresses strong confidence in tumor type, say certainty = high. "
    "If the report mentions a tumor type but expresses significant uncertainty about it, say certainty = low. "
    "If the report does not mention the tumor type, say certainty = U. \n"
    "\nAttenuation: "
    "For each tumor, inform the attenuation if the report mentions it. Attenuation shoud be hyperenhancing, hypoenchanging, or isoenhancing.You may read synonyms like hypermetabolic and hypoattenuating, but you must only answer me hyperenhancing, hypoenchanging, isoenhancing or U. \n"
    "\nUnknow tumor number:"
    "Disregard tumors only if none of the following details are provided: size, location, organ, type or attenuation. \n"
    "If a report mentions multiple tumors in a organ, but does NOT specify how many tumors exist in the organ, use size = multiple. Just use size = multiple when it is impossible to know the number of tumors in the organ, otherwise, you add en entry for each tumor. \n"
    "For example, if a report says a '2 cm metastasis in the liver sub-segment 3, and multiple other small metastases in the liver', you should say: \n"
    "tumor 1: type = metastasis; certainty = high; size = 2 cm; organ = liver; location = segment 3; attenuation = U; \n"
    "tumor 2: type = metastasis; certainty = high; size = multiple; organ = liver; location = U; attenuation = U;\n"
    "Just use an entry with size = multiple instead. As in the example above, even when you have write an entry with size = multiple, you must have an individual entry for each tumor that is individually described in the report. \n"
    "Every time the report says there are multiple tumors in an organ but you cannot know the number, you MUST have an entry with size = multiple for the organ. \n"
    "\nJustification: "
    "Besides filling the template, justify your answer, carefully mentioning each section of the report if present: history, findings, and impressions. "
    "Explain from which sentences you got each size, location, and type.\n"
    "Some reports may refer to past measurements (using words like previously, before, or giving dates). Ignore previous measurements. "
    "Provide me a syntactic analysis of the report sentences mentioning tumor sizes. In this analysis, explain which measurement refers to which tumor, if the measurement is current or past, and if the corresponding tumor is malignant or benign.\n"
    "I will provide an example of a report and a correct answer for a %(organ)s tumor:\n"
    "Example report: \n"
    "%(example_report)s \n"
    "Example answer: \n"
    "%(example_answer)s \n"
    "\n"
    "End of the example. \n"
    "\n"
)

summary_of_terms_pancreas="""Adenocardinoma or similar terms indicate Pancreatic Ductal Adenocarcinoma (PDAC). Neuroendocrine tumor or similar terms indicate Pancreatic Neuroendocrine Tumor (PNET).
6- If the report mentions 'pancreatic cancer' or a diagnosis of 'pancreatic adenocarcinoma' and metastases in other organs (e.g., liver), classify the pancreatic lesion as the primary tumor (e.g., type = cancer or type = Pancreatic Ductal Adenocarcinoma (PDAC)). The pancreatic tumor should not be classified as metastasis in these cases, as it is the origin of the metastatic disease. Only consider type = metastasis if the report mentions a metastatic tumor from other organ, like metastatic RCC.
7- If the report says only 'pancreatic cancer', and gives no other indication of the specific type, you should classify the tumor as type = cancer."""


report_pancreas = """
#Private.
"""

answer_pancreas = """
#Private.
# """

answer_pancreas_multi_organ="""
#Private.
"""

report_liver="""
#Private.
"""

answer_liver= """
#Private.
"""
report_kidney="""#Private."""

answer_kidney="""#Private."""

liver_locationsliver_segments = "Segment I (Caudate Lobe), Segment II, Segment III, Segment IV, Segment V, Segment VI, Segment VII, Segment VIII"
pancreas_subdivisions = "Head, Uncinate Process, Neck, Body, Tail"
kidney_subdivisions = "Right Kidney Renal Cortex, Right Kidney Renal Medulla, Right Kidney Renal Pyramids, Right Kidney Renal Papilla, Right Kidney Renal Columns, Right Kidney Renal Pelvis, Right Kidney Minor Calyces, Right Kidney Major Calyces, Right Kidney Hilum, Left Kidney Renal Cortex, Left Kidney Renal Medulla, Left Kidney Renal Pyramids, Left Kidney Renal Papilla, Left Kidney Renal Columns, Left Kidney Renal Pelvis, Left Kidney Minor Calyces, Left Kidney Major Calyces, Left Kidney Hilum"
organ_part={"liver":liver_locationsliver_segments,"pancreas":pancreas_subdivisions,"kidney":kidney_subdivisions}

# Liver Tumors
benign_liver = "Hepatic Hemangioma (HH), Focal Nodular Hyperplasia (FNH), Bile Duct Adenoma, Simple Liver Cyst (SLC), Cyst"
malignant_liver = "Hepatocellular Carcinoma (HCC), Cholangiocarcinoma (CCA), metastasis"
both_liver = "Hepatic Adenoma (HA), Mucinous Cystic Neoplasm (MCN)"

# Pancreas Tumors
benign_pancreas = "Serous Cystadenoma (SCA), Cyst"
malignant_pancreas = "Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC), metastasis"
both_pancreas = "Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET)"

# Kidney Tumors
benign_kidney = "Renal Oncocytoma (RO), Angiomyolipoma (AML), Simple Renal Cyst, Cyst"
malignant_kidney = "Renal Cell Carcinoma (RCC), Transitional Cell Carcinoma (TCC), Wilms Tumor (Nephroblastoma), metastasis"
both_kidney = "Cystic Nephroma (CN), Multilocular Cystic Renal Neoplasm of Low Malignant Potential (MCRNLMP)"

observationsV0=(" Follow these rules for interpreting radiology reports, and always check the entire report carefully, including any clinical history (especially of cancer), the findings section (if present) and the impressions section (if present): \n "
              "1- 'unremarkable' means that an organ has no tumor. \n "
              "2- Multiple words can be used to describe benign and malignant tumors, such as metastasis, tumor, lesion, mass, cyst, neoplasm, growth and cancer. Consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. Examples of lesions that are not tumors: ulcers, wounds, infections, inflammations or scars.\n"
              "3- You should consider that a certain tumor is certainly benign if this information is explicit in the report (ususally, in the findings or impressions). Examples of benign tumors are: cysts (SLC, PLD, etc.), hepatic hemangioma (HH), Focal Nodular Hyperplasia (FNH), Bile Duct Adenoma, Serous Cystadenoma (SCA), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), . \n "
              "4- You should consider that a certain tumor is certainly malignant if this information is explicit in the report. Examples of benign tumors: . \n "#oncologic findings, index lesions
              #we may have both malignant and benign
              #I want the largest malignant tumor
              #If it is a cancer patient
              "4- Organs never mentioned in the report have no tumors. \n "
                "5- Treat anything that is 'too small to characterize' or uncertain as 0. \n "
                "6- Renal calculi, nephrolithiasis and renal stones are not tumors. \n "
                "7- Organs with no tumor but other pathologies should be reported as 0.")

observations=("")

#, hypodensity and hyperdensity, too small to characterize: uncertain

instructions1ShotFast=("Instructions: Discover if a CT scan radiology report indicates the presence "
                   "of liver tumors, pancreas tumors or kidney tumors. Output binary labels for "
                   "each of these categories, where 1 indicates tumor presence and 0 tumor absence. "
                   "Example: liwhatsappver tumor=1; kidney tumor=0; pancreas tumor=0. "
                   "Answer with only the labels, do not repeat this prompt. "
                   "Report 1 is an example for you, and its labels are provided. "
                   "I want you to give me the labels for Report 2.\n ")

instructionsFewShotFast=("Instructions: Discover if a CT scan radiology report indicates the presence "
                   "of liver tumors, pancreas tumors or kidney tumors. Output binary labels for "
                   "each of these categories, where 1 indicates tumor presence and 0 tumor absence."
                   "Example: liver tumor=1; kidney tumor=0; pancreas tumor=0. "
                   "Answer with only the labels, do not repeat this prompt. "
                   "I will provide a %(examples)s reports and labels as examples for you. "
                   "I want you to give me the labels for Report %(last)s.\n ")

question="Binary labels for Report%(last)s:"



time_machine_solver = ("I am sending two CT radiology reports with respective dates. The first report is from an earlier exam, expressing either no lesion or uncertainty about the presence or nature (benign or malignant) of a %(organ)s lesion (or lesions). "
                        "The second report is from a more recent exam, and it indicates more clearly the presence of a malignant tumor in the %(organ)s. Read both reports carefully, paying attention to the findings, impressions, and clinical history sections (if present). "
                        "Your task is to determine if any %(organ)s lesion reported in the first report is very likely the same as a malignant tumor in the second report. "
                        "Report 1 (earlier exam, %(date1)s): \n"
                        "%(report1)s \n"
                        "Report 2 (more recent exam, %(date2)s): \n"
                        "%(report2)s \n"
                        "Fill out the template below by answering yes, no (if the lesion in report 1 is clearly not the same as in report 2), or uncertain: \n"
                        "very likely malignancy in %(organ)s in the first exam = _ \n"
                        "In case you answer yes, also provide the location and size of the likely malignant lesion (or lesions) in the first report. Do so by filling out the template below, using one line per malignant tumor in the %(organ)s (you may add or remove lines from the template). Substitute the first _ in each line by the the tumor size, and the second by its location (Use 'U' if the report doesn’t specify size or location): \n "
                        "%(organ)s malignant tumor size = _; location = _;\n"
                        "%(organ)s malignant tumor size = _; location = _;\n"
                        "... \n"
                        "I am only interested in measurements that are 'current' at the time of report 1. Reports can write the size of the tumor in 1D, 2D or 3D measurements, and you should use the same standards used in the report."
                        "Write 1D measurements as: 15 mm; 2D measurements as: 15 x 10 mm; and 3D measurements as: 40 x 30 x 30 mm. You may use either cm or mm, but you MUST WRITE in each line of the filled template the unit you are using (cm or mm). If a report does not specify the unit, assume it is mm. \n"
                        "For location, chose one of these options for each tumor: %(organ_locations)s \n"
                        "In your justification for malignancy, explain why a lesion in report 1 is likely the same as in report 2. Refer to relevant sections (findings, clinical history, impressions), and carefully check tumor location. \n"
                        "If you provide measurements, also provide a synthatic analysis of the report sentences mentioning %(organ)s tumor sizes. In this analysis, explain which measurement refers to which tumor, if the measurement is current at the time of exam 1, and if the tumor is malignant or benign.\n"
                        "Follow these rules for interpreting radiology reports: \n"
                        "1- If report 1 mentions absolutely no abnormality in the %(organ)s, answer 'very likely malignancy in %(organ)s in the first exam = no' \n"
                        "2- Some words are only used for describing malignant tumors, for example: metastasis, cancer, growing, or any oncologic lesion and index lesion in cancer patients. \n"
                        "Reports may sometimes mention the specific tumor type. In the %(organ)s, benign tumors are: %(benign_tumors)s. \n "
                        "Malignant %(organ)s tumors are: %(malignant_tumors)s. \n "
                        "Tumors that may be both benign or malignant are: %(both_tumors)s. \n "
                        "3- If the report does not mention that the tumor is benign or does not specify lesion type, but the tumor is growing in relation to a past measurement, consider it malignant. \n"
                        "4- Especially check if the location of a malignant tumor in report 2 matches a lesion in report 1. \n"
)


instructionsNormal=("Instructions: Read the radiology report below carefully, paying special attention to the findings, clinical history, and impressions sections (if available). Answer these questions: \n"
"1- Does the report mention any cancer or tumor (in any organ) or any history of cancer? \n"
"2- Does the report mention any metastasis (in any organ) or any history of metastasis?\n"
"3- Does the report mention any tumor suspicion in the pancreas?\n"
"4- Does the report mention that the patient has an abnormality in the pancreas, which is for sure NOT a tumor?\n"
"5- Does the report mention that the patient has a healthy pancreas?\n"
"Answer me by just filling out the template below, substituting _ by yes or no: \n "
"tumor=_; metastasis=_; pancreatic tumor suspicion=_; pancreatic non-tumor abnormality=_; healthy pancreas=_;\n"
"Besides filling the template, justify your answer, carefully mentioning each section of the report if present: clinical history, findings and impressions. Consider the following conservative rules to interpret the reports:\n"
"a- Consider the %(organ)s unhealthy if ANY abnormality is present.\n"
"b- Consider metastasis if ANY any signal, suspicion, symptom or history of metastatic cancer is present.\n"
"c- If the pancreas is not mentioned or described as unremarkable, consider it healthy.\n"
"d- Multiple words can be used to describe tumors, and you may check both the findings and impressions sections of the report (if present) to understand if an organ has tumors. Some words are: as metastasis, tumor, lesion, mass, cyst, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding"
"e- Consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. "
"Many conditions are not tumors, and should not be interpreted as so, unless a tumor is also reported along with the diease. Examples of conditions that are not tumors: Pancreatitis, Pancreatic Insufficiency, Cystic Fibrosis (CF), Diabetes Mellitus (DM), Exocrine Pancreatic Insufficiency (EPI), pancreatectomy, and Pancreatic Pseudocyst. \n"
"Now some exmples of specific tumor names are: Serous Cystadenoma (SCA), Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET). \n "
"f- Consider any benign (e.g., cyst) or malingnat tumor a tumor. Thus, any type of cyst is a tumor. \n ")

def get_longitudinal_reports(df, patient_id):
    """
    For a given patient ID, extract all exam reports (from the 'Findings' column) and order them by exam date.
    Returns:
      - A list of ordered Encrypted Accession Numbers.
      - A list of strings, where each string is formatted as:
            Report {i} date: month/year
            Report {i} text: findings
        with i starting at 1.
    """
    # Filter the DataFrame to include only rows for the specified patient ID.
    patient_df = df[df["Patient ID"] == patient_id].copy()

    # Ensure 'Exam Completed Date' is in datetime format, then sort the rows by this date.
    patient_df["Exam Completed Date"] = pd.to_datetime(patient_df["Exam Completed Date"])
    patient_df = patient_df.sort_values("Exam Completed Date")

    # Extract the ordered list of Encrypted Accession Numbers.
    accession_numbers = patient_df["Encrypted Accession Number"].tolist()

    # Create the list of formatted report strings.
    report_strings = []
    for i, (_, row) in enumerate(patient_df.iterrows(), start=1):
        # Format the exam date as month/year.
        date_str = row["Exam Completed Date"].strftime("%m/%Y")
        findings = row["Findings"]
        report_str = f"Report {i} date: {date_str}\nReport {i} text: {findings}"
        report_strings.append(report_str)

    return accession_numbers, report_strings

def get_report_n_label(data,i,row_name='Anon Report Text',get_date=False,id_col='Accession Number'):
    print(i)
    if isinstance(i,str):
        # get row with accession number i
        row=data[data[id_col]==i].to_dict('records')[0]
    else:
        row=data.iloc[i]
    if isinstance(row[row_name],str):
        report=row[row_name]
    else:
        report=None

    print(row)

    if get_date:
        
        print(row['Exam Started Date'])
        try:
            # Try parsing it with date and time
            date=row['Exam Started Date'].split()[0]
        except ValueError:
            # If it's just a date, return the string itself
            date=row['Exam Started Date']
        return report,date
    
    try:
        if not math.isnan(row['Liver Tumor']):
            label=f"liver tumor={int(row['Liver Tumor'])}; kidney tumor={int(row['Kidney Tumor'])}; pancreas tumor={int(row['Pancreas Tumor'])}"
        else:
            label=None
    except:
        label=None
    return report,label

def get_instuctions(fast,step,examples=0,organ='liver',all_data=None):
    if step=='tumor detection':
        if fast:
            if len(examples)==0:
                instructions=instructions0ShotFast
            elif len(examples)==1:
                instructions=instructions1ShotFast
            else:
                instructions=instructionsFewShotFast % {'examples':len(examples),'last':len(examples)+1}
        else:
            if len(examples)==0:
                instructions=instructions0Shot
            elif len(examples)==1:
                instructions=instructions1Shot
            else:
                instructions=instructionsFewShot % {'examples':len(examples),'last':len(examples)+1}
    elif step=='pre-diagnostic confirmation':
        instructions=preDiagnositcConfirmation
    elif step=='normal confirmation':
        instructions=instructionsNormal
    elif step=='malignancy detection':
        if len(examples)>0:
            raise ValueError('Only 0 or 1 examples allowed for malignancy detection')
        if not fast:
            if organ=='liver':
                instructions=instructions0ShotMalignancy % {'organ':organ,
                                                        'benign_tumors':benign_liver,
                                                        'malignant_tumors':malignant_liver,
                                                        'both_tumors':both_liver}
            elif organ=='pancreas':
                instructions=instructions0ShotMalignancy % {'organ':organ,
                                                        'benign_tumors':benign_pancreas,
                                                        'malignant_tumors':malignant_pancreas,
                                                        'both_tumors':both_pancreas}
            elif organ=='kidney':
                instructions=instructions0ShotMalignancy % {'organ':organ,
                                                        'benign_tumors':benign_kidney,
                                                        'malignant_tumors':malignant_kidney,
                                                        'both_tumors':both_kidney}
        else:
            if organ=='liver':
                instructions=instructions0ShotMalignancyFast % {'organ':organ,
                                                        'benign_tumors':benign_liver,
                                                        'malignant_tumors':malignant_liver,
                                                        'both_tumors':both_liver}
            elif organ=='pancreas':
                instructions=instructions0ShotMalignancyFast % {'organ':organ,
                                                        'benign_tumors':benign_pancreas,
                                                        'malignant_tumors':malignant_pancreas,
                                                        'both_tumors':both_pancreas}
            elif organ=='kidney':
                instructions=instructions0ShotMalignancyFast % {'organ':organ,
                                                        'benign_tumors':benign_kidney,
                                                        'malignant_tumors':malignant_kidney,
                                                        'both_tumors':both_kidney}
    elif step=='malignant size':
        if len(examples)>0:
            raise ValueError('Examples not implemented for tumor size')
        if not fast:
            instructions=instructions0ShotMalignantSize % {'organ':organ,
                                                        'benign_tumors':benign_kidney,
                                                        'malignant_tumors':malignant_kidney,
                                                        'both_tumors':both_kidney,
                                                        'organ_locations':organ_part[organ]}
        else:
            raise ValueError('Fast not implemented for tumor size')
    elif step=='type and size':
        reports_and_answers={'kidney':[report_kidney,answer_kidney],
                             'liver':[report_liver,answer_liver],
                             'pancreas':[report_pancreas,answer_pancreas]}
        malignant_benign_both={'kidney':[benign_kidney,malignant_kidney,both_kidney],
                                 'liver':[benign_liver,malignant_liver,both_liver],
                                 'pancreas':[benign_pancreas,malignant_pancreas,both_pancreas]}
        instructions=instructions0ShotSizenType % {'organ':organ,
                                                    'benign_tumors':malignant_benign_both[organ][0],
                                                    'malignant_tumors':malignant_benign_both[organ][1],
                                                    'both_tumors':malignant_benign_both[organ][2],
                                                    'organ_locations':organ_part[organ],
                                                    'example_report':reports_and_answers[organ][0],
                                                    'example_answer':reports_and_answers[organ][1],
                                                    'extra_info':(summary_of_terms_pancreas if organ=='pancreas' else '')}
    elif step == 'HCC':
        malignant_benign_both={'liver':[benign_liver,malignant_liver,both_liver]}
        organ='liver'
        instructions=instructionsHCC % {'benign_tumors':malignant_benign_both[organ][0],
                                        'malignant_tumors':malignant_benign_both[organ][1],
                                        'both_tumors':malignant_benign_both[organ][2],
                                        'organ_locations':organ_part[organ],
                                        'example_report':reportHCC,
                                        'example_answer':answerHCC,
                                        'extra_info':''}
        

    elif step=='type and size pathology':
        reports_and_answers={'pancreas':[pathologyReportPancreas,pathologyReportPancreasAnswer]}
        malignant_benign_both={'pancreas':[benign_pancreas,malignant_pancreas,both_pancreas]}
        instructions=instructions0ShotSizenTypePathology % {'organ':organ,
                                                    'benign_tumors':malignant_benign_both[organ][0],
                                                    'malignant_tumors':malignant_benign_both[organ][1],
                                                    'both_tumors':malignant_benign_both[organ][2],
                                                    'organ_locations':organ_part[organ],
                                                    'example_report':reports_and_answers[organ][0],
                                                    'example_answer':reports_and_answers[organ][1],
                                                    'extra_info':(summary_of_terms_pancreas if organ=='pancreas' else '')}
    elif step=='type and size multi-organ':
        reports_and_answers={'kidney':[report_kidney,answer_kidney],
                             'liver':[report_liver,answer_liver],
                             'pancreas':[report_pancreas,answer_pancreas]}
        instructions=instructions0ShotSizenTypeMultiOrgan % {'organ':'pancreas',#used only as example
                                                             'example_report':report_pancreas,
                                                             'example_answer':answer_pancreas_multi_organ}
    elif step=='diagnoses':
        instructions=abnormality_prompt
        
    return instructions

def create_conversation(data,target,target_data=None,examples=[],fast=True, 
                        step='tumor detection',organ='liver',row_name='Anon Report Text',
                        future_report=None):
    if target_data is None:
        target_data=data

    if step=='time machine':
        report,date=get_report_n_label(target_data,target,row_name=row_name,get_date=True)   
        print('Future report:',future_report)
        future_report,future_date=get_report_n_label(target_data,future_report,row_name=row_name,get_date=True)

        usr=time_machine_solver % {'organ':organ,
                                            'benign_tumors':benign_liver,
                                            'malignant_tumors':malignant_liver,
                                            'both_tumors':both_liver,
                                            'organ_locations':organ_part[organ],
                                            'date1':date,
                                            'date2':future_date,
                                            'report1':report,
                                            'report2':future_report}
    elif step=='longitudinal pancreas' or step=='longitudinal pancreas diagnosis':
        accessions,reports=get_longitudinal_reports(data,target)
        #concatenate the list of strings report
        reports='\n'.join(reports)
        if step=='longitudinal pancreas':
            usr = instructionsLongitudinalPancreas +'\n'+ reports
        else:
            usr = instructionsLongitudinalPancreasDiagnosis +'\n'+ reports
        message= [{"role": "system", "content": system+' \n '+observations},
                  {"role": "user", "content": usr}]
        return message,accessions
        
    else:
        usr=get_instuctions(fast,step,examples=examples,organ=organ)
        #add clinical notes
        usr+=' \n '
        usr+=observations
        usr+=' \n '
        #examples
        i=0
        print('Examples:',examples)
        for i,ex in enumerate(examples,1):
            report,label=get_report_n_label(data,ex,row_name=row_name)
            if report is None or label is None:
                raise ValueError('No label or report available for index '+str(ex))
            usr+='Report '+str(i)+': '+report+'\n '
            usr+='Report '+str(i)+' labels: '+label
            usr+=' \n --- \n '
        #target report
        i+=1
        if len(examples)==0:
            num=''
        else:
            num=' '+str(i)
        report,_=get_report_n_label(target_data,target,row_name=row_name)
        if report is None:
            raise ValueError('No report available for index '+str(target))
        usr+='Report'+num+': '+report+'\n '
        #question
        usr+=question % {'last':num}

    message= [{"role": "system", "content": system+' \n '+observations},
                  {"role": "user", "content": usr}]

    #print('Report:',report)
    
    return message

def multi_prompt_message(data,target,target_data,
                         per_message_examples=5,examples=[]):
    assert ((len(examples)+1)%(per_message_examples+1))==0
    num_examples=per_message_examples
    step=per_message_examples+1
    
    message= [
        {"role": "system", "content": system+' \n '+observations},
        #{"role": "user", "content": usr},
    ]
    #print(int(len(examples)/step))
    for i in range(int(len(examples)/step)):
        ex=examples[i*step:(i+1)*step-1]
        t=examples[(i+1)*step-1]
        m=create_conversation(data=data,target=t,examples=ex)[1]["content"]
        message.append({"role": "user", "content": m})
        _,l=get_report_n_label(data,t)
        message.append({"role": "assistant", "content": l})
        
    ex=examples[-per_message_examples:]
    m=create_conversation(data=data,target_data=target_data,
                          target=target,examples=ex)[1]["content"]
    message.append({"role": "user", "content": m})
    
    return message

    
        
    
def run_model(message,base_url='http://0.0.0.0:8000/v1',labels=None,id=None,batch=1):
    conver,answer=SendMessageAPI(text=None,conver=message,base_url=base_url,labels=labels,id=id,batch=batch)
    return answer

def run(target,examples,data,target_data=None,base_url='http://0.0.0.0:8000/v1',print_message=False,
        step='tumor detection',organ='liver',fast=True,row_name='Anon Report Text',id_column='Anon Acc #',
        future_report=None):
    if target_data is None:
        target_data=data

    if isinstance(target,list):
        message=[]
        labels=[]
        id=[]
        for tgt in target:
            message.append(create_conversation(data=data,target=tgt,examples=examples, target_data=target_data,step=step,organ=organ,fast=fast,
                                               row_name=row_name,future_report=future_report))
            if 'Pancreas Tumor' not in data.columns:
                labels=None
            else:
                #labels.append(data.iloc[target][['Liver Tumor','Kidney Tumor','Pancreas Tumor']])
                labels=None
            id.append(data.iloc[target][id_column])
        batch=len(message)
        print('Batch:',batch)
    else:
        message=create_conversation(data=data,target=target,examples=examples,
                                    target_data=target_data,step=step,organ=organ,fast=fast,
                                               row_name=row_name,future_report=future_report)
        if step == 'longitudinal pancreas' or step == 'longitudinal pancreas diagnosis':
            message,accessions=message
        print()
        print('Message:',message)
        print()
        #check if the columns are present
        if 'Pancreas Tumor' not in data.columns:
            labels=None
        else:
            #labels=data.iloc[target][['Liver Tumor','Kidney Tumor','Pancreas Tumor']]
            labels=None
        #print('ID column:',id_column)
        try:
            id=data.iloc[target][id_column]
        except:
            id=None
        batch=1
    if print_message:
        print(message)

    if step == 'longitudinal pancreas' or step == 'longitudinal pancreas diagnosis':
        return run_model(message,base_url=base_url,labels=None,id=None,batch=batch),accessions
    else:
        return run_model(message,base_url=base_url,labels=labels,id=id,batch=batch)


def run_multi_prompt(target,examples,data,target_data=None,
                     base_url='http://0.0.0.0:8000/v1',print_message=False,
                     per_message_examples=5):
    if target_data is None:
        target_data=data
    message=multi_prompt_message(data=data,target=target,examples=examples,
                                 per_message_examples=per_message_examples,
                                 target_data=target_data)
    if print_message:
        print(message)
    return run_model(message,base_url=base_url)

def get_value_old(pattern, string):
    match = re.search(pattern, string)
    if match:
        return int(match.group(1))
    else:
        return np.nan
    

def interpret_output_old(string):
    liver_pattern = r'liver tumor=(\d+)'
    kidney_pattern = r'kidney tumor=(\d+)'
    pancreas_pattern = r'pancreas tumor=(\d+)'

    return {'Liver Tumor':get_value(liver_pattern,string),
            'Kidney Tumor':get_value(kidney_pattern,string),
            'Pancreas Tumor':get_value(pancreas_pattern,string)}

def get_value(pattern, string,step='tumor detection'):
    matches = re.findall(pattern, string.lower())
    print('Matches:',matches)

    if len(matches)==0:
        return np.nan

    if step == 'malignant size' or step == 'all sizes':
        sizes = []

        for match in matches:
            # Extract both integers and floating point numbers
            match,unit=match
            print('Match:',match)
            print('Unit:',unit)
            numbers = [float(num) for num in re.findall(r'\d+\.\d+|\d+', match)]
            print('Numbers:',numbers)
            
            if len(numbers) == 0:
                print('No numbers found')
                continue

            # Convert to mm depending on whether 'cm' or 'mm' is present
            for num in numbers:
                if unit=='cm':
                    sizes.append(num * 10)  # Convert cm to mm
                elif unit=='mm':
                    sizes.append(num)  # Already in mm
            print('Sizes:',sizes)

        # Return the largest size in mm, or np.nan if no sizes were found
        if len(sizes) == 0:
            return np.nan
        else:
            if step == 'malignant size':
                return np.max(sizes)
            else:
                return sizes
                #sz=''
                #for s in sizes:
                #    sz+=str(s)+' '
                #return str(sizes).replace('[','').replace(']','')

    else:
        if 'yes' in matches[0]:
            return 1
        elif 'no' in matches[0]:
            return 0
        else:
            return np.nan
        
def extract_liver_tumors(answer_text, organ="liver"):
    """
    1) Splits the answer by lines that contain "liver tumor N:" to isolate each chunk.
    2) For each chunk, parses the mandatory fields: type, certainty, size, location,
       arterial enhancement, washout, capsule, threshold growth, LI-RADS.
    3) Raises an error if any field is missing.
    4) Returns a dict like:
         {
           "liver tumor 1": {
              "type": ...,
              "certainty": ...,
              "size": ...,
              ...
           },
           "liver tumor 2": {...},
           ...
         }
    """

    # Regex to split on lines like "liver tumor 1:", capturing the line as a separate token.
    # Use re.IGNORECASE so it catches "Liver Tumor 1:" or "LIVER TUMOR 1:", etc.
    split_pattern = rf"(?i)(?=(?:{organ}\s+tumor\s+\d+:))"

    # Split the answer into chunks, each starting with "liver tumor N:"
    # The first chunk could be empty if the text starts with "liver tumor 1:" pattern. 
    chunks = re.split(split_pattern, answer_text)
    # Filter out empty chunks or whitespace.
    chunks = [c.strip() for c in chunks if c.strip()]

    tumors_dict = {}
    for chunk in chunks:
        # The first line in chunk should look like "liver tumor 1:"
        # Extract the tumor number from that line.
        # We'll do a quick check:
        first_line_match = re.match(rf"(?i){organ}\s+tumor\s+(\d+):", chunk)
        if not first_line_match:
            # This chunk doesn't start with "liver tumor N:", skip or continue
            continue

        tumor_number = first_line_match.group(1).strip()
        tumor_key = f"{organ} tumor {tumor_number}"

        # For each field, we look for e.g. "type = X;"
        # We use a small helper function.
        def find_field(field_name):
            # pattern: e.g. "type = ???;"
            # we allow optional whitespace.  We capture up to the next semicolon
            field_pattern = rf"{field_name}\s*=\s*([^;]+);"
            m = re.search(field_pattern, chunk, re.IGNORECASE)
            if not m:
                raise ValueError(f"Missing field '{field_name}' in chunk:\n{chunk}")
            return m.group(1).strip()

        # Now we get each mandatory field
        type_raw        = find_field("type")
        certainty_raw   = find_field("certainty")
        size_raw        = find_field("size")
        location_raw    = find_field("location")
        arterial_raw    = find_field("arterial enhancement")
        washout_raw     = find_field("washout")
        capsule_raw     = find_field("capsule")
        threshold_raw   = find_field("threshold growth")
        lirads_raw      = find_field("LI-RADS")

        # parse size (like your earlier code)
        if 'multiple' in size_raw.lower():
            size_parsed = 'multiple'
        else:
            size_parsed = get_value(r"(.*?)(cm|mm)", size_raw, step='malignant size')

        tumors_dict[tumor_key] = {
            "type"               : type_raw,
            "certainty"          : certainty_raw,
            "size"               : size_parsed,
            "location"           : location_raw,
            "arterial enhancement": arterial_raw,
            "washout"            : washout_raw,
            "capsule"            : capsule_raw,
            "threshold growth"   : threshold_raw,
            "LI-RADS"            : lirads_raw,
        }

    return tumors_dict

def interpret_output(string,step='tumor detection',organ='liver'):

    if step=='tumor detection':
        liver_pattern = r'liver tumor presence\s*[=:]\s*.*?(?:;|$|,|/|yes|no|u)'
        kidney_pattern = r'kidney tumor presence\s*[=:]\s*.*?(?:;|$|,|/|yes|no|u)'
        pancreas_pattern = r'pancreas tumor presence\s*[=:]\s*.*?(?:;|$|,|/|yes|no|u)'

        return {'Liver Tumor':get_value(liver_pattern,string),
                'Kidney Tumor':get_value(kidney_pattern,string),
                'Pancreas Tumor':get_value(pancreas_pattern,string)}
    elif step=='normal confirmation':
        
        # Regex patterns to capture yes/no after "key ="
        tumor_pattern                    = r'tumor\s*=\s*(yes|no)'
        metastasis_pattern              = r'metastasis\s*=\s*(yes|no)'
        pancreatic_tumor_suspicion      = r'pancreatic tumor suspicion\s*=\s*(yes|no)'
        pancreatic_non_tumor_abnormality= r'pancreatic non-tumor abnormality\s*=\s*(yes|no)'
        healthy_pancreas_pattern        = r'healthy pancreas\s*=\s*(yes|no)'

        # Return dictionary of results by calling get_value (already defined by you)
        
        #'Any Tumor', 'Metastasis', 'Pancreatic Tumor Suspicion', 'Pancreatic Non-Tumor Abnormality', 'Healthy Pancreas',
        return {
            'Any Tumor':                        get_value(tumor_pattern, string, step='tumor detection'),
            'Metastasis':                   get_value(metastasis_pattern, string, step='tumor detection'),
            'Pancreatic Tumor Suspicion':   get_value(pancreatic_tumor_suspicion, string, step='tumor detection'),
            'Pancreatic Non-Tumor Abnormality': get_value(pancreatic_non_tumor_abnormality, string, step='tumor detection'),
            'Healthy Pancreas':             get_value(healthy_pancreas_pattern, string, step='tumor detection')
        }

    elif step=='pre-diagnostic confirmation':
        # Define regex patterns to capture yes/no answers.
        tumor_pattern = r'pancreatic tumor suspicion\s*[=:]\s*?(?:;|$|,|/|yes|no)'
        surgery_pattern = r'pancreas surgery\s*[=:]\s*?(?:;|$|,|/|yes|no)'
        
        return {'Pancreatic Tumor Suspicion': get_value(tumor_pattern, string),
                'Pancreas Surgery': get_value(surgery_pattern, string)}
    elif step=='malignancy detection':
        pattern = r"malignant tumor in %s\s*[=:]\s*.*?(?:;|$|,|/|yes|no|u)" % organ  
        return {'Malignant Tumor in '+organ:get_value(pattern,string)}
    elif step=='malignant size':
        pattern = r"%s malignant tumor size\s*[=:]\s*(.*?)(cm|mm)" % organ
        y={'Malignant Tumor in '+organ:get_value(pattern,string,step=step)}
        print(y)
        return y
    elif step=='time machine':
        malignancy_pattern = r"very likely malignancy in %s in the first exam\s*[=:]\s*.*?(?:;|$|,|/|yes|no|u)" % organ
        size_pattern = r"%s malignant tumor size\s*[=:]\s*(.*?)(cm|mm)" % organ
        return {'very likely malignancy in '+organ:get_value(malignancy_pattern,string),
                'very likely malignant tumor in '+organ:get_value(size_pattern,string,step='malignant size')}
    elif step == 'type and size' or step=='type and size pathology':
        # Extracting multiple tumors from the LLM output
        tumor_pattern = rf"{organ} tumor \d+: type = (?P<type>.+?); certainty = (?P<certainty>.+?); size = (?P<size>.+?); location = (?P<location>.+?);"
        matches = re.finditer(tumor_pattern, string.lower())
        
        tumors = {}
        for match in matches:
            tumor_key = f"{organ} tumor {len(tumors) + 1}"
            size_raw = match.group('size').strip()
            if 'multiple' in size_raw:
                size_numbers = 'multiple'
            else:
                size_numbers = get_value(r"(.*?)(cm|mm)", size_raw, step='malignant size')
            tumors[tumor_key] = {
                'type': match.group('type').strip(),
                'certainty': match.group('certainty').strip(),
                'size': size_numbers,
                'location': match.group('location').strip(),
            }
        return tumors
    elif step == 'HCC':
        out = extract_liver_tumors(string.lower(),organ='liver')
        #print(out)
        #raise ValueError('Stop here')
        return out
    elif step == 'type and size multi-organ':
        # Extracting multiple tumors from the LLM output
        tumor_pattern = rf"tumor \d+: type = (?P<type>.+?); certainty = (?P<certainty>.+?); size = (?P<size>.+?); organ = (?P<organ>.+?); location = (?P<location>.+?); attenuation = (?P<attenuation>.+?);"
        matches = re.finditer(tumor_pattern, string.lower())
        
        tumors = {}
        for match in matches:
            tumor_key = f"tumor {len(tumors) + 1}"
            size_raw = match.group('size').strip()
            if 'multiple' in size_raw:
                size_numbers = 'multiple'
            else:
                size_numbers = get_value(r"(.*?)(cm|mm)", size_raw, step='all sizes')
            tumors[tumor_key] = {
                'type': match.group('type').strip(),
                'certainty': match.group('certainty').strip(),
                'size': size_numbers,
                'location': match.group('location').strip(),
                'organ': match.group('organ').strip(),
                'attenuation': match.group('attenuation').strip(),
            }
        return tumors
    elif step == 'diagnoses':
        if "abnormalities =" in string:
            start_index = string.rfind("abnormalities =") + len("abnormalities =")
        elif "abnormalities=" in string:
            start_index = string.rfind("abnormalities=") + len("abnormalities=")
        elif "[" in string:
            start_index = string.find("[")
        else:
            return None
        end_index = string.rfind("]", start_index) + 1  # Include the closing bracket
        abnormalities_str = string[start_index:end_index].strip()
        
        # Safely evaluate the string as a Python object
        #abnormalities = ast.literal_eval(abnormalities_str)
        return abnormalities_str
    
    elif step == 'synonyms':
        if "synonyms =" in string:
            start_index = string.rfind("synonyms =") + len("synonyms =")
        elif "synonyms=" in string:
            start_index = string.rfind("synonyms=") + len("synonyms=")
        elif "{" in string:
            start_index = string.find("{")
        else:
            return None
        end_index = string.rfind("}", start_index) + 1
        synonyms_str = string[start_index:end_index].strip()
        return synonyms_str
    
    elif step == 'longitudinal pancreas':
        first_diagnosis_pattern = r'first diagnosis report\s*[=:]\s*(\d+|none)(?=[;\n.]|$)'
        pre_diagnosis_pattern = r'pre-diagnosis reports\s*[=:]\s*([\d,]+|none)(?=[;\n.]|$)'

        first_diagnosis_match = re.search(first_diagnosis_pattern, string, re.IGNORECASE)
        pre_diagnosis_match = re.search(pre_diagnosis_pattern, string, re.IGNORECASE)

        return {
            'First Diagnosis Report': first_diagnosis_match.group(1) if first_diagnosis_match else None,
            'Pre-Diagnosis Reports': pre_diagnosis_match.group(1) if pre_diagnosis_match else None
        }
        
    elif step == 'longitudinal pancreas diagnosis':
        tumor_pattern = r"tumor types\s*:\s*(.*?)(?=$|\n)"
        match = re.search(tumor_pattern, string, re.IGNORECASE)
        if not match:
            return None  # no "tumor types:" line found
        
        # Extract the entire substring after "tumor types:"
        raw_tumor_str = match.group(1).strip()
        # Example raw_tumor_str might be "PDAC; Cyst; Unknown;" or "none;"
        
        return {'Tumor Types': raw_tumor_str}
        
                
    #output: {'liver tumor 1': {'type': 'hcc', 'certainty': 'high', 'size': [12.0, 5.0], 'location': 'segment 2'}, 'liver tumor 2': {'type': 'benign cyst', 'certainty': 'certain', 'size': [30.0, 20.0], 'location': 'segment 3'}}

    else:
        raise ValueError('Invalid step')
    
def get_random_examples(target,limit,num,data):
    examples=[]
    for i in range(num):
        example=random.randint(0,limit)
        while example==i:
            example=random.randint(0,data.shape[0])
        examples.append(example)
    return examples

def generate_metrics(data, dnn_outputs, id_column='Anon Acc #', columns_to_evaluate=None,MRNs=None,step='tumor detection'):
    """
    Generates and prints confusion matrices and evaluation metrics for specified columns in two DataFrames.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing ground truth labels.
    dnn_outputs (pd.DataFrame): DataFrame containing predicted labels.
    id_column (str): The column name used to match rows between the DataFrames (default is 'Anon Acc #').
    columns_to_evaluate (list): List of column names to evaluate. If None, defaults to ['Liver Tumor', 'Kidney Tumor', 'Pancreas Tumor'].
    """

    if step=='malignancy detection':
        # Step 1: Create a new DataFrame with the selected columns
        dnn_outputs = copy.deepcopy(dnn_outputs[[id_column, 'Malignant Tumor in liver', 'Malignant Tumor in pancreas', 'Malignant Tumor in kidney']])
        # Step 2: Rename the columns
        dnn_outputs.columns = [id_column, 'Liver Tumor', 'Pancreas Tumor', 'Kidney Tumor']

    original_dnn_outputs=copy.deepcopy(dnn_outputs)
    original_data=copy.deepcopy(data)

    # Ensure both DataFrames are sorted by the identifier column
    data = data.sort_values(id_column).reset_index(drop=True)
    dnn_outputs = dnn_outputs.sort_values(id_column).reset_index(drop=True)

    #drop any row with nan in the dnn_outputs
    dnn_outputs=dnn_outputs.dropna()
    #check all rows in data and drop them if they are not present in dnn_outputs
    data=data[data[id_column].isin(dnn_outputs[id_column])]
    
    # Drop any row that has a MRN value not in the MRNs list
    if MRNs is not None:
        data = data[data[id_column].isin(MRNs)]
        dnn_outputs = dnn_outputs[dnn_outputs[id_column].isin(MRNs)]

    # Check if the identifier columns match
    if not data[id_column].equals(dnn_outputs[id_column]):
        raise ValueError(f"The '{id_column}' columns do not match between the DataFrames.")
    
    # Default columns to evaluate if not provided
    if columns_to_evaluate is None:
        columns_to_evaluate = ['Liver Tumor', 'Kidney Tumor', 'Pancreas Tumor']

    #print FPs and FNs
    for column in columns_to_evaluate:
        
        #drop in y_pred and y_true any row where y_true is nan
        data_no_nan=data.dropna(subset=[column])
        #get the same rows in dnn_outputs
        dnn_outputs_no_nan=dnn_outputs[dnn_outputs[id_column].isin(data_no_nan[id_column])]

        y_true = data_no_nan[column]
        y_pred = dnn_outputs_no_nan[column]

        # print False Positives and False Negatives
        print('False Positives and False Negatives for',column)
        for case in data_no_nan[(y_true==0) & (y_pred==1)][[id_column,column]].values:
            print('False Positives for',column,':',case[0])
            #print content of the report
            print(data_no_nan[data_no_nan[id_column]==case[0]]['Anon Report Text'].values[0])
            print(' \n ')
        for case in data_no_nan[(y_true==1) & (y_pred==0)][[id_column,column]].values:
            print('False Negatives for',column,':',case[0])
            #print content of the report
            print(data_no_nan[data_no_nan[id_column]==case[0]]['Anon Report Text'].values[0])
            print(' \n ')
    
    # Compute and print confusion matrices and metrics for each specified column
    for column in columns_to_evaluate:
        #drop in y_pred and y_true any row where y_true is nan
        data_no_nan=data.dropna(subset=[column])
        #get the same rows in dnn_outputs
        dnn_outputs_no_nan=dnn_outputs[dnn_outputs[id_column].isin(data_no_nan[id_column])]

        y_true = data_no_nan[column]
        y_pred = dnn_outputs_no_nan[column]

        print('Organ:',column)
        print('Gorund Truth:',y_true)
        print('Predictions:',y_pred)

        cm = confusion_matrix(y_true, y_pred,labels=[0,1])
        
        tn, fp, fn, tp = cm.ravel()
        
        # Manually calculate metrics, setting to NaN if division by zero occurs
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
        
        print(f"Metrics for {column}:")
        print(f"Confusion Matrix:\n{cm}\n")
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Negatives (FN): {fn}")
        print(f"Sensitivity: {sensitivity if not np.isnan(sensitivity) else 'NaN (0/0)'}")
        print(f"Specificity: {specificity if not np.isnan(specificity) else 'NaN (0/0)'}")
        print(f"PPV (Precision): {ppv if not np.isnan(ppv) else 'NaN (0/0)'}")
        print(f"F1-score: {f1 if not np.isnan(f1) else 'NaN (0/0)'}\n")

    #analyze nan cases
    for column in columns_to_evaluate:
        #print report for any row where original_dnn_output is nan and data is not
        j=0
        for case in original_dnn_outputs[original_dnn_outputs[column].isna()][id_column].values:
            #check if case is nan in original_data:
            if original_data[original_data[id_column]==case][column].isna().values[0]:
                continue
            if 'too small' in original_data[original_data[id_column]==case]['Anon Report Text'].values[0].lower():
                continue
            if 'ill-defined' in original_data[original_data[id_column]==case]['Anon Report Text'].values[0].lower():
                continue
            #print content of the report for case
            for i in list(range(4)):
                print('\n')
            if step=='tumor detection':
                print('NaN tumor detection case but tumor label not nan for',column,':',case)
            elif step=='malignancy detection':
                print('NaN malignancy detection case but label not nan for',column,':',case)
            print(original_data[original_data[id_column]==case]['Anon Report Text'].values[0])
            j+=1
        print('Total of',j,'cases with NaN in the predictions but not in the ground truth for',column)

def get_first_malignancy(accession_number, df, id_column='Accession Number'):
    """
    Function to get the Accession Number of the first malignancy diagnosis
    for a patient based on the given pre-diagnosis Accession Number using the 
    'pancreatic cancer timeline' column.
    
    Parameters:
    - accession_number: The Accession Number of the pre-diagnosis report
    - df: The dataframe containing the reports
    
    Returns:
    - The Accession Number of the first malignancy diagnosis for the patient, 
      or None if no malignancy is found.
    """
    
    # Get the patient based on the provided Accession Number
    patient_row = df[df[id_column] == accession_number]
    
    if patient_row.empty:
        raise ValueError(f"No report found with Accession Number {accession_number}")
    
    # Get the patient's Assigned Number
    assigned_number = patient_row['Assigned Number'].values[0]
    
    # Filter the reports for the same patient (Assigned Number)
    patient_reports = df[df['Assigned Number'] == assigned_number]
    
    # Sort the patient's reports by 'Exam Started Date' to ensure chronological order
    patient_reports = patient_reports.sort_values(by='Exam Started Date')
    
    # Find the first report where 'pancreatic cancer timeline' is 'first diagnosis'
    first_diagnosis_row = patient_reports[patient_reports['pancreatic cancer timeline'] == 'first positive'].head(1)
    
    if not first_diagnosis_row.empty:
        print(f"First malignancy diagnosis found for patient with Accession Number {accession_number}:")
        return first_diagnosis_row[id_column].values[0]
    else:
        raise ValueError(f"No first malignancy diagnosis found for patient with Accession Number {accession_number}")
            

def write_tumor_multi_rows(writer, sample, tumors, answer, multi_organ=False,report=None, step=None):
    for tumor_id, tumor_data in tumors.items():
        size = tumor_data.get('size', [])
        
        if isinstance(size, (float, int)):  # Single numeric value
            #check if nan
            if np.isnan(size):
                size_str='u'
            else:
                size_str = f"{size} mm"
        elif isinstance(size, list):  # List of numeric values
            size_str = " x ".join(map(str, size))
        elif size == 'multiple':  # Handle 'multiple' cases
            size_str = "multiple"
        else:  # Handle unexpected cases
            size_str = "U"

        if step=='HCC':
            #{'liver tumor 1': {'type': 'hcc', 'certainty': 'certain', 'size': nan, 'location': 'segment 6', 'arterial enhancement': 'absent', 'washout': 'absent', 'capsule': 'u', 'threshold growth': 'u', 'LI-RADS': 'lr-tr nonviable'}}

            # Build the output row based on the step.
            # For HCC, we require all fields.
            row = [
                sample,
                tumor_id,
                tumor_data.get('type', 'U'),
                tumor_data.get('certainty', 'U'),
                size_str,
                tumor_data.get('location', 'U'),
                tumor_data.get('arterial enhancement', 'U'),
                tumor_data.get('washout', 'U'),
                tumor_data.get('capsule', 'U'),
                tumor_data.get('threshold growth', 'U'),
                tumor_data.get('LI-RADS', 'U'),
                answer
            ]
            if report is not None:
                row.append(report)
        else:
            row = [
                sample,
                tumor_id,
                tumor_data.get('organ', np.nan),
                tumor_data.get('type', np.nan),
                tumor_data.get('location', np.nan),
                size_str,
                tumor_data.get('attenuation', np.nan),
                tumor_data.get('certainty', np.nan),
                answer  # Add the raw LLM answer to the row
            ]
        if report is not None:
            row.append(report)
        writer.writerow(row)



def inference_loop(data, base_url='http://0.0.0.0:8888/v1', step='tumor detection', outputs={}, examples=0, fast=True,
                   institution='UCSF', save_name=None, restart=False,item_list=None):
    """
    ### Function Documentation: 'inference_loop'
    
    #### Summary:
    This function processes medical data, performs inference on tumor detection or classification tasks, and saves or updates results. It supports various steps, such as tumor detection, malignancy detection, and type/size analysis, and includes logic to handle different data structures and institutions.
    
    #### Parameters:
    - **data** ('pd.DataFrame'): Input dataset containing radiology/pathology records.
    - **base_url** ('str'): URL of the inference API, the 4 numbers at the end are the ones you used in VLLM serve. Default is ''http://0.0.0.0:8888/v1''. 
    - **step** ('str'): Task to perform (e.g., ''tumor detection'', ''malignancy detection'', ''type and size', 'type and size pathology').
    - **outputs** ('dict' or 'list'): Previous results to be updated. Defaults to an empty dictionary.
    - **examples** ('int'): Number of examples for contextual inference. Default is '0'.
    - **fast** ('bool'): Flag to enable fast processing by reducing prompt size. Not available to all steps. Reduces accuracy. Default is 'True'.
    - **institution** ('str'): Institution name, which affects column naming and processing logic. Default is ''UCSF'', can be COH.
    - **save_name** ('str'): Name of the CSV file to save results. Default is 'None'.
    - **restart** ('bool'): Whether to restart processing by clearing saved results. Default is 'False'. Careful.
    - **item_list** ('list'): List of specific items to process. Default is 'None'.
    
    #### Returns:
    - **'pd.DataFrame'**: Updated outputs as a DataFrame containing processed results.
    
    #### Steps:
    1. **Institution-Specific Initialization**:
       - Sets column names and organ list based on the institution.
       - Validates the presence of necessary columns.
    
    2. **Save File Handling**:
       - Initializes or reads a save file if specified.
       - Manages restarting logic by clearing or creating the file.
    
    3. **Outputs Preparation**:
       - Converts 'outputs' to a dictionary if necessary.
       - Handles duplicate entries.
    
    4. **Processing Loop**:
       - Iterates over organs (if applicable) and rows of the dataset.
       - Skips already processed or irrelevant samples.
       - Retrieves relevant report text for inference.
       - Selects examples if needed.
    
    5. **Inference and Interpretation**:
       - Sends data to an inference API.
       - Interprets and updates outputs based on the specified step.
    
    6. **Saving Results**:
       - Appends results to the specified save file in the appropriate format.
    
    7. **Return Results**:
       - Returns updated outputs as a DataFrame.
    
    #### Supported Steps:
    - ''tumor detection'': Detects tumors across specified organs.
    - ''malignancy detection'': Identifies malignancy in detected tumors.
    - ''malignant size'': Determines the size of malignant tumors.
    - ''type and size'': Analyzes tumor type and size.
    - ''type and size multi-organ'': Multi-organ tumor type and size analysis.
    - ''diagnoses'': Adds abnormality and inference results to the dataset.
    - ''time machine'': Performs longitudinal analysis using future reports.
    
    #### Example Usage:
    '''python
    # Example: Tumor detection with a save file
    results = inference_loop(
        data=my_data,
        base_url='http://127.0.0.1:8000',
        step='tumor detection',
        save_name='tumor_detection_results.csv',
        restart=False
    )
    """
    outputs = copy.deepcopy(outputs)

    if institution == 'UCSF':
        id_column = 'Anon Acc #'
        #check if the columns are present
        if id_column not in data.columns:
            id_column='Acc #'
        if id_column not in data.columns:
            id_column='id'
        if id_column not in data.columns:
            id_column='Encrypted Accession Number'
        if id_column not in data.columns:
            id_column='Accession Number'
        if id_column not in data.columns:
            id_column='BDMAP ID'
        if id_column not in data.columns:
            id_column='BDMAP_ID'
        organs = ['liver', 'kidney', 'pancreas']
        
        if step=='normal confirmation':
            organs=['pancreas']
        report_column = 'Anon Report Text'
        if report_column not in data.columns:
            report_column='Findings'
        if report_column not in data.columns:
            report_column='Report Text'
        if report_column not in data.columns:
            report_column='Report'
        if report_column not in data.columns:
            report_column='report'
        if step=='HCC':
            organs=['liver']
        if step=='longitudinal pancreas' or step=='longitudinal pancreas diagnosis':
            organs=['pancreas']
            id_column='Patient ID'
        if step=='pre-diagnostic confirmation':
            organs=['pancreas']
            report_column='oldest pre-diagnostic report'
            id_column='Patient ID'
            if report_column not in data.columns:
                report_column='Report'
                id_column='Encrypted Accession Number'
        labels = True
        if report_column not in data.columns:
            raise ValueError('No report column found')
        if id_column not in data.columns:
            raise ValueError('No ID column found')
    else:
        if step=='longitudinal pancreas diagnosis':
            organs=['pancreas']
            id_column='Patient ID'
            report_column='Findings'
            print('Columns:',data.columns)
        else:
            id_column = 'Accession Number'
            organs = ['pancreas']
            report_column = 'Report Text'
            labels = False

    if step == 'tumor detection' or step == 'diagnoses':
        lp=['']#no need to loop multiple times
    else:
        lp=organs

    saved = set()  # Use a set to store processed samples for fast lookup

    if step == 'diagnoses':
        header=data.columns.tolist()+['Abnormalities', 'DNN answer']

    # Check if the save file exists and restart is False
    if save_name is not None:
        if save_name[-4:]=='.csv':
            save_name=save_name[:-4]
        if step.replace(' ', '_') not in save_name:
            save_name = save_name + '_' + step.replace(' ', '_')
        save_name+='.csv'
        file_path = save_name
        print('Save path:',file_path)

        if os.path.exists(file_path) and not restart:
            # Check if file is not empty
            if os.stat(file_path).st_size > 0:
                # Read the first column (Anon Acc # or Accession Number) to skip already processed samples
                print(pd.read_csv(file_path))
                #raise ValueError('Save not implemented')
                saved = set(pd.read_csv(file_path)[id_column].tolist())
                #saved = set(pd.read_csv(file_path).iloc[:, 0].tolist())
            else:
                saved = set()
        else:
            saved = set()
        
        print('Number of saved samples:', len(saved))

        # If restart is True, delete the existing file and reset
        if restart and os.path.exists(file_path):
            os.remove(file_path)

        # If the file does not exist or was deleted, create it and write headers
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                if step == 'tumor detection':
                    writer.writerow([id_column, 'Liver Tumor', 'Kidney Tumor', 'Pancreas Tumor', 'DNN answer'])
                elif step == 'normal confirmation':
                    
                    writer.writerow([id_column, 'Any Tumor', 'Metastasis', 'Pancreatic Tumor Suspicion', 'Pancreatic Non-Tumor Abnormality', 'Healthy Pancreas', 'report', 'DNN answer'])
                elif step=='pre-diagnostic confirmation':
                    if id_column=='Patient ID':
                        writer.writerow([id_column, 'Pancreatic Tumor Suspicion in First Pre-diagnositc', 'Pancreas Surgery in First Pre-diagnositc', 'report', 'DNN answer'])
                    else:
                        writer.writerow([id_column, 'Pancreatic Tumor Suspicion', 'Pancreas Surgery ', 'DNN answer'])
                elif step == 'malignancy detection':
                    writer.writerow([id_column, 'Liver Tumor', 'Kidney Tumor', 'Pancreas Tumor', 'DNN answer', 'Malignant Tumor in '+organs[0], 'DNN answer 2'])
                elif step == 'malignant size':
                    writer.writerow([id_column, 'Liver Tumor', 'Kidney Tumor', 'Pancreas Tumor', 'DNN answer', 'Malignant Tumor in '+organs[0], 'DNN answer 2', 'Size of Largest Malignant Tumor in '+organs[0], 'DNN answer 3'])
                elif step == 'time machine':#use information from future report to understand if past report shows a malignant tumor
                    writer.writerow([id_column, 'Liver Tumor', 'Kidney Tumor', 'Pancreas Tumor', 'DNN answer', 'Malignant Tumor in '+organs[0], 'DNN answer 2', 'Size of Largest Malignant Tumor in '+organs[0], 'DNN answer 3', 'Longitudinal Analysis: Very Likely Malignancy in '+organs[0], 'Longitudinal Analysis: Size of Very Likely Largest Malignant Tumor in '+organs[0], 'DNN answer 4'])
                elif step == 'type and size' or step=='type and size pathology':
                    writer.writerow([id_column, "Tumor ID", "Tumor Type", "Type Certainty", "Tumor Size", "Tumor Location"])
                elif step == 'type and size multi-organ':
                    writer.writerow([id_column, "Tumor ID", "Organ", "Tumor Type", "Tumor Location", "Tumor Size (mm)", "Tumor Attenuation", "Type Certainty","DNN Answer","Report"])
                elif step == 'HCC':
                    writer.writerow([id_column, "Tumor ID", "Tumor Type", "Type Certainty", "Tumor Size", "Tumor Location", "Arterial Enhancement", "Washout", "Capsule", "Threshold Growth", "LI-RADS","DNN Answer","Report"])
                elif step == 'diagnoses':
                    header=data.columns.tolist()+['Abnormalities', 'DNN answer']
                    writer.writerow(header)
                elif step == 'longitudinal pancreas':
                    writer.writerow([id_column, "First Diagnosis", "Pre-diagnosis", "Ordered Accessions","reports","DNN Answer"])
                elif step == 'longitudinal pancreas diagnosis': 
                    writer.writerow([id_column, "Tumor Types","reports","DNN Answer"])
                    
    # Convert outputs to dictionary if needed
    if isinstance(outputs, list):
        pass
    elif not isinstance(outputs, dict):
        # Check for duplicates and handle them
        if outputs[id_column].duplicated().any():
            print(f"Warning: There are duplicate values in the {id_column} column. We are taking only the first occurrence.")
            raise ValueError('Duplicated values in the id column')
            outputs = outputs.drop_duplicates(subset=id_column)

       #print('outputs:',outputs)
        outputs = outputs.set_index(id_column).to_dict(orient='index')

    if step == 'type and size' or step == 'type and size multi-organ' or step=='type and size pathology':
        old_step=copy.deepcopy(outputs)
        outputs = {}
        

    # Loop over organs and data
    if step == 'longitudinal pancreas' or step == 'longitudinal pancreas diagnosis':
        #here, we loop over patients, not over reports
        cases = data['Patient ID'].unique().tolist()
    else:
        cases = range(data.shape[0])
    
    for organ in lp:
        for i in cases:
            start=time.time()
            if step != 'longitudinal pancreas' and step != 'longitudinal pancreas diagnosis':
                sample = data.iloc[i][id_column]
                report,_=get_report_n_label(data,i,row_name=report_column,get_date=False,id_col=id_column) 

                # Skip if the sample is in the saved list
                if sample in saved:
                    print(f'Skipping sample, already saved: {sample}')
                    continue
                if item_list is not None:
                    if sample not in item_list:
                        continue
                else:
                    if step == 'malignancy detection':
                        # Skip if the outputs do show no tumor in the organ
                        if outputs[sample][organ.capitalize()+' Tumor'] != 1.0:
                            print(f'Skipping sample, no certain tumor in {organ}: {sample}')
                            continue

                    if step == 'type and size' or step=='type and size pathology':
                        if isinstance(old_step, dict):
                            # Skip if the outputs do show no tumor in the organ
                            if sample in old_step and old_step[sample][organ.capitalize()+' Tumor'] != 1.0:
                                print(f'Skipping sample, no certain tumor in {organ}: {sample}')
                                continue
                        elif isinstance(old_step, list):
                            if sample not in old_step:
                                print(f'Skipping sample, no certain tumor in {organ}: {sample}')
                                continue

                    if step == 'malignant size':
                        if sample not in outputs:
                            print(f'Skipping sample, not yet predicted for malignancy: {sample}')
                            continue
                        # Skip if the outputs do show no tumor in the organ
                        if outputs[sample][organ.capitalize()+' Tumor'] != 1.0 or outputs[sample]['Malignant Tumor in '+organ] != 1.0:#measuring certain and uncertain tumors
                            print(f'Skipping sample, no certain malignant tumor in {organ}: {sample}')
                            continue

            if step == 'time machine':
                print(data.iloc[i]['pancreatic cancer timeline'])
                #check if data.iloc[i]['pancreatic cancer timeline'] is nan
                if not isinstance(data.iloc[i]['pancreatic cancer timeline'],str):
                    print(f'Skipping sample, no pre-diagnosis report: {sample}')
                    continue
                elif 'pre-diagnosis' not in data.iloc[i]['pancreatic cancer timeline']:
                    print(f'Skipping sample, no pre-diagnosis report: {sample}')
                    continue
                
                #get report of first diagnosis
                first_diagnosis=get_first_malignancy(sample, data, id_column=id_column)

                print('First diagnosis:',first_diagnosis)
            else:
                first_diagnosis=None
                
            if step == 'longitudinal pancreas' or step == 'longitudinal pancreas diagnosis':
                sample = i
                _,reports = get_longitudinal_reports(data, sample)
                #concat list of strings
                reports = '\n\n'.join(reports)
                print(f"Processing patient {sample}...")



            # Example selection logic
            if examples == 0:
                ex = []
            else:
                if institution != 'UCSF':
                    raise ValueError('Only UCSF institution is supported for examples')
                ex = get_random_examples(target=i, num=examples, limit=data.shape[0] - 1, data=data, step=step, organ=organ, outputs=outputs)



            answer = run(target=i, examples=ex, data=data, print_message=False, base_url=base_url, step=step, organ=organ, fast=fast,
                         row_name=report_column, id_column=id_column,future_report=first_diagnosis)
            
            if step == 'longitudinal pancreas' or step == 'longitudinal pancreas diagnosis':
                answer,accessions=answer
            
            # Interpret the output
            print('Arrived here')
            print('Answer:',answer)

            out = interpret_output(answer, step=step, organ=organ)
            
            print('Out:',out)

            # Update the outputs based on the step
            if step == 'tumor detection' or step == 'type and size' or step == 'type and size multi-organ' or step=='type and size pathology' or step == 'HCC' or step == 'longitudinal pancreas' or step == 'longitudinal pancreas diagnosis' or step == 'pre-diagnostic confirmation' or step == 'normal confirmation':
                outputs[sample] = out
            elif step == 'malignancy detection' or step == 'malignant size' or step == 'time machine':
                #print('Outputs:',outputs)
                outputs[sample].update(out)
            elif step=='diagnoses':
                row=data.iloc[i].to_dict()
                row['Abnormalities']=out
                row['DNN answer']=answer
                outputs[sample]=row
            
                
            if step == 'diagnoses':
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        tmp=[]
                        for h in header:
                            tmp.append(outputs[sample][h])
                        writer.writerow(tmp)
            elif step == 'normal confirmation':
                if save_name is not None:
                #id_column, 'Any Tumor', 'Metastasis', 'Pancreatic Tumor Suspicion', 'Pancreatic Non-Tumor Abnormality', 'Healthy Pancreas', 'report', 'DNN answer'
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        writer.writerow([sample]+[outputs[sample]['Any Tumor'],outputs[sample]['Metastasis'],outputs[sample]['Pancreatic Tumor Suspicion'],outputs[sample]['Pancreatic Non-Tumor Abnormality'],outputs[sample]['Healthy Pancreas'],report,answer])
            elif step == 'longitudinal pancreas':
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        writer.writerow([sample]+[outputs[sample]['First Diagnosis Report'],outputs[sample]['Pre-Diagnosis Reports'],accessions,reports,answer])
            elif step == 'pre-diagnostic confirmation':
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        writer.writerow([sample]+[outputs[sample]['Pancreatic Tumor Suspicion'],outputs[sample]['Pancreas Surgery'],report,answer])
                else:
                    raise ValueError('not saving')
            elif step == 'longitudinal pancreas diagnosis':
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        writer.writerow([sample]+[outputs[sample]['Tumor Types'],reports,answer])
            elif step != 'type and size' and step != 'type and size multi-organ' and step!='type and size pathology' and step != 'HCC':
                # Append the new result to the CSV
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        writer.writerow([sample] + list(outputs[sample].values()) + [answer])
            elif step == 'type and size' or step=='type and size pathology' or step == 'HCC':
                 with open(file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    print('Outputs:',outputs[sample])
                    write_tumor_multi_rows(writer, sample, outputs[sample], answer,step=step,report=report)
            elif step == 'type and size multi-organ':
                with open(file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    print('Outputs:',outputs[sample])
                    write_tumor_multi_rows(writer, sample, outputs[sample], answer, multi_organ=True,report=report)

            print(f"Processed sample {sample} in {time.time() - start:.2f} seconds")

    if step != 'type and size' and step != 'type and size multi-organ' and step!='type and size pathology' and step != 'HCC':
        # Return updated outputs as a DataFrame
        outputs = pd.DataFrame.from_dict(outputs, orient='index')
        outputs.reset_index(inplace=True)
        outputs.rename(columns={'index': id_column}, inplace=True)

        # If institution is UCSF, generate metrics
        #if institution == 'UCSF' and 'size' not in step:
        #    generate_metrics(data, outputs, step=step)

    return outputs



# extract abnormal findings

abnormality_prompt_v0="""Your task is to extract and list all abnormal findings from a CT report provided below. 
Desired output format: provide me a python list of python dictionaries. Each dictionary should refer to an abnormality in the report. Each dictionary should have the following keys: abnormality, organ, location inside organ, size, and certainty. If you cannot infer any of these characteristics from the report, leave them as None.
Consider the following guidelines:
1- Abnormalities can be image findings, like lesions or ground-glass opacity or lesions, or diagnoses, like pneumonia or cancer.
2- If the report mentions a finding, but does not specify whether it is normal or abnormal, assume it is abnormal.
3- If a report presents both image findings and diagnoses, list them separately, even if they refer to the same abnormality.
4- If the report mentions no abnormalities, return an empty list.

"""

abnormality_prompt = """
Your task is to extract and list all **abnormal findings** from the CT report provided below ("CT report to analyze").  

**Output Format**  
Return a **Python list of dictionaries**, name it "abnormalities". Each dictionary should represent one abnormality and include the following keys:  
- **abnormality**:  type abnormality (e.g., ground-glass opacity, lesion, pneumonia). Avoid descriptions here. For example, use "lesion" instead of "large hypodense lesion". Be susinct and use the most standard terminology (in the singular), if possible.
- **organ**: The organ where the abnormality is found (e.g., lung, liver).  
- **location_inside_organ**: Specific location within the organ, if provided (e.g., upper lobe, segment VI).  
- **size**: The size of the abnormality, if mentioned (e.g., 2.5 cm).  
- **certainty**: Level of certainty mentioned in the report, characterize as high, medium or low. If nothing suggestest uncertainty, consider it high. 
- **description**: All the report's sentences related to the abnormality. The sentences should be **directly copied** from the report, just remove any personal name, patient MRN or Accession Number, if present. Copy all the sentences that refer to the abnormality or are even remotely related to it, even if they are not contiguous.

If organ, location_inside_organ, description or size cannot be inferred from the report, leave it as 'None'.

**Guidelines**  
1. **Types of Abnormalities**: Abnormalities can be **image findings** (e.g., lesions, ground-glass opacity) or **diagnoses** (e.g., pneumonia, cancer).  
2. **Assume Abnormal**: If the report mentions a finding but does not clarify whether it is normal or abnormal, assume it is **abnormal**.  
3. **Separate Listings**: If a report mentions both **image findings** and **diagnoses** referring to the same abnormality, list them **separately** as distinct entries.
4. **Lesion**: Do not use very generic terms as lesion if a more specific term is available (e.g., cyst, nodule, mass, trauma).
5. **No Abnormalities**: If the report mentions no abnormalities, return an **empty list**. "Unremarkable" findings are not considered abnormalities.
6. **Organ**: Use the most common organ name (e.g., “lungs” instead of “pulmonary parenchyma”). Avoid system names or very broad terms like “GI tract”, “reproductive organs” or "abdomen". Instead, use specific terms like “esophagus” or “uterus.”


**Example Input**  
CT Report:  
"There is a 2.5 cm hypodense lesion in segment VI of the liver involving the hepatic artery. Findings are suspicious for metastasis. Bilateral ground-glass opacities are noted in the lungs."

**Example Output**  
abnormalities = [
    {"abnormality": "lesion", "organ": "liver", "location_inside_organ": "segment VI", "size": "2.5 cm", "certainty": "high", "description": "There is a 2.5 cm hypodense lesion in segment VI of the liver involving the hepatic artery. Findings are suspicious for metastasis."},  
    {"abnormality": "ground-glass opacities", "organ": "lungs", "location_inside_organ": "bilateral", "size": None, "certainty": "high", "description": "Bilateral ground-glass opacities are noted in the lungs."},  
    {"abnormality": "metastasis", "organ": "liver", "location_inside_organ": None, "size": None, "certainty": "medium", "description": "There is a 2.5 cm hypodense lesion in segment VI of the liver involving the hepatic artery. Findings are suspicious for metastasis."}  
]

**Example Input**
CT Report:
Visualized lung bases: For chest findings, please see the separately dictated report from the CT of the chest of the same date.
Liver: Unremarkable
Gallbladder: Unremarkable
Spleen:  Unremarkable
Pancreas:  Unremarkable
Adrenal Glands:  Unremarkable
Kidneys:  Unremarkable
GI Tract:  Scattered colonic diverticula without evidence of diverticulitis.
Vasculature:  Unremarkable
Lymphadenopathy: Absent
Peritoneum: No ascites
Bladder: Unremarkable
Reproductive organs: Reproductive organs are surgically absent. Left pelvic sidewall multiloculated cystic lesion measures 3.5 x 2.2 cm on series 5 image 134. This was unchanged when compared to prior outside MRI from 10/12/2015.
Bones:  No suspicious lesions
Extraperitoneal soft tissues: Unremarkable
Lines/drains/medical devices: None
Impressions:
Joseph Hopkins has scattered colonic diverticula and multiloculated lesion on the left pelvic sidewall, unchanged from prior MRI.

**Example Output** 
abnormalities = [
    {"abnormality": "diverticulum", 
     "organ": "colon", 
     "location_inside_organ": None, 
     "size": None, 
     "certainty": "high",
     "description": "GI Tract:  [name removed] has scattered colonic diverticula without evidence of diverticulitis. Scattered colonic diverticula and multiloculated lesion on the left pelvic sidewall, unchanged from prior MRI."},
    
    {"abnormality": "cyst", 
     "organ": "pelvis", 
     "location_inside_organ": "left pelvic sidewall", 
     "size": "3.5 x 2.2 cm", 
     "certainty": "high",
     "description": Left pelvic sidewall multiloculated cystic lesion measures 3.5 x 2.2 cm on series 5 image 134. This was unchanged when compared to prior outside MRI from 10/12/2015. Scattered colonic diverticula and multiloculated lesion on the left pelvic sidewall, unchanged from prior MRI."}
]

CT report to analyze:
"""

group_synonyms = """I will provide you a list of diseases and findings taken from CT scan reports. Some of the names are synonyms or abbreviations of the same disease/finding. Your task is to group them together.
Output format: provide me a python dictionary where each key is a group of synonyms and the value is a list of all the synonyms in that group. You can take as key the most usual term in the group."""

def get_diagnoses(diagnoses_csv):
    # Load the CSV file
    diag = pd.read_csv(diagnoses_csv)
    diagnoses=[]
    errors=0
    for item in diag['Abnormalities'].str.replace('\n','').to_list():
        try:
            x=ast.literal_eval(item)
        except:
            errors+=1
            continue
        if type(x)==list and len(x)>0:
            for item in x:
                try:
                    diagnoses.append(item['abnormality'])
                except:
                    errors+=1

    print('Errors:',errors)

    return list(set(diagnoses))





#analyze outputs

def load(path):
    df=pd.read_csv(path)
    df=df.drop_duplicates(subset=['id'])
    return df

def get_abnormalities(diag,diag_save_path=None):
    if isinstance(diag,str):
        diag=load(diag)
    diagnoses=[]
    errors=0
    for item in diag['Abnormalities'].str.replace('\n','').to_list():
        try:
            x=ast.literal_eval(item)
        except:
            errors+=1
            continue
        if type(x)==list and len(x)>0:
            for item in x:
                try:
                    diagnoses.append(item['abnormality'])
                except:
                    errors+=1

    diagnoses=list(set(diagnoses))

    print(f'Errors: {errors}')
    print(f'Number of abnormal findings: {len(diagnoses)}')

    if diag_save_path is not None:
        df = pd.DataFrame(diagnoses, columns=["Diagnoses"])
        df.to_csv(diag_save_path, index=False)

    return diagnoses


get_diagnoses_0 = """
Below is a large list of radiological findings and diagnoses. Many items in this list may be synonyms or near-synonyms referring to the same underlying concept. I would like you to produce a Python dictionary that groups these terms together. Specifically, follow these instructions:

- For each concept that has synonyms or closely related terms, choose one standard radiological term as the key.
- Under that key, list all variants, synonyms, or closely related terms from the provided list as the dictionary value (a list of strings).
- If a term does not have any synonyms, it should appear as a key with a single-value list containing just that term.
- Focus on grouping terms that radiologists would consider essentially synonymous or describing the same imaging finding.
- Preserve all unique terms from the original list so that every term appears in the final dictionary, either as a single-item list or grouped under one of the synonyms.
- The values of the dictionary should include all terms from the original list, and no term should be repeated in multiple groups.
- Unless necessary, the organ name should not be included in the key term. For example, use "cyst" instead of "renal cyst" or "liver cyst".
- For each key in the dictionary, the corresponding value should contain all synonyms for that key, **including the key itself**.

Here is the list of findings/diagnoses to process:

%(diagnoses)s

Please provide the dictionary as Python code, using a dictionary literal with keys as strings and values as lists of strings. Call the dictionary "synonyms". Remember that ALL findings above must appear in the dictionary.
"""

get_diagnoses = """
Below is a large list of radiological findings and diagnoses. Many items in this list may be synonyms or near-synonyms referring to the same underlying concept. I would like you to produce a Python dictionary that groups these terms together. Specifically, follow these instructions:


- For each concept that has synonyms or closely related terms, choose one standard radiological term as the key.
- Under that key, list all variants, synonyms, or closely related terms from the provided list as the dictionary value (a list of strings).
- If a term does not have any synonyms, it should appear as a key with a single-value list containing just that term.
- Focus on grouping terms that radiologists would consider essentially synonymous or describing the same imaging finding.
- Preserve all unique terms from the original list so that every term appears in the final dictionary, either as a single-item list or grouped under one of the synonyms.
- The values of the dictionary should include all terms from the original list, and no term should be repeated in multiple groups.
- Unless necessary, the organ name should not be included in the key term. For example, use "cyst" instead of "renal cyst" or "liver cyst".
- For each key in the dictionary, the corresponding value should contain all synonyms for that key, **including the key itself**.

Here is the list of findings/diagnoses to process:

%(diagnoses)s

Please try to get keys for your dictionary from the finsings below. In case some findings are not synonyms with any of the findings below, you can create new keys for them. Remember that ALL findings above must appear in the dictionary.

%(synonyms)s
"""
def merge_dicts(dict1, dict2):
    """
    Merges two dictionaries. For shared keys, combines values and removes duplicates.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        dict: Merged dictionary with combined and deduplicated values for shared keys.
    """
    merged_dict = {}

    # Get all keys from both dictionaries
    all_keys = set(dict1.keys()).union(set(dict2.keys()))

    for key in all_keys:
        # Fetch values from both dictionaries; default to empty list
        values1 = dict1.get(key, [])
        values2 = dict2.get(key, [])
        
        # Ensure values are lists for easy merging
        if not isinstance(values1, list):
            values1 = [values1]
        if not isinstance(values2, list):
            values2 = [values2]
        
        # Combine values and remove duplicates
        merged_dict[key] = list(set(values1 + values2))

    return merged_dict

def summarize_diagnoses(diagnoses,base_url='http://0.0.0.0:8000/v1',batch=100,save_name=None):
    if isinstance(diagnoses,str):
        diagnoses=get_abnormalities(diagnoses)
    start=0
    end=batch
    non_added_values=[]
    while True:
        print('Start:',start)
        if end>len(diagnoses):
            end=len(diagnoses)
        d=diagnoses[start:end]+non_added_values

        keys=''
        for item in d:
            keys+=item+', '

        if start==0:
            prompt=get_diagnoses_0 % {'diagnoses':str(d)}
        else:
            prompt=get_diagnoses % {'diagnoses':str(d),'synonyms':keys}
        message= [{"role": "system", "content": system+' \n '+observations},
                  {"role": "user", "content": prompt}]
        
        if start!=0:
            old_syn=copy.deepcopy(synonyms)

        conver,answer=SendMessageAPI(text=None, conver=message, base_url=base_url)
            
        new_syns=interpret_output(answer,step='synonyms')
        try:
            new_syns=ast.literal_eval(new_syns)
        except:
            prompt="""You returned a non-valid python dictionary, I had an error using ast.literal_eval() for it. Please provide a valid python dictionary named synonyms. Answer with just the disctionary."""
            conver.append({"role": "user", "content": prompt})
            conver,answer=SendMessageAPI(text=None, conver=conver, base_url=base_url)
            new_syns=interpret_output(answer,step='synonyms')
            new_syns=ast.literal_eval(new_syns)

        if start!=0:
            synonyms=merge_dicts(synonyms, new_syns)
        else:
            synonyms=new_syns

        if start!=0:

            #check if all values are in synonyms
            new_values = list(chain.from_iterable(synonyms.values()))
            non_added_values=[]
            for value in d:
                if value not in new_values:
                    non_added_values.append(value)

            if len(non_added_values)>0:
                prompt="""The synonyms dictionary you provide is incomplete. Please add to it the itens below and send me the updated dictionary (the entire dictionary, name it synonyms). 
                        If the synonyms below are not synonyms with any of the itens in the dictionary, you can create new keys for them. Otherwise, add them to the existing keys. \n"""
                prompt+='Add these findings to values:'
                prompt+=str(non_added_values)
                print('Missed values:',non_added_values)
                conver.append({"role": "user", "content": prompt})

                _,answer=SendMessageAPI(text=None, conver=conver, base_url=base_url)
                print('Answer:',answer)
                new_syns=interpret_output(answer,step='synonyms')
                try:
                    new_syns=ast.literal_eval(new_syns)
                except:
                    prompt="""You returned a non-valid python dictionary, I had an error using ast.literal_eval() for it. Please provide a valid python dictionary named synonyms. Answer with just the disctionary."""
                    conver.append({"role": "user", "content": prompt})
                    conver,answer=SendMessageAPI(text=None, conver=conver, base_url=base_url)
                    new_syns=interpret_output(answer,step='synonyms')
                    new_syns=ast.literal_eval(new_syns)
                synonyms=merge_dicts(synonyms, new_syns)

                new_values = list(chain.from_iterable(synonyms.values()))
                non_added_values=[]
                for value in d:
                    if value not in new_values:
                        non_added_values.append(value)
                if len(non_added_values)>0:
                    print('Still not added values:',non_added_values)

        #check if all values previously in synonyms are still there


        print('# of synonym groups:',len(synonyms))
        print('# of words in synonyms:',len(list(chain.from_iterable(synonyms.values()))))

        if save_name is not None:
            #remove file if it exists
            if start==0 and os.path.exists(save_name):
                os.remove(save_name)
            with open(save_name, 'w') as file:
                file.write(str(synonyms))

        start+=batch
        end+=batch
        if start>=len(diagnoses):
            break

    return synonyms

def get_standard_key(finding, synonyms_dict,sub_organ=None):
    """
    Given a finding (string) and the synonyms_dict (a dictionary where
    keys are 'standard' terms and values are lists of synonym strings),
    this function returns the key whose value list contains the given finding,
    ignoring case. If no match is found, it returns None.
    """
    finding_lower = finding.lower()
    if sub_organ is not None:
        sub_organ=sub_organ.lower()
    for key, synonym_list in synonyms_dict.items():
        if sub_organ is not None:
            if any(sub_organ == synonym.lower() for synonym in synonym_list):
                return key
        # Check if the lowercase version of finding matches any lowercase synonym
        if any(finding_lower == synonym.lower() for synonym in synonym_list):
            return key
    return None

def count_findings(diagnoses_csv,synonyms_dict,organ='all'):
    import ast
    # Load the CSV file
    diag = load(diagnoses_csv)
    diagnoses={}
    LLM_errors=0
    report=0
    missing_from_synonym_dict=[]
    for item in diag['Abnormalities'].str.replace('\n','').to_list():
        try:
            x=ast.literal_eval(item)
        except:
            LLM_errors+=1
            #print('Error here')
            continue
        general_diag=[]
        if type(x)==list and len(x)>0:
            for item in x:
                if organ!='all':
                    if 'organ' not in item:
                        continue
                    if item['organ'] not in organ:
                        continue
                try:
                    y=item['abnormality']
                except:
                    LLM_errors+=1
                    continue
                if synonyms_dict is not None:
                    d=get_standard_key(y, synonyms_dict)
                else:
                    d=y
                if d is None:
                    missing_from_synonym_dict.append(y)
                    continue
                general_diag.append(d)
                
            general_diag=list(set(general_diag))
            #print(general_diag)
            for d in general_diag:
                if d not in diagnoses:
                    diagnoses[d]=1
                else:
                    diagnoses[d]+=1
            report+=1

    print('LLM errors:',LLM_errors)
    missing_from_synonym_dict=list(set(missing_from_synonym_dict))
    print('Diagnoses:',len(diagnoses))
    print('Missing from synonym dict:',len(missing_from_synonym_dict))
    print('Reports:',report)

    return diagnoses,missing_from_synonym_dict


def plot_top_diseases(results_LLM, N=10, minimum=1, flip_axes=False,organ='all',synonyms_dict=None,font=10):
    """
    Plots a bar chart of the top N diseases by occurrence.
    The largest-occurrence disease will be at the top and bars extend from left to right,
    unless flip_axes is True, in which case the occurrences are plotted on the y-axis.
    """
    disease_dict,_=count_findings(results_LLM,synonyms_dict,organ=organ)
    print('Disease dict:',len(disease_dict))
    # Sort diseases by occurrences in descending order
    sorted_items = sorted(disease_dict.items(), key=lambda x: x[1], reverse=True)

    # Select top N and filter by minimum threshold
    top_items = [item for item in sorted_items[:N] if item[1] >= minimum]

    # Unpack into lists for plotting
    diseases, occurrences = zip(*top_items)

    # Adjust figure size dynamically
    long = max(6, len(top_items) * 0.2)
    short = 6
    if flip_axes:
        plt.figure(figsize=(long, short))
    else:
        plt.figure(figsize=(short, long))

    if flip_axes:
        # Vertical bar plot (occurrences on y-axis)
        plt.bar(diseases, occurrences, color='skyblue')
        plt.ylabel('Occurrences',fontsize=font)
        plt.xlabel('Diseases',fontsize=font)
        title=f'Top {str(N)} Diseases by Occurrence'
        if organ!='all':
            title+=' for '+organ[0].capitalize()
        plt.title(title,fontsize=font)
        plt.xticks(rotation=90, ha="center")  # Rotate x-axis labels for better readability
        plt.gca().margins(x=0.01, y=0.1)  # Reduce internal padding
    else:
        # Horizontal bar plot (default behavior)
        plt.barh(diseases, occurrences, color='skyblue')
        plt.xlabel('Occurrences')
        plt.title(f'Top {N} Diseases by Occurrence',fontsize=font)
        plt.gca().invert_yaxis()  # Largest at the top
        plt.gca().margins(y=0.01, x=0.1)  # Reduce internal padding

    # Apply minimal padding
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    plt.xticks(fontsize=font)  # X-axis ticks font size
    plt.yticks(fontsize=font)  # Y-axis ticks font size
    
    # Show the plot
    plt.show()

possible_cancers= [
    "tumor",
    "mass",
    "metastasis",
    "metastases",
    "carcinomatosis",
    "hepatocellular carcinoma",
    "tumor thrombus",
    "lesion",
    "lesions",
    "nodule",
    "nodules",
    "cystic lesion",
    "sclerotic lesion",
    "sclerotic lesions",
    "lytic lesion",
    "nodular contour",
    "nodular density",
    "nodular opacity",
    "ground-glass nodule",
    "soft tissue mass",
    "cancer",
    "neoplasm",
    'hyperdensity',
    'hypodensity'
]


organ_dict = {
    "liver": ["liver"],
    "kidney": ["kidney", "left kidney","nephrectomy bed"],
    "pancreas": ["pancreas", "pancreatic head"],
    "adrenal gland": ["adrenal gland"],
    "lung": ["lung"],
    "reproductive organs": ["reproductive organs"],
    "vagina": ["vagina"],
    "uterus": ["uterus"],
    "prostate": ["prostate"],
    "spleen": ["spleen"],
    "gi tract": ["gi tract","gastric remnant"],
    "stomach": ["stomach"],
    "duodenum": ["duodenum"],
    "small bowel": ["small bowel", "small bowel mesentery"],
    "rectum": ["rectum"],
    "peritoneum": ["peritoneum", "peritoneum/retroperitoneum"],
    "bone": ["bone", "pubic bone", "right iliac bone", "vertebral body", "skeleton", "sacrum", "acetabulum"],
    "soft tissue": ["soft tissue", "subcutaneous tissue", "retroperitoneal soft tissue",
                    "extraperitoneal soft tissue", "subcutaneous soft tissue"],
    "retroperitoneum": ["retroperitoneum", "left retroperitoneum", "retroperitoneal", "retroperitoneal space"],
    "ovary": ["ovary", "ovarie"],
    "bladder": ["bladder"],
    "gallbladder": ["gallbladder"],
    "pelvis": ["pelvis", "right pelvi"],
    "mesentery": ["mesentery", "mesenteric", "mesentery or omentum", "small bowel mesentery"],
    "lymphatic system": ["lymph node", "lymphatic system"],
    "bile duct": ["bile duct", "common bile duct"],
    "vasculature": ["vasculature", "vein", "artery"],
    "abdomen": ["abdomen", "anterior abdominal wall", "hemidiaphragm"],
    "skin": ["skin"],
    "omentum": ["omentum"],
    "pericardium": ["pericardium"],
    "breast": ["breast"],
    "bartholin's gland": ["bartholin's gland"],
    "musculature": ["musculature", "left iliopsoas muscle"],
    "presacral": ["presacral"],
    "endometrium": ["endometrium"],
    "diaphragm": ["diaphragm"],
    "colon": ["colon"],
    "esophagus": ["esophagus"],
}

def count_organs(diagnoses_csv,synonyms_dict,diseases=['cancer','lesion','tumor','hypodensities','hyperdensities'],organ_dict=organ_dict):
    import ast
    # Load the CSV file
    diag = load(diagnoses_csv)
    organ_counts={}
    LLM_errors=0
    report=0
    missing_from_synonym_dict=[]
    for item in diag['Abnormalities'].str.replace('\n','').to_list():
        try:
            x=ast.literal_eval(item)
        except:
            LLM_errors+=1
            #print('Error here')
            continue
        #print(x)
        if type(x)==list and len(x)>0:
            organs_tumor=[]
            for item in x:
                #print(item)
                if diseases!='all':
                    try:
                        disease=item['abnormality']
                        if synonyms_dict is not None:
                            disease=get_standard_key(disease, synonyms_dict)
                        #print(disease)
                        if disease not in diseases:
                            continue
                    except:
                        continue
                try:
                    y=item['organ'].lower()
                    y=get_standard_key(y, organ_dict)
                    if y[-1]=='s' and y not in ['pancreas','uterus','reproductive organs','pelvis']:
                        y=y[:-1]
                except:
                    LLM_errors+=1
                    continue
                organs_tumor.append(y)
            organs_tumor=list(set(organs_tumor))
            for y in organs_tumor:
                if y not in organ_counts:
                    organ_counts[y]=1
                else:
                    organ_counts[y]+=1
            report+=1
                

    print('LLM errors:',LLM_errors)
    print('Reports:',report)

    return organ_counts

def plot_cancer_organs(results_LLM, N=10, minimum=1, flip_axes=False, organ='all', synonyms_dict=None,
                       diseases=possible_cancers, font=20, log_scale=False):
    """
    Plots a bar chart of the top N diseases by occurrence.
    The largest-occurrence disease will be at the top and bars extend from left to right,
    unless flip_axes is True, in which case the occurrences are plotted on the y-axis.
    """
    disease_dict = count_organs(results_LLM, synonyms_dict, diseases=diseases)
    print('Disease dict:', disease_dict)
    print('Disease dict:', len(disease_dict))
    
    # Sort diseases by occurrences in descending order
    sorted_items = sorted(disease_dict.items(), key=lambda x: x[1], reverse=True)

    # Select top N and filter by minimum threshold
    top_items = [item for item in sorted_items[:N] if item[1] >= minimum]

    # Unpack into lists for plotting
    diseases, occurrences = zip(*top_items)

    # Adjust figure size dynamically
    long = max(6, len(top_items) * 0.2)
    short = 6
    if flip_axes:
        plt.figure(figsize=(long, short))
    else:
        plt.figure(figsize=(short, long))

    if flip_axes:
        # Vertical bar plot (occurrences on y-axis)
        plt.bar(diseases, occurrences, color='skyblue', log=log_scale)
        plt.ylabel('Occurrences', fontsize=font)
        plt.xlabel('Organs', fontsize=font)
        title = f'Number of tumor reports per organ'
        plt.title(title)
        plt.xticks(rotation=90, ha="center")  # Rotate x-axis labels for better readability
        plt.gca().margins(x=0.01, y=0.1)  # Reduce internal padding
    else:
        # Horizontal bar plot (default behavior)
        plt.barh(diseases, occurrences, color='skyblue', log=log_scale)
        plt.xlabel('Occurrences', fontsize=font)
        plt.title(f'Number of tumor reports per organ', fontsize=font)
        plt.gca().invert_yaxis()  # Largest at the top
        plt.gca().margins(y=0.01, x=0.1)  # Reduce internal padding

    # Apply minimal padding
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    
    # Tick Labels
    plt.xticks(fontsize=font)  # X-axis ticks font size
    plt.yticks(fontsize=font)  # Y-axis ticks font size

    # Show the plot
    plt.show()



def select_disease_organ(data, diseases, organs, organ_dict=organ_dict, synonyms_dict=None):
    llm_out = load(data)
    header = llm_out.columns.to_list()

    LLM_errors = 0
    cases = {}

    rows_to_add = []

    # Use itertuples with name=None to get plain tuples
    for tup in tqdm.tqdm(llm_out.itertuples(index=False, name=None), total=len(llm_out)):
        # Create a dict mapping header -> value directly from the tuple
        row_dict = dict(zip(header, tup))
        
        abnormalities = row_dict.get('Abnormalities')
        if not isinstance(abnormalities, str):
            LLM_errors += 1
            continue

        item = abnormalities.replace('\n', '')

        try:
            x = ast.literal_eval(item)
        except:
            LLM_errors += 1
            continue

        add = False
        row_added = False
        organs_added = []

        try:
            if isinstance(x, list) and len(x) > 0:
                for sub_item in x:
                    organ = sub_item.get('organ', '')
                    sub_organ=sub_item.get('location_inside_organ', '')
                    description = sub_item.get('description', '')
                    if "unremarkable" in description.lower() or "post-operative" in description.lower() or "absen" in description.lower() or "enlarged" in description.lower():
                        continue
                    organ = get_standard_key(organ, organ_dict,sub_organ)
                    diag = sub_item.get('abnormality', '')
                    if synonyms_dict is not None:
                        diag = get_standard_key(diag, synonyms_dict)

                    if diag in diseases and organ not in organs_added:
                        add = True
                        if organ not in cases:
                            cases[organ] = []
                        cases[organ].append(row_dict)
                        organs_added.append(organ)
        except:
            LLM_errors += 1
            continue

        if add:
            rows_to_add.append(row_dict)

    df = pd.DataFrame(rows_to_add, columns=header)
    return df, cases