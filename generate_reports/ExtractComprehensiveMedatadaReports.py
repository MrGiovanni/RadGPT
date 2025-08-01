import pandas as pd
import re
import os
import argparse
#from fpdf import FPDF
import tqdm


import re
import numpy as np


def extract_findings(report):
        """Extract the findings section."""
        findings, _, _ = report.partition("IMPRESSION:")
        return findings.strip() if findings else report.strip()

def extract_organ_sections(findings):
    """Extract sections for Kidney, Liver, and Pancreas."""
    sections = {}
    organ_names = ["Pancreas", "Liver", "Kidney", "Colon", "Spleen"]
    end_markers = ["Spleen", "IMPRESSION"]
    
    for organ in organ_names:
        # Find start of the organ section
        start = findings.find(f"{organ}:")
        if start != -1:
            # Find the end of the section
            end_candidates = [findings.find(marker, start) for marker in organ_names + end_markers if findings.find(marker, start) > start]
            end = min(end_candidates) if end_candidates else len(findings)
            sections[organ] = findings[start:end].strip()
    return sections

header=['BDMAP ID',
        #from metadata file 
        'spacing', 
        'shape', 
        'sex',
        'age',
        'scanner',
        'contrast',
        #from reports
        'liver volume (cm^3)',
        'total liver lesion volume (cm^3)',
        'total liver tumor volume (cm^3)',
        'total liver cyst volume (cm^3)',
        'number of liver lesion instances',
        'number of liver tumor instances',
        'number of liver cyst instances',
        'largest liver lesion diameter (cm)',
        'largest liver cyst diameter (cm)',
        'largest liver tumor diameter (cm)',
        'largest liver lesion location (1-8)',
        'largest liver cyst location (1-8)',
        'largest liver tumor location (1-8)',
        'largest liver lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
        'largest liver cyst attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
        'largest liver tumor attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
        'pancreas volume (cm^3)',
        'total pancreatic lesion volume (cm^3)',
        'total pancreatic tumor volume (cm^3)',
        'total pancreatic PDAC volume (cm^3)',
        'total pancreatic cyst volume (cm^3)',
        'number of pancreatic lesion instances',
        'number of pancreatic tumor instances',
        'number of pancreatic PDAC instances',
        'number of pancreatic cyst instances',
        'largest pancreatic lesion diameter (cm)',
        'largest pancreatic cyst diameter (cm)',
        'largest pancreatic tumor diameter (cm)',
        'largest pancreatic PDAC diameter (cm)',
        'largest pancreatic lesion location (head, body, tail)',
        'largest pancreatic cyst location (head, body, tail)',
        'largest pancreatic tumor location (head, body, tail)',
        'largest pancreatic PDAC location (head, body, tail)',
        'largest pancreatic lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
        'largest pancreatic cyst attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
        'largest pancreatic tumor attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
        'largest pancreatic PDAC attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
        'pancreatic tumor staging (T1-T4)',
        'left kidney volume (cm^3)',
        'right kidney volume (cm^3)',
        'kidney volume (cm^3)',
        'total kidney lesion volume (cm^3)',
        'total kidney cyst volume (cm^3)',
        'total kidney tumor volume (cm^3)',
        'number of kidney lesion instances',
        'number of kidney cyst instances',
        'number of kidney tumor instances',
        'largest kidney lesion diameter (cm)',
        'largest kidney cyst diameter (cm)',
        'largest kidney tumor diameter (cm)',
        'largest kidney lesion location (left, right)',
        'largest kidney cyst location (left, right)',
        'largest kidney tumor location (left, right)',
        'largest kidney lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
        'largest kidney cyst attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
        'largest kidney tumor attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
        'spleen volume (cm^3)',
        'total colon lesion volume (cm^3)',
        'number of colon lesion instances',
        'largest colon lesion diameter (cm)',
        'largest colon lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
        'total esophagus lesion volume (cm^3)',
        'number of esophagus lesion instances',
        'largest esophagus lesion diameter (cm)',
        'total uterus lesion volume (cm^3)',
        'number of uterus lesion instances',
        'largest uterus lesion diameter (cm)',
        'radiologist note',
        'structured report',
        'narrative report',
        'fusion structured report',
        'fusion narrative report',
        ]

def parse_lesions(organ_sections):
    """Extract sections for cysts, lesion, and tumor."""
    organs=['liver','pancreas','kidney','colon']
    #print(organ_sections)
    data={}
    for organ in organs:
        if organ == "Spleen":
            continue
        #print('Organ:',organ)
        sections = {}

        
        #print('Section for:',organ, 'Inside parse_lesions')
        #print(findings)
        lesion_names = ["malignant tumors", "cysts", "lesions","PDACs"]
        end_markers = ["Pancreas:", "Liver:", "Kidney:", "Colon:","Spleen:", "IMPRESSION"]
        #remove organ name from the list
        end_markers.remove(organ.capitalize()+':')

        
        for lesion in lesion_names:
            if organ.capitalize() not in organ_sections or f"{lesion}:" not in organ_sections[organ.capitalize()]:
                print(f"No {lesion} for:",organ)
                data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} diameter (cm)"]=np.nan
                data[f"'total {organ} {lesion.replace('malignant ','')[:-1]} volume (cm^3)"]=np.nan
                data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} attenuation (hyperattenuating, isoattenuating, hypoattenuating)"]=np.nan
                if organ=='pancreas':
                    data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} location (head, body, tail)"]=np.nan
                elif organ=='kidney':
                    data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} location (left, right)"]=np.nan
                elif organ=='liver':
                    data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} location (1-8)"]=np.nan
                else:
                    data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} location"]=np.nan
                data[f"number of {organ} {lesion.replace('malignant ','')[:-1]} instances"]=np.nan
                continue
            findings=organ_sections[organ.capitalize()]
            start = findings.find(f"{lesion}:")
            if start != -1:
                # Find the end of the section
                end_candidates = [findings.find(marker, start) for marker in lesion_names + end_markers if findings.find(marker, start) > start]
                end = min(end_candidates) if end_candidates else len(findings)
                sections[lesion] = findings[start:end].strip()

        #parse each individual lesion in each section[lesion]
        for lesion in sections:
            if sections[lesion] is np.nan:
                continue
            i=1
            tmp={}
            while True:
                lesion_name = f"{organ.capitalize()} tumor {i}:" 
                if lesion_name not in sections[lesion]:
                    lesion_name = f"{organ.capitalize()} lesion {i}:"
                if lesion_name not in sections[lesion]:
                    break

                start = sections[lesion].find(lesion_name)
                if start == -1:
                    break
                #end = sections[lesion].find(f"{organ} {lesion} {i+1}")
                next_starts = [
                    sections[lesion].find(f"{organ} {lesion} {i+1}:"),
                    sections[lesion].find(f"{organ.capitalize()} lesion {i+1}:"),
                    sections[lesion].find(f"{organ.capitalize()} tumor {i+1}:")
                ]
                next_starts = [pos for pos in next_starts if pos != -1]
                end = min(next_starts) if next_starts else len(sections[lesion])
                if end == -1:
                    end = len(sections[lesion])
                tmp[lesion_name] = sections[lesion][start:end].strip()
                i+=1
            sections[lesion]=tmp

            lesion_name = f"{organ.capitalize()} tumor 1:" 
            if lesion_name not in sections[lesion]:
                lesion_name = f"{organ.capitalize()} lesion 1:"

            #largest lesion:
            try:
                largest=tmp[lesion_name]
            except:
                print('organ section:', organ_sections)
                print('lesion name:',lesion)
                print('Tmp:',tmp)
                print('sections:',sections)
                largest=tmp[f"{organ.capitalize()} tumor 1:"]
            #largest lesion size
            size = extract_measurements(largest)
            #largest lesion volume
            #print('Largest lesion text:',largest)
            volume = extract_value(r"Volume:\s*([\d\.]+)",largest)
            total_vol = sum(extract_value(r"Volume:\s*([\d\.]+)", tmp[key], default=0) for key in tmp)
            location = extract_value(r"Location:\s*(.+?)\.", largest, str(np.nan),str)
            #hu value
            if 'hyperattenuating' in largest.lower():
                hu='hyperattenuating'
            elif 'hypoattenuating' in largest.lower():
                hu='hypoattenuating'
            elif 'isoattenuating' in largest.lower():
                hu='isoattenuating'
            else:
                hu=np.nan

            data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} diameter (cm)"]=(size if not isinstance(size,list) else size[0])
            data[f"total {organ} {lesion.replace('malignant ','')[:-1]} volume (cm^3)"]=total_vol
            data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} attenuation (hyperattenuating, isoattenuating, hypoattenuating)"]=hu
            if organ=='pancreas':
                data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} location (head, body, tail)"]=location.replace('pancreas ','').replace(' ','').replace('.','').replace('/',',')
            elif organ=='kidney':
                data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} location (left, right)"]=location.replace('kidney','').replace(' ','').replace('.','').replace('/',',')
            elif organ=='liver':
                data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} location (1-8)"]=location.replace('hepatic segments ','').replace('hepatic segment ','').replace(' ','').replace('.','').replace('/',',')
            else:
                data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} location"]=location.replace('/',',')
            data[f"largest {organ} {lesion.replace('malignant ','')[:-1]} volume"]=volume
            data[f"number of {organ} {lesion.replace('malignant ','')[:-1]} instances"]=i-1

    return data

def extract_measurements(section):
    """Extract measurements in cm (number x number cm)."""
    matches = re.findall(r"(\d+(\.\d+)?)\s*x\s*\d+(\.\d+)?\s*cm", section)
    return [float(match[0]) for match in matches]  # Return the first number of each measurement

def extract_value_old(pattern, text, default=np.nan, value_type=float):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return value_type(match.group(1))
            except ValueError:
                return default
        return default

def extract_value(pattern, text, default=np.nan, value_type=float):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return value_type(match.group(1).strip())
        except ValueError:
            print(f"ValueError for pattern '{pattern}' in text")
            return default
    else:
        print(f"No match for pattern '{pattern}' in text")
    return value_type(default)


def parse_radiology_report(report):
    """Parse a radiology report and extract relevant information."""
    data = {}
    findings = extract_findings(report)
    sections = extract_organ_sections(findings)
    
    for organ, section in sections.items():
        # volumes
        if organ != "Kidney":
            if "(volume:" in section:
                start = section.find("(volume:") + len("(volume:")
                end = section.find("cm^3", start)
                data[f"{organ.lower()} volume (cm^3)"] = np.round(float(section[start:end].strip()),1)
            else:
                data[f"{organ.lower()} volume (cm^3)"] = np.nan
        else:
            data[f"right {organ.lower()} volume (cm^3)"] = extract_value(rf"right {organ} volume:\s*([\d\.]+)",section)
            data[f"left {organ.lower()} volume (cm^3)"] = extract_value(rf"left {organ} volume:\s*([\d\.]+)",section)
            data[f"{organ.lower()} volume (cm^3)"] = extract_value(rf"total {organ} volume:\s*([\d\.]+)",section)

    if "Clinical stage: " in report:
        print('Report:',report)
        data["pancreatic tumor staging (T1-T4)"] = report[report.find("Clinical stage: ") + len("Clinical stage: "):report.find("Clinical stage: ") + len("Clinical stage: ")+2].split()[0]
        
    tmp=parse_lesions(sections)

    for key in tmp:
        data[key]=tmp[key]

    out={}
    for head in header:
        if head not in data:
            if head.replace('pancreatic', 'pancreas') in data:
                out[head] = data[head.replace('pancreatic', 'pancreas')]
            else:
                out[head] = np.nan
        else:
            out[head]=data[head]
    return out


def create_big_table(metadata, reports, narrative_reports,metadata_df2,atlas_df,ts_df,itens_df,fusion):
    # Ensure column names are correctly identified
    if metadata is not None:
        metadata = metadata.rename(columns={"BDMAP ID": "Case"})  # Align naming for joining

    if not isinstance(itens_df, list):
        cases=itens_df['Case'].to_list()
    else:                                 # assume DataFrame
        cases = itens_df

    big_table={}
    for case in tqdm.tqdm(cases):
        row={}
        try:
            report=reports[reports["Case"]==case]["Report"].values[0]
            parsed=parse_radiology_report(report)
            no_report=False
        except:
            parsed={}
            no_report=True
            
        for head in header:
            #print(head)
            if head =='BDMAP ID':
                row[head]=case
            elif head=='radiologist note':
                if (fusion is not None) and case in fusion["BDMAP ID"].values:
                    row[head]=fusion[fusion["BDMAP ID"]==case]["radiologist notes"].values[0]
                else:
                    row[head]=np.nan
            elif head=='fusion structured report' and (fusion is not None):
                if case in fusion["BDMAP ID"].values:
                    row[head]=fusion[fusion["BDMAP ID"]==case]["fusion structured report"].values[0]
                else:
                    row[head]=np.nan
            elif head=='fusion narrative report' and (fusion is not None):
                if case in fusion["BDMAP ID"].values:
                    row[head]=fusion[fusion["BDMAP ID"]==case]["fusion narrative report"].values[0]
                else:
                    row[head]=np.nan
            elif head == 'age' or head == 'sex':
                if (atlas_df is not None) and case in atlas_df["Case"].values:
                    value=atlas_df[atlas_df["Case"]==case][head].values[0]
                    if head=='age':
                        if value[0]=='0':
                            value=value[1:]
                        if value[-1]=='Y':
                            value=value[:-1]
                    row[head]=value
                elif (ts_df is not None) and case in ts_df["Case"].values:
                    if head=='age':
                        value=ts_df[ts_df["Case"]==case][head].values[0]
                        value=str(int(value))
                        row[head]=value
                    if head=='sex':
                        value=ts_df[ts_df["Case"]==case]['gender'].values[0]
                        if value=='f':
                            value='F'
                        elif value=='m':
                            value='M'
                        else:
                            value=np.nan
                        row[head]=value
                else:
                    row[head]=np.nan
            elif head=='scanner':
                if (ts_df is not None) and case in ts_df["Case"].values:
                    row[head]=ts_df[ts_df["Case"]==case]['manufacturer'].values[0]
                else:
                    row[head]=np.nan
            elif head=='structured report':
                try:
                    row[head]=reports[reports["Case"]==case]["Report"].values[0]#.replace('cm^3','cc')
                except:
                    row[head]=np.nan
            elif head=='narrative report' and (narrative_reports is not None):
                try:
                    row[head]=narrative_reports[narrative_reports["Case"]==case]["Generated_Report"].values[0]#.replace('cm^3','cc')
                except:
                    row[head]=np.nan
            elif (metadata_df2 is not None)  and (head in metadata_df2.columns):
               # print('Columns:',metadata_df2.columns)
                row[head]=metadata_df2[metadata_df2["Case"]==case][head].values
                if len(row[head])==0:
                    row[head]=np.nan
                else:
                    row[head]=row[head][0]
            elif (metadata is not None) and head in metadata.columns:
               # print('I am here')
                row[head]=metadata[metadata["Case"]==case][head].values[0]
            elif head in parsed:
                row[head]=parsed[head]
            elif not no_report:  
                raise ValueError(f"Column '{head}' not found in metadata, reports, or parsed data.")
        big_table[case]=row

    big_table=pd.DataFrame.from_dict(big_table,orient='index')
    return big_table

def create_reports_pdfs(df, output_dir="output_reports"):
    """
    Reads a DataFrame and converts the reports into PDFs, one per BDMAP ID.

    Parameters:
    - df (pd.DataFrame): Input dataframe with the columns:
        'BDMAP ID', 'structured report', 'narrative report',
        'fusion structured report', 'fusion narrative report'.
    - output_dir (str): Base output directory where the folders will be created.

    Creates:
    - Two folders: AtlasStructuredReports and AtlasNarrativeReports.
    - One PDF per BDMAP ID in each folder (structured and narrative reports).
    """
    # Define output folders
    structured_folder = os.path.join(output_dir, "AtlasStructuredReports")
    narrative_folder = os.path.join(output_dir, "AtlasNarrativeReports")

    # Create folders if they don't exist
    os.makedirs(structured_folder, exist_ok=True)
    os.makedirs(narrative_folder, exist_ok=True)


def fill_nan_in_specific_columns(df):
    # Identify the target columns
    target_columns = [
        col for col in df.columns 
        if any(keyword in col for keyword in ['lesion', 'cyst', 'tumor']) 
        and any(keyword in col for keyword in ['cm', 'number'])
    ]
    
    # Fill NaN with 0 in the identified columns
    df[target_columns] = df[target_columns].fillna(0)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Create a unified big table from metadata, reports, and narrative reports.")
    parser.add_argument("--metadata", help="Path to the metadata CSV file", default=None)
    parser.add_argument("--metadata2", required=False, help="Path to the metadata CSV file", default=None)
    parser.add_argument("--reports", required=False, help="Path to the reports CSV file")
    parser.add_argument("--narrative_reports", required=False, help="Path to the narrative reports CSV file",default=None)
    parser.add_argument("--output", required=False, help="Path to save the output CSV file")
    parser.add_argument("--dap_metadata", required=False, help="Path to dap atlas processed metadata", default=None)
    parser.add_argument("--ts_metadata", required=False, help="Path to total segmentator processed metadata", default=None)
    parser.add_argument("--itens", required=False, help="Path to the AbdomenAtlas ids", default=None)
    parser.add_argument("--fusion_reports", required=False, help="Path to the AbdomenAtlas ids", default=None)
    

    args = parser.parse_args()

    # Load dataframes
    if args.metadata is not None:
        metadata_df = pd.read_csv(args.metadata)
    else:
        metadata_df = None
    if args.metadata2 is not None:
        metadata_df2 = pd.read_csv(args.metadata2)
    else:
        metadata_df2 = None
    #reports_df = pd.read_csv(args.reports)

    #load and clean structured reports
    try:
        structured_reports = pd.read_csv(args.reports, usecols=['Case', 'Report'])
    except:
        structured_reports = pd.read_csv(args.reports, usecols=['Case', ' Report'])
        structured_reports.rename(columns={' Report':'Report'}, inplace=True)
    structured_reports.drop_duplicates(subset=['Case'], inplace=True)
    #drop nan
    structured_reports.dropna(subset=['Case'], inplace=True)
    structured_reports.dropna(subset=['Report'], inplace=True)
    #remove any column where case does not start with BDMAP_ followed by 6 numbers
    structured_reports = structured_reports[structured_reports['Case'].str.match('BDMAP_\d{6}')]
    #order by case
    structured_reports.sort_values(by='Case', inplace=True)

    reports_df=structured_reports

    #rename columns if necessary:
    if ' Report' in reports_df.columns:
        reports_df.rename(columns={' Report':'Report'},inplace=True)
    if args.narrative_reports is not None:
        narrative_reports_df = pd.read_csv(args.narrative_reports)
    else:
        narrative_reports_df = None
    if args.dap_metadata is not None:
        atlas_df = pd.read_csv(args.dap_metadata)
    else:
        atlas_df = None
    if args.ts_metadata is not None:
        ts_df = pd.read_csv(args.ts_metadata)
    else:
        ts_df = None
    if args.itens is not None:
        itens_df = pd.read_csv(args.itens)
    else:
        itens_df = structured_reports['Case'].tolist()
    if args.fusion_reports is not None:
        fusion_df = pd.read_csv(args.fusion_reports)
    else:
        fusion_df = None

    # Create the big table
    big_table = create_big_table(metadata_df, reports_df, narrative_reports_df,metadata_df2,atlas_df,ts_df,itens_df,fusion_df)

    #sort by case
    big_table.sort_values(by='BDMAP ID',inplace=True)

    print('Header of the table:',big_table.columns)

    #re-order by header
    big_table=big_table[header]

    #remove spaces around report
    big_table['structured report'] = big_table['structured report'].str.strip().str.replace('cm^3', 'cc', regex=False).str.replace('cm³', 'cc', regex=False).str.replace('kindeys', 'kidneys', regex=False)
    if args.narrative_reports is not None:
        big_table['narrative report'] = big_table['narrative report'].str.strip().str.replace('cm^3', 'cc', regex=False).str.replace('cm³', 'cc', regex=False).str.replace('kindeys', 'kidneys', regex=False)
    if args.fusion_reports is not None:
        big_table['fusion structured report'] = big_table['fusion structured report'].str.strip().str.replace('cm^3', 'cc', regex=False).str.replace('cm³', 'cc', regex=False).str.replace(r'\nFINDINGS:', r'\n\nFINDINGS:', regex=True).str.replace(r'\nIMPRESSION:', r'\n\nIMPRESSION:', regex=True).str.replace('kindeys', 'kidneys', regex=False)
        big_table['fusion narrative report'] = big_table['fusion narrative report'].str.strip().str.replace('cm^3', 'cc', regex=False).str.replace('cm³', 'cc', regex=False).str.replace(r'\nFINDINGS:', r'\n\nFINDINGS:', regex=True).str.replace(r'\nIMPRESSION:', r'\n\nIMPRESSION:', regex=True).str.replace('kindeys', 'kidneys', regex=False)

    #for columns containing lesion, replace nan by 0
    big_table=fill_nan_in_specific_columns(big_table)
        
    # Save to the output path
    big_table.round(decimals=1).to_csv(args.output, index=False)
    print(f"Big table saved to {args.output}")

    create_reports_pdfs(big_table)


if __name__ == "__main__":
    main()
