import pandas as pd
import RadGPT_Style as rgpt
import argparse

def main(args):

    clinical_notes=pd.read_csv(args.radiology_notes)
    reports=pd.read_csv(args.reports)    

    reps=rgpt.iterate_add_info(reports, clinical_notes,base_url=f'http://0.0.0.0:{args.port}/v1',
                            prt=True,save_path=args.output,report_col='structured report')
    reps=rgpt.iterate_add_info(reps, clinical_notes,base_url=f'http://0.0.0.0:{args.port}/v1',
                            prt=True,save_path=args.output,report_col='narrative report')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuse structured and narrative radiology reports')

    parser.add_argument('--reports', type=str, help='CSV file with narrative and structured reports')
    parser.add_argument('--radiology_notes', type=str, help='CSV file with radiology notes')
    parser.add_argument('--output', type=str, help='CSV for fusion reports')
    parser.add_argument('--port', type=str, help='VLLM port')



    args = parser.parse_args()
    main(args)