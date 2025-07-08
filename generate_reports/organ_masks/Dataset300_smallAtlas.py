from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import tqdm
import os

if __name__ == '__main__':
    """
    How to train our submission to the JHU benchmark
    
    1. Execute this script here to convert the dataset into nnU-Net format. Adapt the paths to your system!
    2. Run planning and preprocessing: `nnUNetv2_plan_and_preprocess -d 224 -npfp 64 -np 64 -c 3d_fullres -pl 
    nnUNetPlannerResEncL_torchres`. Adapt the number of processes to your System (-np; -npfp)! Note that each process 
    will again spawn 4 threads for resampling. This custom planner replaces the nnU-Net default resampling scheme with 
    a torch-based implementation which is faster but less accurate. This is needed to satisfy the inference speed 
    constraints.
    3. Run training with `nnUNetv2_train 224 3d_fullres all -p nnUNetResEncUNetLPlans_torchres`. 24GB VRAM required, 
    training will take ~28-30h.
    """


    #base = '/fastwork/psalvador/JHU/data/AbdomenAtlas1.1Mini/uncompressed/'
    #base_mask = '/fastwork/psalvador/JHU/data/AbdomenAtlas3.0MiniSegmentations/AbdomenAtlas3.0Mini/'
    #cases = subdirs(base, join=False, prefix='BDMAP')

    target_dataset_id = 300
    target_dataset_name = f'Dataset300_smallAtlas'
    raw_dir = '/mnt/ccvl15/pedro/nnUNet_raw/'
    #maybe_mkdir_p(join(raw_dir, target_dataset_name))
    imagesTr = join(raw_dir, target_dataset_name, 'imagesTr')
    labelsTr = join(raw_dir, target_dataset_name, 'labelsTr')
    #maybe_mkdir_p(imagesTr)
    #maybe_mkdir_p(labelsTr)

    #reference mapping form combine_labels.py. This shows how each class (including overlaps) was annotated in the combined labels
    ids={'background': 0,
        'kidney_right': 1,
        'kidney_left': 2,
        'kidney_lesion': 3,
        'kidney_lesion_kidney_right': 4,
        'kidney_lesion_kidney_left': 5,
        'pancreas': 6,
        'pancreas_head': 7,
        'pancreas_body': 8,
        'pancreas_tail': 9,
        'pancreatic_lesion': 10,
        'pancreatic_lesion_pancreas_head': 11,
        'pancreatic_lesion_pancreas_body': 12,
        'pancreatic_lesion_pancreas_tail': 13,
        'liver': 14,
        'liver_segment_1': 15,
        'liver_segment_2': 16,
        'liver_segment_3': 17,
        'liver_segment_4': 18,
        'liver_segment_5': 19,
        'liver_segment_6': 20,
        'liver_segment_7': 21,
        'liver_segment_8': 22,
        'liver_lesion': 23,
        'liver_lesion_liver_segment_1': 24,
        'liver_lesion_liver_segment_2': 25,
        'liver_lesion_liver_segment_3': 26,
        'liver_lesion_liver_segment_4': 27,
        'liver_lesion_liver_segment_5': 28,
        'liver_lesion_liver_segment_6': 29,
        'liver_lesion_liver_segment_7': 30,
        'liver_lesion_liver_segment_8': 31,
        'spleen': 32,
        'colon': 33,
        'stomach': 34,
        'duodenum': 35,
        'common_bile_duct': 36,
        'intestine': 37,
        'aorta': 38,
        'postcava': 39,
        'adrenal_gland_left': 40,
        'adrenal_gland_right': 41,
        'gall_bladder': 42,
        'bladder': 43,
        'celiac_trunk': 44,
        'esophagus': 45,
        'hepatic_vessel': 46,
        'portal_vein_and_splenic_vein': 47,
        'lung_left': 48,
        'lung_right': 49,
        'prostate': 50,
        'rectum': 51,
        'femur_left': 52,
        'femur_right': 53,
        'superior_mesenteric_artery': 54,
        'veins': 55}

    #for case in tqdm.tqdm(cases):
    #    shutil.copy(join(base, case, 'ct.nii.gz'), join(imagesTr, case + '_0000.nii.gz'))
    #    shutil.copy(join(base_mask, case, 'combined_with_subseg.nii.gz'), join(labelsTr, case + '.nii.gz'))

    #the labels in a grop are all integer values in the combined labels which that class encompasses.
    #the first value in the list is the one used to annotate that class in "combine_labels.py"
    #You can read this: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
    #here, the keys are only the classes you really want the model to train on, not all superpositions
    superposing_groups={
         'kidney_right':  [ids['kidney_right'],ids['kidney_lesion_kidney_right']],#i.e.: integers 1,3 in the combined labels are annotated as kidney_right, and this is sent to the model as the GT for kidney_right
         'kidney_left':   [ids['kidney_left'],ids['kidney_lesion_kidney_left']],
         'kidney_lesion': [ids['kidney_lesion'],ids['kidney_lesion_kidney_right'],ids['kidney_lesion_kidney_left']],

         'pancreas':[ids['pancreas'],ids['pancreas_head'],ids['pancreas_body'],ids['pancreas_tail'],
                     ids['pancreatic_lesion'],ids['pancreatic_lesion_pancreas_head'],ids['pancreatic_lesion_pancreas_body'],ids['pancreatic_lesion_pancreas_tail']],
         'pancreas_head': [ids['pancreas_head'],ids['pancreatic_lesion_pancreas_head']],
         'pancreas_body': [ids['pancreas_body'],ids['pancreatic_lesion_pancreas_body']],
         'pancreas_tail': [ids['pancreas_tail'],ids['pancreatic_lesion_pancreas_tail']],
         'pancreatic_lesion':[ids['pancreatic_lesion'],ids['pancreatic_lesion_pancreas_head'],ids['pancreatic_lesion_pancreas_body'],ids['pancreatic_lesion_pancreas_tail']],

         'liver':[ids['liver'],ids['liver_segment_1'],ids['liver_segment_2'],ids['liver_segment_3'],ids['liver_segment_4'],
                   ids['liver_segment_5'],ids['liver_segment_6'],ids['liver_segment_7'],ids['liver_segment_8'],
                   ids['liver_lesion'],ids['liver_lesion_liver_segment_1'],ids['liver_lesion_liver_segment_2'],ids['liver_lesion_liver_segment_3'],ids['liver_lesion_liver_segment_4'],
                   ids['liver_lesion_liver_segment_5'],ids['liver_lesion_liver_segment_6'],ids['liver_lesion_liver_segment_7'],ids['liver_lesion_liver_segment_8']],
         'liver_segment_1': [ids['liver_segment_1'],ids['liver_lesion_liver_segment_1']],
         'liver_segment_2': [ids['liver_segment_2'],ids['liver_lesion_liver_segment_2']],
         'liver_segment_3': [ids['liver_segment_3'],ids['liver_lesion_liver_segment_3']],
         'liver_segment_4': [ids['liver_segment_4'],ids['liver_lesion_liver_segment_4']],
         'liver_segment_5': [ids['liver_segment_5'],ids['liver_lesion_liver_segment_5']],
         'liver_segment_6': [ids['liver_segment_6'],ids['liver_lesion_liver_segment_6']],
         'liver_segment_7': [ids['liver_segment_7'],ids['liver_lesion_liver_segment_7']],
         'liver_segment_8': [ids['liver_segment_8'],ids['liver_lesion_liver_segment_8']],
         'liver_lesion': [ids['liver_lesion'],ids['liver_lesion_liver_segment_1'],ids['liver_lesion_liver_segment_2'],ids['liver_lesion_liver_segment_3'],ids['liver_lesion_liver_segment_4'],
                          ids['liver_lesion_liver_segment_5'],ids['liver_lesion_liver_segment_6'],ids['liver_lesion_liver_segment_7'],ids['liver_lesion_liver_segment_8']]}

    background={
    'background': 0}


    non_overlapping=[
    'spleen',
    'colon',
    'stomach',
    'duodenum',
    'common_bile_duct',
    'intestine',
    'aorta',
    'postcava',
    'adrenal_gland_left',
    'adrenal_gland_right',
    'gall_bladder',
    'bladder',
    'celiac_trunk',
    'esophagus',
    'hepatic_vessel',
    'portal_vein_and_splenic_vein',
    'lung_left',
    'lung_right',
    'prostate',
    'rectum',
    'femur_left',
    'femur_right',
    'superior_mesenteric_artery',
    'veins',
    ]

    labels=background
    for key in superposing_groups:
        labels[key]=superposing_groups[key]
    for key in non_overlapping:
        labels[key]=ids[key]

    region_class_order=[ids[key] for key in labels if key != 'background']
    #as we have less labels than IDs, this reconstruction is LOSSY. Thus, to get the true region segmentations, look at the soft predictions of the model.  
		 
    generate_dataset_json(
        join(raw_dir, target_dataset_name),
        {0: 'CT'},  # this was a mistake we did at the beginning and we keep it like that here for consistency
        labels,
        len(os.listdir(labelsTr)),
        '.nii.gz',
        region_class_order,
        target_dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient',
    )
