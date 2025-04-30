# Image-level evaluation of Segmentations

1- Generate segmentations with DiffTumor or the Flagship Model.

2- Convert saved segmentations to binary outputs: tumor/no tumor
```bash
python Segmentation2BinaryLabels.py --outputs_folder /path/to/output/folder/ --ct_folder /path/to/ct/folder/ --th 30
```

Output: csv file with binary outputs at /path/to/outputs_folder/tumor_detection_results.csv. 

**Setting the threshold**: run the command above multiple times in the validation dataset, with the following thresholds: --th 10, --th 20, --th 30, --th 40, --th 50, --th 60, --th 70, --th 80, --th 90, --th 100, --th 150, --th 200, --th 250, --th 300, --th 500. Check, **for each organ**, which th gives the best sensitivity and specificity balance. Then, use these thresholds in the test dataset.

Input: this code was designed for DiffTumor. So, you may need to convert the Flagship model's output to the DiffTumor format to run it on the Flagship model (or you can modify the code loading process).

<details>
  <summary>Details about mask post-processing and the converseion to binary labels</summary>

Post-processing details:

  a- We remove any tumor detection outside of the organ mask. E.g., we certify that all liver tumors in liver_tumor.nii.gz are inside the liver (liver.nii.gz).

  b- We apply [binary erosion](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_erosion.html), using a 3x3x3 mm cube as the structuring element. This operation denoises the segmentation mask, removing detections that are smaller than the structuring element. Thus, binary erosion helps avoid false positives.

  c- After erosion, we calculate the total volume of the tumors in the tumor mask, in mm^3. If it is above a threshold parameter, the sample is considered positive for tumor. 
  
  Higher thresholds (th) and binary erosion reduce the number of false positives but may reduce the model capacity to detect very small tumors, and increase false negatives.

</details>

3- Calculate tumor detection specificity and sensitivity using the outputs from the step above, comparing them to the ground-truth labels extracted by RadGPT from human-made reports (Part 1, Step 2). These results represent both the accuracy of our segmenters (Difftumor and Flagship model), and the accuracy of our automatic reports (our reports are consistent with the segmentation results). Thus, you actually do not need to generate the reports to know their accuracy!
