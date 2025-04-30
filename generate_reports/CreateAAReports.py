import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import scipy
import os
import pandas as pd
import scipy.ndimage as ndimage
from skimage import measure
from scipy.spatial.distance import pdist, squareform
from skimage.transform import rotate
import pathlib
import csv
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
import staging as stg
import traceback
import re

# Use multiprocessing Lock for process-safe access to the file
lock = multiprocessing.Lock()


def load_canonical(path, orientation=('R', 'A', 'S')):
    """
    Loads a NIfTI file and reorients it to the specified canonical orientation (default: RAS).

    Parameters:
    - path: str, path to the NIfTI file
    - orientation: tuple of desired axis labels (default: ('R', 'A', 'S'))

    Returns:
    - reoriented_img: Nifti1Image reoriented to the specified orientation
    """
    # Load the image
    img = nib.load(path)

    # Get the orientation transform to reorient the image
    transform = get_orientation_transform(img, orientation)

    # Apply the transform to the data
    transformed_data = apply_transform(img.get_fdata(), transform)

    # Update the affine matrix to reflect the new orientation
    new_affine = nib.orientations.inv_ornt_aff(transform, img.shape).dot(img.affine)

    # Create a new Nifti1Image with the reoriented data, updated affine, and original header
    reoriented_img = nib.Nifti1Image(transformed_data, new_affine, img.header)

    # Update the header to reflect the new shape and affine
    reoriented_img.header.set_data_shape(transformed_data.shape)
    reoriented_img.header.set_zooms(tuple(abs(new_affine[i, i]) for i in range(3)))

    return reoriented_img

# Supporting functions
def get_orientation_transform(img, orientation=('L', 'P', 'S')):
    """
    Computes the orientation transform to reorient the image to the desired orientation.

    Parameters:
    - img: Nifti1Image
    - orientation: tuple of desired axis labels

    Returns:
    - transform: numpy array representing the orientation transform
    """
    # Get the current orientation
    current_ornt = nib.orientations.io_orientation(img.affine)
    # Get the desired orientation
    desired_ornt = nib.orientations.axcodes2ornt(orientation)
    # Compute the transform
    transform = nib.orientations.ornt_transform(current_ornt, desired_ornt)
    return transform

def apply_transform(data, transform):
    """
    Applies the orientation transform to the data array.

    Parameters:
    - data: numpy array
    - transform: numpy array representing the orientation transform

    Returns:
    - transformed_data: numpy array with the orientation transformed
    """
    # Apply the orientation transform to the data
    transformed_data = nib.orientations.apply_orientation(data, transform)
    return transformed_data



def count_unconnected_objects(binary_array,erode=0):
    if erode>0:
        #erosion
        binary_array=scipy.ndimage.binary_erosion(binary_array,structure=np.ones((erode,erode)),
                                                iterations=1)
    # Label connected components in the binary array
    labeled_array, num_features = scipy.ndimage.label(binary_array)

    # Create a mask to identify white objects (labeled regions)
    white_object_mask = (labeled_array > 0)  # Consider all labeled regions

    # Count the number of unique labels (excluding background label 0)
    num_objects = np.max(labeled_array)  # This gives the number of unique labels

    return num_objects

def plot_slices(data,z=-1, limit0=0, limit1=1000909):
    # Assuming `array` is your 3D numpy array containing image data
    #array = load_canonical(data).get_fdata()
    array[array<-100]=-100
    array[array>200]=200
    # Plot multiple slices along the z-axis
    num_slices = array.shape[z]
    rows = 4
    cols = 4

    # Calculate the total number of subplots needed
    total_plots = rows * cols

    # Determine slice indices to display
    slice_indices = np.linspace(limit0, min(num_slices - 1,limit1), total_plots, dtype=int)

    # Create the figure and subplots
    fig, axs = plt.subplots(rows, cols, figsize=(12, 12))

    # Plot each slice on a separate subplot
    for i, idx in enumerate(slice_indices):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        if z==-1:
            ax.imshow(array[:, :, idx], cmap='gray')
        elif z==0:
            ax.imshow(array[idx, :, :], cmap='gray')
        elif z==1:
            ax.imshow(array[:, idx, :], cmap='gray')
            
        ax.axis('off')
        connected=str(count_unconnected_objects(array[:, :, idx]))
        ax.set_title(f"Slice {idx}")#+' '+connected)

    # Remove any unused subplots
    for j in range(total_plots, rows * cols):
        fig.delaxes(axs.flatten()[j])

    plt.tight_layout()
    plt.show()
    
def is_binary_array(array):
    """
    Check if a NumPy array is binary (contains only 0s and 1s).

    Parameters:
    array (numpy.ndarray): Input array to check.

    Returns:
    bool: True if the array is binary, False otherwise.
    """
    return np.isin(array, [0, 1]).all()    

def plot_slices_union(data,data2,z=-1, limit0=0, limit1=1000909):
    # Assuming `array` is your 3D numpy array containing image data
    #array = load_canonical(data).get_fdata()
    #array2 = load_canonical(data2).get_fdata()
    array=array*array2
    
    # Plot multiple slices along the z-axis
    num_slices = array.shape[z]
    rows = 4
    cols = 4

    total_plots = rows * cols

    # Determine slice indices to display
    slice_indices = np.linspace(limit0, min(num_slices - 1,limit1), total_plots, dtype=int)
    
    # Create the figure and subplots
    fig, axs = plt.subplots(rows, cols, figsize=(12, 12))

    # Plot each slice on a separate subplot
    for i, idx in enumerate(slice_indices):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        if z==-1:
            ax.imshow(array[:, :, idx], cmap='gray')
        elif z==0:
            ax.imshow(array[idx, :, :], cmap='gray')
        elif z==1:
            ax.imshow(array[:, idx, :], cmap='gray')
            
        ax.axis('off')
        connected=str(count_unconnected_objects(array[:, :, idx]))
        ax.set_title(f"Slice {idx}"+' '+connected)

    # Remove any unused subplots
    for j in range(total_plots, rows * cols):
        fig.delaxes(axs.flatten()[j])

    plt.tight_layout()
    plt.show()
#dataframe=pd.read_csv('/run/media/pedro/e911bf59-fe8e-4ddb-8938-5dc4f40b094f/DiffTumor/file_mapping.csv')
    
def apply_rotation(slice, rotation):
    if rotation == '90':
        return np.rot90(slice)
    elif rotation == '180':
        return np.rot90(slice, 2)
    elif rotation == '270':
        return np.rot90(slice, 3)
    else:
        return slice

def get_rotation(affine):
    rotations = []
    for i in range(3):
        max_index = np.argmax(np.abs(affine[:3, i]))
        if affine[max_index, i] < 0:
            rotations.append('90')
        elif max_index == 1:
            rotations.append('0')
        elif max_index == 2:
            rotations.append('180')
        else:
            rotations.append('270')
    return rotations

def plot_slices_overlay(array, array2, liver_mask=None, z=-1, limit0=0, limit1=1000909,
                        color='red', name='Tumors'):
    # Load the CT scan and segmentation mask data
    #array = load_canonical(data).get_fdata()
    array[array < -100] = -100
    array[array > 200] = 200
    
    #array2 = load_canonical(data2).get_fdata()
    if liver_mask is not None:
        array2 = array2 * load_canonical(liver_mask).get_fdata()

    # Get affine transform from header
    affine = load_canonical(data).affine

    # Determine the necessary rotations
    rotations = get_rotation(affine)

    # Calculate the total number of subplots needed
    num_slices = array.shape[z]
    rows = 4
    cols = 4
    total_plots = rows * cols

    # Determine slice indices to display
    slice_indices = np.linspace(limit0, min(num_slices - 1, limit1), total_plots, dtype=int)
    
    # Create the figure and subplots
    fig, axs = plt.subplots(rows, cols, figsize=(12, 12))

    # Plot each slice on a separate subplot
    for i, idx in enumerate(slice_indices):
        row = i // cols
        col = i % cols
        ax = axs[row, col]

        if z == -1:
            slice_ct = array[:, :, idx]
            slice_mask = array2[:, :, idx]
        elif z == 0:
            slice_ct = array[idx, :, :]
            slice_mask = array[idx, :, :]
        elif z == 1:
            slice_ct = array[:, idx, :]
            slice_mask = array2[:, idx, :]

        # Apply the necessary rotations
        slice_ct = apply_rotation(slice_ct, rotations[0])
        slice_mask = apply_rotation(slice_mask, rotations[0])

        # Normalize CT scan values to range [0, 1] if necessary
        slice_ct = (slice_ct - np.min(slice_ct)) / (np.max(slice_ct) - np.min(slice_ct))

        # Create an RGB image
        rgb_image = np.stack([slice_ct, slice_ct, slice_ct], axis=-1)

        if color == 'red':
            rgb_image[..., 1] = np.where(slice_mask > 0, rgb_image[..., 1] * 0.5, rgb_image[..., 1])
            rgb_image[..., 2] = np.where(slice_mask > 0, rgb_image[..., 2] * 0.5, rgb_image[..., 2])
        elif color == 'green':
            rgb_image[..., 0] = np.where(slice_mask > 0, rgb_image[..., 0] * 0.5, rgb_image[..., 0])
            rgb_image[..., 2] = np.where(slice_mask > 0, rgb_image[..., 2] * 0.5, rgb_image[..., 2])
        elif color == 'blue':
            rgb_image[..., 0] = np.where(slice_mask > 0, rgb_image[..., 0] * 0.5, rgb_image[..., 0])
            rgb_image[..., 1] = np.where(slice_mask > 0, rgb_image[..., 1] * 0.5, rgb_image[..., 1])
        else:
            raise ValueError('Color not implemented')

        # Plot the composite image
        ax.imshow(rgb_image)
        ax.axis('off')
        connected = str(count_unconnected_objects(slice_mask))
        ax.set_title(f"Slice {idx} " + name + ': ' + connected)

    # Remove any unused subplots
    for j in range(total_plots, rows * cols):
        fig.delaxes(axs.flatten()[j])

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_slices_overlay('path_to_ct_scan.nii.gz', 'path_to_segmentation_mask.nii.gz', liver_mask='path_to_liver_mask.nii.gz', z=-1, limit0=0, limit1=100, color='red', name='Tumors')
    

    
def plot_slice_overlay(data, data2, idx, spacing,
                       liver_mask=None, z=-1, color='red', name='Tumors',
                       overlay=True):
    # Load the CT scan and segmentation mask data
    img = load_canonical(data)
    array = img.get_fdata()
    
    array[array < -100] = -100
    array[array > 200] = 200
    
    img2 = load_canonical(data2)
    array2 = img2.get_fdata()
    
    # Get affine transform from header
    affine = load_canonical(data).affine

    # Determine the necessary rotations
    rotations = get_rotation(affine)

    
    if liver_mask is not None:
        array2 = array2 * load_canonical(liver_mask).get_fdata()
    
    # Get voxel spacing from the NIfTI header
    voxel_spacing = spacing
    #print('spacing:',voxel_spacing)

    # Determine slice orientation and get corresponding slices
    if z == -1:
        slice_ct = array[:, :, idx]
        slice_mask = array2[:, :, idx]
        x_spacing, y_spacing = voxel_spacing[0], voxel_spacing[1]
    elif z == 0:
        slice_ct = array[idx, :, :]
        slice_mask = array2[idx, :, :]
        x_spacing, y_spacing = voxel_spacing[1], voxel_spacing[2]
    elif z == 1:
        slice_ct = array[:, idx, :]
        slice_mask = array2[:, idx, :]
        x_spacing, y_spacing = voxel_spacing[0], voxel_spacing[2]
        
    # Apply the necessary rotations
    slice_ct = apply_rotation(slice_ct, rotations[0])
    slice_mask = apply_rotation(slice_mask, rotations[0])

    # Transpose and flip slices for correct orientation
    #slice_ct = np.transpose(slice_ct)
    #slice_ct = np.flip(slice_ct, axis=(0, 1))

    #slice_mask = np.transpose(slice_mask)
    #slice_mask = np.flip(slice_mask, axis=(0, 1))

    # Normalize CT scan values to range [0, 1] if necessary
    slice_ct = (slice_ct - np.min(slice_ct)) / (np.max(slice_ct) - np.min(slice_ct))

    # Create an RGB image
    rgb_image = np.stack([slice_ct, slice_ct, slice_ct], axis=-1)
    
    if overlay:
        if color == 'red':
            rgb_image[..., 1] = np.where(slice_mask > 0, rgb_image[..., 1] * 0.5, rgb_image[..., 1])
            rgb_image[..., 2] = np.where(slice_mask > 0, rgb_image[..., 2] * 0.5, rgb_image[..., 2])
        elif color == 'green':
            rgb_image[..., 0] = np.where(slice_mask > 0, rgb_image[..., 0] * 0.5, rgb_image[..., 0])
            rgb_image[..., 2] = np.where(slice_mask > 0, rgb_image[..., 2] * 0.5, rgb_image[..., 2])
        elif color == 'blue':
            rgb_image[..., 0] = np.where(slice_mask > 0, rgb_image[..., 0] * 0.5, rgb_image[..., 0])
            rgb_image[..., 1] = np.where(slice_mask > 0, rgb_image[..., 1] * 0.5, rgb_image[..., 1])
        else:
            raise ValueError('Color not implemented')


    # Plot the composite image
    plt.figure(figsize=(12, 12))
    plt.imshow(rgb_image)
    
    # Set axis labels to represent millimeters
    plt.xlabel(f'X (mm)')
    plt.ylabel(f'Y (mm)')
    
    # Set axis ticks to represent millimeters
    x_ticks = np.arange(0, slice_ct.shape[1], int(slice_ct.shape[1] / 10))
    y_ticks = np.arange(0, slice_ct.shape[0], int(slice_ct.shape[0] / 10))
    plt.xticks(x_ticks, labels=np.round(x_ticks * x_spacing).astype(int))
    plt.yticks(y_ticks, labels=np.round(y_ticks * y_spacing).astype(int))
    
    plt.title(f'Slice {idx} - {name}')
    plt.tight_layout()
    plt.show()

def measure_diameter(binary_image):
    """
    Measures the diameter of an arbitrary shape in a binary image and returns the diameter
    along with the two extreme points that define this diameter.

    Parameters:
    binary_image (numpy.ndarray): 2D binary array where the shape is represented by 1s.

    Returns:
    tuple: The diameter of the shape and the coordinates of the two extreme points.
    """
    # Find contours of the shape
    contours = measure.find_contours(binary_image, 0.5)

    if not contours:
        raise ValueError("No contours found in the binary image")

    # Assuming there's only one shape, take the first contour
    contour = contours[0]

    # Compute pairwise distances between all contour points
    distances = pdist(contour)
    distance_matrix = squareform(distances)

    # Find the indices of the maximum distance
    max_distance_idx = np.unravel_index(np.argmax(distance_matrix, axis=None), distance_matrix.shape)

    # Get the coordinates of the extreme points
    point1 = contour[max_distance_idx[0]]
    point2 = contour[max_distance_idx[1]]

    # Find the maximum distance
    max_distance = distance_matrix[max_distance_idx]

    return max_distance, point1, point2

def largest_connected_component_size(array_3d):
    """
    Calculate the size of the largest connected object in a 3D array.

    Parameters:
    array_3d (numpy.ndarray): Input 3D array with binary values (0s and 1s).

    Returns:
    int: Size of the largest connected component.
    """
    import scipy.ndimage as ndimage
    # Define the connectivity (3x3x3 for 26-connectivity)
    structure = np.ones((3, 3, 3), dtype=int)
    
    # Label the connected components
    labeled_array, num_features = ndimage.label(array_3d, structure=structure)
    
    # Calculate the size of each connected component
    sizes = ndimage.sum(array_3d, labeled_array, range(1, num_features + 1))
    
    # Find the largest connected component
    largest_component_size = sizes.max() if sizes.size > 0 else 0
    
    return int(largest_component_size)

def resample_image(image, original_spacing, target_spacing=(1, 1, 1),order=1):
    """
    Resample the image to the target spacing.

    Parameters:
    image (nibabel.Nifti1Image): Input image to resample.
    target_spacing (tuple): Target spacing in x, y, z directions.

    Returns:
    numpy.ndarray: Resampled image data.
    """
    # Get original spacing
    resize_factor = np.array(original_spacing) / np.array(target_spacing)
    #new_shape = np.round(image.shape * resize_factor).astype(int)

    # Resample image
    try:image=image.get_fdata()
    except:pass
    resampled_image = ndimage.zoom(image, resize_factor, order=order)

    return resampled_image, resize_factor   
    
def detection(tumor_mask, organ_mask, spacing,th=10,erode=True):
    """
    Returns true if there are tumors in the ct scan.
    
    tumor_mask: file path for the tumor segmentation mask, e.g., liver_tumor.nii.gz
    organ_mask: file path for the organ segmentation mask, e.g., liver.nii.gz
    th: detection thresold, in mm^3. Only considers a sample as positive if the total tumor 
    mask volume is bigger than th
    erode: performs binary erosion if True, denoises mask, avoiding false positives
    """
    # Load the CT scan and segmentation mask data
    array = load_canonical(tumor_mask).get_fdata()
    
    if organ_mask is not None:
        array = array*load_canonical(organ_mask).get_fdata()
        
    array, _ = resample_image(array, original_spacing=spacing,
                              target_spacing=(1, 1, 1),order=0)
    if erode:
        array=scipy.ndimage.binary_erosion(array,structure=np.ones((3,3,3)), iterations=1)
   
    if array.sum()>th:
        return True
    else: 
        return False
        
def get_new_name(original_name, dataframe):
    match = dataframe[dataframe['OriginalName'] == original_name]
    if not match.empty:
        return match['NewName'].values[0][:-len('.nii.gz')]
    else:
        return None
    
def get_first_last_slices(data,erode=0,z=-1):
    """
    Get the indices of the first and last slices in the z-axis with at least one voxel valued 1.

    Parameters:
    data (numpy.ndarray): 3D array representing the CT scan segmentation mask.

    Returns:
    tuple: (first_index, last_index) of the slices containing at least one voxel with a value of 1.
    """
    array = load_canonical(data).get_fdata()
    # Plot multiple slices along the z-axis
    num_slices = array.shape[z]
    
    first_slice=10000
    last_slice=0
    
    # Plot each slice on a separate subplot
    for idx in list(range(num_slices)):
        if z==-1:
            slc=array[:, :, idx]
        elif z==0:
            slc=array[idx, :, :]
        elif z==1:
            slc=array[:, idx, :]
            
        connected=count_unconnected_objects(slc,erode=erode)
        if connected>0:
            if idx<first_slice:
                first_slice=idx
            if idx>last_slice:
                last_slice=idx
    
    return first_slice, last_slice

#info=pd.read_csv('/run/media/pedro/e911bf59-fe8e-4ddb-8938-5dc4f40b094f/Oncology_autoReport-log_anon.xlsx - Studies.csv')



def getLiverReport(filename):
    acc=filename[filename.find('_')+1:filename.rfind('_')]
    print(acc)
    for index, row in info.iterrows():
        if row['Anon Acc #']==acc:
            report=row['Anon Report Text']
            break
    
    if 'Index lesions' in report:
        st=report[report.find('Index lesions'):]
        a=st[:st[len('Index lesions')+20:].find(':')]
        print(a)
    print(report[report.find('Liver:'):report.find('Gallbladder:')])
    print(report[report.find('IMPRESSION:'):])
    
def print_report(filename):
    acc=filename[filename.find('_')+1:filename.rfind('_')]
    print(acc)
    for index, row in info.iterrows():
        if row['Anon Acc #']==acc:
            report=row['Anon Report Text']
            print(report)
            return
        


def rotate_image(binary_image, angle):
    import cv2
    h, w = binary_image.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Convert to uint8 before rotation
    binary_image_uint8 = (binary_image * 255).astype(np.uint8)
    rotated_image = cv2.warpAffine(binary_image_uint8, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    # Convert back to binary after rotation
    rotated_image_binary = (rotated_image > 127).astype(np.uint8)
    return rotated_image_binary

def measure_vertical_span(binary_image):
    y_coords, x_coords = np.where(binary_image == 1)
    vertical_span = np.max(y_coords) - np.min(y_coords)
    return vertical_span


def print_slice(slc,spacing):
    plt.figure(figsize=(12, 12))
    plt.imshow(slc)
    
    # Set axis labels to represent millimeters
    plt.xlabel(f'X (mm)')
    plt.ylabel(f'Y (mm)')
    
    # Set axis ticks to represent millimeters
    x_ticks = np.arange(0, slc.shape[1], int(slc.shape[1] / 10))
    y_ticks = np.arange(0, slc.shape[0], int(slc.shape[0] / 10))
    plt.xticks(x_ticks, labels=np.round(x_ticks).astype(int))
    plt.yticks(y_ticks, labels=np.round(y_ticks).astype(int))
    
    plt.tight_layout()
    plt.show()
    
def measure_volume(array_3d,spacing,check_border=False):
    # Load the NIfTI file
    #img = load_canonical(file_path)
    #assert np.array_equal(img.get_fdata(), img.get_fdata().astype(bool))
    #img_data = img.get_fdata().astype('uint8')
    #img_data = np.where(img_data > 0.5, 1, 0)
    #img = nib.Nifti1Image(img_data, img.affine, img.header)  
    
    # print('Pre-reize volume:', int(img.get_fdata().sum()/1000))
    
    # Resample the image to have 1mm voxel spacing in all dimensions
    #array_3d, resize_factor = resample_image(img, original_spacing=spacing,
    #                                         target_spacing=(1, 1, 1),order=0)
    
    #array_3d=np.where(array_3d < 0.5, 0, 1)
    #print(np.unique(array_3d))
    
    if check_border:
        #assert if binary and 3d
        eroded = scipy.ndimage.binary_erosion(array_3d, structure=np.ones((1, 1, 1)), iterations=1)
        dilated = scipy.ndimage.binary_dilation(eroded, structure=np.ones((1,1,1)), iterations=2)
        has_ones_on_borders = (
            np.any(dilated[0, :, :]) or  # Front face (x = 0)
            np.any(dilated[-1, :, :]) or # Back face (x = -1)
            np.any(dilated[:, 0, :]) or  # Left face (y = 0)
            np.any(dilated[:, -1, :]) or # Right face (y = -1)
            np.any(dilated[:, :, 0]) or  # Top face (z = 0)
            np.any(dilated[:, :, -1])    # Bottom face (z = -1)
        )
        #check if organ touches the ct boders
        if has_ones_on_borders:
            return None

    assert np.array_equal(array_3d, array_3d.astype(bool))
    assert len(array_3d.shape)==3
    
    return array_3d.sum()





def measure_organ_hu(organ,tumor,ct):

    if tumor is None:
        tumor = np.zeros_like(organ)  # Create a mask of zeros if no tumor is provided
    
    mask=organ-tumor
    mask=np.where(mask > 0.5, 1, 0)
    eroded = scipy.ndimage.binary_erosion(mask, structure=np.ones((1,1,1)), iterations=1)
    if eroded.sum()!=0:
        mask=eroded
    assert len(mask.shape)==3
    
    ct=ct*mask
    mean_hu=ct.sum()/(mask.sum()+1e-6)
    std_hu = ct[mask != 0].std()
    return np.round(mean_hu,1),np.round(std_hu,1)

def load_segments_liver(segments_path,spacing):
    joint=None
    for segment in os.listdir(segments_path):
        if 'liver_segment' not in segment:
            continue
        #print('Segment path:', segment,segments_path)
        seg_path = os.path.join(segments_path, segment)
        seg=load_canonical(seg_path).get_fdata()
        seg = np.where(seg > 0.5, 1, 0)
        if joint is None:
            joint=seg
        else:
            if joint.max()!=(joint+seg).max():
                #solve overlap: zero seg where joint !=0
                seg=np.where(joint > 0, 0, seg)
                #raise ValueError('Overlapping segments')
            #print(seg_path)
            joint+=seg*(int(seg_path[-1-len('.nii.gz')]))
    
    joint, _ = resample_image(joint,original_spacing=spacing,
                             target_spacing=(1, 1, 1),order=0)

    return joint

def load_segments_pancreas(segments_path,spacing):
    #loads head, body and tail of the pancreas
    joint=None
    
    for i,segment in enumerate(['pancreas_head.nii.gz','pancreas_body.nii.gz','pancreas_tail.nii.gz'],1):
        #print('Segment path:', segment,segments_path)
        seg_path = os.path.join(segments_path, segment)
        seg=load_canonical(seg_path).get_fdata()
        seg = np.where(seg > 0.5, 1, 0)
        if joint is None:
            joint=seg
        else:
            joint+=seg*i
            #threshold to i
            joint=np.where(joint > i, i, joint)

    joint, _ = resample_image(joint,original_spacing=spacing,
                             target_spacing=(1, 1, 1),order=0)
    #print('loaded pancreas segments from:',segments_path)
    return joint

def load_segments_kidney(segments_path,spacing):
    joint=None
    for i,segment in enumerate(['kidney_left.nii.gz','kidney_right.nii.gz'],1):
        #print('Segment path:', segment,segments_path)
        seg_path = os.path.join(segments_path, segment)
        seg=load_canonical(seg_path).get_fdata()
        seg = np.where(seg > 0.5, 1, 0)
        if joint is None:
            joint=seg
        else:
            joint+=seg*i
            #threshold to i
            joint=np.where(joint > i, i, joint)

    joint, _ = resample_image(joint,original_spacing=spacing,
                             target_spacing=(1, 1, 1),order=0)
    return joint
    
segment_labels = {
    'liver': {
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8'
    },
    'pancreas': {
        1: 'head',
        2: 'body',
        3: 'tail'
    },
    'kidney': {
        1: 'left kidney',
        2: 'right kidney'
    }
}

def get_tumor_segment(segments,tumor):
    uni=segments*tumor
    non_zero_elements = uni[uni != 0]
    unique_values, counts = np.unique(non_zero_elements, return_counts=True)
    # Calculate threshold: 10% of the number of non-zero elements
    threshold = int(0.10 * len(non_zero_elements))
    # Filter values that appear more than the threshold
    frequent_values = unique_values[counts > threshold]

    if frequent_values.size==0:
        return None
    else:
        return frequent_values


def analyze_nth_largest_connected_component(array_3d, ns=[1],th=None,erode=0,
                                            ct=None,segments=None,resize_factor=1):
    #erode: any tumor that disapears afer the binary erosion is ignored.
    #print('Received segments:',segments)
    
    from math import atan2, degrees
    
    # Define the connectivity (3x3x3 for 26-connectivity)
    structure = np.ones((3, 3, 3), dtype=int)
    
    # Label the connected components
    labeled_array, num_features = ndimage.label(array_3d, structure=structure)
    
    if num_features == 0:
        return None
    
    # Calculate the size of each connected component
    sizes = ndimage.sum(array_3d, labeled_array, range(1, num_features + 1))
    sorted_indices = np.argsort(sizes)[::-1]
    
    outputs = {}
    if ns is None:
        ns=list(range(1,len(sorted_indices)+1))
    included=0
    for n in ns:
        if n > num_features:
            continue
        # Find the label of the n-th largest connected component
        nth_largest_label = sorted_indices[n-1] + 1

        # Get a boolean mask of the n-th largest connected component
        nth_largest_component_mask = (labeled_array == nth_largest_label)
        
        if erode>0:
            #ignores tumors that disappear upon binary erosion
            eroded=scipy.ndimage.binary_erosion(nth_largest_component_mask,
                                                structure=np.ones((erode,erode,erode)),
                                                iterations=1)
            if eroded.sum()==0:
                continue
        
        # Measure the volume (number of voxels in the n-th largest component)
        volume = nth_largest_component_mask.sum()
        
        if segments is not None:
            msk = np.where(nth_largest_component_mask > 0.5, 1, 0)
            seg=get_tumor_segment(segments,msk)
        else:
            seg=None

        if ct is not None:
            msk = np.where(nth_largest_component_mask > 0.5, 1, 0)
            msk = scipy.ndimage.binary_erosion(msk,
                                                structure=np.ones((1,1,1)),
                                                iterations=1)
            segmented = ct*msk
            mean_hu = segmented.sum()/msk.sum()
            std_hu = segmented[msk != 0].std()
            #print(ct.shape,segmented[msk != 0].shape,msk.sum())
        else:
            mean_hu = None
            std_hu = None
            
        if th is not None:
            if volume<th:
                break

        # Find the indices of the n-th largest connected component
        component_indices = np.where(nth_largest_component_mask)
        
        # Iterate through the z-axis to find the longest diameter
        max_longest_diameter = 0
        longest_points = None
        for z in range(array_3d.shape[2]):
            slice_mask = nth_largest_component_mask[:, :, z]
            if np.any(slice_mask):
                diam, point1, point2=measure_diameter(slice_mask)
                if diam > max_longest_diameter:
                    max_longest_diameter = diam
                    longest_points = (point1, point2)
                    longest_diameter_slice=z

        longest_diameter = max_longest_diameter

        if longest_points is not None:
            # Compute perpendicular distances to the line defined by the longest points
            z=longest_diameter_slice
            slice_mask = nth_largest_component_mask[:, :, z]
            # Calculate the angle to rotate the image so the line between point1 and point2 is parallel to the x-axis
            point1, point2 = longest_points
            angle = degrees(atan2(point2[0] - point1[0], point2[1] - point1[1]))
            
            # Rotate the image
            #rotated_image = rotate_image(slice_mask, angle)
            rotated_image = rotate(slice_mask, angle, resize=True, order=0,
                                   preserve_range=True, mode='constant', cval=0)
            assert np.array_equal(rotated_image, rotated_image.astype(bool))
            
            #print_slice(rotated_image,spacing=resize_factor[1])

            # Measure the vertical span
            perpendicular_diameter = measure_vertical_span(rotated_image)
            perpendicular_point=None
            
        else:
            perpendicular_diameter = 0
            perpendicular_point = None

        # Measure the size along the x, y, and z axes (canonical meaning)
        x = component_indices[0].max() - component_indices[0].min() + 1
        center_x = int(component_indices[0].min() + x/2)
        
        y = component_indices[1].max() - component_indices[1].min() + 1
        center_y = int(component_indices[1].min() + y/2)
        center_y = array_3d.shape[1] - center_y

        included+=1
        outputs[included] = {
            "center_x": center_x,
            "center_y": center_y,
            "slice": np.round(longest_diameter_slice/resize_factor[-1],1),
            "volume": volume,
            "longest_diameter": int(longest_diameter),
            "perpendicular_diameter": int(perpendicular_diameter),
            "mean_hu": np.round(mean_hu,1),
            "std_hu":np.round(std_hu,1),
            "tumor_segment":seg
            }

    return outputs

def get_paths(folder,anno_folder,item,clss):
    tumor=None
    cyst=None
    lesion=None

    if clss=='liver' or clss=='kidney' or clss=='pancreas' or clss=='colon':
        #load tumor
        if clss=='liver':
            classes=['liver','hepatic']
        elif clss=='kidney':
            classes=['kidney','renal']
        elif clss=='pancreas':
            classes=['pancreas','pancreatic']
        elif clss=='colon':
            if os.path.isfile(os.path.join(anno_folder,item,f'segmentations/colon_lesion.nii.gz')):
                lesion=os.path.join(anno_folder,item,f'segmentations/colon_lesion.nii.gz')
        if clss!='colon':
            for c in classes:
                #load lesions
                if os.path.isfile(os.path.join(anno_folder,item,f'segmentations/{c}_tumor.nii.gz')):
                    if tumor is None:
                        tumor=os.path.join(anno_folder,item,f'segmentations/{c}_tumor.nii.gz')
                if os.path.isfile(os.path.join(anno_folder,item,f'segmentations/{c}_cyst.nii.gz')):
                    if cyst is None:
                        cyst=os.path.join(anno_folder,item,f'segmentations/{c}_cyst.nii.gz')
                if os.path.isfile(os.path.join(anno_folder,item,f'segmentations/{c}_lesion.nii.gz')):
                    if lesion is None:
                        lesion=os.path.join(anno_folder,item,f'segmentations/{c}_lesion.nii.gz')

        if clss=='pancreas':
            pdac=None
            pnet=None
            for c in classes:
                if os.path.isfile(os.path.join(anno_folder,item,f'segmentations/{c}_pdac.nii.gz')):
                    pdac=os.path.join(anno_folder,item,f'segmentations/{c}_pdac.nii.gz')
                if os.path.isfile(os.path.join(anno_folder,item,f'segmentations/{c}_pnet.nii.gz')):
                    pnet=os.path.join(anno_folder,item,f'segmentations/{c}_pnet.nii.gz')
        else:
            pdac=None
            pnet=None

    #load organ
    if clss=='kidney':
        if os.path.isfile(os.path.join(anno_folder,item,'segmentations/kidney_right.nii.gz')):
            organ=os.path.join(anno_folder,item,'segmentations/kidney_right.nii.gz')
        else:
            organ=os.path.join(anno_folder,item,'segmentations/_kidney_right.nii.gz')
    elif clss=='spleen':
        if os.path.isfile(os.path.join(anno_folder,item,'segmentations/spleen.nii.gz')):
            organ=os.path.join(anno_folder,item,'segmentations/spleen.nii.gz')
        else:
            organ=os.path.join(anno_folder,item,'segmentations/__spleen.nii.gz')

        organ=load_canonical(organ).get_fdata().astype('uint8')
        ct=os.path.join(folder,item,'ct.nii.gz')
        ct=load_canonical(ct)
        spacing=ct.header.get_zooms()
        ct=ct.get_fdata()
        organ, _ = resample_image(organ, original_spacing=spacing,
                                             target_spacing=(1, 1, 1),order=0)
        ct, _ = resample_image(ct, original_spacing=spacing,
                                                target_spacing=(1, 1, 1))
        organ=organ.astype('float32')
        text, segments, vol, organ_hu, organ_hu_std=organ_text(False,None,ct,organ,spacing,clss,skip_incomplete=None,item=None)
        return text
        #vol=measure_volume(organ,spacing=spacing,check_border=True)
        #organ_hu,organ_hu_std=measure_organ_hu(organ,0*organ,ct)
        #text=''
        #if vol is not None:
        #    text+=f"{clss.capitalize()}: "
        #    text+=f"Volume: {np.round(vol/1000,1)} cm^3. "
        #    text+=f"Mean HU value: {organ_hu} +/- {organ_hu_std}.\n"
        #return text
    else:
        if os.path.isfile(os.path.join(anno_folder,item,f'segmentations/{clss}.nii.gz')):
            organ=os.path.join(anno_folder,item,f'segmentations/{clss}.nii.gz')
        else:
            #raise ValueError(f'No {clss} segmentation found for {item}')
            organ=os.path.join(anno_folder,item,f'segmentations/_{clss}_.nii.gz')
    return tumor, cyst, lesion, organ, pdac, pnet

def load_n_resize_ct_n_organ(folder, item, clss, organ):
    ct=os.path.join(folder,item,'ct.nii.gz')
    ct=load_canonical(ct)
    spacing=ct.header.get_zooms()
    ct=ct.get_fdata()

    if 'kidney' not in clss:
        organ=load_canonical(organ).get_fdata().astype('uint8')
        # Resample the image to have 1mm voxel spacing in all dimensions
        organ, resize_factor = resample_image(organ, original_spacing=spacing,
                                                target_spacing=(1, 1, 1),order=0)
        organ_right,organ_left=None,None
    else:
        organ_right=load_canonical(organ).get_fdata().astype('uint8')
        organ_left=load_canonical(organ.replace('kidney_right','kidney_left')).get_fdata().astype('uint8')
        # Resample the image to have 1mm voxel spacing in all dimensions
        organ_right, resize_factor = resample_image(organ_right, original_spacing=spacing,
                                                target_spacing=(1, 1, 1),order=0)
        organ_left, _ = resample_image(organ_left, original_spacing=spacing, target_spacing=(1, 1, 1),order=0)
        organ=np.maximum(organ_right,organ_left)
    
        
    ct, resize_factor = resample_image(ct, original_spacing=spacing,
                                             target_spacing=(1, 1, 1))
    
    if not np.array_equal(organ, organ.astype(bool)):
        organ = np.where(organ > 0.5, 1, 0)
    
    organ = organ.astype('float32')

    return organ, ct, spacing, resize_factor,organ_right,organ_left


def pdac_resectability(interaction,location):
    #https://pdf.sciencedirectassets.com/273440/1-s2.0-S0016508513X00125/1-s2.0-S0016508513015886/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjECcaCXVzLWVhc3QtMSJGMEQCIEi12xGEUoRJrzEBL37kxFe%2FLVPVG%2B96mrAomZ3Sd6drAiBvGHDnvkfe%2BmIHT%2FWY%2BlTw5FpM7k4V%2B8xaaXzKJGlB4Cq8BQiw%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMTQ6IgTPO5GvOPMsTKpAFPgfiLQmONpuLR2qbe5%2BN0v5y4Reb9AOeHQsuE9aFB9awTwHCclNE0RUqhhlaZQO2rtq8NzbKaS3CeioqfeTFzyzDQu1RYlkXAsFQQB8RpG63Qt6AsOzJ05q3%2FHBA3wDsJM%2B4EUpTEMiz%2FCA9F0I3XwpFFxPVr5%2BUQgUJgK3zLAQIBVGVy9ulufZ5%2BMDksVilpRH%2BMrmSLD60d2GRcP3RKOwYsDwXmCE9yWxSn0GDJBVli8WB%2BJlUozcREFgFSc5DIG%2Fxy0hM5QKIdoNn%2FPSVf2rs6qg%2BpfkOyIVD2BPbBPTM5XwFL%2BsoVuvyg2hQmB9EZr7zvIz8bI98RQlzNinl3DdUAMOPFLTnr1LcodrKjnDDdCHikHrklKMszXxCU3djjPdGtBAUBvwTIQyvEdD6t535FUAWJ3M7vwZjqQT%2B%2B8dN4DRZNjcRhAlMJ1cUc4COBUFHVsAEKWTpIOpStObz%2BYR1r3kYQWNAA7mPUiFOooBKSqweXRy%2F1TxV7SDwwMpRg4SEn4hHuPUB%2F3g49PP4Opa5zp15mx0UrMq9loo6Eumamgv7bJmmWMW3YjK9q8Ytp%2FBtRghgewDSAqncZjYwWME6Gc%2FkfSDjJLUWwT7w7Nwf9znqaKpV4Y4QFJmJmhWbq3rRXib8cQOIUwFkxlRxb0It9Zc4yiP4zG6z3A1FrkArZxTfYPUdujPBexeozXxvsIlQ6DMWJ7bmqQeuNi6XYtRlz7O%2BwotBJbukCewErDduDvs%2F0i9SwLDcS7wPM5mmCbCAvpdTNtqpICf5Jkoqsi2RPGw8IJvWTfp17d2JEej2FzQQ0a%2BP%2FSWp8rhTjw4Be5xbil%2FSs4IDt6SqbYMMJDi7kFIjeag68dRZY73MwQYwoY7KuQY6sgFn4KyQMJ%2BrFKSNalsjcV3Spyhns%2BCuYqxQ5SNCIa0PMsu4odhy5x%2FZpjVg%2FtVw3dHL1Ww1C72Hi3QJb8Tw7FjOogcGdsjbZyDhI2DcLORalvpPnEGvZHLVEN8I5b%2FFhrDzZvewTfYhDlIzUb6rAyvn4XSvAyQkiv6%2BUaqZyYGCSjCqpXWFjNJcTzLNa9ZuaG1zEd4pM8WV4rAmRlw6qPWW9SlHLAoNURg3XaSAdgDZeNIa&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241111T232022Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRXXJHFK2%2F20241111%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=62324ad55fafd5880e480c0249178a3673387b29efc069d347236aa110d5afe0&hash=46a444302b873626b09e7396f2769ced82bb4076460a9b4b9d3b38cf8fa4ac21&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0016508513015886&tid=spdf-98be5b3a-4516-44a7-a1af-88e623e47b0f&sid=b64d08038638c849fc09d36-95a0255531edgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=14135a055552565651&rr=8e120332ab34421d&cc=us
    if int(interaction['SMA'])==0 and int(interaction['CHA'])==0 and int(interaction['CA'])==0 and int(interaction['portal vein and SMV'])==0:
        return 'resectable (no abutment of SMA, CHA, or portal vein or SMV).'
    elif 'head' in location and (int(interaction['SMA'])>180 or int(interaction['CA'])>0 or int(interaction['IVC'])>0):
        t= 'unresectable ('
        if int(interaction['SMA'])>180:
            t+='SMA encasement, '
        if int(interaction['CA'])>0:
            t+='CA abutment, '
        if int(interaction['IVC'])>0:
            t+='IVC abutment, '
        t=t[:-2]+').'
        return t
    elif 'head' not in location and (int(interaction['SMA'])>180 or int(interaction['CA'])>180):
        t= 'unresectable ('
        if int(interaction['SMA'])>180:
            t+='SMA encasement, '
        if int(interaction['CA'])>180:
            t+='CA encasement, '
        t=t[:-2]+').'
        return t
    elif int(interaction['portal vein and SMV'])==0 and int(interaction['CA'])==0 and int(interaction['SMA'])<180:
        t=' borderline resectable ('
        if int(interaction['CHA'])>0 and int(interaction['CHA'])<180:
            t+='CHA abutment, '
        elif int(interaction['CHA'])>180:
            t+='CHA encasement, '
        if int(interaction['SMA'])>0:
            t+='SMA abutment, '
        t=t[:-2]+').'
        return t
    else:
        return '.'
    
def get_largest_stage_key(data):
    # Define a function to parse stage into numeric and letter parts
    def parse_stage(stage):
        number_part = int(stage[1])  # Extracts the number after 'T'
        letter_part = stage[2:] if len(stage) > 2 else ''  # Extracts letter part if it exists
        return number_part, letter_part

    # Use max with a custom key based on parsed numeric and letter parts from the 'stage' key
    largest_stage_key = max(data.keys(), key=lambda k: parse_stage(data[k]['stage']))
    return largest_stage_key

def write_lesion_report(tumor,ct,organ,organ_hu,item,spacing,resize_factor,erode,th,
                        clss,skip_incomplete=False,plot=False,
                        lesion_type='lesion',segments=None,path=None,
                        organ_hu_std=None):
    print('Writing report for:', item, clss, lesion_type)

    if lesion_type not in ['lesion','malignant tumor','cyst','PDAC','PNET']:
        raise ValueError('Lesion type not implemented')

    #if plot:
    #    m,n=get_first_last_slices(organ,erode=5)
    #    plot_slices_overlay(ct,organ,None,-1,m,n,name='Components')
    #    plot_slices_overlay(ct,tumor,organ,-1,m,n,name='Tumors')

    text=''
    

    sizes=analyze_nth_largest_connected_component(tumor, ns=None,th=th,ct=ct,segments=segments,
                                                resize_factor=resize_factor,erode=erode)

    large=0
    large_hypo=0
    large_hyper=0
    small_hypo=0
    small_hyper=0
    small_iso=0
    large_iso=0
    stages={}
    interactions={}
    locations={}
    staged=True
    largest_stage=None
    largest_resectability=None
    local_largest=''

    for n in sizes:
        if clss=='pancreas':
            print(f'Pancreas tumor {n} segment is:', sizes[n]['tumor_segment'])

        #print(sizes[n])
        z=int(sizes[n]['slice'])
        s=f"{clss.capitalize()} lesion {n}: \n"
        if lesion_type in ['PDAC','PNET','cyst','malignant tumor']:
            s= s[:-2]+f" appearance consistent with {lesion_type}.\n"

        print('Tumor segment:', sizes[n]['tumor_segment'],", clss:",clss)

        if sizes[n]['tumor_segment'] is not None and len(sizes[n]['tumor_segment'])!=0 and clss!='colon':
            print('Inside')
            s+=f"Location: "
            if (clss=='liver'):
                s+='hepatic segment '
                if n==1:
                    local_largest='hepatic segment '
                for seg in sizes[n]['tumor_segment']:
                    s+=str(seg)+'/'
                    if n==1:
                        local_largest+=str(seg)+'/'
                s=s[:-1]
                if n==1:
                    local_largest=local_largest[:-1]
            elif (clss=='pancreas'):
                s+='pancreas '
                if n==1:
                    local_largest=''
                for seg in sizes[n]['tumor_segment']:
                    print(f'Tumor {n} pancreas segment:', seg)
                    s+=segment_labels['pancreas'][seg]+'/'
                    if n==1:
                        local_largest+=segment_labels['pancreas'][seg]+'/'
                s=s[:-1]
                if n==1:
                    local_largest=local_largest[:-1]
            elif (clss=='kidney'):
                #print('Segment:', sizes[n]['tumor_segment'])
                s+=segment_labels['kidney'][sizes[n]['tumor_segment'][0]]
                if n==1:
                    local_largest=segment_labels['kidney'][sizes[n]['tumor_segment'][0]]
            s+='.\n'

        if n==1 and local_largest!='':
            local_largest=f'({local_largest})'

        s+=f"Size: {np.round(sizes[n]['longest_diameter']/10,1)} x {np.round(sizes[n]['perpendicular_diameter']/10,1)} cm (image {z}). "
        s+=f"Volume: {np.round(sizes[n]['volume']/1000,1)} cm^3.\n"

        s+=f"Enhancement relative to {clss}: "

        
            
        if sizes[n]['mean_hu']>(organ_hu-organ_hu_std/10) and sizes[n]['mean_hu']<(organ_hu+organ_hu_std/10):
            s+=f"Isoattenuating "
        elif sizes[n]['mean_hu']<organ_hu:
            s+=f"Hypoattenuating " 
        else:
            s+=f"Hyperattenuating "
        #s+=f"{clss.capitalize()} lesion measuring {sizes[n]['longest_diameter']/10} x {sizes[n]['perpendicular_diameter']/10} cm.\n"
        s+=f"(HU value is {sizes[n]['mean_hu']}+/-{sizes[n]['std_hu']}).\n"

        #try:
            #pancreatic cancer staging
        if lesion_type!='cyst' and clss=='pancreas':
            try:
                interaction,tx,stage,contacted_organs=stg.stage(path,debug=False,size=sizes[n]['longest_diameter']/10,pnet=(lesion_type=='PNET'))
                stages[n]=stage
                interactions[n]=interaction
                locations[n]=sizes[n]['tumor_segment']
                sizes[n]['stage']=stage
                if len(contacted_organs)==0:
                    s+='Contact with adjacent organs: No. \n'
                else:
                    s+='Contact with adjacent organs: '
                    for organ in contacted_organs:
                        s+=organ+', '
                    s=s[:-2]+'.\n'
                s+=tx
                resec=pdac_resectability(interactions[n],locations[n])
                if resec!='.':
                    s+='Surgical resectability: '+resec+'\n'
                sizes[n]['resectability']=resec
            except Exception as e:
                print(f"Error staging {item}: {e}")
                with open('error_log_report_creation.txt', 'a') as f:
                    f.write(f"Error staging {item}: {str(e)}\n")
                    f.write(traceback.format_exc() + '\n')
                staged=False
        else:
            staged=False

        #except:
        #    print('Error staging lesion:', item, clss, lesion_type)
        #    staged=False

        s+="\n"
        
        text+=s
        if sizes[n]['longest_diameter']/10>5:
            large+=1
            if sizes[n]['mean_hu']>(organ_hu-organ_hu_std/10) and sizes[n]['mean_hu']<(organ_hu+organ_hu_std/10):
                large_iso+=1
            elif sizes[n]['mean_hu']<organ_hu:
                large_hypo+=1
            else:
                large_hyper+=1
        else:
            if sizes[n]['mean_hu']>(organ_hu-organ_hu_std/10) and sizes[n]['mean_hu']<(organ_hu+organ_hu_std/10):
                small_iso+=1
            elif sizes[n]['mean_hu']<organ_hu:
                small_hypo+=1
            else:
                small_hyper+=1

        if plot:
            print(s)
            plot_slice_overlay(ct,tumor,z,spacing,organ,-1,name='Tumors')
            plot_slice_overlay(ct,tumor,z,spacing,organ,-1,name='Tumors',
                                overlay=False)

                                
            
    if len(sizes) != 0:
        text=f"{clss.capitalize()} {lesion_type}s:\n"+text
        text+='IMPRESSION: \n'

        if large_hypo>0 and large_hyper==0 and large_iso==0:
            typ_L='hypoattenuating'
        elif large_hypo==0 and large_hyper>0 and large_iso==0:
            typ_L='hyperattenuating'
        elif large_hypo==0 and large_hyper==0 and large_iso>0:
            typ_L='isoattenuating'
        else:
            typ_L=''

        if small_hypo>0 and small_hyper==0 and small_iso==0:
            typ_S='hypoattenuating'
        elif small_hypo==0 and small_hyper>0 and small_iso==0:
            typ_S='hyperattenuating'
        elif small_hypo==0 and small_hyper==0 and small_iso>0:
            typ_S='isoattenuating'
        else:
            typ_S=''

        if large>1:

            if lesion_type=='PDAC' or lesion_type=='PNET':
                text+=f'Multiple ({large}) large {typ_L} {clss} masses, consistent with biopsy-proven {lesion_type}. '
            elif lesion_type=='cyst':
                text+=f'Multiple ({large}) large {typ_L} {clss} masses of cystic appearance. '
            elif lesion_type=='malignant tumor':
                text+=f'Multiple ({large}) large {typ_L} {clss} masses of malignant appearance. '
            else:
                text+=f'Multiple ({large}) large {typ_L} {clss} masses. '

            text+=f'Largest one {local_largest} measures {np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)} cm.'
            if staged:
                text+=f'\nClinical stage: {sizes[get_largest_stage_key(sizes)]['stage']}NxMx.\n'
                ad=sizes[get_largest_stage_key(sizes)]['resectability']
                if ad!='.':
                    text=text+'Surgical resectability: '+ad+'\n'
            if len(sizes)>large:
                if (len(sizes)-large)==1:
                    text+=f' Additionally, a smaller {typ_S} {clss} {lesion_type} present.'
                else:
                    text+=f' Additionally, {len(sizes)-large} smaller {typ_S} {clss} {lesion_type}s present.'
        
        elif large==1:

            if lesion_type=='PDAC' or lesion_type=='PNET':
                text+=f'A large {typ_L} {clss} {local_largest} mass ({np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)}, cm),  consistent with biopsy-proven {lesion_type}. '
            elif lesion_type=='cyst':
                text+=f'A large {typ_L} {clss} {local_largest} mass of cystic appearance ({np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)} cm). '
            elif lesion_type=='malignant tumor':
                text+=f'A large {typ_L} {clss} {local_largest} mass of malignant appearance ({np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)} cm). '
            else:
                text+=f'A large {typ_L} {clss} {local_largest} mass ({np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)} cm). '
            if staged:
                text+=f'\nClinical stage: {sizes[get_largest_stage_key(sizes)]['stage']}NxMx.\n'
                ad=sizes[get_largest_stage_key(sizes)]['resectability']
                if ad!='.':
                    text=text+f'Surgical resectability: '+ad+'\n'
            if len(sizes)>large:
                if (len(sizes)-large)==1:
                    text+=f' Additionally, a smaller {typ_S} {clss} {lesion_type} present.'
                else:
                    text+=f' Additionally, {len(sizes)-large} smaller {typ_S} {clss} {lesion_type}s present.'
        
        else:

            if len(sizes)==1:
                if lesion_type=='PDAC' or lesion_type=='PNET':
                    text+=f'A {typ_S} {clss} {local_largest} mass ({np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)} cm), consistent with biopsy-proven {lesion_type}. '
                elif lesion_type=='cyst':
                    text+=f'A {typ_S} {clss} {local_largest} mass of cystic appearance ({np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)} cm). '
                elif lesion_type=='malignant tumor':
                    text+=f'A {typ_S} {clss} {local_largest} mass of malignant appearance ({np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)} cm). '
                else:
                    text+=f'A {typ_S} {clss} {local_largest} mass ({np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)} cm).'
            else:
                if lesion_type=='PDAC' or lesion_type=='PNET':
                    text+=f'Multiple ({len(sizes)}) {typ_S} {clss} masses consistent with biopsy-proven {lesion_type}. Largest one {local_largest} measures {np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)} cm. '
                elif lesion_type=='cyst':
                    text+=f'Multiple ({len(sizes)}) {typ_S} {clss} masses of cystic appearance. Largest one {local_largest} measures {np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)} cm. '
                elif lesion_type=='malignant tumor':
                    text+=f'Multiple ({len(sizes)}) {typ_S} {clss} masses of malignant appearance. Largest one {local_largest} measures {np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)} cm. '
                else:
                    text+=f'Multiple ({len(sizes)}) {typ_S} {clss} masses. Largest one {local_largest} measures {np.round(sizes[1]["longest_diameter"]/10,1)} x {np.round(sizes[1]["perpendicular_diameter"]/10,1)} cm. '
            
            if staged:
                text+=f'\nClinical stage: {sizes[get_largest_stage_key(sizes)]['stage']}NxMx.\n'
                ad=sizes[get_largest_stage_key(sizes)]['resectability']
                if ad!='.':
                    text=text+f'Surgical resectability: '+ad+'\n'

            text=text.replace('  ',' ')

    
    text+='\n'
    if len(sizes)>1:
        if lesion_type=='PDAC' or lesion_type=='PNET':
            text+=f'Total volume of all {clss} {lesion_type}s: '
        elif lesion_type=='cyst':
            text+=f'Total volume of all {clss} masses of cystic appearance: '
        elif lesion_type=='malignant tumor':
            text+=f'Total volume of all {clss} masses of malignant appearance: '
        else:
            text+=f'Total volume of all {clss} masses: '
        text+=f'{np.round(sum([sizes[n]["volume"] for n in sizes])/1000,1)} cm^3.\n'
    return text


def organ_text(healthy,tumors,ct,organ,spacing,clss,skip_incomplete,item,organ_right=None,organ_left=None,seg_pth=None,
               spleen_hu=None,phase=None):
    global anno_folder
    
    if tumors is not None:
        organ_hu,organ_hu_std=measure_organ_hu(organ,tumors,ct)
    else:
        organ_hu,organ_hu_std=measure_organ_hu(organ,0*organ,ct)

    if clss=='liver':
        segments=load_segments_liver(anno_folder+f'/{item}/segmentations/',spacing)
    elif clss=='pancreas':
        try:
            segments=load_segments_pancreas(anno_folder+f'/{item}/segmentations/',spacing)
        except:
            segments=None
    elif clss=='kidney':
        segments=load_segments_kidney(anno_folder+f'/{item}/segmentations/',spacing)
    else:
        segments=None

        

    if 'kidney' not in clss:
        vol=measure_volume(organ,spacing=spacing,check_border=True)
        vol_right=None
        vol_left=None
    else:
        vol_right=measure_volume(organ_right,spacing=spacing,check_border=True)
        vol_left=measure_volume(organ_left,spacing=spacing,check_border=True)
        if vol_right is None or vol_left is None:
            vol=None
        else:
            vol=vol_right+vol_left

    att='normal'
    if vol is not None:
        size='normal'
        size_right='normal'
        size_left='normal'
        
        if clss=='spleen':
            #Taylor A, Dodds W, Erickson S, Stewart E. CT of Acquired Abnormalities of the Spleen. AJR Am J Roentgenol. 1991;157(6):1213-9. doi:10.2214/ajr.157.6.1950868 - Pubmed
            if vol/1000>314.5:
                size='large'
            if vol/1000>430.8:
                size='massive'
        elif clss=='kidney':
            #https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5837692/%23:~:text%3DThresholds%2520for%2520low%2520and%2520high,415.2%2520cm3%2520for%2520men.&ved=2ahUKEwi95pPEuauJAxVIFVkFHaTvDCUQ-tANegQIChAF&usg=AOvVaw21JO0ABiIFJajSWmLMvI76
            if vol_right/1000>(415.2/2):
                size_right='large'
            if vol_left/1000>(415.2/2):
                size_left='large'
        elif clss=='liver':
            #highly depends on weight, this is about the maximum for a 150kg man
            if vol/1000>3000:
                size='large'
            if organ_hu<=40:
                att='fatty'
        elif clss=='pancreas':
            #https://pubmed.ncbi.nlm.nih.gov/29972077/
            if vol/1000>83:
                size='large'
            if spleen_hu is not None and organ_hu/spleen_hu<=0.7:
                att='fatty'
        else:
            size=None
    else:
        size=None

    
    text=''
    text+=f"{clss.capitalize()}: \n"

    if vol is not None:
        if 'kidney' not in clss:
            if size=='normal':
                text+='Normal size '
            elif size=='large':
                text+=f"{clss.capitalize()} is enlarged "
            elif size=='massive':
                text+=f"{clss.capitalize()} is massively enlarged "

        else:
            if size_right=='normal' and size_left=='normal':
                text+='Normal size '
            elif size_right=='large' or size_left=='large':
                text+=f"Bilateral {clss}s are enlarged "
            elif size_right=='massive' or size_left=='massive':
                text+=f"Bilateral {clss}s are massively enlarged "
            else:
                text+='Right kidney size is '
                if size_right=='normal':
                    text+='normal, '
                elif size_right=='large':
                    text+='enlarged, '
                elif size_right=='massive':
                    text+='massively enlarged, '
                text+='left kidney size is '
                if size_left=='normal':
                    text+='normal '
                elif size_left=='large':
                    text+='enlarged '
                elif size_left=='massive':
                    text+='massively enlarged '
        if 'kidney' in clss:
            text+=f"(right kidney volume: {np.round(vol_right/1000,1)} cm^3; "
            text+=f"left kidney volume: {np.round(vol_left/1000,1)} cm^3; "
            text+=f"total kidney volume: {np.round(vol/1000,1)} cm^3). "
        elif 'colon' in clss:
            text+=f"Volume: {np.round(vol/1000,1)} cm^3.\n"
        else:
            text+=f"(volume: {np.round(vol/1000,1)} cm^3).\n"

    #print('clss:', clss)
    #if clss!='spleen':
    #    print('phase:', phase)
    #    print('spleen_hu:', spleen_hu)
    #    print('organ_hu:', organ_hu)
        #print('att:', att)

    
    if phase is not None and phase=='Plain':
        if clss=='liver' and (spleen_hu is not None) and (organ_hu is not None):
            if att=='fatty':
                text+=f"Fatty infiltration (Mean HU value: {organ_hu} +/- {organ_hu_std}).\n"
            else:
                text+=f"Normal attenuation (Mean HU value: {organ_hu} +/- {organ_hu_std}).\n"
        elif clss=='pancreas' and (spleen_hu is not None) and (organ_hu is not None):
            if att=='fatty':
                text+=f"Fatty infiltration, mean HU value: {organ_hu} +/- {organ_hu_std}, pancreatic index (P/S): {np.round(organ_hu/spleen_hu,2)}.\n"
            else:
                text+=f"Normal attenuation, mean HU value: {organ_hu} +/- {organ_hu_std}, pancreatic index (P/S): {np.round(organ_hu/spleen_hu,2)}.\n"
        else:
            text+=f"Mean HU value: {organ_hu} +/- {organ_hu_std}.\n"
    else:
        text+=f"Mean HU value: {organ_hu} +/- {organ_hu_std}.\n"

    if clss=='colon':
        text='Colon:\n'
    
    
    return text, segments, vol, organ_hu, organ_hu_std
    

def create_report(folder,anno_folder,item,clss,names,N=0,plot=False,skip_incomplete=False,th=None,
                  phase=None,spleen_hu=None):
    if not os.path.isdir(os.path.join(folder,item)):
        return None
    
    if item in names:
        print('repeated:', item)
        return None
    
    #get paths for lesion and organ
    tmp=get_paths(folder,anno_folder,item,clss)
    if isinstance(tmp,str):
        return tmp
    tumor, cyst, lesion, organ, pdac, pnet =tmp
    
    if th is None or th==0:
        if tumor is not None and '/_' in tumor:
            th=50
            erode=3
        else:
            th=0
            erode=0
    else:
        th=th
        erode=3
    
    if not os.path.exists(organ):
        #return None
        raise ValueError('Organ not found:', organ)
    
    #load ct and organ
    organ, ct, spacing, resize_factor,organ_right,organ_left=load_n_resize_ct_n_organ(folder, item, clss, organ)

    if tumor is not None:
        tumor=load_canonical(tumor).get_fdata().astype('uint8')
        if not np.array_equal(tumor, tumor.astype(bool)):
            tumor = np.where(tumor > 0.5, 1, 0)
        tumor, resize_factor = resample_image(tumor, original_spacing=spacing,target_spacing=(1, 1, 1),order=0)
        tumor=tumor.astype('float32')
    if cyst is not None:
        cyst=load_canonical(cyst).get_fdata().astype('uint8')
        if not np.array_equal(cyst, cyst.astype(bool)):
            cyst = np.where(cyst > 0.5, 1, 0)
        cyst, resize_factor = resample_image(cyst, original_spacing=spacing,target_spacing=(1, 1, 1),order=0)
        cyst=cyst.astype('float32')
    if lesion is not None:
        lesion=load_canonical(lesion).get_fdata().astype('uint8')
        if not np.array_equal(lesion, lesion.astype(bool)):
            lesion = np.where(lesion > 0.5, 1, 0)
        lesion, resize_factor = resample_image(lesion, original_spacing=spacing,target_spacing=(1, 1, 1),order=0)
        lesion=lesion.astype('float32')
    if pdac is not None:
        pdac=load_canonical(pdac).get_fdata().astype('uint8')
        if not np.array_equal(pdac, pdac.astype(bool)):
            pdac = np.where(pdac > 0.5, 1, 0)
        pdac, resize_factor = resample_image(pdac, original_spacing=spacing,target_spacing=(1, 1, 1),order=0)
        pdac=pdac.astype('float32')
    if pnet is not None:
        pnet=load_canonical(pnet).get_fdata().astype('uint8')
        if not np.array_equal(pnet, pnet.astype(bool)):
            pnet = np.where(pnet > 0.5, 1, 0)
        pnet, resize_factor = resample_image(pnet, original_spacing=spacing,target_spacing=(1, 1, 1),order=0)
        pnet=pnet.astype('float32')

    if lesion is not None:
        if tumor is not None:
            lesion=lesion-tumor
        if cyst is not None:
            lesion=lesion-cyst
        if pdac is not None:
            lesion=lesion-pdac
        if pnet is not None:
            lesion=lesion-pnet
        if tumor is not None or cyst is not None or pdac is not None or pnet is not None:
            #binary erosion
            lesion=np.where(lesion > 0.5, 1, 0)
            lesion=scipy.ndimage.binary_erosion(lesion,structure=np.ones((1,1,1)), iterations=1)
            lesion=np.where(lesion > 0.5, 1, 0)
            if lesion.sum()<10:
                lesion=None

    #assert no intersection between pnet and lesion
    if pnet is not None and lesion is not None:
        print('PNET:', np.sum(pnet))
        print('Lesion:', np.sum(lesion))
        if np.sum(pnet*lesion)>0:
            raise ValueError('Intersection between pnet and lesion')

    all_lesions=0*organ
    if tumor is not None:
        all_lesions+=tumor
    if cyst is not None:
        all_lesions+=cyst
    if lesion is not None:
        all_lesions+=lesion
    if pdac is not None:
        all_lesions+=pdac
    if pnet is not None:
        all_lesions+=pnet
    all_lesions=np.where(all_lesions > 0.5, 1, 0)
    healthy=all_lesions.sum()==0.0
    if tumor is None and cyst is None and lesion is None:
        healthy=False

    if tumor is None and cyst is None and lesion is None:
        if skip_incomplete:
            return ''  # No lesions found, return empty string

    if healthy and skip_incomplete:
        return f"{clss.capitalize()}: Unremarkable."
    


    
    
    texts=[]
    if tumor is None and cyst is None and lesion is None and pdac is None and pnet is None:
        #not clear if healthy
        text, segments, vol, organ_hu, organ_hu_std=organ_text(False,None,ct,organ,spacing,clss,skip_incomplete,item,organ_right,organ_left,
                                                                spleen_hu=spleen_hu,phase=phase)
        if clss=='colon':
            text=''
        return text
    elif healthy:
        #healthy
        text, segments, vol, organ_hu, organ_hu_std=organ_text(True,all_lesions,ct,organ,spacing,clss,skip_incomplete,item,organ_right,organ_left,
                                                                spleen_hu=spleen_hu,phase=phase)
        if clss=='colon':
            text=''
        return text
    else:
        #not healthy
        text, segments, vol, organ_hu, organ_hu_std=organ_text(False,all_lesions,ct,organ,spacing,clss,skip_incomplete,item,organ_right,organ_left,
                                                                spleen_hu=spleen_hu,phase=phase)
        for problem,lesion_type in zip([pdac,pnet,tumor,cyst,lesion],['PDAC','PNET','malignant tumor','cyst','lesion']):
            if problem is not None and problem.sum()>0:
                texts.append(write_lesion_report(problem,ct,organ,organ_hu,item,spacing,resize_factor,erode,th,clss,skip_incomplete,plot,
                        lesion_type,segments,path=os.path.join(anno_folder,item,'segmentations'),organ_hu_std=organ_hu_std))
        impressions='IMPRESSION: \n'
        findings=''
        for t in texts:
            if "IMPRESSION:" in t:
                finding, separator, impression = t.partition("IMPRESSION:")
                print('Impression:', impression)#impression correct here!
                findings += finding.strip() + "\n"
                impressions += impression.strip() + "\n"
            else:
                findings += t.strip() + "\n"
        print('Impressions:', impressions)
        text += '\n' + findings + impressions
        
    return text

def multi_organ_report(folder,names,anno_folder,item,skip_incomplete,plot,N,phase=None,th=None):
    try:
        report = real_multi_organ_report(folder,names,anno_folder,item,skip_incomplete,plot,N,phase,th)
        return report
    except Exception as e:
        print(f"Error processing {item}: {e}")
        with open('error_log_report_creation.txt', 'a') as f:
            f.write(f"Error processing {item}: {str(e)}\n")
            f.write(traceback.format_exc() + '\n')

def real_multi_organ_report(folder,names,anno_folder,item,skip_incomplete,plot,N,phase=None,th=None):

    start=time.time()
    print('Processing:', item)
    print('Phase:', phase)
    findings=''
    impressions=''
    spleen_hu=None
    for clss in ['spleen','liver','pancreas','kidney','colon']:
        print('Processing:', item, clss)
        report=create_report(folder=folder,anno_folder=anno_folder,item=item,clss=clss,
                                names=names,N=N,plot=plot,skip_incomplete=skip_incomplete,th=th,
                                phase=phase,spleen_hu=spleen_hu)
        #print(report)
        if report is not None:
            if clss=='spleen':
                print('Report for spleen:', report)
                spleen_hu=re.search(r"Mean HU value: ([\d\.]+)", report)
                if spleen_hu is not None:
                    spleen_hu=float(spleen_hu.group(1))
                    print('Spleen HU is', spleen_hu)
            div=report.find('IMPRESSION: \n')
            if div!=-1:
                findings+=report[:report.find('IMPRESSION: \n')]+'\n'
                if clss!='colon':
                    #kidneys
                    if 'Bilateral kidneys are enlarged' in report:
                        impressions += 'Enlarged kidneys. '
                    elif 'Right kidney size is enlarged' in report:
                        impressions += 'Enlarged right kidney. '
                    elif 'Left kidney size is enlarged' in report:
                        impressions += 'Enlarged left kidney. '
                    elif 'Bilateral kidneys are massively enlarged' in report:
                        impressions += 'Massively enlarged kidneys. '
                    elif 'Right kidney size is massively enlarged' in report:
                        impressions += 'Massively enlarged right kidney. '
                    elif 'Left kidney size is massively enlarged' in report:
                        impressions += 'Massively enlarged left kidney. '
                    #other organs
                    elif 'massively enlarged' in findings:
                        impressions += f'Massively enlarged {clss}. '
                    elif 'enlarged' in findings:
                        impressions += f'Enlarged {clss}. '
                impressions+=report[report.find('IMPRESSION: \n')+len('IMPRESSION: \n'):]#+'\n'
            else:
                findings+=report+'\n'
                if clss!='colon':
                    #kidneys
                    if 'Bilateral kidneys are enlarged' in report:
                        impressions += 'Enlarged kidneys. '
                    elif 'Right kidney size is enlarged' in report:
                        impressions += 'Enlarged right kidney. '
                    elif 'Left kidney size is enlarged' in report:
                        impressions += 'Enlarged left kidney. '
                    elif 'Bilateral kidneys are massively enlarged' in report:
                        impressions += 'Massively enlarged kidneys. '
                    elif 'Right kidney size is massively enlarged' in report:
                        impressions += 'Massively enlarged right kidney. '
                    elif 'Left kidney size is massively enlarged' in report:
                        impressions += 'Massively enlarged left kidney. '
                    #other organs
                    elif 'massively enlarged' in findings:
                        impressions += f'Massively enlarged {clss}. \n'
                    elif 'enlarged' in findings:
                        impressions += f'Enlarged {clss}. \n'
    if findings=='':
        return
    if impressions=='':
        impressions='No tumor observed in the liver, pancreas or kidneys.'

    report=findings+'IMPRESSION:\n'+impressions
    print('Report for:', item,'; time:', time.time()-start)

    report='FINDINGS: \n'+report
    if phase is not None:
        if phase=='Plain':
            report='Non-contrast CT \n'+report
        else:
            report='CT '+phase+' Phase \n\n'+report
    
    report=report.replace('  ',' ').replace('\n\n\n','\n\n')
    if report[:-2]=='\n':
        report=report[:-2]
    if report[:-2]=='\n':
        report=report[:-2]
        
    return report



def process_item(case, csv_file, folder, names, anno_folder, skip_incomplete, plot, N, metadata=None,th=None):
    phase=None
    if metadata is not None:
        if isinstance(metadata,str) and metadata=='JHH':
            if 'A' in case:
                phase='Arterial'
            elif 'V' in case:
                phase='Venous'
            else:
                phase=None
        else:
            try:
                phase=metadata.loc[metadata['BDMAP ID'] == case, 'CT Phase'].values[0]
            except:
                phase=metadata.loc[metadata['BDMAP ID'] == case, 'CT Phase'].values
            print('Phase in metadata:', phase)
            #check if string
            if not isinstance(phase,str):
                phase=None

    if phase is None:
        #get only cases where 'MSD' or 'Decathlon' is in original_id (getting from FLARE) or original_dataset (getting from FLARE)
        mapping=pd.read_csv('AbdomenAtlas1.0_id_mapping.csv')
        msd=mapping[(mapping['original_id (getting from FLARE)'].str.contains('MSD')) | (mapping['original_id (getting from FLARE)'].str.contains('Decathlon')) | (mapping['original_dataset (getting from FLARE)'].str.contains('MSD')) | (mapping['original_dataset (getting from FLARE)'].str.contains('Decathlon'))]['AbdomenAtlas_id'].to_list()
        if case in msd:
            phase='Venous'

    report = real_multi_organ_report(folder, names, anno_folder, case, skip_incomplete, plot, N,phase=phase,th=th)
    if report is not None:
        with open(csv_file, 'a', newline='', encoding='utf-8') as csv_dsc:
            writer = csv.writer(csv_dsc, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            writer.writerow([case, report])

    print(f"Case {case} processed.")
    print(f'Report for case {case}:', report)
    #except Exception as e:
    #    # Handle and log errors
    #    print(f"An error occurred with case {case}: {e}")
    #    with open('BUGS.txt', 'a') as f:
    #       f.write(f"BAD CASE: {case}\n")

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





def AbdomenAtlasReport(plot=False,restart_csv=False,skip_incomplete=True,csv_file='AbdomenAtlasReports.csv',num_workers=10,
                       current_part=0,parts=1,dataset='AA',th=None,args=None):
    """
    Generates a report for AbdomenAtlas data.
    """
    
    folder='./AbdomenAtlas/'
    global anno_folder
    anno_folder='./AbdomenAtlas/'

    if dataset=='AAMini':
        folder='./AbdomenAtlasMini/'
        anno_folder='./AbdomenAtlasMini/'

    if dataset=='AA' or dataset=='AAMini':
        ids=sorted([x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder,x))])
    else:
        raise ValueError('Dataset not implemented')
    
    metadata=pd.read_csv('AbdomenAtlas_metadata.csv')

    #remove duplicates and sort
    ids=list(set(ids))
    ids.sort()

    #print('hi')
    #print('cases before split: ', len(ids))
    #print('Parts:', parts)
    if parts>1:
        print('part split:', current_part, parts)
        #split the ids into parts (parts is the number of parts)
        ids=get_part(ids, parts, current_part)

    print('Number of cases:', len(ids))

    N=0

    names=[]

    if restart_csv or not os.path.isfile(csv_file):
        with open(csv_file, 'w') as f:
            f.write('Case, Report\n')
    else:
        with open(csv_file, 'r') as f:
            lines=f.readlines()
            for line in lines[1:]:
                names.append(line.split(',')[0])

    items=[x for x in ids if x not in names]
    print('Items to process:', len(items))

    if num_workers > 1:
        # Parallel execution using Pool
        pool = multiprocessing.Pool(processes=num_workers)
        for case in items:
            pool.apply_async(process_item, (case, csv_file, folder, names, anno_folder, skip_incomplete, plot, N, metadata, th))
        pool.close()
        pool.join()
    else:
        # Sequential execution for cases
        for case in items:
            process_item(case, csv_file, folder, names, anno_folder, skip_incomplete, plot, N, metadata=metadata, th=th)



def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run AbdomenAtlasReport with specified options.")
    
    parser.add_argument('--skip_incomplete', action='store_true', default=False, help='Skip evaluating cases (default: False)')
    parser.add_argument('--restart_csv', action='store_true', default=False, help='Restart the CSV file (default: True)')
    parser.add_argument('--plot', action='store_true', default=False, help='Enable plotting (default: False)')
    parser.add_argument('--csv_file', default='AbdomenAtlasReports.csv')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--current_part', default=0, type=int)
    parser.add_argument('--parts', default=1, type=int)
    parser.add_argument('--dataset', default='AA', type=str)
    parser.add_argument('--th', default=0, type=int)#ignores tumors smaller than 50 mm3 (noise removal)
    parser.add_argument('--pancreas_only', action='store_true', default=False, help='Run only pancreatic cancer cases in abdomen atlas (Datastet must be AA)')
    parser.add_argument('--colon_only', action='store_true', default=False, help='Run only colon cancer cases in abdomen atlas (Datastet must be AA)')
    
    #remove error_log_report_creation.txt if it exists
    if os.path.exists('error_log_report_creation.txt'):
        os.remove('error_log_report_creation.txt')
    

    # Parse the arguments
    args = parser.parse_args()

    # Run AbdomenAtlasReport with the provided arguments
    AbdomenAtlasReport(skip_incomplete=args.skip_incomplete, restart_csv=args.restart_csv, plot=args.plot, csv_file=args.csv_file, num_workers=args.num_workers,
                       current_part=args.current_part, parts=args.parts, dataset=args.dataset,th=args.th,args=args)

if __name__ == "__main__":
    main()
