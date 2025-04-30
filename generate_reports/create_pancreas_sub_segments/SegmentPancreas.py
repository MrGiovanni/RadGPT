import nibabel as nib
import numpy as np
import time
from scipy.ndimage import label, binary_erosion, binary_dilation
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import rotate, shift
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import nibabel.orientations as nio
from scipy.ndimage import affine_transform
import os
import argparse
from pathlib import Path
from multiprocessing import Pool
import traceback



def overlay_sma_pancreas(sma, pancreas, x_line=None, axis=1, flip_vertically=True, vector=None, activated=False):
    if not activated:
        return
    sma=sma.copy()
    pancreas=pancreas.copy()

    
    if vector is not None:
        print('Vector:', vector)
        print('SMA shape: ', sma.shape)
        v=[(i//2) for i in pancreas.shape]
        n=0
        while True:
            if (int(pancreas.shape[0]//2+n*vector[0]) < 0 or int(pancreas.shape[0]//2+n*vector[0] >= pancreas.shape[0]) or
                int(pancreas.shape[1]//2+n*vector[1]) < 0 or int(pancreas.shape[1]//2+n*vector[1] >= pancreas.shape[1]) or
                int(pancreas.shape[2]//2+n*vector[2]) < 0 or int(pancreas.shape[2]//2+n*vector[2] >= pancreas.shape[2])):
                break
            
            try:
                pancreas[int(pancreas.shape[0]//2+n*vector[0]),
                        int(pancreas.shape[1]//2+n*vector[1]),
                        int(pancreas.shape[2]//2+n*vector[2])] = True
                #pancreas[int(pancreas.shape[0]//2-n*vector[0]),
                #        int(pancreas.shape[1]//2-n*vector[1]),
                #        int(pancreas.shape[2]//2-n*vector[2])] = True
            except:
                break
            n+=1

            

    # Project to 2D based on specified axis
    print( 'SMA shape: ', sma.shape)
    print( 'Pancreas shape: ', pancreas.shape)
    sma_projection = sma.sum(axis=axis).T
    pancreas_projection = pancreas.sum(axis=axis).T

    # Make binary for clear overlay
    sma_projection = (sma_projection > 0).astype(int)
    pancreas_projection = (pancreas_projection > 0).astype(int)

    # Create an RGB image with pancreas in cyan and SMA in red
    overlay_image = np.zeros((sma_projection.shape[0], sma_projection.shape[1], 3), dtype=np.uint8)
    overlay_image[..., 0] = sma_projection * 255
    overlay_image[..., 1] = pancreas_projection * 255
    overlay_image[..., 2] = pancreas_projection * 255

    # Flip image vertically only if needed
    if flip_vertically:
        overlay_image = np.flipud(overlay_image)

    # Display image with optional line
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_image)
    plt.axis('off')
    plt.title("SMA (Red) Overlaid on Pancreas Projection")

    if x_line is not None:
        plt.axvline(x=x_line, color='yellow', linestyle='--', linewidth=2)
        plt.text(x_line + 5, overlay_image.shape[0] - 10, f"x = {x_line}", color='yellow', fontsize=12, ha='left', va='top')

    

    plt.show()

def print_x_span(data, axis=0):
    """
    Prints the x-span (extent in the specified axis) of a 3D binary object.

    Parameters:
    - data (ndarray): 3D binary numpy array representing the object.
    - axis (int): Axis along which to calculate the span (default is x-axis).
    """
    # Find the non-zero indices along the specified axis
    non_zero_indices = np.where(data.any(axis=(1, 2)))[0] if axis == 0 else \
                       np.where(data.any(axis=(0, 2)))[0] if axis == 1 else \
                       np.where(data.any(axis=(0, 1)))[0]

    min_index, max_index = non_zero_indices.min(), non_zero_indices.max()
    return max_index - min_index + 1


def crop_pancreas_with_bounding_box(pancreas, center_of_mass, box_size):
    """
    Crops the pancreas volume around a bounding box centered on the center of mass,
    with the bounding box size scaled to 1.5 times the maximum span of the pancreas.
    
    Parameters:
    - pancreas: 3D numpy array representing the binary pancreas mask.
    - center_of_mass: Tuple or array-like of length 3 (x, y, z) representing the center of mass coordinates.
    
    Returns:
    - cropped_volume: 3D numpy array representing the cropped (and possibly padded) pancreas volume.
    """
    
    # Calculate the bounding box limits, centered on the center of mass
    x_min = int(max(center_of_mass[0] - box_size // 2, 0))
    x_max = int(min(center_of_mass[0] + box_size // 2, pancreas.shape[0] - 1))
    
    y_min = int(max(center_of_mass[1] - box_size // 2, 0))
    y_max = int(min(center_of_mass[1] + box_size // 2, pancreas.shape[1] - 1))
    
    z_min = int(max(center_of_mass[2] - box_size // 2, 0))
    z_max = int(min(center_of_mass[2] + box_size // 2, pancreas.shape[2] - 1))
    
    # Crop the pancreas volume within the bounding box
    cropped_pancreas = pancreas[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    
    # Calculate padding needed if bounding box extends beyond volume limits
    pad_x = ((int(max(0, box_size / 2 - center_of_mass[0])), int(max(0, center_of_mass[0] + box_size / 2 - pancreas.shape[0] + 1))))
    pad_y = ((int(max(0, box_size / 2 - center_of_mass[1])), int(max(0, center_of_mass[1] + box_size / 2 - pancreas.shape[1] + 1))))
    pad_z = ((int(max(0, box_size / 2 - center_of_mass[2])), int(max(0, center_of_mass[2] + box_size / 2 - pancreas.shape[2] + 1))))
    #print('Center of mass:', center_of_mass)
    #print('Padding:', pad_x, pad_y, pad_z)
    #print('Pancreas shape:', pancreas.shape)
    #print('Box size:', box_size)
    # Apply padding if needed
    cropped_volume = np.pad(cropped_pancreas, (pad_x, pad_y, pad_z), mode='constant', constant_values=0)
    
    return cropped_volume

def center_of_mass(pancreas,downsample=True,downsample_factor=100):
    # Get the coordinates of non-zero voxels in the pancreas
    non_zero_coords = np.column_stack(np.nonzero(pancreas))
    # Downsample if necessary
    if downsample and non_zero_coords.shape[0] > downsample_factor:
        non_zero_coords = non_zero_coords[::downsample_factor]
    # Compute center of mass and shift vector
    original_center_of_mass = non_zero_coords.mean(axis=0)
    return original_center_of_mass, non_zero_coords

def rotate_3d_array(data, principal_component,reverse=False):
    # Step 1: Normalize the principal component
    principal_component = principal_component / np.linalg.norm(principal_component)
    
    # Define the target vector (x-axis)
    target_vector = np.array([1, 0, 0])
    
    # Check the dot product to ensure the principal component aligns with the target vector
    #print('Principal component:', principal_component)
    #print('Target vector:', target_vector)
    if np.dot(principal_component, target_vector) < 0:
        principal_component = -principal_component  # Flip to ensure positive alignment with x-axis
    
    # Calculate the rotation axis (cross product) and normalize it
    rotation_axis = np.cross(principal_component, target_vector)
    
    # Handle the case where the principal component is already aligned with the x-axis
    if np.allclose(rotation_axis, 0):
        return data, principal_component, np.eye(3), np.array(data.shape) / 2  # No rotation needed

    rotation_axis /= np.linalg.norm(rotation_axis)
    
    # Calculate the rotation angle
    angle = np.arccos(np.clip(np.dot(principal_component, target_vector), -1.0, 1.0))
    
    # Create the rotation matrix using the rotation axis and angle
    rotation = R.from_rotvec(angle * rotation_axis)
    rotation_matrix = rotation.as_matrix()

    if reverse:
        rotation_matrix = rotation_matrix.T
    
    # Step 2: Define the center of the data
    center = np.array(data.shape) / 2
    
    # Step 3: Apply the affine transformation
    # Since affine_transform applies the inverse of the provided matrix, use the inverse of the rotation matrix
    rotated_data = affine_transform(
        data,
        rotation_matrix.T,  # Transpose to invert for affine_transform
        offset=center - np.dot(rotation_matrix.T, center),  # Offset to keep the rotation around the center
        order=0  # Linear interpolation (you can also use `order=0` for nearest-neighbor if binary data)
    )
    
    return rotated_data, principal_component, rotation_matrix, center

def maximize_x_span(pancreas, sma, downsample=True, downsample_factor=100,limit_angle=45):
    original_shape=pancreas.shape
    
    original_center_of_mass,_=center_of_mass(pancreas,downsample,downsample_factor)
    #print('Original center of mass:', original_center_of_mass)
    #print('Original x span:',print_x_span(pancreas))
    # Shift pancreas and SMA data to center
    #translated_pancreas = binary_shift_pad(pancreas, shift_vector)
    #translated_sma = binary_shift_pad(sma, shift_vector)

    #bounding box around the center of mass
    # Calculate the span in each dimension
    non_zero_indices = np.nonzero(pancreas)
    x_span = non_zero_indices[0].max() - non_zero_indices[0].min() + 1
    y_span = non_zero_indices[1].max() - non_zero_indices[1].min() + 1
    z_span = non_zero_indices[2].max() - non_zero_indices[2].min() + 1
    # Find the maximum span and scale it by 1.5
    max_span = max(x_span, y_span, z_span)
    box_size = int(2 * max_span)
    translated_pancreas=crop_pancreas_with_bounding_box(pancreas, original_center_of_mass,box_size)
    if sma is not None:
        translated_sma=crop_pancreas_with_bounding_box(sma, original_center_of_mass,box_size)
    else:
        translated_sma=None

    
    # new center of mass and coordinates
    new_center_of_mass, non_zero_coords = center_of_mass(translated_pancreas,downsample,
                                                         downsample_factor)

    # Perform PCA to get the principal component
    pca = PCA(n_components=3)
    pca.fit(non_zero_coords)
    principal_axes = pca.components_
    #print('Principal axes:', principal_axes[0])




    # Target vector (x-axis)
    target_vector = np.array([1, 0, 0])
    if np.dot(principal_axes[0], target_vector) < 0:
        pa = -principal_axes[0]
    else:
        pa = principal_axes[0]
    angle = np.degrees(np.arccos(np.dot(pa, target_vector)))
    #print( 'Angle of rotation is:', angle)
    if angle > limit_angle:
        print('Angle is too large, skipping rotation:', angle)
        return translated_pancreas, translated_sma, original_center_of_mass, new_center_of_mass,  np.array([1,0,0]), original_shape



    
    overlay_sma_pancreas(translated_pancreas, 0*translated_pancreas,vector=principal_axes[0])
    overlay_sma_pancreas(translated_pancreas, 0*translated_pancreas,vector=principal_axes[0],axis=2)

    # Rotate both masks using the calculated rotation matrix
    rotated_pancreas, principal_component, rotation_matrix, center = rotate_3d_array(translated_pancreas, principal_axes[0])
    
    
    if sma is not None:
        rotated_sma = affine_transform( translated_sma,
                                        rotation_matrix.T,  # Transpose to invert for affine_transform
                                        offset=center - np.dot(rotation_matrix.T, center),  # Offset to keep the rotation around the center
                                        order=0  # Linear interpolation (you can also use `order=0` for nearest-neighbor if binary data)
                                    )
        #rotated_sma = rotate_3d_array(translated_sma, principal_axes[0])
    else:
        rotated_sma = None

    #print('x span after rotation:', print_x_span(rotated_pancreas))

    _, non_zero_coords = center_of_mass(rotated_pancreas,downsample,downsample_factor)
    pca_check = PCA(n_components=3)
    pca_check.fit(non_zero_coords)
    principal_axes_check = pca_check.components_[0]
    dot2=np.dot(principal_axes_check, target_vector)
    if dot2 < 0:
        principal_axes[0] = -principal_axes[0]
        dot2 = -dot2
    #print('Principal axes after rotation:', principal_axes_check)
    overlay_sma_pancreas(rotated_pancreas, 0*rotated_pancreas,vector=principal_axes_check)
    overlay_sma_pancreas(rotated_pancreas, 0*rotated_pancreas,vector=principal_axes_check,axis=2)
    #print('Angle after rotation: ', np.degrees(np.arccos(dot2))) # Should be close to 0

    return rotated_pancreas, rotated_sma, original_center_of_mass, new_center_of_mass, principal_component, original_shape




def revert_rotation(rotated_data, original_center_of_mass, new_center_of_mass,
                                    principal_axes, original_shape):
    x, _, _, _=rotate_3d_array(rotated_data, principal_axes,reverse=True)
    return x


def revert_translation(rotated_data, original_center_of_mass, new_center_of_mass, original_shape):
    # Shift back 
    reverted_data = np.zeros( original_shape, dtype=bool)# Calculate offset to place `rotated_back_data` at `original_center_of_mass` in `reverted_data`
    offset = np.round(original_center_of_mass - new_center_of_mass).astype(int)

    # Determine the bounds for insertion, handling edges if they go out of bounds
    x_min = max(0, offset[0])
    x_max = min(original_shape[0], offset[0] + rotated_data.shape[0])
    
    y_min = max(0, offset[1])
    y_max = min(original_shape[1], offset[1] + rotated_data.shape[1])
    
    z_min = max(0, offset[2])
    z_max = min(original_shape[2], offset[2] + rotated_data.shape[2])

    # Calculate corresponding indices for `rotated_back_data`
    x_rot_min = max(0, -offset[0])
    x_rot_max = x_rot_min + (x_max - x_min)

    y_rot_min = max(0, -offset[1])
    y_rot_max = y_rot_min + (y_max - y_min)

    z_rot_min = max(0, -offset[2])
    z_rot_max = z_rot_min + (z_max - z_min)

    # Place the `rotated_back_data` within `reverted_data` at the correct location
    reverted_data[x_min:x_max, y_min:y_max, z_min:z_max] = rotated_data[x_rot_min:x_rot_max, 
                                                                        y_rot_min:y_rot_max, 
                                                                        z_rot_min:z_rot_max]

    return reverted_data


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

def refine_body_mask(body_mask, head_mask):
    """
    Refine the body_mask by performing connected component analysis and morphological operations
    on each x-plane intersecting with the pancreas body. Reclassify parts of the body as head
    based on specific criteria.

    Parameters:
    - body_mask: 3D numpy array of the body mask.
    - head_mask: 3D numpy array of the head mask (to be updated).

    Returns:
    - refined_body_mask: Updated body mask after refinement.
    - updated_head_mask: Updated head mask after reclassification.
    """
    refined_body_mask = np.copy(body_mask)
    updated_head_mask = np.copy(head_mask)

    # Get the x-indices where the body_mask is present
    x_indices = np.where(np.any(body_mask, axis=(1, 2)))[0]

    # Iterate from tail to head
    x_indices = x_indices[::-1]

    previous_body_slice = None

    for x in x_indices:
        # Step 0: Copy the slice
        slice_2d = refined_body_mask[x, :, :].copy()

        if previous_body_slice is None:
            # First slice, assume all components are body
            eroded_slice = slice_2d.copy()
            body_components_mask = eroded_slice.copy()
            previous_body_slice = body_components_mask.copy()
        else:
            # Step 1: Erode the slice
            #eroded_slice = binary_erosion(slice_2d, iterations=2)

            # Step 2: Count connected components
            labeled_slice, num_features = label(eroded_slice)
            #print(f"Slice {x}: Found {num_features} components after erosion.")

            body_components_mask = np.zeros_like(eroded_slice, dtype=bool)

            # Dilate previous_body_slice to account for small gaps
            dilated_previous_body_slice = binary_dilation(previous_body_slice)

            # For each component, check if it touches the previous body slice
            for component in range(1, num_features + 1):
                component_mask = (labeled_slice == component)

                # Check if this component touches the previous body slice
                connected = np.any(component_mask & dilated_previous_body_slice)
                if connected:
                    body_components_mask |= component_mask

        # Step 4: Dilate the body components to revert erosion
        #dilated_slice = binary_dilation(body_components_mask,iterations=6)
        dilated_slice=slice_2d

        # Step 5: AND operation between this slice and the original slice
        refined_slice = np.logical_and(dilated_slice, slice_2d)

        # Step 6: Update the body and head masks
        # Update refined_body_mask with refined_slice
        refined_body_mask[x, :, :] = refined_slice

        # Reclassify pixels not in refined_slice as head
        non_body_components = slice_2d & (~refined_slice)
        updated_head_mask[x, :, :][non_body_components] = True

        # Update previous_body_slice for next iteration
        previous_body_slice = refined_slice.copy()

    return refined_body_mask, updated_head_mask

def print_axis_span(data, axis_name):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis_name not in axis_map:
        raise ValueError("Invalid axis name. Choose from 'x', 'y', or 'z'.")
    
    axis = axis_map[axis_name]
    # Sum along the other two axes to find non-zero slices along the specified axis
    non_zero_indices = np.where(data.sum(axis=tuple(i for i in range(3) if i != axis)) > 0)[0]

    if non_zero_indices.size > 0:
        min_index, max_index = non_zero_indices.min(), non_zero_indices.max()
        print(f"Pancreas span along {axis_name}-axis: {min_index} to {max_index} (span: {max_index - min_index + 1})")
    else:
        print(f"No pancreas data along {axis_name}-axis.")

def image_2_largest_cc(image):
    # Step 3: Find the largest connected component in overlap_xz
    labeled_array, num_features = label(image)  # Label connected components
    if num_features > 0:
        # Calculate the size of each component
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0  # Ignore background label

        # Find the label of the largest component
        largest_component_label = component_sizes.argmax()

        # Create a binary mask for the largest component only
        image = labeled_array == largest_component_label
    else:
        # No components found, use an empty array
        image = np.zeros_like(image, dtype=bool)
    return image

def sma_top(SMA_data,pancreas_data,downsample=3):
    #start_time = time.time()

    # Step 1: Get 2D projections of SMA and pancreas along the x-axis (by summing over the y-axis)
    SMA_projection_xz = SMA_data.sum(axis=1)  # Sum over y-axis to get an x-z projection
    pancreas_projection_xz = pancreas_data.sum(axis=1)  # Sum over y-axis for pancreas

    # Step 2: Calculate the overlap in this 2D projection
    overlap_xz = (SMA_projection_xz > 0) & (pancreas_projection_xz > 0)  # Binary overlap
    #take largest conncted component
    if overlap_xz.sum()==0:
        overlap_xz = SMA_projection_xz

    if overlap_xz.sum()==0:
        raise ValueError('No SMA')


    # Step 3: Handle cases where there is no overlap in the x-z projection
    #take the top 20 slices of the overlap
    highest_sma_z = np.max(np.where(overlap_xz.any(axis=(0)))[0])
    # Define the z range to keep (topmost centimeter)
    start_z = max(highest_sma_z - int(30/downsample), 0)  # Up to 10 slices below `highest_sma_z`, clamped to zero
    # Zero out all SMA data outside the topmost centimeter range
    overlap_xz[:, :start_z] = 0

    #take largest connected component of 2D overlap

    overlap_xz=image_2_largest_cc(overlap_xz)

    # Step 4: Expand overlap to match the 3D shape of SMA_data along the y-axis
    overlap_xz_expanded = overlap_xz[:, np.newaxis, :]  # Add new axis for y
    overlap_xz_expanded = np.repeat(overlap_xz_expanded, SMA_data.shape[1], axis=1)  # Repeat along y-axis

    # Step 5: Apply the expanded 2D overlap to the 3D SMA data
    SMA_data = SMA_data & overlap_xz_expanded  # Retain only SMA regions within the 2D overlap
    #zero sma everywhere where its projection is out of the overlap
    #take only the highest part of the sma overlap with the pancreas, so that you avoid any overlap between it and the head bottom
        
    #print('SMA after cut:')
    overlay_sma_pancreas(SMA_data, pancreas_data*0)    


    # Save rotated images as NIfTI for debugging
    #pancreas_nifti = nib.Nifti1Image(pancreas_data.astype(np.int8), pancreas_nii.affine, pancreas_nii.header)
    #sma_nifti = nib.Nifti1Image(SMA_data.astype(np.int8), pancreas_nii.affine, pancreas_nii.header)
    #nib.save(pancreas_nifti, f'rotated_pancreas.nii.gz')
    #nib.save(sma_nifti, f'rotated_sma.nii.gz')

    #print('Zeroed SMA outside of the region touching pancreas (ideally) or its top cm: ', time.time() - start_time)
    return SMA_data

def apply_inverse_orientation(data, original_ornt, original_affine, original_header, las_ornt=('L', 'P', 'S')):
    """
    Applies the inverse orientation to the data, based on the original and LAS orientations.
    Uses the original affine and header to create a new NIfTI image.
    
    Parameters:
    - data: 3D numpy array, data to reorient.
    - original_ornt: Original orientation code from the NIfTI file (e.g., RAS).
    - las_ornt: Target orientation code (e.g., LAS).
    - original_affine: Original affine matrix from the NIfTI file.
    - original_header: Original header from the NIfTI file.
    
    Returns:
    - transformed_img: NIfTI image object with the reoriented data and original affine.
    """
    # Calculate the transformation from LAS back to the original orientation
    las_to_original_transform = nib.orientations.ornt_transform(
        nib.orientations.axcodes2ornt(las_ornt),
        nib.orientations.axcodes2ornt(original_ornt)
    )

    # Apply the orientation transform
    reverted_data = nib.orientations.apply_orientation(data, las_to_original_transform)

    # Create a new NIfTI image with the original affine and header
    transformed_img = nib.Nifti1Image(reverted_data.astype(np.uint8), original_affine, original_header)
    transformed_img.set_data_dtype(np.uint8)
    
    return transformed_img

def upsample(head_data, body_data, tail_data, original_pancreas, downsample, original_size,
             original_center_of_mass,new_center_of_mass,spacing,debug):
    #start_time = time.time()
    #binary dilation and avoid intersections
    head_data = binary_dilation(head_data, iterations=2)
    #remove intersection with body or tail
    head_data[body_data] = False
    head_data[tail_data] = False

    tail_data = binary_dilation(tail_data, iterations=2)
    tail_data[body_data] = False
    tail_data[head_data] = False

    body_data = binary_dilation(body_data, iterations=2)
    body_data[head_data] = False
    body_data[tail_data] = False

    if debug:
            print('Resized size: ', original_size)
            print('Original spacing: ', spacing)
            print('Downsample: ', downsample)   

    #revert to original 3D shape
    head_data =  zoom(head_data, zoom=[downsample/spacing[0],
                                    downsample/spacing[1],
                                    downsample/spacing[2]], order=0)#[:original_size[0],:original_size[1],:original_size[2]].astype(np.bool_)
    body_data =  zoom(body_data, zoom=[downsample/spacing[0],
                                    downsample/spacing[1],
                                    downsample/spacing[2]], order=0)#[:original_size[0],:original_size[1],:original_size[2]].astype(np.bool_)
    tail_data =  zoom(tail_data, zoom=[downsample/spacing[0],
                                    downsample/spacing[1],
                                    downsample/spacing[2]], order=0)#[:original_size[0],:original_size[1],:original_size[2]].astype(np.bool_)
        
    if debug:
        #print new sizes
        print('New size: ', head_data.shape)

    head_data[body_data] = False
    head_data[tail_data] = False
    tail_data[body_data] = False
    tail_data[head_data] = False
    
    if debug:
        print('Upsampled and removed intersections: ')
        overlay_sma_pancreas(head_data, body_data, activated=True)
        overlay_sma_pancreas(head_data, tail_data, activated=True)
    

    #print('resized back: ', time.time() - start_time)

    # Revert translation
    #start_time = time.time()
    com=np.array([c*downsample/s for c,s in zip(original_center_of_mass,spacing)])
    n_com= np.array([c*downsample/s for c,s in zip(new_center_of_mass,spacing)])
    if debug:
        print('Translation parameters:')
        print('Original center of mass:', com)
        print('New center of mass:', n_com)
        print('Original size:', original_pancreas.shape)
        print('New size:', body_data.shape)
        print('Original spacing:', spacing)


    body_data=revert_translation(body_data, com, n_com, original_pancreas.shape)
    head_data=revert_translation(head_data, com, n_com, original_pancreas.shape)
    tail_data=revert_translation(tail_data, com, n_com, original_pancreas.shape)

    if debug:
        print('Translated back: ')
        overlay_sma_pancreas(head_data, body_data, activated=True)
        overlay_sma_pancreas(head_data, tail_data, activated=True)

    #crop to original size
    head_data = head_data[:original_size[0],:original_size[1],:original_size[2]].astype(np.bool_)
    body_data = body_data[:original_size[0],:original_size[1],:original_size[2]].astype(np.bool_)
    tail_data = tail_data[:original_size[0],:original_size[1],:original_size[2]].astype(np.bool_)

    if debug:
        print('Cropped to original size: ')
        overlay_sma_pancreas(head_data, original_pancreas, activated=True)
        overlay_sma_pancreas(body_data, original_pancreas, activated=True)
        overlay_sma_pancreas(tail_data, original_pancreas, activated=True)

    #use and between the dilated sub-segments and the original pancreas mask
    head_data[~original_pancreas]=False
    body_data[~original_pancreas]=False
    tail_data[~original_pancreas]=False
    #print('resized back: ', time.time() - start_time)
    return head_data, body_data, tail_data

def process_pancreas_SMA(pancreas_path, SMA_path, output_dir='.',downsample=3,two_rotations=True,
                         debug=False):
    """
    Processes the pancreas and SMA images by reorienting them to LAS orientation and zeroing
    the SMA volume for all z-slices where the pancreas is absent.

    Additionally, splits the pancreas into head, body, and tail based on the SMA data span
    over the left-right (x) axis, and saves the resulting NIfTI images. It also refines the
    body mask to reclassify certain regions as head.

    Parameters:
    - pancreas_path: File path to the pancreas NIfTI image.
    - SMA_path: File path to the SMA NIfTI image.
    - output_dir: Directory to save the output NIfTI images (default is current directory).

    Returns:
    - modified_SMA_img: NIfTI image with the modified SMA data.
    """

    
    # Load both images using nibabel and cast to uint8
    SMA_nii = nib.load(SMA_path)
    SMA_data = np.asanyarray(SMA_nii.dataobj).astype(np.bool_)
    if SMA_data.sum() == 0:
        return
    start_time = time.time()
    print('Started processing: ', pancreas_path)
    pancreas_nii = nib.load(pancreas_path)
    pancreas_data = np.asanyarray(pancreas_nii.dataobj).astype(np.bool_)
    #print('Loaded pancreas and converted to bool: ', time.time() - start_time)
    #start_time = time.time()
    #print('Loaded SMA and converted to bool: ', time.time() - start_time)
    #start_time = time.time()

    # Calculate the orientation transformation based on the pancreas image
    original_ornt = nib.orientations.aff2axcodes(pancreas_nii.affine)
    #print('Original orientation: ', original_ornt)


    # Get the original spacing and orientation
    spacing = pancreas_nii.header.get_zooms()[:3]
    #print("Original spacing:", spacing)

    if original_ornt != ('L', 'P', 'S'):

        transform_pancreas = get_orientation_transform(pancreas_nii, orientation=('L', 'P', 'S'))
        transform_SMA = get_orientation_transform(SMA_nii, orientation=('L', 'P', 'S'))

        #print('Got orientation transforms: ', time.time() - start_time)
        #start_time = time.time()
        
        # Apply the transformation to the data
        pancreas_data = apply_transform(pancreas_data, transform_pancreas)
        SMA_data = apply_transform(SMA_data, transform_SMA)
        #print('Re-oriented: ', time.time() - start_time)

        spacing = np.array(spacing)
        #make spacing 1x1x1
        # Reorder the spacing according to the reorientation transform
        try:
            spacing = spacing[np.abs(transform_pancreas[:, 0]).astype(int)]
        except:
            print('spacing:', spacing)
            print('Transform pancreas:', transform_pancreas[:, 0])
            print('Pancreas:', pancreas_path)
            #print('Pancreas nii:', pancreas_nii)
            spacing = spacing[np.abs(transform_pancreas[:, 0]).astype(int)]
        #print("Reoriented spacing:", spacing)


    #start_time = time.time()

    original_size = pancreas_data.shape
    original_pancreas=pancreas_data.copy()

    if debug:
        print('Original size: ', original_size)
        print('Original spacing: ', spacing)
        print('Downsample: ', downsample)

    #if downsample>1:
    SMA_data = zoom(SMA_data, zoom=[spacing[0]/(downsample),
                                    spacing[1]/(downsample),
                                    spacing[2]/(downsample)], order=0)
    pancreas_data = zoom(pancreas_data, zoom=[spacing[0]/(downsample),
                                                spacing[1]/(downsample),
                                                spacing[2]/(downsample)], order=0)
    
    if debug:
        print('Downsampled: ', pancreas_data.shape)


    
    overlay_sma_pancreas(SMA_data, pancreas_data,activated=debug)
    #print('downsampled: ', time.time() - start_time)
    #start_time = time.time()
    
    SMA_data=sma_top(SMA_data,pancreas_data,downsample)

    #print('SMA cut: ', time.time() - start_time)
    
    #start_time = time.time()


    #rotate pancreas and sma using PCA to allign the pancreas along the x-axis
    #print('Pancreas x span: ', pancreas_data.sum(axis=(1, 2)))
    pancreas_data, SMA_data, original_center_of_mass, new_center_of_mass, principal_axes, \
        original_shape = maximize_x_span(pancreas_data, SMA_data)
    #print('Maximized x span: ', time.time() - start_time)

    overlay_sma_pancreas(SMA_data, pancreas_data,activated=debug)


    #start_time = time.time()


    # Get the span of the SMA_data over the left-right axis
    SMA_projection_x = SMA_data.sum(axis=(1, 2))  # Sum over y and z axes
    non_zero_x_indices = np.where(SMA_projection_x > 0)[0]
    if non_zero_x_indices.size > 0:
        leftmost_x = non_zero_x_indices.min()
        rightmost_x = non_zero_x_indices.max()
        sep = (rightmost_x + leftmost_x) // 2  # Separation point
        #print(f"SMA data spans from x = {leftmost_x} to x = {rightmost_x}")
        #print(f"Separating at x = {sep}")
    else:
        raise ValueError("SMA data is zero everywhere along the x-axis.")

    #print('Computed SMA span over left-right axis: ', time.time() - start_time)
    #start_time = time.time()


    overlay_sma_pancreas(SMA_data, pancreas_data, x_line=sep,activated=debug)


    # Head: x >= sep
    left_mask = pancreas_data.copy()
    left_mask[:sep] = False
    head_mask = pancreas_data.copy()
    head_mask[sep:] = False


    # Left part (x < sep)
    #left_mask = pancreas_nonzero & (x_array < sep)

    overlay_sma_pancreas(head_mask, pancreas_data,activated=debug)
    overlay_sma_pancreas(left_mask, pancreas_data,activated=debug)

    
    redo=False
    if two_rotations:
        backup_left_mask = left_mask.copy()
        # rotate body and tail
        start_time = time.time()
        #rotate pancreas and sma using PCA to allign the pancreas along the x-axis
        #print('Pancreas x span: ', pancreas_data.sum(axis=(1, 2)))
        left_mask, _, left_original_center_of_mass, left_new_center_of_mass, left_principal_axes, \
            left_original_shape = maximize_x_span(left_mask, None,limit_angle=45)
        #print('Maximized x span of body and tail: ', time.time() - start_time)
        #cut in half point:
        # Body: middle_point_left <= x < sep
        sep=left_mask.shape[0]//2
        tail_mask = left_mask.copy()
        tail_mask[:sep] = False
        body_mask = left_mask.copy()
        body_mask[sep:] = False

        #revert rotation
        tail_mask= revert_rotation(tail_mask,left_original_center_of_mass,left_new_center_of_mass,left_principal_axes, left_original_shape)
        body_mask= revert_rotation(body_mask,left_original_center_of_mass,left_new_center_of_mass,left_principal_axes, left_original_shape)
        #revert translation
        tail_mask=revert_translation(tail_mask, left_original_center_of_mass, left_new_center_of_mass, head_mask.shape)
        body_mask=revert_translation(body_mask, left_original_center_of_mass, left_new_center_of_mass, head_mask.shape)

        overlay_sma_pancreas(tail_mask, body_mask,activated=debug)#, x_line=sep)
        #check if head touches tail
        #dilation
        dilated_tail = binary_dilation(tail_mask, iterations=1, structure=np.ones((6,6,6)))
        dilated_head = binary_dilation(head_mask, iterations=1, structure=np.ones((6,6,6)))
        if np.any(dilated_tail[dilated_head]):
            redo=True
            left_mask=backup_left_mask

    if redo or not two_rotations:
        if redo:
            print('Redoing without two rotations')

        left_mask_x = left_mask.sum(axis=(1, 2))  # Sum over y and z axes
        non_zero_x_indices = np.where(left_mask_x > 0)[0]
        leftmost_x = non_zero_x_indices.min()
        rightmost_x = non_zero_x_indices.max()
        sep = (rightmost_x + leftmost_x) // 2  # Separation point
        #print(f"Left pancreas spans from x = {leftmost_x} to x = {rightmost_x}")
        #print(f"Separating at x = {sep}")
    

        overlay_sma_pancreas(SMA_data, pancreas_data, x_line=sep,activated=debug)

        # Body: middle_point_left <= x < sep
        tail_mask = left_mask.copy()
        tail_mask[:sep] = False
        body_mask = left_mask.copy()
        body_mask[sep:] = False
    if debug:
        print('Head, body and tail before refinement:')
    #print('Head, body and tail:')
    overlay_sma_pancreas(head_mask, pancreas_data,activated=debug)
    overlay_sma_pancreas(body_mask, pancreas_data,activated=debug)
    overlay_sma_pancreas(tail_mask, pancreas_data,activated=debug)

    # Refine the body_mask
    body_mask, head_mask = refine_body_mask(body_mask, head_mask)

    if debug:
        print('Head, body and tail after refinement:')

    overlay_sma_pancreas(head_mask, pancreas_data,activated=debug)
    overlay_sma_pancreas(body_mask, pancreas_data,activated=debug)
    overlay_sma_pancreas(tail_mask, pancreas_data,activated=debug)

    #print('Refined the body mask: ', time.time() - start_time)
    #start_time = time.time()

    head_data = np.zeros_like(pancreas_data, dtype=np.bool_)
    body_data = np.zeros_like(pancreas_data, dtype=np.bool_)
    tail_data = np.zeros_like(pancreas_data, dtype=np.bool_)
    head_data[head_mask] = pancreas_data[head_mask]
    body_data[body_mask] = pancreas_data[body_mask]
    tail_data[tail_mask] = pancreas_data[tail_mask]

    #print('Split pancreas into head, body, and tail: ', time.time() - start_time)

    #start_time = time.time()

    #revert rotation
    head_data = revert_rotation(head_data,original_center_of_mass,new_center_of_mass,principal_axes, original_shape)
    body_data = revert_rotation(body_data,original_center_of_mass,new_center_of_mass,principal_axes, original_shape)
    tail_data = revert_rotation(tail_data,original_center_of_mass,new_center_of_mass,principal_axes, original_shape)
    #split the pancreas data into head, body, and tail
    # Create the head, body, and tail data
    #head_data = np.where(pancreas_data == 1, 1, 0)
    #body_data = np.where(pancreas_data ==2, 1, 0)
    #tail_data = np.where(pancreas_data ==3, 1, 0)
    #print( 'Reverted rotation: ', time.time() - start_time)
    if debug:
        print('Head, body and tail after rotating back:')
        overlay_sma_pancreas(head_data, 0*head_data,activated=debug)
        overlay_sma_pancreas(body_data, 0*body_data,activated=debug)
        overlay_sma_pancreas(tail_data, 0*tail_data,activated=debug)
        
    
    head_data, body_data, tail_data=upsample(head_data, body_data, tail_data, 
                                                 original_pancreas, downsample, original_size,
                                                 original_center_of_mass,
                                                 new_center_of_mass,spacing,
                                                 debug=debug)

    #if downsample>1:
    #    #revert translation and resize
    #    head_data, body_data, tail_data=upsample(head_data, body_data, tail_data, 
    #                                             original_pancreas, downsample, original_size,
    #                                             original_center_of_mass,
    #                                             new_center_of_mass,spacing)
    #else:
    #    # Revert translation
    #    com=np.array([c*downsample*s for c,s in zip(original_center_of_mass,spacing)])
    #    n_com= np.array([c*downsample*s for c,s in zip(new_center_of_mass,spacing)])
    #    body_data=revert_translation(body_data, com,n_com, original_pancreas.shape)
    #    head_data=revert_translation(head_data, com,n_com, original_pancreas.shape)
    #    tail_data=revert_translation(tail_data, com,n_com, original_pancreas.shape)

    if debug:
        print('Final head, body, and tail after resize: ')
    overlay_sma_pancreas(head_data, original_pancreas, axis=2,activated=debug)

    #print('Final head, body, and tail: ')
    overlay_sma_pancreas(head_data, original_pancreas,activated=debug)
    overlay_sma_pancreas(body_data, original_pancreas,activated=debug)
    overlay_sma_pancreas(tail_data, original_pancreas,activated=debug)

    # Adjust the affine for the reoriented pancreas image
    #affine_transform_pancreas = nib.orientations.inv_ornt_aff(transform_pancreas, pancreas_data.shape)
    #new_affine_pancreas = pancreas_nii.affine @ affine_transform_pancreas

    if original_ornt != ('L', 'P', 'S'):
        #start_time = time.time()
        # Revert data to original orientation and save
        head_img = apply_inverse_orientation(head_data, original_ornt, pancreas_nii.affine, pancreas_nii.header)
        body_img = apply_inverse_orientation(body_data, original_ornt, pancreas_nii.affine, pancreas_nii.header)
        tail_img = apply_inverse_orientation(tail_data, original_ornt, pancreas_nii.affine, pancreas_nii.header)
        #print( 'Reverted orientation: ', time.time() - start_time)
    else:
        head_img = nib.Nifti1Image(head_data.astype(np.uint8), pancreas_nii.affine, pancreas_nii.header)
        body_img = nib.Nifti1Image(body_data.astype(np.uint8), pancreas_nii.affine, pancreas_nii.header)
        tail_img = nib.Nifti1Image(tail_data.astype(np.uint8), pancreas_nii.affine, pancreas_nii.header)
    
    #start_time = time.time()
    # Save the images
    nib.save(head_img, f'{output_dir}/pancreas_head.nii.gz')
    nib.save(body_img, f'{output_dir}/pancreas_body.nii.gz')
    nib.save(tail_img, f'{output_dir}/pancreas_tail.nii.gz')

    # Save original pancreas
    #original_pancreas_img = nib.Nifti1Image(original_pancreas, pancreas_nii.affine, pancreas_nii.header)
    #nib.save(original_pancreas_img, f'{output_dir}/original_pancreas.nii.gz')

    print(output_dir,' -- saved head, body, and tail images in ', np.round(time.time() - start_time,2), ' s')
    #start_time = time.time()

def outputs_exist(output_dir):
    """Check if all required output files exist in the output directory."""
    required_files = ["pancreas_head.nii.gz", "pancreas_body.nii.gz", "pancreas_tail.nii.gz"]
    return all((output_dir / filename).is_file() for filename in required_files)

def process_case(case, destination_dir, restart,pancreas_dir,sma_dir,debug=False,downsample=2):
    """Helper function to process a single case."""

    pancreas_path = Path(os.path.join(pancreas_dir,case,'segmentations',"pancreas.nii.gz"))
    sma_path = Path(os.path.join(sma_dir,case.replace('BDMAP_','BDMAP'),'segmentations',"superior_mesenteric_artery.nii.gz"))
    output_subdir = Path(os.path.join(destination_dir,case, "segmentations"))

    # Skip if already processed and restart is not set
    if not restart and outputs_exist(output_subdir) and os.path.isfile(os.path.join(output_subdir,'pancreas_head.nii.gz')):
        print(f"Skipping {output_subdir} -- already exists.")
        return

    # Ensure the output directory exists
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Run the processing function
    #process_pancreas_SMA(pancreas_path, sma_path, output_dir=output_subdir, downsample=2)
    if debug:
        process_pancreas_SMA(pancreas_path, sma_path, output_dir=output_subdir, downsample=downsample,debug=debug)
    else:
        try:
            process_pancreas_SMA(pancreas_path, sma_path, output_dir=output_subdir, downsample=downsample,debug=debug)
        except Exception as e:
            print(f"Error processing {case}: {e}")
            with open(os.path.join(destination_dir, 'error_log_pancreas_segments.txt'), 'a') as f:
                f.write(f"Error processing {case}: {str(e)}\n")
                f.write(traceback.format_exc() + '\n')

def main(source_dir, destination_dir, restart, parts, current_part, num_processes,sma_dir):
    if 'Atlas' in source_dir:
        cases_folder='/ccvl/net/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.1/AbdomenAtlas1.1/'
    else:
        cases_folder=source_dir

    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    sma_dir = Path(sma_dir)

    

    # Find cases
    cases=[]
    for x in os.listdir(cases_folder):
        pth_pancreas=Path(os.path.join(source_dir,x,"segmentations","pancreas.nii.gz"))
        pth_sma=Path(os.path.join(sma_dir,x,"segmentations","superior_mesenteric_artery.nii.gz").replace('BDMAP_','BDMAP'))
        #print(pth_pancreas, pth_sma)
        if pth_pancreas.is_file() and pth_sma.is_file():
            #print('dentro')
            cases.append(x)

    # Split cases into parts
    part_size = len(cases) // parts
    start = current_part * part_size
    end = (current_part + 1) * part_size if current_part < parts - 1 else len(cases)
    cases = cases[start:end]

    print('Total cases to process:', len(cases))

    # Filter out already processed cases if not restarting
    if not restart:
        cases = [case for case in cases if not outputs_exist(destination_dir / case / "segmentations")]
    print('Cases to process after filtering:', len(cases))

    # Process cases with multiprocessing if num_processes > 1
    if num_processes > 1:
        with Pool(processes=num_processes) as pool:
            pool.starmap(process_case, [(case, destination_dir, restart,source_dir,sma_dir) for case in cases])
    else:
        # Process cases serially if num_processes == 1
        for case in cases:
            process_case(case, destination_dir, restart,source_dir,sma_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pancreas and SMA segmentations.")
    parser.add_argument("--source_dir", type=str, help="Path to the source directory.")
    parser.add_argument("--sma_dir", type=str, help="Path to the source directory.")
    parser.add_argument("--destination_dir", type=str, help="Path to the destination directory.")
    parser.add_argument("--restart", action="store_true", help="Reprocess existing outputs.")
    parser.add_argument("--parts", type=int, help="Option to divide inference in multiple processes.", default=1)
    parser.add_argument("--current_part", type=int, help="Current part to process, from 0 to parts-1.", default=0)
    parser.add_argument("--num_processes", type=int, help="Number of parallel processes to run.", default=10)
    
    args = parser.parse_args()
    main(args.source_dir, args.destination_dir, args.restart, args.parts, args.current_part, args.num_processes,args.sma_dir)