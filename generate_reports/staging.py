import nibabel as nib
import os
from nibabel.processing import resample_from_to
import numpy as np
import copy
from scipy.ndimage import label, binary_dilation, binary_erosion
from skimage.morphology import skeletonize_3d
import torch
import torch.nn.functional as F

import numpy as np
#from skimage.morphology import skeletonize_3d, binary_dilation
from scipy.ndimage import label
from sklearn.decomposition import PCA
from scipy.ndimage import affine_transform
from scipy.linalg import svd
from scipy.spatial.transform import Rotation as R

from skimage.morphology import skeletonize_3d, dilation, ball
from scipy.ndimage import label, generate_binary_structure, convolve
from nibabel.orientations import aff2axcodes



import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label, generate_binary_structure
from skimage.morphology import ball
from nibabel.orientations import aff2axcodes

import CreateAAReports as cr




def divide_CA(aorta_data, celiac_data, affine):
    original_volume = celiac_data.copy()

    # Extract the proper celiac axis (CA) using existing function
    proper_ca, remaining_celiac, cut_celiac = extract_proper_celiac_axis(aorta_data, celiac_data, affine)

    # Convert to boolean
    proper_ca = proper_ca.astype(bool)
    cut_celiac = cut_celiac.astype(bool)

    # Print for debugging
    print('Shape of proper ca:', proper_ca.shape)
    nib.save(nib.Nifti1Image(proper_ca.astype(np.uint8), affine), 'debug_proper_ca.nii.gz')

    # Label connected components in the remaining celiac data
    labeled_celiac, num_features = label(cut_celiac)

    # Determine the left-right axis
    axcodes = aff2axcodes(affine)
    left_right_axis = next(idx for idx, code in enumerate(axcodes) if code in ('L', 'R'))

    # Initialize hepatic and splenic artery arrays
    hepatic_artery = np.zeros_like(celiac_data, dtype=bool)
    splenic_artery = np.zeros_like(celiac_data, dtype=bool)

    # Calculate mean position of the CA along the left-right axis
    ca_coords = np.argwhere(proper_ca)
    ca_mean_position = np.mean(ca_coords[:, left_right_axis])

    # Classify each component based on position
    for component_label in range(1, num_features + 1):
        component = (labeled_celiac == component_label)
        component_coords = np.argwhere(component)
        mean_position = np.mean(component_coords[:, left_right_axis])

        if mean_position > ca_mean_position:
            splenic_artery |= component
        else:
            hepatic_artery |= component

    return proper_ca, hepatic_artery, splenic_artery


def clean_CA_end(proper_ca, all_ca, affine):
    all_ca_copy = all_ca.copy()

    # Determine left-right axis from affine
    axcodes = aff2axcodes(affine)
    left_right_axis = next(idx for idx, code in enumerate(axcodes) if code in ('L', 'R'))

    # Initialize output volume
    out_volume = np.zeros_like(proper_ca, dtype=bool)

    # Process each slice along the left-right axis
    for idx in range(proper_ca.shape[left_right_axis]):
        proper_ca_slice = get_slice(proper_ca, left_right_axis, idx)
        all_ca_slice = get_slice(all_ca, left_right_axis, idx)

        if np.any(proper_ca_slice & all_ca_slice):
            labeled_components, num_features = label(all_ca_slice)
            for component_label in range(1, num_features + 1):
                component_slice = (labeled_components == component_label)
                if np.any(component_slice & proper_ca_slice):
                    proper_ca_slice |= component_slice

        set_slice(out_volume, left_right_axis, idx, proper_ca_slice)

    cut_ca = np.clip(all_ca_copy.astype(np.uint8) - out_volume.astype(np.uint8), 0, 1).astype(bool)
    out_volume = out_volume.astype(bool)

    return out_volume, cut_ca


def clean_CA_end_simple(volume, slice_indices, ap_axis):
    slice_areas = []
    occupied_slices = []
    for idx in slice_indices:
        area = get_slice(volume, ap_axis, idx).astype(np.uint8).sum()
        if area > 0:
            slice_areas.append(area)
            occupied_slices.append(idx)
    # Area statistics
    area_mean = np.mean(slice_areas)
    out = np.zeros_like(volume)
    num_slices = len(occupied_slices)
    cutoff = int(0.9 * num_slices)
    for i, idx in enumerate(occupied_slices):
        area = slice_areas[i]
        if i < cutoff or (area < 2 * area_mean):
            set_slice(out, ap_axis, idx, get_slice(volume, ap_axis, idx))
    return out


def extract_proper_celiac_axis(aorta_data, celiac_data, affine):
    """
    Extracts the proper celiac axis by removing the splenic and hepatic arteries.
    
    Parameters:
    - aorta_data: Boolean NumPy array of the aorta image data.
    - celiac_data: Boolean NumPy array of the celiac artery annotation.
    - affine: Affine matrix of the images (same for both aorta and celiac).
    
    Returns:
    - output_volume: Boolean NumPy array of the extracted proper celiac axis.
    """
    celica_data = celiac_data.copy()
    aorta_data = binary_dilation(aorta_data, structure=ball(7))
    # Step 1: Clean small branches
    combined = celiac_data | aorta_data
    eroded = binary_erosion(combined, structure=ball(3))
    dilated = binary_dilation(eroded, structure=ball(4))
    #cleaned_ca = celiac_data
    cleaned_ca = dilated & celiac_data
    #cleaned_ca = eroded & celiac_data


    #save ca and aorta for debugging
    nib.save(nib.Nifti1Image(cleaned_ca.astype(np.uint8), affine), 'debug_ca.nii.gz')
    nib.save(nib.Nifti1Image(aorta_data.astype(np.uint8), affine), 'debug_aorta.nii.gz')

    # Determine the anterior-posterior axis and direction
    axcodes = aff2axcodes(affine)
    ap_axis = None
    for idx, code in enumerate(axcodes):
        if code in ('A', 'P'):
            ap_axis = idx
            break
    if ap_axis is None:
        raise ValueError("Could not determine anterior-posterior axis from affine.")
    if axcodes[ap_axis] == 'A':
        slice_indices = range(cleaned_ca.shape[ap_axis])
    elif axcodes[ap_axis] == 'P':
        slice_indices = range(cleaned_ca.shape[ap_axis] - 1, -1, -1)
    else:
        raise ValueError("Unexpected axis code for anterior-posterior direction.")

    # Initialize empty output volume
    output_volume = np.zeros_like(cleaned_ca, dtype=bool)

    # Initialize variables
    last_aorta_slice = None
    last_output_slice = None
    intersection_found = False
    max_gap_size=0
    gap_size=0

    for idx in slice_indices:
        aorta_slice = get_slice(aorta_data, ap_axis, idx)
        ca_slice = get_slice(cleaned_ca, ap_axis, idx)
        output_slice = get_slice(output_volume, ap_axis, idx)

        if last_aorta_slice is None:
            last_aorta_slice=aorta_slice
        if last_output_slice is None:
            last_output_slice=output_slice

        if not intersection_found:
            # Check if the aorta and celiac artery intersect
            intersection = np.logical_and(binary_dilation(last_aorta_slice), ca_slice)
            if np.any(intersection):
                intersection_found = True
                print('Intersection found at slice', idx)
                # Connected Component Analysis on ca_slice
                labeled_ca_slice, num_features = label(ca_slice)
                touching_labels = np.unique(labeled_ca_slice[intersection])
                touching_labels = touching_labels[touching_labels > 0]
                if touching_labels.size == 0:
                    continue
                # Add the largest component that touches the aorta
                max_label = max(
                    touching_labels,
                    key=lambda l: np.sum(labeled_ca_slice == l)
                )
                output_slice = (labeled_ca_slice == max_label)
                set_slice(output_volume, ap_axis, idx, output_slice)
        else:
            #already after intersection
            # Dilate the previous main component once to allow for connections
            dilated_previous_component = binary_dilation(last_output_slice)

            # Label components in the current slice
            labeled_slice, num_features = label(ca_slice)
            
            # Check if there are any components in the current slice
            if num_features == 0:
                continue  # No components to process, skip this slice

            largest_touching_component = np.zeros_like(ca_slice, dtype=bool)
            found_connection = False

            # Iterate over all components sorted by size (largest first)
            component_sizes = np.bincount(labeled_slice.ravel())[1:]  # Exclude the background
            sorted_components = np.argsort(component_sizes)[::-1] + 1  # Largest to vesselllest component labels
            connections=[]
            #get the largest component that touches the last one
            for component_label in sorted_components:
                component_mask = (labeled_slice == component_label)
                
                # Check if this component touches the dilated previous component
                #if np.any(component_mask & dilated_previous_component):
                if np.any(component_mask & dilated_previous_component):
                    #argest_touching_component = component_mask
                    connections.append(component_mask)
                    found_connection = True
                    
            # If a connection was found, update previous_main_component
            if found_connection:
                if len(connections)==1:
                    output_slice=connections[0]
                    set_slice(output_volume, ap_axis, idx, output_slice)
                elif len(connections)>1:
                    if connections[0].astype(np.uint8).sum()>connections[1].astype(np.uint8).sum()*10:
                        output_slice=connections[0]
                        set_slice(output_volume, ap_axis, idx, output_slice)
                    else:
                        print('Stopping at slice', idx)
                        break
                in_gap=False
                gap_size=0
            else:
                # If no connection, add the largest component without updating previous_main_component
                #main_component = (labeled_slice == sorted_components[0])
                gap_size+=1
                if gap_size>max_gap_size:
                    max_gap_size=gap_size


        last_aorta_slice=aorta_slice.copy()
        last_output_slice=output_slice.copy()
                
    
    ca_not_clean=output_volume.copy()
    output_volume=clean_CA_end_simple(output_volume,slice_indices, ap_axis)
    output_volume, cut_ca=clean_CA_end(output_volume,celica_data.copy(), affine)

    #binary erosion and dilation
    # Apply binary erosion and dilation with the structuring element
    eroded_volume = binary_erosion(output_volume, structure=ball(4))
    dilated_volume = binary_dilation(eroded_volume, structure=ball(5))
    output_volume = np.logical_and(dilated_volume, celica_data)

    #return intersected_volume.astype(bool)
    # Keep only the largest 3D connected component
    labeled_intersection, num_features = label(output_volume)
    if num_features > 0:
        # Identify the largest connected component
        component_sizes = np.bincount(labeled_intersection.ravel())[1:]  # Exclude background
        largest_component_label = np.argmax(component_sizes) + 1
        final_output = (labeled_intersection == largest_component_label)
    else:
        # No components were found, return an empty volume
        raise ValueError('No components were found')


    return final_output, ca_not_clean, cut_ca

def get_slice(volume, axis, index):
    """
    Extracts a 2D slice from a 3D volume along a specified axis at a given index.
    """
    slicer = [slice(None)] * 3
    slicer[axis] = index
    return volume[tuple(slicer)]

def set_slice(volume, axis, index, slice_data):
    """
    Sets a 2D slice in a 3D volume along a specified axis at a given index.
    """
    slicer = [slice(None)] * 3
    slicer[axis] = index
    volume[tuple(slicer)] = slice_data

def stack_slices(slice1, slice2):
    """
    Stacks two 2D slices to create a small 3D volume for connected component analysis.
    """
    return np.stack([slice1, slice2], axis=-1)







def remove_splenic_artery(
    aorta_data,
    celiac_data,
    affine,
    delta_x=10.0,
    struct_elem_radius=10
):
    """
    Separates the splenic artery from the celiac artery annotation.

    Parameters:
    - aorta_data: Boolean NumPy array of the aorta image data.
    - celiac_data: Boolean NumPy array of the celiac artery annotation.
    - affine: Affine matrix of the images (same for both aorta and celiac).
    - header: Header of the images (same for both aorta and celiac).
    - delta_x: Threshold for 'significantly left' in physical units (e.g., millimeters).
    - struct_elem_radius: Radius for the dilation structuring element (in voxels).

    Returns:
    - intersection: Boolean NumPy array of the result.
    - affine: Affine matrix to be used for saving or further processing.
    - header: Header to be used for saving or further processing.
    """
    # Ensure that the data is boolean
    aorta_data = aorta_data.astype(bool)
    celiac_data = celiac_data.astype(bool)
    
    # Step 1: Skeletonize the celiac_aa annotation
    skeleton = skeletonize_3d(celiac_data)
    
    # Remove branching points to separate branches using 2D projection
    skeleton_no_branches = remove_branch_points_projection(skeleton)
    
    # Perform connected component analysis to separate branches
    struct = generate_binary_structure(3, 1)  # 3D connectivity
    labeled_skeleton, num_features = label(skeleton_no_branches, structure=struct)
    
    # Step 2: Delete any branch that reaches significantly left of the aorta
    # Compute the center of mass of the aorta in voxel indices
    aorta_center_voxel = np.array(np.nonzero(aorta_data)).mean(axis=1)
    
    # Transform the center of mass to world coordinates using the affine matrix
    aorta_center_world = nib.affines.apply_affine(affine, aorta_center_voxel)
    
    # Determine which axis corresponds to the left-right direction using the image orientation
    axcodes = aff2axcodes(affine)
    # Map the axis codes to indices
    left_right_axis = None
    for idx, code in enumerate(axcodes):
        if code in ['L', 'R']:
            left_right_axis = idx
            break
    
    if left_right_axis is None:
        raise ValueError("Could not determine left-right axis from image orientation.")
    
    # Prepare a list of labels to keep
    labels_to_keep = identify_branches_to_keep(
        labeled_skeleton, num_features, affine, axcodes, left_right_axis,
        aorta_center_world, delta_x
    )
    
    # Create the mask of branches to keep
    keep_mask = np.isin(labeled_skeleton, labels_to_keep)
    
    # Step 3: Perform binary dilation over the remaining skeleton branches
    struct_elem = ball(struct_elem_radius)
    dilated_skeleton = dilation(keep_mask, struct_elem)
    
    # Calculate the intersection between the dilated skeleton and the original celiac_aa
    intersection = np.logical_and(dilated_skeleton, celiac_data)
    
    # Return the result along with the affine and header
    return intersection

def remove_branch_points_projection(skeleton):
    """
    Removes branching points from a skeleton using a 2D projection to identify possible branch points,
    then confirms them in 3D, and removes them from the skeleton.

    Parameters:
    - skeleton: A binary 3D NumPy array representing the skeletonized image.

    Returns:
    - A binary 3D NumPy array with branching points removed.
    """
    # Sum the skeleton along the z-axis to get a 2D projection (assuming z is axis 2)
    projection_2d = skeleton.sum(axis=2)
    # Binarize the projection
    projection_2d = (projection_2d > 0).astype(np.uint8)
    
    # Find branch points in the 2D projection
    # Define the 2D connectivity for neighbors (8-connectivity)
    struct_2d = generate_binary_structure(2, 2)
    neighbor_count_2d = convolve(projection_2d, struct_2d, mode='constant', cval=0)
    # Identify branch points in 2D (pixels with more than two neighbors)
    branch_points_2d = ((neighbor_count_2d > 3) & (projection_2d > 0))
    
    # Get the coordinates of the branch points in 2D
    branch_coords_2d = np.array(np.nonzero(branch_points_2d)).T  # Shape (N, 2)
    
    # Initialize the skeleton without branch points
    skeleton_no_branches = skeleton.copy()
    
    # Map back to 3D: for each (x, y), find all z where skeleton[x, y, z] == 1
    for x, y in branch_coords_2d:
        z_indices = np.nonzero(skeleton[x, y, :])[0]
        for z in z_indices:
            # Confirm if this point is a branch point in 3D
            neighbor_count_3d = count_neighbors_3d(skeleton, x, y, z)
            if neighbor_count_3d > 3:
                # Remove this branch point
                skeleton_no_branches[x, y, z] = 0
    return skeleton_no_branches

def count_neighbors_3d(skeleton, x, y, z):
    """
    Counts the number of neighboring voxels for a given voxel in 3D.

    Parameters:
    - skeleton: The binary 3D skeleton array.
    - x, y, z: The voxel indices.

    Returns:
    - neighbor_count: The number of neighboring voxels (excluding the center voxel).
    """
    x_min = max(x - 1, 0)
    x_max = min(x + 2, skeleton.shape[0])
    y_min = max(y - 1, 0)
    y_max = min(y + 2, skeleton.shape[1])
    z_min = max(z - 1, 0)
    z_max = min(z + 2, skeleton.shape[2])
    
    # Extract the neighborhood
    neighborhood = skeleton[x_min:x_max, y_min:y_max, z_min:z_max]
    # Count the number of neighbors
    neighbor_count = np.sum(neighborhood) - 1  # Exclude the center voxel
    return neighbor_count

def identify_branches_to_keep(
    labeled_skeleton, num_features, affine, axcodes, left_right_axis,
    aorta_center_world, delta_x
):
    """
    Identify labels of branches to keep based on their spatial relation to the aorta.

    Returns:
    - labels_to_keep: A list of labels representing branches to keep.
    """
    labels_to_keep = []
    # For each label, check if any part of the branch extends significantly left
    for label_idx in range(1, num_features + 1):
        # Get the voxel indices for the current branch
        coords_voxel = np.array(np.nonzero(labeled_skeleton == label_idx)).T
        
        # Transform voxel coordinates to world coordinates
        coords_world = nib.affines.apply_affine(affine, coords_voxel)
        
        # Check the left-right condition
        if axcodes[left_right_axis] == 'L':
            left_condition = coords_world[:, left_right_axis] < (aorta_center_world[left_right_axis] - delta_x)
        elif axcodes[left_right_axis] == 'R':
            left_condition = coords_world[:, left_right_axis] > (aorta_center_world[left_right_axis] + delta_x)
        else:
            raise ValueError("Unexpected axis code for left-right direction.")
        
        if not np.any(left_condition):
            # Keep this branch
            labels_to_keep.append(label_idx)
    
    return labels_to_keep



def remove_branch_points(skeleton):
    """
    Remove branching points from a 3D skeleton.
    
    Parameters:
    - skeleton: 3D binary array of the skeletonized volume.

    Returns:
    - pruned_skeleton: Skeleton with branching points removed.
    """
    pruned_skeleton = skeleton.copy()
    neighbors_offsets = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
    ]
    
    for idx, val in np.ndenumerate(skeleton):
        if val == 1:
            # Count the number of neighboring skeleton elements
            num_neighbors = sum(
                skeleton[idx[0] + dx, idx[1] + dy, idx[2] + dz]
                for dx, dy, dz in neighbors_offsets
                if 0 <= idx[0] + dx < skeleton.shape[0]
                and 0 <= idx[1] + dy < skeleton.shape[1]
                and 0 <= idx[2] + dz < skeleton.shape[2]
            )
            # Remove branching points (more than 2 neighbors)
            if num_neighbors > 2:
                pruned_skeleton[idx] = 0

    return pruned_skeleton



def calculate_thickness(volumetric_branch, skeleton_branch):
    """
    Calculate the thickness of a branch as the ratio of branch volume to skeleton length.
    
    Parameters:
    - volumetric_branch: 3D binary array representing the volumetric branch.
    - skeleton_branch: 3D binary array representing the skeleton of the branch.

    Returns:
    - thickness: The calculated thickness as volume / skeleton length.
    """
    volume = np.sum(volumetric_branch)  # Count of voxels in the volumetric branch
    skeleton_length = np.sum(skeleton_branch)  # Count of voxels in the skeleton branch
    thickness = volume / skeleton_length if skeleton_length > 0 else 0  # Avoid division by zero
    return thickness



def isolate_main_branch_skeleton(volume, dilation_radius=3):
    """
    Isolates the main branch of a 3D volume by skeletonizing it, removing branch points,
    dilating each path, and keeping the thickest path per z-slice. 
    Only branches in the final output_volume are included in the filtered skeleton.

    Parameters:
    - volume: 3D binary array representing the original structure.
    - dilation_radius: Radius of the structuring element for dilation.

    Returns:
    - output_volume: 3D binary array with only the thickest branch on each z-slice.
    - filtered_skeleton: Skeleton of only the branches in output_volume.
    """
    # Step 1: Skeletonize the volume
    skeleton = skeletonize_3d(volume > 0)
    
    # Step 2: Remove branching points from the skeleton
    pruned_skeleton = remove_branch_points(skeleton)

    # Step 3: Label each unbranched path in the pruned skeleton
    labeled_skeleton, num_labels = label(pruned_skeleton)

    # Prepare structuring element for dilation
    structuring_element = np.ones((dilation_radius, dilation_radius, dilation_radius), dtype=bool)

    # Step 4: Dilate each labeled path, intersect with original volume, and calculate thickness
    volumetric_branches = []
    thicknesses = []
    
    for label_id in range(1, num_labels + 1):
        # Create a binary mask for the current path
        path_mask = (labeled_skeleton == label_id)
        
        # Dilate the path mask
        dilated_path = binary_dilation(path_mask, structure=structuring_element)
        
        # Intersect with the original volume to get the volumetric branch
        volumetric_branch = np.logical_and(dilated_path, volume)
        volumetric_branches.append(volumetric_branch)
        
        # Calculate thickness for this volumetric branch using volume/skeleton length
        thickness = calculate_thickness(volumetric_branch, path_mask)
        thicknesses.append(thickness)

    # Step 5: For each z-slice, keep only the thickest branch
    output_volume = np.zeros_like(volume, dtype=bool)
    skeleton_mask = np.zeros_like(volume, dtype=bool)  # For filtered skeleton output
    
    for z in range(volume.shape[2]):
        max_thickness = 0
        selected_branch = None
        selected_skeleton_branch = None
        for branch_idx, volumetric_branch in enumerate(volumetric_branches):
            if np.any(volumetric_branch[:, :, z]):  # Check if branch has voxels on this z-slice
                if thicknesses[branch_idx] > max_thickness:
                    max_thickness = thicknesses[branch_idx]
                    selected_branch = volumetric_branch
                    selected_skeleton_branch = (labeled_skeleton == (branch_idx + 1))  # Original skeleton branch

        # Add the selected branch to the output for this z-slice
        if selected_branch is not None:
            output_volume[:, :, z] = np.logical_or(output_volume[:, :, z], selected_branch[:, :, z])
            skeleton_mask[:, :, z] = np.logical_or(skeleton_mask[:, :, z], selected_skeleton_branch[:, :, z])

    return output_volume, skeleton_mask




def isolate_main_branch_final(volume, affine):
    """
    For each slice along the z-axis from top to bottom, isolates the main branch by keeping the largest connected 
    component that touches the largest component in the previous slice. In the first 5% of slices, only the largest 
    component is kept. If no component touches the previous main branch, the largest component in the slice is added 
    to the output, but the previous main component remains unchanged. After processing, apply binary erosion and 
    dilation with a 5x5x5 structuring element, intersect with the original volume, and retain only the largest 3D connected component.

    Parameters:
        volume (numpy.ndarray): 3D binary mask of the vessel.
        affine (numpy.ndarray): Affine matrix from the loaded NIfTI file to determine orientation.

    Returns:
        numpy.ndarray: Processed 3D binary volume with the main branch isolated and the largest 3D component retained.
    """
    # Deep copy the input volume to ensure the original is not modified
    original_volume = copy.deepcopy(volume)

    # Determine which axis corresponds to z (height) using the affine matrix
    orientation = nib.orientations.aff2axcodes(affine)
    z_axis_index = orientation.index('S') if 'S' in orientation else orientation.index('I')
    
    # Check if the z-axis direction in the affine matrix is superior-to-inferior or inferior-to-superior
    z_direction = affine[z_axis_index, z_axis_index]
    top_to_bottom = True if z_direction < 0 else False

    # Ensure volume is initialized correctly
    tmp = volume.copy().astype('uint8')  # Initialize tmp as a copy of volume

    # Sum over the two axes that are not the z-axis
    axes_to_sum = tuple(i for i in [0, 1, 2] if i != z_axis_index)
    tmp = tmp.sum(axis=axes_to_sum)

    # Find indices along the z-axis where there are non-zero values
    z_indices = np.where(tmp > 0)[0]

    # Ensure we iterate from top to bottom based on orientation
    if top_to_bottom:
        z_indices = z_indices[::-1]

    # Initialize an empty binary volume for the output
    output_volume = np.zeros_like(volume, dtype=bool)

    # Determine the threshold for the first 5% of non-zero slices
    initial_slices = int(0.05 * len(z_indices))
    #initial_slices=1000

    # Initialize a variable to store the main component from the previous slice
    previous_main_component = None

    gap_size=0
    max_gap_size=0
    
    for i, z in enumerate(z_indices,0):

        # Extract the current z slice based on the determined z-axis index
        if z_axis_index == 0:
            slice_2d = volume[z, :, :].astype(bool)
        elif z_axis_index == 1:
            slice_2d = volume[:, z, :].astype(bool)
        else:
            slice_2d = volume[:, :, z].astype(bool)

        if i < initial_slices:
            # In the first 5% of slices, keep only the largest component
            labeled_slice, num_features = label(slice_2d)
            if num_features > 1:
                # Identify the largest connected component
                component_sizes = np.bincount(labeled_slice.ravel())[1:]  # Exclude the background
                largest_component = (labeled_slice == (np.argmax(component_sizes) + 1))
                main_component = largest_component
                previous_main_component = largest_component
            else:
                main_component = slice_2d
                previous_main_component = slice_2d
        else:
            # Dilate the previous main component once to allow for connections
            try:
                dilated_previous_component = binary_dilation(previous_main_component)
            except:
                print(previous_main_component)
                raise ValueError('Error in binary dilation')

            # Label components in the current slice
            labeled_slice, num_features = label(slice_2d)
            
            # Check if there are any components in the current slice
            if num_features == 0:
                continue  # No components to process, skip this slice

            largest_touching_component = np.zeros_like(slice_2d, dtype=bool)
            found_connection = False

            # Iterate over all components sorted by size (largest first)
            component_sizes = np.bincount(labeled_slice.ravel())[1:]  # Exclude the background
            sorted_components = np.argsort(component_sizes)[::-1] + 1  # Largest to vesselllest component labels
            
            #get the largest component that touches the last one
            for component_label in sorted_components:
                component_mask = (labeled_slice == component_label)
                
                # Check if this component touches the dilated previous component
                #if np.any(component_mask & dilated_previous_component):
                if np.any(component_mask & dilated_previous_component):
                    largest_touching_component = component_mask
                    found_connection = True
                    break  # Stop once we find the largest component that touches

            # If a connection was found, update previous_main_component
            if found_connection:
                main_component = largest_touching_component
                previous_main_component = main_component
                in_gap=False
                gap_size=0
            else:
                # If no connection, add the largest component without updating previous_main_component
                #main_component = (labeled_slice == sorted_components[0])
                gap_size+=1
                if gap_size>max_gap_size:
                    max_gap_size=gap_size

        # Update the output volume with the main component found for this slice
        if z_axis_index == 0:
            output_volume[z, :, :] = main_component
        elif z_axis_index == 1:
            output_volume[:, z, :] = main_component
        else:
            output_volume[:, :, z] = main_component


    print('Main branch isolated 2D')
    # Define a 5x5x5 structuring element for erosion and dilation
    
    structuring_element = np.ones((3,3,3), dtype=bool)
    # Apply binary erosion and dilation with the structuring element
    eroded_volume = binary_erosion(output_volume, structure=structuring_element)
    print('Eroded')
    #check if empty
    if not np.any(eroded_volume):
        return output_volume
    n=min(3,max_gap_size*2)
    n=max(3,n)
    print('n is:', n)
    structuring_element = np.ones((n,n,n), dtype=bool)
    dilated_volume = binary_dilation(eroded_volume, structure=structuring_element)
    print('Dilated')

    # Find the intersection with the original volume
    intersected_volume = np.logical_and(dilated_volume, original_volume)

    #return intersected_volume.astype(bool)
    # Keep only the largest 3D connected component
    labeled_intersection, num_features = label(intersected_volume)
    if num_features > 0:
        # Identify the largest connected component
        component_sizes = np.bincount(labeled_intersection.ravel())[1:]  # Exclude background
        largest_component_label = np.argmax(component_sizes) + 1
        final_output = (labeled_intersection == largest_component_label)
    else:
        # No components were found, return an empty volume
        final_output = np.zeros_like(volume, dtype=bool)

    return final_output

def tumor_reaches_left_of_aorta(tumor_data, aorta_data, affine, delta_x=0.0):
    """
    Checks if the tumor reaches the left side of the aorta.

    Parameters:
    - tumor_data: Boolean NumPy array of the tumor segmentation.
    - aorta_data: Boolean NumPy array of the aorta segmentation.
    - affine: Affine matrix of the images (same for both tumor and aorta).
    - delta_x: Optional threshold in millimeters to consider as 'reaching' left of the aorta.

    Returns:
    - reaches_left: Boolean value indicating whether the tumor reaches the left side of the aorta.
    """
    # Ensure that the data is boolean
    tumor_data = tumor_data.astype(bool)
    aorta_data = aorta_data.astype(bool)
    
    # Determine which axis corresponds to the left-right direction using the image orientation
    axcodes = aff2axcodes(affine)
    # Map the axis codes to indices
    left_right_axis = None
    for idx, code in enumerate(axcodes):
        if code in ['L', 'R']:
            left_right_axis = idx
            break
    if left_right_axis is None:
        raise ValueError("Could not determine left-right axis from image orientation.")
    
    # Get voxel coordinates of the aorta and tumor
    aorta_voxels = np.array(np.nonzero(aorta_data)).T  # Shape (N, 3)
    tumor_voxels = np.array(np.nonzero(tumor_data)).T  # Shape (M, 3)
    
    # Transform voxel coordinates to world coordinates
    aorta_world_coords = nib.affines.apply_affine(affine, aorta_voxels)
    tumor_world_coords = nib.affines.apply_affine(affine, tumor_voxels)
    
    # Determine the leftmost coordinate of the aorta along the left-right axis
    if axcodes[left_right_axis] == 'L':
        # Left increases in the positive direction
        aorta_leftmost = np.min(aorta_world_coords[:, left_right_axis]) - delta_x
        # Check if any tumor voxels are left of the aorta's leftmost coordinate
        reaches_left = np.any(tumor_world_coords[:, left_right_axis] < aorta_leftmost)
    elif axcodes[left_right_axis] == 'R':
        # Right increases in the positive direction
        aorta_leftmost = np.max(aorta_world_coords[:, left_right_axis]) + delta_x
        # Check if any tumor voxels are left of the aorta's leftmost coordinate
        reaches_left = np.any(tumor_world_coords[:, left_right_axis] > aorta_leftmost)
    else:
        raise ValueError("Unexpected axis code for left-right direction.")
    
    return reaches_left

def skeletonize_main_branch(volume):
    """
    Skeletonizes a 3D structure, removes branching points using GPU-accelerated convolution if available,
    separates each branch, and retains only the largest branch at each z-slice.

    Parameters:
    - volume: 3D binary array representing the original structure.

    Returns:
    - output_volume: 3D binary array with only the largest branch retained at each z-slice.
    """
    # Step 1: Skeletonize the volume on the CPU
    skeleton = skeletonize_3d(volume > 0).astype(np.uint8)
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Move to GPU if available and count neighbors with a 3D convolution
    skeleton_tensor = torch.tensor(skeleton, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Define a 3x3x3 kernel with 1s around the center (excluding the center voxel itself)
    kernel = torch.ones((1, 1, 3, 3, 3), device=device)
    kernel[0, 0, 1, 1, 1] = 0

    # Perform 3D convolution to count neighbors
    neighbor_count = F.conv3d(skeleton_tensor, kernel, padding=1).squeeze()

    # Remove branching points (keep only voxels with <= 2 neighbors)
    pruned_skeleton_tensor = torch.where((skeleton_tensor.squeeze() == 1) & (neighbor_count <= 2), 1, 0)

    # Move the pruned skeleton back to CPU if it was processed on the GPU
    pruned_skeleton = pruned_skeleton_tensor.cpu().numpy().astype(np.uint8)

    # Step 3: Perform 3D connected component analysis on the pruned skeleton
    labeled_skeleton, num_labels = label(pruned_skeleton)

    # Calculate the size of each branch (connected component) by counting voxels
    branch_sizes = np.bincount(labeled_skeleton.ravel())[1:]  # Skip the background count (label 0)
    
    # Step 4: Iterate through each z-slice and keep only the largest branch
    output_volume = np.zeros_like(volume, dtype=bool)
    
    for z in range(volume.shape[2]):
        # Find labels that intersect this z-slice
        slice_labels = np.unique(labeled_skeleton[:, :, z])
        slice_labels = slice_labels[slice_labels > 0]  # Ignore the background (label 0)
        
        if slice_labels.size > 0:
            # Get sizes of intersecting branches and find the largest
            slice_branch_sizes = {label: branch_sizes[label - 1] for label in slice_labels}
            largest_branch_label = max(slice_branch_sizes, key=slice_branch_sizes.get)
            
            # Retain only the largest branch in this slice
            output_volume[:, :, z] = (labeled_skeleton[:, :, z] == largest_branch_label)

    return output_volume




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

def center_of_mass(pancreas,downsample=True,downsample_factor=100):
    # Get the coordinates of non-zero voxels in the pancreas
    non_zero_coords = np.column_stack(np.nonzero(pancreas))
    # Downsample if necessary
    if downsample and non_zero_coords.shape[0] > downsample_factor:
        non_zero_coords = non_zero_coords[::downsample_factor]
    # Compute center of mass and shift vector
    original_center_of_mass = non_zero_coords.mean(axis=0)
    return original_center_of_mass, non_zero_coords

def crop_pancreas_with_bounding_box(pancreas, center_of_mass, box_size):
    """
    Crops the pancreas volume around a bounding box centered on the center of mass
    
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

def align_volume_x(volume, skeleton, volume2):
    #align vessel skeleton with pca
    original_center_of_mass, non_zero_coords = center_of_mass(skeleton)
        #bounding box around the center of mass
    # Calculate the span in each dimension
    non_zero_indices = np.nonzero(skeleton)
    x_span = non_zero_indices[0].max() - non_zero_indices[0].min() + 1
    y_span = non_zero_indices[1].max() - non_zero_indices[1].min() + 1
    z_span = non_zero_indices[2].max() - non_zero_indices[2].min() + 1
    # Find the maximum span and scale it by 1.5
    max_span = max(x_span, y_span, z_span)
    box_size = int(2 * max_span)
    translated_skeleton=crop_pancreas_with_bounding_box(skeleton, original_center_of_mass,box_size)
    translated_volume=crop_pancreas_with_bounding_box(volume, original_center_of_mass,box_size)
    translated_volume2=crop_pancreas_with_bounding_box(volume2, original_center_of_mass,box_size)

    # new center of mass and coordinates
    new_center_of_mass, non_zero_coords = center_of_mass(skeleton,False)
    if len(non_zero_coords) < 2:
        principal_axes=[np.array([1, 0, 0])]
    elif len(non_zero_coords) < 3:
        principal_axes=[non_zero_coords[1]-non_zero_coords[0]]
    else:
        # Perform PCA to get the principal component
        pca = PCA(n_components=3)
        #print('Non zero coords:', non_zero_coords)
        pca.fit(non_zero_coords)
        principal_axes = pca.components_
    #print('Principal axes:', principal_axes[0])
    target_vector = np.array([1, 0, 0])
    if np.dot(principal_axes[0], target_vector) < 0:
        pa = -principal_axes[0]
    else:
        pa = principal_axes[0]

    rotated_skeleton, principal_component, rotation_matrix, center = rotate_3d_array(translated_skeleton, principal_axes[0])
    rotated_volume = affine_transform( translated_volume,
                                        rotation_matrix.T,  # Transpose to invert for affine_transform
                                        offset=center - np.dot(rotation_matrix.T, center),  # Offset to keep the rotation around the center
                                        order=0  # Linear interpolation (you can also use `order=0` for nearest-neighbor if binary data)
                                    )
    rotated_volume2 = affine_transform( translated_volume2,
                                        rotation_matrix.T,  # Transpose to invert for affine_transform
                                        offset=center - np.dot(rotation_matrix.T, center),  # Offset to keep the rotation around the center
                                        order=0  # Linear interpolation (you can also use `order=0` for nearest-neighbor if binary data)
                                    )
    return rotated_volume, rotated_skeleton, rotated_volume2
    
def check_vessel_tumor_intersection(vessel, tumor,skeleton,n=5,affine=None,debug=False,vessel_type=None):
    intersection_percentages=[]
    start=0
    max_intersection=0
    while start<vessel.shape[0]:
        if start+n>vessel.shape[0]:
            end=vessel.shape[0]-1
        else:
            end=start+n

        if start-n<0:
            padded_start=0
        else:
            padded_start=start-n
        if end+n>vessel.shape[0]:
            padded_end=vessel.shape[0]-1
        else:
            padded_end=end+n

        # Get padded slice
        padded_vessel = vessel[padded_start:padded_end, :, :]
        padded_tumor = tumor[padded_start:padded_end, :, :]
        padded_skeleton = skeleton[padded_start:padded_end, :, :]

        # Check if vessel intersects with tumor in the padded region
        if not np.any(padded_vessel & padded_tumor):
            #print('vessel does not intersect with tumor in slice:', start)
            start += 1
            continue

        if padded_skeleton.astype('uint8').sum()>1:
            # Align the padded slices
            #print('Shape of padded slice:', padded_vessel.shape)
            #aligned_vessel=padded_vessel
            #aligned_skeleton=padded_skeleton
            #aligned_tumor=padded_tumor
            aligned_vessel, aligned_skeleton, aligned_tumor = align_volume_x(padded_vessel, padded_skeleton, padded_tumor)
        else:
            print('No skeleton found in slice:', start)
            start += 1
            continue

        # Crop aligned volume at x, ensuring final thickness is exactly `n`
        mid_index = aligned_vessel.shape[0] // 2
        crop_start = max(mid_index - n // 2, 0)
        crop_end = crop_start + n
        aligned_vessel = aligned_vessel[crop_start:crop_end, :, :]
        aligned_skeleton = aligned_skeleton[crop_start:crop_end, :, :]
        aligned_tumor = aligned_tumor[crop_start:crop_end, :, :]
        
        borders = np.zeros_like(aligned_vessel)
        #erode 2D, slice by slice
        for i in range(aligned_vessel.shape[0]):
            borders[i]=aligned_vessel[i] ^ binary_erosion(aligned_vessel[i])
        # Get borders of the aligned_vessel: original - eroded
        #eroded_vessel = binary_erosion(aligned_vessel)
        #borders = aligned_vessel ^ eroded_vessel

        # Check number of border voxels that intersect with tumor
        intersection_voxels = np.sum(borders & aligned_tumor)
        border_voxels = np.sum(borders)

        #print('Shape of slice:', aligned_vessel.shape)
        
        # Calculate intersection percentage, avoid division by zero
        if border_voxels > 0:
            intersection_percentages.append(intersection_voxels / border_voxels)
            #print('Slice:', start, 'Intersection percentage:', intersection_voxels / border_voxels)
            if intersection_voxels / border_voxels>max_intersection:
                max_intersection=intersection_voxels / border_voxels
                #save the slice
                if debug:
                    nib.save(nib.Nifti1Image(aligned_vessel.astype(np.uint8)+2*aligned_tumor.astype(np.uint8),affine), vessel_type+'_aligned_volume_max_intersection.nii.gz')
                    nib.save(nib.Nifti1Image(borders.astype(np.uint8),affine), vessel_type+'_borders.nii.gz')
        
        if debug:
            print('Intersection in slice:', start, 'is:', intersection_voxels / border_voxels, 'intersection voxels:', intersection_voxels, 'border voxels:', border_voxels)
            
        # Increment start to move to the next segment
        start += 1

    return np.max(intersection_percentages)*360

type2name={'SMA':'superior_mesenteric_artery.nii.gz','aorta':'aorta.nii.gz','CA and CHA':'celiac_aa.nii.gz','portal vein and SMV':'veins.nii.gz','IVC':'postcava.nii.gz'}

def load_pancreatic(folder_path):
    pancreas_path = os.path.join(folder_path, 'pancreas.nii.gz')
    pancreas = cr.load_canonical(pancreas_path)


    pdac=os.path.join(folder_path, 'pancreatic_pdac.nii.gz')
    if os.path.exists(pdac):
        pdac = cr.load_canonical(pdac)
        #check if orientation is the same
        if not np.allclose(pancreas.affine, pdac.affine):
            #reorient pdac to pancreas
            pdac = resample_from_to(pdac, (pancreas.shape, pancreas.affine))
        pnet = os.path.join(folder_path, 'pancreatic_pnet.nii.gz')
        pnet = cr.load_canonical(pnet)
        #check if orientation is the same
        if not np.allclose(pancreas.affine, pnet.affine):
            #reorient pnet to pancreas
            pnet = resample_from_to(pnet, (pancreas.shape, pancreas.affine))

        pancreas_data = pancreas.get_fdata().astype(bool)
        pdac_data = pdac.get_fdata().astype(bool)
        pnet_data = pnet.get_fdata().astype(bool)
        tumor_data = pdac_data | pnet_data
    else:
        lesion=os.path.join(folder_path, 'pancreatic_lesion.nii.gz')
        lesion = cr.load_canonical(lesion)
        if not np.allclose(pancreas.affine, lesion.affine):
            #reorient lesion to pancreas
            lesion = resample_from_to(lesion, (pancreas.shape, pancreas.affine))
        pancreas_data = pancreas.get_fdata().astype(bool)
        tumor_data = lesion.get_fdata().astype(bool)
    
    vessels={}
    for vessel_type in type2name.keys():
        vessel = os.path.join(folder_path, type2name[vessel_type])
        #print('Folder path:', folder_path)
        if not os.path.exists(vessel):
            vessel=os.path.join('/ccvl/net/ccvl15/pedro/AtlasVessels/',folder_path[folder_path.rfind('BDMAP_'):folder_path.rfind('BDMAP_')+len('BDMAP_00000010')].replace('_',''),'segmentations',
                                type2name[vessel_type])
        if not os.path.exists(vessel):
            vessel=os.path.join('/mnt/ccvl15/pedro/AtlasVessels/', folder_path[folder_path.rfind('BDMAP_'):folder_path.rfind('BDMAP_')+len('BDMAP_00000010')].replace('_',''),'segmentations',
                                type2name[vessel_type])
        vessel = cr.load_canonical(vessel)
        #check if orientation is the same
        if not np.allclose(pancreas.affine, vessel.affine):
            #reorient vessel to pancreas
            vessel = resample_from_to(vessel, (pancreas.shape, pancreas.affine))
        #get data
        vessel_data = vessel.get_fdata().astype(bool)
        vessels[vessel_type]=vessel_data

    dilated_tumor = binary_dilation(tumor_data)

    if np.any(vessels['CA and CHA'] & dilated_tumor):
        #break CA into 3 vessels
        celiac_data = vessels['CA and CHA']
        aorta = vessels['aorta'].copy()
        ca,ha,sa=divide_CA(aorta_data=aorta, celiac_data=celiac_data, affine=pancreas.affine)
        vessels['CA']=ca
        vessels['CHA']=ha
        vessels['SA']=sa
        #remove 'CA and CHA'
        del vessels['CA and CHA']
    else:
        #no intersection anyway
        vessels['CA']=vessels['CA and CHA']*0
        vessels['CHA']=vessels['CA and CHA']*0
        vessels['SA']=vessels['CA and CHA']*0
        del vessels['CA and CHA']
    return pancreas_data, vessels, tumor_data, pancreas.affine

def crop_to_tumor_box(tumor, vessel):
    """
    Crops both the tumor and vessel volumes using a bounding box that is slightly larger than the tumor.

    Parameters:
    - tumor: 3D binary array representing the tumor volume.
    - vessel: 3D binary array representing the vessel volume.
    - padding: Number of voxels to expand the bounding box around the tumor.

    Returns:
    - cropped_tumor: Cropped tumor volume.
    - cropped_vessel: Cropped vessel volume, with the same shape as `cropped_tumor`.
    """
    # Find bounding box of the tumor
    tumor_coords = np.argwhere(tumor)
    min_coords = tumor_coords.min(axis=0)
    max_coords = tumor_coords.max(axis=0)

    #set padding as tumor size
    padding = max_coords-min_coords

    # Add padding to the bounding box
    min_coords = np.maximum(min_coords - padding, 0)
    max_coords = np.minimum(max_coords + padding, np.array(tumor.shape) - 1)

    # Crop both volumes using the padded bounding box
    cropped_tumor = tumor[min_coords[0]:max_coords[0]+1,
                          min_coords[1]:max_coords[1]+1,
                          min_coords[2]:max_coords[2]+1]
    
    cropped_vessel = vessel[min_coords[0]:max_coords[0]+1,
                          min_coords[1]:max_coords[1]+1,
                          min_coords[2]:max_coords[2]+1]
    
    return cropped_tumor, cropped_vessel

def stage(path,debug=False,size=None,pnet=False):
    pancreas, vessels, tumor, affine = load_pancreatic(path)
    print('Staging data loaded')

    #dilate tumor
    tumor_dilated = binary_dilation(tumor,iterations=2)
    interaction={'SMA':0,'aorta':0,'CA':0,'portal vein and SMV':0,'IVC':0}
    #for key in vessel.keys():
    if debug:
        #save the isolated vessel
        nib.save(nib.Nifti1Image(tumor.astype(np.uint8), affine), 'tumor.nii.gz')

    tumor_uncropped=tumor_dilated.copy()
        
    for vessel_type in vessels.keys():
        print('Analyzing: ', vessel_type)
        vessel = vessels[vessel_type].copy()
        if debug:
            #save the isolated vessel
            nib.save(nib.Nifti1Image(vessel.astype(np.uint8), affine), vessel_type+'_original.nii.gz')
        tumor = tumor_dilated.copy()
        #check if vessel intersects with tumor

        if debug and vessel_type=='portal vein and SMV':
            main_vessel = isolate_main_branch_final(vessel, affine)
            nib.save(nib.Nifti1Image(main_vessel.astype(np.uint8), affine), vessel_type+'_main_branch_uncropped.nii.gz')
            
        if not np.any(vessel & tumor):
            print('vessel does not intersect with tumor:', vessel_type)
            interaction[vessel_type]=0
            if not debug:
                continue

        #if vessel_type=='CA and CHA':
            #get celiac axis only
        #    aorta = vessels['aorta'].copy()
        #    vessel=extract_proper_celiac_axis(aorta_data=aorta, celiac_data=vessel, affine=affine)
        #    if debug:
                #save the isolated vessel
        #        nib.save(nib.Nifti1Image(vessel.astype(np.uint8), affine), 'isolated_CA.nii.gz')

        #crop to tumor box
        tumor, vessel = crop_to_tumor_box(tumor_uncropped, vessel)

        

        if vessel_type != 'aorta' and vessel_type != 'IVC':
            main_vessel = isolate_main_branch_final(vessel, affine)
        else:
            main_vessel = vessel

        if debug:
            #save the isolated vessel
            nib.save(nib.Nifti1Image(main_vessel.astype(np.uint8), affine), vessel_type+'_main_branch.nii.gz')

        if not np.any(main_vessel & tumor):
            print('Main vessel branch does not intersect with tumor:', vessel_type)
            interaction[vessel_type]=0
            continue
        
        #print('Shapes:', tumor.shape, vessel.shape)
       

        skeleton=skeletonize_main_branch(main_vessel)
        print('Skeletonized')
        if debug:
            #save the skeleton
            nib.save(nib.Nifti1Image(skeleton.astype(np.uint8), affine), vessel_type+'_skeleton.nii.gz')

        aligned_vessel, aligned_skeleton, aligned_tumor = align_volume_x(main_vessel, skeleton, tumor)
        if debug:
            #save aligned volume
            nib.save(nib.Nifti1Image(aligned_vessel.astype(np.uint8)+2*aligned_tumor.astype(np.uint8), affine), vessel_type+'_aligned_volume.nii.gz')

        intersection_angle = check_vessel_tumor_intersection(aligned_vessel, aligned_tumor, aligned_skeleton,n=5,affine=affine,
                                                             debug=debug,vessel_type=vessel_type)
        print('Intersection angle:', int(np.round(intersection_angle,0)))
        interaction[vessel_type]=int(np.round(intersection_angle,0))

    text='Vascular Involvement:\n'
    contact_text=''
    no_contact_text='Tumor does not contact: '
    for vessel_type in vessels.keys():
        if interaction[vessel_type]==0:
            no_contact_text+=f'{vessel_type}, '
        elif interaction[vessel_type]>0 and interaction[vessel_type]<180:
            contact_text+=f'{vessel_type[0].capitalize()+vessel_type[1:]}: tumor contact but not encasement ({interaction[vessel_type]} degree contact with the tumor). \n'
        elif interaction[vessel_type]>=180:
            contact_text+=f'{vessel_type[0].capitalize()+vessel_type[1:]}: tumor encasement ({interaction[vessel_type]} degree contact with the tumor). \n'
    if contact_text!='':
        text+=contact_text
    if no_contact_text!='Tumor does not contact: ':
        text+=no_contact_text[:-2]+'. \n'

    contacted_organs=[]
    organs = {
            'left adrenal gland': 'adrenal_gland_left.nii.gz',
            'spleen': 'spleen.nii.gz',
            'colon': 'colon.nii.gz',
            'left kidney': 'kidney_left.nii.gz',
            'stomach': 'stomach.nii.gz',
            'duodenum': 'duodenum.nii.gz',
            'bile duct': 'common_bile_duct.nii.gz',
            'intestine': 'intestine.nii.gz',
        }
    for name, organ_file in organs.items():
        # Load the organ file
        organ_path = os.path.join(path, organ_file)
        #check if file exists
        if not os.path.exists(organ_path):
            continue
        organ = cr.load_canonical(organ_path)
        
        # Check if orientation is the same
        if not np.allclose(affine, organ.affine):
            # Reorient organ to match the tumor
            organ = resample_from_to(organ, (tumor.shape, affine))
        
        # Load organ data and check for overlap with the tumor
        organ_data = organ.get_fdata().astype(bool)

        tumor, organ_data = crop_to_tumor_box(tumor_uncropped, organ_data)

        interaction[name] = np.any(organ_data & tumor)
        if np.any(organ_data & tumor):
            contact_text += f'{name}, '
            contacted_organs.append(name)
        else:
            no_contact_text += f'{name}, '

    if pnet:
        if contact_text!='Tumor contact with organs: ':
            text+=contact_text[:-2]+'. \n'
        if no_contact_text!='Tumor does not contact: ':
            text+=no_contact_text[:-2]+'. \n'

    #get stage
    #PDAC
	#•	T1: Tumor ≤2 cm, limited to the pancreas.
	#•	T1a: Tumor ≤0.5 cm.
	#•	T1b: Tumor >0.5 cm and ≤1 cm.
	#•	T1c: Tumor >1 cm and ≤2 cm.
	#•	T2: Tumor >2 cm and ≤4 cm, limited to the pancreas.
	#•	T3: Tumor >4 cm, limited to the pancreas.
	#•	T4: Tumor involves the celiac axis (CA) or superior mesenteric artery (SMA), rendering it unresectable. Kang: or CHA.

    #PNET
	#•	T1: Tumor ≤2 cm, limited to the pancreas.
	#•	T2: Tumor >2 cm and ≤4 cm, limited to the pancreas.
	#•	T3: Tumor >4 cm, limited to the pancreas or any size that invades the duodenum or bile duct.
	#•	T4: Tumor invades adjacent organs (e.g., stomach, spleen, colon, adrenal glands, kidneys) or major arteries (CA or SMA).


    if interaction['SMA']>=180 or interaction['CA']>=180 or interaction['CHA']>180:
        stage='T4'
    elif pnet and (interaction['stomach'] or interacion['spleen'] or interaction['colon'] or interaction['adrenal gland left'] or interaction['kidney left']):
        stage='T4'
    elif pnet and (interaction['duodenum'] or interaction['bile duct']):
        stage='T3'
    else:
        #check size
        if size is None:
            stage=None
        elif (not pnet) and size<=0.5:
            stage='T1a'
        elif (not pnet) and size>0.5 and size<=1:
            stage='T1b'
        elif (not pnet) and size>1 and size<=2:
            stage='T1c'
        elif (pnet) and size<2:
            stage='T1'
        elif size>2 and size<=4:
            stage='T2'
        elif size>4:
            stage='T3'

    if stage is not None:
        text+=f'Tumor Stage (T stage): {stage}. \n'

    return interaction,text,stage, contacted_organs


