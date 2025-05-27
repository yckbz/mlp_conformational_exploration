# selection.py

import numpy as np
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from tqdm import tqdm


# In selection.py

def _calculate_soap_descriptors(structures: list, soap_config: dict):
    """
    Calculates the average SOAP descriptors for a list of structures.
    """
    print("Calculating SOAP descriptors for all candidate structures...")

    # --- NEW: Clean the structures before parallel processing ---
    # The attached calculator contains un-pickle-able objects (thread locks).
    # We remove it from each structure because dscribe only needs positions/symbols.
    for atm in structures:
        atm.calc = None
    # --- End of Change ---

    # Setup the SOAP descriptor generator from dscribe
    soap_generator = SOAP(
        species=soap_config['species'],
        periodic=False,
        r_cut=soap_config['r_cut'],
        n_max=soap_config['n_max'],
        l_max=soap_config['l_max'],
        average="outer",
        sparse=False
    )
    
    # This parallel call will now work correctly.
    descriptors = soap_generator.create(structures, n_jobs=-1)
    print(f"SOAP calculation complete. Descriptor matrix shape: {descriptors.shape}")
    return descriptors

def select_fps(structures: list, n_select: int, fps_params: dict, soap_params: dict):
    """
    Selects structures using Farthest Point Sampling (FPS). [cite: 3]

    Args:
        structures (list): The pool of candidate structures.
        n_select (int): The total number of structures to select.
        fps_params (dict): Parameters for FPS, like the distance metric.
        soap_params (dict): Parameters for SOAP descriptor calculation.

    Returns:
        list: A list of selected ASE Atoms objects.
    """
    print("Starting Farthest Point Sampling (FPS)...")
    if len(structures) <= n_select:
        print("Warning: Number of candidates is less than or equal to the number to select. Returning all candidates.")
        return structures

    descriptors = _calculate_soap_descriptors(structures, soap_params)
    n_candidates = descriptors.shape[0]
    
    # Choose the distance metric
    metric = fps_params.get('distance_metric', 'euclidean')
    if metric == 'cosine':
        distance_matrix_func = cosine_distances
    else:
        distance_matrix_func = euclidean_distances
        
    # Start with a random structure [cite: 3]
    selected_indices = [np.random.randint(n_candidates)]
    remaining_indices = list(range(n_candidates))
    remaining_indices.remove(selected_indices[0])

    # Initialize the minimum distance from each remaining point to the selected set
    # For the first iteration, this is just the distance to the single starting point
    selected_descs = descriptors[selected_indices]
    dist_to_set = distance_matrix_func(descriptors[remaining_indices], selected_descs).min(axis=1)

    # Iteratively select the farthest point [cite: 3]
    for _ in tqdm(range(1, n_select), desc="FPS Selection"):
        # Find the point with the maximum minimum distance [cite: 3]
        farthest_point_idx_in_remaining = np.argmax(dist_to_set)
        farthest_point_global_idx = remaining_indices.pop(farthest_point_idx_in_remaining)
        
        # Add the new point to the selected set [cite: 4]
        selected_indices.append(farthest_point_global_idx)

        # If we have selected all we need, break
        if len(selected_indices) == n_select:
            break
            
        # Update distances for the next iteration
        # We only need to remove the distance entry for the point just selected
        dist_to_set = np.delete(dist_to_set, farthest_point_idx_in_remaining)
        
        # Calculate distance from remaining points to the *newly added* point
        new_point_desc = descriptors[farthest_point_global_idx].reshape(1, -1)
        dist_to_new_point = distance_matrix_func(descriptors[remaining_indices], new_point_desc).flatten()
        
        # Update the overall minimum distance to the set
        dist_to_set = np.minimum(dist_to_set, dist_to_new_point)

    selected_structures = [structures[i] for i in selected_indices]
    print(f"FPS complete. Selected {len(selected_structures)} structures.")
    return selected_structures


def select_pca_grid(structures: list, n_select: int, pca_grid_params: dict, soap_params: dict):
    """
    Selects structures using a SOAP+PCA gridding strategy.

    Args:
        structures (list): The pool of candidate structures.
        n_select (int): The total number of structures to select (acts as a target).
        pca_grid_params (dict): Parameters for the PCA gridding method.
        soap_params (dict): Parameters for SOAP descriptor calculation.

    Returns:
        list: A list of selected ASE Atoms objects.
    """
    print("Starting SOAP+PCA+Gridding selection...")
    descriptors = _calculate_soap_descriptors(structures, soap_params)

    # 1. Perform PCA for dimensionality reduction
    n_comp = pca_grid_params.get('n_components', 2)
    pca = PCA(n_components=n_comp)
    principal_components = pca.fit_transform(descriptors)
    print(f"PCA complete. Explained variance by {n_comp} components: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    # 2. Establish the grid
    grid_size = pca_grid_params['grid_size']
    if len(grid_size) != n_comp:
        raise ValueError("grid_size must have the same number of dimensions as n_components.")

    # Determine the range for each principal component [cite: 5]
    pc_min = principal_components.min(axis=0)
    pc_max = principal_components.max(axis=0)

    # Create bins for each dimension
    grid_bins = [np.linspace(pc_min[i], pc_max[i], grid_size[i] + 1) for i in range(n_comp)]
    
    # 3. Assign each structure to a grid cell
    # A dictionary where keys are grid cell indices (e.g., (1, 5)) and
    # values are lists of structure indices belonging to that cell.
    grid_map = {}
    
    # Use np.digitize for efficient binning
    binned_indices = np.zeros_like(principal_components, dtype=int)
    for i in range(n_comp):
        # np.digitize is 1-based, subtract 1 for 0-based index
        binned_indices[:, i] = np.digitize(principal_components[:, i], grid_bins[i]) - 1
        # Clamp values to be within grid bounds
        binned_indices[:, i] = np.clip(binned_indices[:, i], 0, grid_size[i] - 1)

    for i, pc_coords in enumerate(binned_indices):
        cell_tuple = tuple(pc_coords)
        if cell_tuple not in grid_map:
            grid_map[cell_tuple] = []
        grid_map[cell_tuple].append(i)

    # 4. Select one structure from each non-empty cell
    selected_indices = []
    for cell in grid_map.values():
        # Randomly select one structure from this grid cell
        selected_index = np.random.choice(cell)
        selected_indices.append(selected_index)

    selected_structures = [structures[i] for i in selected_indices]
    print(f"PCA Gridding complete. Selected {len(selected_structures)} structures from {len(grid_map)} non-empty grid cells.")
    return selected_structures
