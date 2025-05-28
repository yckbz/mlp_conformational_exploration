# selection.py

import numpy as np
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from tqdm import tqdm


def _calculate_soap_descriptors(structures: list, soap_config: dict):
    """
    Calculates average SOAP descriptors for a list of atomic structures.

    SOAP (Smooth Overlap of Atomic Positions) is a descriptor that characterizes
    local atomic environments.
    """
    print("Calculating SOAP descriptors for all candidate structures...")

    # Detach ASE calculators from structures before parallel processing with dscribe.
    # Calculators can contain non-serializable objects (e.g., thread locks)
    # that prevent pickling for multiprocessing. dscribe only needs positions/symbols.
    for atm in structures:
        atm.calc = None

    # Initialize the SOAP descriptor generator from the dscribe library.
    soap_generator = SOAP(
        species=soap_config['species'], # List of chemical species in the system.
        periodic=False,                 # Set to True for periodic systems if applicable.
        r_cut=soap_config['r_cut'],     # Cutoff radius for atomic environments.
        n_max=soap_config['n_max'],     # Number of radial basis functions.
        l_max=soap_config['l_max'],     # Number of angular basis functions.
        average="outer",                # Type of averaging for global descriptors.
        sparse=False                    # Whether to produce sparse output.
    )
    
    # Create SOAP descriptors for all structures. Uses all available CPU cores (n_jobs=-1).
    descriptors = soap_generator.create(structures, n_jobs=-1)
    print(f"SOAP calculation complete. Descriptor matrix shape: {descriptors.shape}")
    return descriptors

def select_fps(structures: list, n_select: int, fps_params: dict, soap_params: dict):
    """
    Selects a diverse subset of structures using Farthest Point Sampling (FPS).

    FPS iteratively selects structures that are farthest from the already selected set
    in the SOAP descriptor space, maximizing diversity.
    """
    print("Starting Farthest Point Sampling (FPS)...")
    if not structures:
        print("Warning: No structures provided for FPS selection. Returning empty list.")
        return []
    if len(structures) <= n_select:
        print("Warning: Number of candidates is less than or equal to the number to select. "
              "Returning all candidates.")
        return structures

    # Calculate SOAP descriptors to represent each structure numerically.
    descriptors = _calculate_soap_descriptors(structures, soap_params)
    n_candidates = descriptors.shape[0]
    
    # Determine the distance metric for comparing descriptors (e.g., 'euclidean' or 'cosine').
    metric_name = fps_params.get('distance_metric', 'euclidean')
    distance_func = cosine_distances if metric_name == 'cosine' else euclidean_distances
        
    # Initialize FPS by selecting a random structure as the first point.
    selected_indices = [np.random.randint(n_candidates)]
    remaining_indices = list(range(n_candidates))
    remaining_indices.remove(selected_indices[0])

    # Calculate initial minimum distances from all remaining points to the first selected point.
    # `dist_to_set[i]` stores the minimum distance of `remaining_indices[i]` to any already selected point.
    selected_descs = descriptors[selected_indices] # Descriptors of already selected structures
    dist_to_set = distance_func(descriptors[remaining_indices], selected_descs).min(axis=1)

    # Iteratively select the farthest points.
    for _ in tqdm(range(1, n_select), desc="FPS Selection"):
        if not remaining_indices: # Stop if no more points to select
            break

        # Find the point in the remaining set that is farthest from the already selected set.
        idx_in_remaining_of_farthest = np.argmax(dist_to_set)
        global_idx_of_farthest = remaining_indices.pop(idx_in_remaining_of_farthest)
        
        selected_indices.append(global_idx_of_farthest)

        if len(selected_indices) == n_select:
            break # Reached the target number of structures.
            
        # Update distances for the next iteration.
        # Remove the entry for the just-selected point from dist_to_set.
        dist_to_set = np.delete(dist_to_set, idx_in_remaining_of_farthest)
        
        if not remaining_indices: # Check again after pop and delete
            break

        # Calculate distances from the remaining points to the *newly added* point.
        newly_selected_desc = descriptors[global_idx_of_farthest].reshape(1, -1)
        dist_to_newly_selected = distance_func(descriptors[remaining_indices], newly_selected_desc).flatten()
        
        # Update the minimum distances to the selected set.
        dist_to_set = np.minimum(dist_to_set, dist_to_newly_selected)

    # Retrieve the selected ASE Atoms objects based on their indices.
    selected_structures_fps = [structures[i] for i in selected_indices]
    print(f"FPS complete. Selected {len(selected_structures_fps)} structures.")
    return selected_structures_fps


def select_pca_grid(structures: list, n_select: int, pca_grid_params: dict, soap_params: dict):
    """
    Selects structures using a SOAP descriptor, PCA, and gridding strategy.

    This method aims to select structures that are representative of different regions
    in a low-dimensional space defined by Principal Component Analysis (PCA) of SOAP descriptors.
    The `n_select` parameter acts as a target; the actual number selected depends on
    the number of non-empty grid cells.
    """
    print("Starting SOAP+PCA+Gridding selection...")
    if not structures:
        print("Warning: No structures provided for PCA-Grid selection. Returning empty list.")
        return []
        
    descriptors = _calculate_soap_descriptors(structures, soap_params)

    # 1. Perform PCA for dimensionality reduction of SOAP descriptors.
    n_components = pca_grid_params.get('n_components', 2) # Number of principal components.
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(descriptors)
    print(f"PCA complete. Explained variance by {n_components} components: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    # 2. Establish a grid in the PCA space.
    grid_resolution = pca_grid_params['grid_size'] # e.g., [10, 10] for a 10x10 grid in 2D.
    if len(grid_resolution) != n_components:
        raise ValueError("grid_size must have the same number of dimensions as n_components.")

    # Determine the data range for each principal component to define grid boundaries.
    pc_min_values = principal_components.min(axis=0)
    pc_max_values = principal_components.max(axis=0)

    # Create bins (edges of grid cells) for each principal component dimension.
    grid_bins_per_dim = [np.linspace(pc_min_values[i], pc_max_values[i], grid_resolution[i] + 1) for i in range(n_components)]
    
    # 3. Assign each structure to a grid cell based on its PCA coordinates.
    # `grid_map` will store: {grid_cell_tuple: [list_of_structure_indices_in_cell]}
    grid_map = {}
    
    # Digitize PC coordinates into grid cell indices.
    binned_pc_indices = np.zeros_like(principal_components, dtype=int)
    for i in range(n_components):
        # `np.digitize` returns 1-based indices; convert to 0-based.
        binned_pc_indices[:, i] = np.digitize(principal_components[:, i], grid_bins_per_dim[i]) - 1
        # Ensure indices are within grid bounds [0, grid_resolution-1].
        binned_pc_indices[:, i] = np.clip(binned_pc_indices[:, i], 0, grid_resolution[i] - 1)

    # Populate the grid_map.
    for struct_idx, pc_coords_binned in enumerate(binned_pc_indices):
        cell_id_tuple = tuple(pc_coords_binned)
        if cell_id_tuple not in grid_map:
            grid_map[cell_id_tuple] = []
        grid_map[cell_id_tuple].append(struct_idx)

    # 4. Select one structure randomly from each non-empty grid cell.
    selected_indices_pca_grid = []
    for cell_content_indices in grid_map.values():
        if cell_content_indices: # If the cell is not empty
            selected_index_from_cell = np.random.choice(cell_content_indices)
            selected_indices_pca_grid.append(selected_index_from_cell)

    # Note: The actual number of selected structures might be less than `n_select`
    # if the number of non-empty grid cells is less than `n_select`.
    # Or it could be more if n_select is very small relative to grid cells.
    # This method prioritizes coverage over exact count matching `n_select`.
    
    selected_structures_pca_grid = [structures[i] for i in selected_indices_pca_grid]
    print(f"PCA Gridding complete. Selected {len(selected_structures_pca_grid)} structures "
          f"from {len(grid_map)} non-empty grid cells.")
    return selected_structures_pca_grid