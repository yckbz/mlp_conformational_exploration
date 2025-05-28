# filters.py

import numpy as np
from ase.geometry import get_distances

class SanityFilters:
    """
    Applies various sanity checks to an atomic structure to ensure physical plausibility.
    """
    def __init__(self, config: dict, potential, relaxed_seed_energy: float):
        """
        Initializes the filters with settings from the configuration.

        Args:
            config (dict): Filter settings from the config.yaml file.
            potential (DeepMDPotential): The calculator object for energy/force evaluations.
            relaxed_seed_energy (float): Energy of the initial, relaxed seed structure,
                                         used as a baseline for the max energy check.
        """
        self.config = config
        self.potential = potential # Storing potential, though not directly used in current filter methods post-ASE cache.
        self.relaxed_seed_energy = relaxed_seed_energy

    def is_valid(self, atoms) -> bool:
        """
        Runs all enabled checks on the ASE Atoms object.
        Expects a calculator to be attached to 'atoms'.
        This function triggers a single calculation (energy and forces)
        and caches the results for subsequent checks.
        """
        try:
            # This call ensures energy and forces are computed once and cached.
            # Subsequent calls to .get_potential_energy() or .get_forces() in the
            # individual check methods will use these cached results.
            atoms.get_forces()
        except Exception as e:
            # If the potential evaluation fails (e.g., for a highly unstable structure),
            # the structure is considered invalid.
            # print(f"Debug: Calculation failed for a structure. Error: {e}") # Optional debug
            return False

        # Run specific checks if they are enabled in the configuration.
        if self.config['min_distance_check']['enabled']:
            if not self._check_min_distance(atoms):
                return False

        if self.config['max_energy_check']['enabled']:
            if not self._check_max_energy(atoms):
                return False

        if self.config['max_force_check']['enabled']:
            if not self._check_max_force(atoms):
                return False

        return True
        
    def _check_min_distance(self, atoms) -> bool:
        """
        Checks if any two atoms are closer than specified minimum distances.
        """
        params = self.config['min_distance_check']
        dist_matrix_config = params['min_dist_matrix']
        symbols = atoms.get_chemical_symbols()
        num_atoms = len(atoms)

        if num_atoms < 2:
            return True # No pairs to check.

        # Efficiently get unique pairs of indices for distance checking.
        i_indices, j_indices = np.triu_indices(num_atoms, k=1)
        
        # Get all distances at once for efficiency.
        all_distances_matrix = atoms.get_all_distances()
        distances_for_pairs = all_distances_matrix[i_indices, j_indices]

        for i, j, d in zip(i_indices, j_indices, distances_for_pairs):
            symbol_pair1 = f"{symbols[i]}-{symbols[j]}"
            symbol_pair2 = f"{symbols[j]}-{symbols[i]}" # Check both orders, e.g., Au-C and C-Au
            
            min_dist = dist_matrix_config.get(symbol_pair1, dist_matrix_config.get(symbol_pair2))
            
            if min_dist is not None and d < min_dist:
                return False
        return True

    def _check_max_energy(self, atoms) -> bool:
        """
        Checks if the structure's energy per atom exceeds a threshold
        relative to the relaxed seed's energy per atom.
        """
        params = self.config['max_energy_check']
        num_atoms = len(atoms)
        if num_atoms == 0:
            return True # Avoid division by zero if atoms object is empty.
        
        # Calculate energy per atom for the relaxed seed structure.
        relaxed_epa = self.relaxed_seed_energy / num_atoms
        # Determine the maximum allowed energy per atom for the current structure.
        threshold_epa = relaxed_epa + params['energy_tolerance']
        
        current_epa = atoms.get_potential_energy() / num_atoms
        
        if current_epa > threshold_epa:
            return False
        return True

    def _check_max_force(self, atoms) -> bool:
        """
        Checks if any component of the force on any atom exceeds a defined threshold.
        """
        params = self.config['max_force_check']
        max_allowed_force_component = params['max_force_component']
        forces_on_atoms = atoms.get_forces()
        
        # Check if the absolute value of any force component is too large.
        if np.any(np.abs(forces_on_atoms) > max_allowed_force_component):
            return False
        return True