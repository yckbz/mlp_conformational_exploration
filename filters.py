# filters.py

import numpy as np
from ase.geometry import get_distances
from ase.calculators.singlepoint import SinglePointCalculator

class SanityFilters:
    """
    A class to apply various sanity checks on an atomic structure.
    """
    def __init__(self, config: dict, potential, relaxed_seed_energy: float):
        """
        Initializes the filters based on the configuration.

        Args:
            config (dict): The filter settings from the config file.
            potential (DeepMDPotential): The potential object for energy/force evals.
            relaxed_seed_energy (float): The energy of the initial, relaxed seed.
        """
        self.config = config
        self.potential = potential
        self.relaxed_seed_energy = relaxed_seed_energy
    def is_valid(self, atoms) -> bool:
        """
        Runs all enabled checks on the given ASE Atoms object.
        The atoms object is expected to have a calculator attached. This function
        will trigger the calculation.
        """
        try:
            atoms.get_forces()
        except Exception as e:
            return False

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
        Checks if any two atoms are closer than a defined minimum distance.
        """
        params = self.config['min_distance_check']
        dist_matrix = params['min_dist_matrix']
        symbols = atoms.get_chemical_symbols()
        num_atoms = len(atoms)

        if num_atoms < 2:
            return True

        # Using np.triu_indices is a very robust way to get unique pairs of indices
        # without relying on the exact return signature of get_distances
        i_indices, j_indices = np.triu_indices(num_atoms, k=1)
        
        # This is more efficient than calculating all distances if not needed
        all_distances = atoms.get_all_distances()
        distances = all_distances[i_indices, j_indices]

        for i, j, d in zip(i_indices, j_indices, distances):
            symbol_pair1 = f"{symbols[i]}-{symbols[j]}"
            symbol_pair2 = f"{symbols[j]}-{symbols[i]}"
            
            min_dist = dist_matrix.get(symbol_pair1, dist_matrix.get(symbol_pair2))
            
            if min_dist and d < min_dist:
                return False
        return True

    def _check_max_energy(self, atoms) -> bool:
        """
        Checks if the energy per atom exceeds a defined threshold.
        """
        params = self.config['max_energy_check']
        num_atoms = len(atoms)
        if num_atoms == 0: return True
        
        # The threshold is defined relative to the relaxed seed's energy per atom
        relaxed_e_per_atom = self.relaxed_seed_energy / num_atoms
        threshold_e_per_atom = relaxed_e_per_atom + params['energy_tolerance']
        
        # Use the standard ASE method to get the energy from the attached calculator
        current_e_per_atom = atoms.get_potential_energy() / num_atoms
        
        if current_e_per_atom > threshold_e_per_atom:
            return False
        return True

    def _check_max_force(self, atoms) -> bool:
        """
        Checks if any component of the force on any atom exceeds a threshold.
        """
        params = self.config['max_force_check']
        max_f = params['max_force_component']
        forces = atoms.get_forces()
        
        if np.any(np.abs(forces) > max_f):
            # print(f"DEBUG: Max force fail: Found force component > {max_f}")
            return False
        return True
