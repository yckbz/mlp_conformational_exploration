# sampling.py

import numpy as np
from ase.optimize import LBFGS
from ase.calculators.singlepoint import SinglePointCalculator
from tqdm import tqdm

# In sampling.py

def run_monte_carlo(atoms_initial, potential, filters, mc_params):
    """
    Performs a Metropolis Monte Carlo simulation using the ASE calculator interface.
    """
    print("Starting Monte Carlo sampling...")
    temperatures = mc_params['temperatures']
    max_disp = mc_params['max_displacement']
    n_steps = mc_params['num_steps']

    k_B = 8.617333262145e-5  # Boltzmann constant in eV/K

    valid_structures = []
    atoms = atoms_initial.copy()
    atoms.calc = potential
    energy_n = atoms.get_potential_energy()

    for temp in temperatures:
        print(f"Running MC at {temp} K...")
        accepted_moves = 0
        for _ in tqdm(range(n_steps), desc=f"MC @ {temp}K"):

            # --- THE FINAL FIX: Ensure state separation ---
            # 1. Detach the calculator to prevent sharing its state during the copy.
            atoms.calc = None
            # 2. Now the copy is clean; it's just positions and symbols.
            trial_atoms = atoms.copy()
            # 3. Re-attach the calculator to the main 'atoms' object and the new 'trial_atoms'.
            atoms.calc = potential
            trial_atoms.calc = potential
            # --- End of Fix ---

            # Generate a trial move
            atom_index = np.random.randint(len(trial_atoms))
            displacement = (np.random.rand(3) - 0.5) * 2 * max_disp
            trial_atoms.positions[atom_index] += displacement

            # The is_valid function will now run a fresh calculation on trial_atoms.
            if not filters.is_valid(trial_atoms):
                valid_structures.append(atoms.copy())
                continue

            energy_n_plus_1 = trial_atoms.get_potential_energy()
            delta_e = energy_n_plus_1 - energy_n

            # Metropolis acceptance criterion
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / (k_B * temp)):
                atoms = trial_atoms
                energy_n = energy_n_plus_1
                accepted_moves += 1

            valid_structures.append(atoms.copy())

        print(f"Finished MC at {temp} K. Acceptance rate: {accepted_moves / n_steps:.2%}")

    return valid_structures

def run_random_walk(atoms_initial, potential, filters, rw_params):
    """
    Performs a Potential Energy Surface Random Walk.
    """
    print("Starting Potential Energy Surface Random Walk...")
    pert_mag = rw_params['perturbation_magnitude']
    n_walks = rw_params['num_walks']
    fmax = rw_params['optimization_fmax']

    valid_structures = []
    current_structure = atoms_initial.copy()
    
    # Since our 'potential' is now a real ASE calculator, we can just attach it.
    current_structure.calc = potential

    for i in tqdm(range(n_walks), desc="Random Walk"):
        perturbed_structure = current_structure.copy()
        
        # Apply a large random perturbation to all atoms
        perturbation = (np.random.rand(len(perturbed_structure), 3) - 0.5) * 2 * pert_mag
        perturbed_structure.positions += perturbation
        
        # Attach the calculator to the new perturbed structure
        perturbed_structure.calc = potential
        
        # The optimizer can now use the attached calculator directly. No more hooks needed.
        optimizer = LBFGS(perturbed_structure, trajectory=None, logfile=None)
        
        try:
            optimizer.run(fmax=fmax, steps=100)
        except Exception as e:
            # This catch is still useful for any unexpected optimization errors.
            print(f"Warning: Optimization failed during walk {i+1}. Skipping. Error: {e}")
            continue

        # After successful optimization, check if the new local minimum is valid
        if filters.is_valid(perturbed_structure):
            valid_structures.append(perturbed_structure)
        
        # The new minimum becomes the starting point for the next walk
        current_structure = perturbed_structure.copy()

    return valid_structures
