# sampling.py

import numpy as np
from tqdm import tqdm
from itertools import product

from ase import units
from ase.optimize import LBFGS
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen # Using Berendsen NPT for stability
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


def _normalize_input(value):
    """Ensures the input (e.g., temperatures, pressures) is a list."""
    if isinstance(value, list):
        return value
    return [value]


def run_monte_carlo(atoms_initial, potential, filters, mc_params):
    """
    Performs Metropolis Monte Carlo (MC) sampling.
    Supports NVT and NPT ensembles, iterating over T and P combinations.
    """
    ensemble = mc_params.get('ensemble', 'NVT').upper()
    print(f"Starting Monte Carlo sampling in the {ensemble} ensemble...")

    temperatures = _normalize_input(mc_params['temperatures'])
    n_steps = mc_params['num_steps'] # Number of MC steps per condition
    max_disp = mc_params['max_displacement'] # Max atomic displacement for a trial move
    k_B = units.kB # Boltzmann constant from ASE

    candidate_structures = []

    # Prepare simulation conditions based on ensemble type
    if ensemble == 'NPT':
        pressures = _normalize_input(mc_params.get('pressures', [1.0])) # Default to 1.0 bar if not specified
        conditions = list(product(temperatures, pressures)) # All (T, P) pairs
        max_vol_change_frac = mc_params.get('max_volume_change_fraction', 0.1)
        move_ratio = 0.5 # Probability ratio for attempting particle vs. volume move
    else: # NVT
        conditions = temperatures # Only loop over temperatures

    # Main loop over each temperature or (temperature, pressure) condition
    for condition_params in conditions:
        atoms = atoms_initial.copy() # Start fresh for each condition
        atoms.calc = potential

        if ensemble == 'NPT':
            temp, pressure_bar = condition_params
            pressure_ase_units = pressure_bar * units.bar # Convert pressure to ASE's internal units
            print(f"Running MC at {temp} K and {pressure_bar} bar...")
        else: # NVT
            temp = condition_params
            print(f"Running MC at {temp} K...")

        accepted_moves = 0
        current_energy = atoms.get_potential_energy()

        # Perform MC steps for the current condition
        desc_text = f"MC @ {temp}K, {f'{pressure_bar}bar' if ensemble == 'NPT' else 'NVT'}"
        for _ in tqdm(range(n_steps), desc=desc_text):
            # Detach/reattach calculator to ensure clean state for trial moves
            atoms.calc = None
            trial_atoms = atoms.copy()
            atoms.calc = potential
            trial_atoms.calc = potential

            # Decide whether to attempt a volume change or particle displacement for NPT
            if ensemble == 'NPT' and np.random.rand() > move_ratio:
                # --- Volume Change Move (NPT) ---
                current_volume = atoms.get_volume()
                # Propose volume change on a log scale for better sampling
                volume_multiplier = np.exp(max_vol_change_frac * (np.random.rand() - 0.5))
                trial_volume = current_volume * volume_multiplier
                
                # Scale cell and atom positions isotropically
                trial_atoms.set_cell(atoms.get_cell() * (trial_volume / current_volume)**(1/3), scale_atoms=True)

                if not filters.is_valid(trial_atoms):
                    candidate_structures.append(atoms.copy()) # Save current state if trial invalid
                    continue
                
                trial_energy = trial_atoms.get_potential_energy()
                delta_energy = trial_energy - current_energy
                delta_volume = trial_volume - current_volume
                
                # NPT acceptance criterion (Metropolis)
                exponent_arg = -(delta_energy + pressure_ase_units * delta_volume - len(atoms) * k_B * temp * np.log(trial_volume / current_volume)) / (k_B * temp)

                if exponent_arg > 0 or np.random.rand() < np.exp(exponent_arg):
                    atoms, current_energy = trial_atoms, trial_energy # Accept move
                    accepted_moves += 1
            else:
                # --- Particle Displacement Move (NVT or NPT) ---
                atom_to_move = np.random.randint(len(trial_atoms))
                displacement_vector = (np.random.rand(3) - 0.5) * 2 * max_disp
                trial_atoms.positions[atom_to_move] += displacement_vector

                if not filters.is_valid(trial_atoms):
                    candidate_structures.append(atoms.copy()) # Save current state if trial invalid
                    continue
                
                trial_energy = trial_atoms.get_potential_energy()
                delta_energy = trial_energy - current_energy

                # NVT acceptance criterion (Metropolis)
                if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / (k_B * temp)):
                    atoms, current_energy = trial_atoms, trial_energy # Accept move
                    accepted_moves += 1
            
            candidate_structures.append(atoms.copy()) # Save current structure (accepted or rejected)

        print(f"Finished condition. Acceptance rate: {accepted_moves / n_steps:.2%}")

    return candidate_structures


def run_molecular_dynamics(atoms_initial, potential, filters, md_params):
    """
    Performs Molecular Dynamics (MD) sampling.
    Supports NVT (Langevin thermostat) and NPT (Berendsen barostat) ensembles.
    Iterates over all combinations of specified temperatures and pressures.
    """
    ensemble = md_params.get('ensemble', 'NVT').upper()
    print(f"Starting Molecular Dynamics sampling in the {ensemble} ensemble...")

    temperatures = _normalize_input(md_params['temperatures'])
    n_steps = md_params['n_steps']         # Total MD steps per condition
    timestep_fs = md_params['timestep']    # Timestep in femtoseconds
    save_interval = md_params['save_interval'] # Save a frame every this many steps

    candidate_structures = []

    # Prepare simulation conditions
    if ensemble == 'NPT':
        pressures = _normalize_input(md_params.get('pressures', [1.0])) # Default to 1.0 bar
        conditions = list(product(temperatures, pressures)) # All (T,P) pairs
    else: # NVT
        conditions = temperatures

    # Main loop over each temperature or (temperature, pressure) condition
    for condition_params in conditions:
        atoms = atoms_initial.copy() # Start fresh for each condition
        atoms.calc = potential
        
        current_temp_K = condition_params[0] if ensemble == 'NPT' else condition_params

        # Initialize atomic velocities to the target temperature
        # and remove any center-of-mass momentum for simulation stability.
        MaxwellBoltzmannDistribution(atoms, temperature_K=current_temp_K)
        Stationary(atoms) # Ensure zero center-of-mass momentum

        dyn = None # Dynamics object

        if ensemble == 'NPT':
            temp_K, pressure_bar = condition_params
            print(f"Running MD at {temp_K} K and {pressure_bar} bar...")

            # Coupling times for thermostat and barostat (fs)
            thermostat_coupling_fs = 100 * units.fs
            barostat_coupling_fs = 1000 * units.fs # Longer barostat coupling for stability
            
            # Estimate compressibility for Berendsen barostat (e.g., for Gold)
            # Bulk Modulus (B) ~180 GPa for Gold. Compressibility = 1/B.
            # Convert to ASE's atomic units: (eV/Angstrom^3)^-1
            bulk_modulus_GPa = 180.0
            bulk_modulus_ase_units = bulk_modulus_GPa * 1e9 / (units.Pascal * units.GPa) # Convert GPa to eV/A^3
            compressibility_ase_units = 1.0 / bulk_modulus_ase_units if bulk_modulus_ase_units > 0 else 0.0

            dyn = NPTBerendsen(
                atoms,
                timestep=timestep_fs * units.fs,
                temperature_K=temp_K,
                pressure_au=pressure_bar * units.bar, # Pressure in ASE units
                taut=thermostat_coupling_fs,
                taup=barostat_coupling_fs,
                compressibility_au=compressibility_ase_units,
                fixcm=True,  # Fix center-of-mass drift
            )
        else: # NVT
            temp_K = condition_params
            print(f"Running MD at {temp_K} K...")
            dyn = Langevin(
                atoms,
                timestep=timestep_fs * units.fs,
                temperature_K=temp_K,
                friction=0.02 # Standard friction coefficient for Langevin
            )

        # Define a callback function to collect structures during the MD run
        def collect_frame():
            # Check if the current structure is valid according to defined filters
            if filters.is_valid(atoms):
                candidate_structures.append(atoms.copy())

        dyn.attach(collect_frame, interval=save_interval) # Attach collector
        dyn.run(n_steps) # Run the MD simulation

    print(f"MD sampling complete. Generated {len(candidate_structures)} total candidate structures.")
    return candidate_structures


def run_random_walk(atoms_initial, potential, filters, rw_params):
    """
    Performs Potential Energy Surface (PES) Random Walk.
    Applies large perturbations followed by local energy minimization.
    """
    print("Starting Potential Energy Surface Random Walk...")
    perturbation_magnitude_A = rw_params['perturbation_magnitude'] # Angstrom
    num_walks = rw_params['num_walks']           # Number of independent walks
    opt_fmax = rw_params['optimization_fmax']    # Force tolerance for LBFGS

    valid_structures = []
    # Each walk starts from the initial (relaxed) seed structure independently.
    for i in tqdm(range(num_walks), desc="Random Walk"):
        current_atoms = atoms_initial.copy()
        current_atoms.calc = potential
        
        # Apply a large random perturbation to all atoms
        random_displacements = (np.random.rand(len(current_atoms), 3) - 0.5) * 2 * perturbation_magnitude_A
        current_atoms.positions += random_displacements
        
        # No need to re-attach calculator as it's on current_atoms, which is then passed to LBFGS
        optimizer = LBFGS(current_atoms, trajectory=None, logfile=None)
        
        try:
            # Perform local energy minimization
            optimizer.run(fmax=opt_fmax, steps=100) # Max 100 steps for optimization
        except Exception as e:
            # Catch any errors during optimization (e.g., if structure becomes too distorted)
            print(f"Warning: Optimization failed during walk {i+1}. Error: {e}. Skipping.")
            continue

        # Check if the newly found local minimum is valid
        if filters.is_valid(current_atoms):
            valid_structures.append(current_atoms.copy()) # Save a copy of the optimized structure

    print(f"Random Walk complete. Found {len(valid_structures)} valid structures.")
    return valid_structures