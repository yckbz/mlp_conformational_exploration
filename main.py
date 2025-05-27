# main.py

import os
import shutil
import numpy as np

# We need to import these here for the main function
from utils import load_config, load_seeds, save_structures
from potential import DeepMDPotential
from filters import SanityFilters
from sampling import run_monte_carlo, run_random_walk
from selection import select_fps, select_pca_grid
from ase.optimize import LBFGS

def main():
    """
    Main execution script for the conformational exploration workflow.
    """
    print("--- Script started. ---", flush=True)

    # --- Initialization ---
    print("Loading configuration file...", flush=True)
    config = load_config('config.yaml')
    print("Configuration loaded.", flush=True)

    print("Loading potential model (this may take a moment)...", flush=True)
    potential = DeepMDPotential(config['initial_potential'])
    print("Potential model loaded.", flush=True)

    print("Loading seed files...", flush=True)
    seeds = load_seeds(config['seed_files'])
    print(f"Found {len(seeds)} seed files.", flush=True)
    
    if not seeds:
        print("Error: No valid seed structures were loaded. Exiting.", flush=True)
        return

    all_selected_structures = []

    # --- Process Each Seed Independently ---
    for i, seed in enumerate(seeds):
        print(f"\n{'='*20} Processing Seed {i+1}/{len(seeds)} ({seed.get_chemical_formula()}) {'='*20}", flush=True)
        
        # 1. Initial Relaxation to get baseline energy for filters
        print("Performing initial relaxation of the seed structure...", flush=True)
        seed_relaxed = seed.copy()
        
        seed_relaxed.calc = potential
        
        optimizer = LBFGS(seed_relaxed, logfile=None)
        optimizer.run(fmax=0.05, steps=50)
        
        relaxed_energy = float(seed_relaxed.get_potential_energy())

        print(f"Relaxed seed energy: {relaxed_energy:.4f} eV", flush=True)

        # 2. Setup Filters for this specific seed
        print("Setting up filters...", flush=True)
        filters = SanityFilters(config['filters'], potential, relaxed_energy)
        print("Filters are ready.", flush=True)

        # 3. Sampling
        sampler_mode = config['sampling_mode']
        print(f"Starting sampling with mode: {sampler_mode}...", flush=True)
        if sampler_mode == 'monte_carlo':
            candidate_pool = run_monte_carlo(seed_relaxed, potential, filters, config['mc_params'])
        elif sampler_mode == 'random_walk':
            candidate_pool = run_random_walk(seed_relaxed, potential, filters, config['rw_params'])
        else:
            raise ValueError(f"Unknown sampling_mode: {sampler_mode}")

        if not candidate_pool:
            print(f"Warning: No valid candidate structures were generated for seed {i+1}. Skipping.", flush=True)
            continue
        print(f"Generated {len(candidate_pool)} valid candidates for seed {i+1}.", flush=True)

        # 4. Selection
        selection_mode = config['selection_method']
        print(f"Starting selection with method: {selection_mode}...", flush=True)
        n_select = config['total_structures_to_select']
        
        if selection_mode == 'fps':
            selected_for_seed = select_fps(candidate_pool, n_select, config['fps_params'], config['soap_params'])
        elif selection_mode == 'pca_grid':
            selected_for_seed = select_pca_grid(candidate_pool, n_select, config['pca_grid_params'], config['soap_params'])
        else:
            raise ValueError(f"Unknown selection_method: {selection_mode}")
            
        all_selected_structures.extend(selected_for_seed)
        print(f"Completed processing for seed {i+1}.", flush=True)

    # --- Final Output ---
    print("\n--- Workflow finalizing... ---", flush=True)
    if all_selected_structures:
        output_file = config['output_file']
        if os.path.exists(output_file):
            os.remove(output_file)
        save_structures(all_selected_structures, output_file)
        print(f"Workflow complete. A total of {len(all_selected_structures)} structures have been saved to {output_file}.", flush=True)
    else:
        print("\nWorkflow complete, but no structures were selected.", flush=True)

if __name__ == '__main__':
    # This is the very first thing that runs
    print("--- main.py execution started ---", flush=True)
    main()