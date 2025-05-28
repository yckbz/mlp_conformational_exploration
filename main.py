# main.py

import os
import numpy as np

# Workflow modules
from utils import load_config, load_seeds, save_structures
from potential import DeepMDPotential
from filters import SanityFilters
from sampling import run_monte_carlo, run_random_walk, run_molecular_dynamics
from selection import select_fps, select_pca_grid

# ASE modules
from ase.optimize import LBFGS


def main():
    """
    Main execution script for the conformational exploration workflow.
    Orchestrates loading, sampling, filtering, and selection of structures.
    """
    print("--- Script started. ---", flush=True)

    # --- Initialization ---
    # Load configuration, potential model, and initial seed structures.
    config = load_config('config.yaml')
    potential = DeepMDPotential(config['initial_potential']) #
    seeds = load_seeds(config['seed_files']) #
    
    if not seeds:
        print("Error: No valid seed structures were loaded. Exiting.", flush=True)
        return

    all_candidates_pool = [] # Used if selection_scope is 'global'
    all_selected_structures = [] # Final list of structures to be saved
    
    # Determine if selection is done per seed or globally on all candidates.
    selection_scope = config.get('selection_scope', 'per_seed') #

    # --- Process Each Seed ---
    # The workflow is repeated for each seed structure provided.
    for i, seed_structure in enumerate(seeds):
        print(f"\n{'='*20} Processing Seed {i+1}/{len(seeds)} ({seed_structure.get_chemical_formula()}) {'='*20}", flush=True)
        
        # 1. Initial Relaxation of the seed structure
        # Provides a consistent starting point and baseline energy for filters.
        print("Performing initial relaxation of the seed structure...", flush=True)
        seed_relaxed = seed_structure.copy()
        seed_relaxed.calc = potential
        optimizer = LBFGS(seed_relaxed, logfile=None) # Using ASE's LBFGS optimizer
        optimizer.run(fmax=0.05, steps=50) # Relax until force criteria met or max steps
        relaxed_energy = float(seed_relaxed.get_potential_energy())
        print(f"Relaxed seed energy: {relaxed_energy:.4f} eV", flush=True)

        # 2. Setup Filters
        # Filters are specific to each seed, using its relaxed energy as a reference.
        filters = SanityFilters(config['filters'], potential, relaxed_energy) #

        # 3. Sampling
        # Generate a pool of candidate structures using the chosen sampling method.
        sampler_mode = config['sampling_mode'] #
        print(f"Starting sampling with mode: {sampler_mode}...", flush=True)
        
        candidate_pool = []
        if sampler_mode == 'monte_carlo':
            candidate_pool = run_monte_carlo(seed_relaxed, potential, filters, config['mc_params']) #
        elif sampler_mode == 'random_walk':
            candidate_pool = run_random_walk(seed_relaxed, potential, filters, config['rw_params']) #
        elif sampler_mode == 'molecular_dynamics':
            candidate_pool = run_molecular_dynamics(seed_relaxed, potential, filters, config['md_params']) #
        else:
            raise ValueError(f"Unknown sampling_mode: {sampler_mode}")

        if not candidate_pool:
            print(f"Warning: No valid candidate structures were generated for seed {i+1}. Skipping.", flush=True)
            continue
        print(f"Generated {len(candidate_pool)} valid candidates for seed {i+1}.", flush=True)

        # --- Selection Logic ---
        if selection_scope == 'per_seed':
            # 4a. Select a subset of structures from the current seed's candidate pool.
            print(f"Starting per-seed selection...", flush=True)
            n_select = config['total_structures_to_select'] #
            selection_method = config['selection_method'] #

            selected_for_this_seed = []
            if selection_method == 'fps':
                selected_for_this_seed = select_fps(candidate_pool, n_select, config['fps_params'], config['soap_params']) #
            elif selection_method == 'pca_grid':
                selected_for_this_seed = select_pca_grid(candidate_pool, n_select, config['pca_grid_params'], config['soap_params']) #
            else:
                raise ValueError(f"Unknown selection_method: {selection_method}")
            all_selected_structures.extend(selected_for_this_seed)
            print(f"Selected {len(selected_for_this_seed)} structures from seed {i+1}.", flush=True)
        
        elif selection_scope == 'global':
            # 4b. Add all candidates from this seed to a global pool for later selection.
            all_candidates_pool.extend(candidate_pool)
            print(f"Added {len(candidate_pool)} candidates from seed {i+1} to the global pool.", flush=True)
        
        print(f"Completed processing for seed {i+1}.", flush=True)

    # --- Global Selection (if applicable) ---
    if selection_scope == 'global':
        if not all_candidates_pool:
             print("\nWarning: No candidates were generated in total for global selection. Exiting.", flush=True)
        else:
            print(f"\n--- Starting global selection on {len(all_candidates_pool)} total candidates... ---", flush=True)
            n_select = config['total_structures_to_select'] #
            selection_method = config['selection_method'] #

            if selection_method == 'fps':
                all_selected_structures = select_fps(all_candidates_pool, n_select, config['fps_params'], config['soap_params']) #
            elif selection_method == 'pca_grid':
                all_selected_structures = select_pca_grid(all_candidates_pool, n_select, config['pca_grid_params'], config['soap_params']) #
            else:
                raise ValueError(f"Unknown selection_method: {selection_method}")
            print(f"Global selection complete. Selected {len(all_selected_structures)} structures.", flush=True)
            
    # --- Final Output ---
    # Save all selected structures (either from per-seed or global selection) to a file.
    print("\n--- Workflow finalizing... ---", flush=True)
    if all_selected_structures:
        output_file = config['output_file'] #
        # Remove existing output file to prevent appending to old results by default.
        if os.path.exists(output_file):
            os.remove(output_file)
        save_structures(all_selected_structures, output_file)
        print(f"Workflow complete. A total of {len(all_selected_structures)} structures have been saved to {output_file}.", flush=True)
    else:
        print("\nWorkflow complete, but no structures were selected from any seed.", flush=True)

if __name__ == '__main__':
    # Entry point of the script.
    print("--- main.py execution started ---", flush=True) # This is the very first print
    main()