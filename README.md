# Conformation Explorer for MLP Training Sets

## Overview

This project provides a modular and configurable Python workflow designed to explore the potential energy surface (PES) of a given molecular system. The core goal is to efficiently sample and select a diverse set of conformations for use in training high-quality Machine Learning Potentials (MLPs) like DeepMD.

Starting with an initial potential (`.pb` file) and one or more seed structures, the workflow generates thousands of candidate structures, filters out unrealistic ones in real-time, and then uses sophisticated selection algorithms to create a final, information-rich, and minimally sized training set[cite: 1].

## Features

- **Multiple Input Formats**: Supports reading initial structures from `POSCAR`, extended `XYZ`, and `CIF` files[cite: 1].
- **Advanced Sampling Engines**:
    - **Metropolis Monte Carlo (MC)**: Samples equilibrium conformations at various user-defined temperatures[cite: 1].
    - **Potential Energy Surface Random Walk**: Explores local minima by applying large perturbations followed by energy minimization[cite: 1].
- **Real-time Sanity Filters**:
    - **Minimum Atomic Distance Check**: Rejects structures where atoms are unrealistically close[cite: 1, 2].
    - **Maximum Energy Check**: Prevents exploration of excessively high-energy regions[cite: 1].
    - **Maximum Force Check**: Avoids unstable structures where forces are too high[cite: 1].
- **Intelligent Selection Algorithms**:
    - **Farthest Point Sampling (FPS)**: Selects structures to maximize geometric diversity[cite: 3].
    - **SOAP+PCA+Gridding**: Ensures uniform coverage across the primary dimensions of conformational space[cite: 5].
- **Modular and Configurable**: All parameters are controlled via a single `config.yaml` file for easy tuning and experimentation.

## Project Structure
mlp_conformational_exploration/
├── seeds/                  # Directory for initial seed structures
│   ├── seed1.cif
│   └── ...
├── graph.pb                # Input DeepMD potential file
├── config.yaml             # Main configuration file
├── main.py                 # Main execution script
├── potential.py            # Wrapper for deepmd-kit
├── sampling.py             # MC and Random Walk samplers
├── filters.py              # Sanity check functions
├── selection.py            # FPS and PCA+Grid selection algorithms
├── utils.py                # Helper functions for I/O
└── README.md               # This documentation file

## Setup and Installation

1.  **Clone the repository (or create the file structure above).**

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install "deepmd-kit[gpu]" # or "deepmd-kit" for CPU
    pip install ase numpy scikit-learn dscribe pyyaml tqdm
    ```

## How to Run

1.  **Prepare Inputs**:
    * Place your initial frozen model file (e.g., `graph.pb`) in the root directory[cite: 1].
    * Place your seed structure files (e.g., `seed1.cif`, `molecule.xyz`) inside the `seeds/` directory.

2.  **Configure the Workflow**:
    * Open `config.yaml` and adjust the parameters to suit your system and goals. Key parameters include `sampling_mode`, `selection_method`, `temperatures` (for MC), and `total_structures_to_select`. See the section below for full details.

3.  **Execute the Script**:
    * Run the main script from the terminal:
    ```bash
    python main.py
    ```

4.  **Get a cup of coffee.** The process, especially the sampling and descriptor calculation, can be computationally intensive.

5.  **Check the Output**:
    * The final selected structures from all seeds will be combined and saved to the file specified by `output_file` in `config.yaml` (e.g., `selected_structures.xyz`)[cite: 1].

## Configuration Details (`config.yaml`)

#### General Settings
- `initial_potential`: Path to your `.pb` model file.
- `seed_files`: A list of paths to your initial structure files.
- `output_file`: Name of the final extended XYZ file.

#### Sampling (`sampling_mode`)
- **`'monte_carlo'`**[cite: 1]:
  - `temperatures`: A list of temperatures (in Kelvin) to run simulations at[cite: 1].
  - `max_displacement`: The maximum distance (in Å) an atom can be moved in a single step[cite: 1].
  - `num_steps`: Number of MC steps to run *per temperature*.
- **`'random_walk'`**[cite: 1]:
  - `perturbation_magnitude`: The strength of the random displacement applied to all atoms before minimization[cite: 1].
  - `optimization_fmax`: The force convergence criterion for the local energy minimization[cite: 1].
  - `num_walks`: The total number of walk/relaxation cycles to perform[cite: 1].

#### Filters (`filters`)
- **`min_distance_check`**[cite: 1]:
  - `enabled`: `true` or `false`.
  - `min_dist_matrix`: A dictionary defining the minimum allowed distance (in Å) between pairs of elements[cite: 1].
- **`max_energy_check`**[cite: 1]:
  - `enabled`: `true` or `false`.
  - `energy_tolerance`: The maximum allowed energy *per atom* (in eV) above the initial relaxed seed's energy[cite: 1].
- **`max_force_check`**[cite: 1]:
  - `enabled`: `true` or `false`.
  - `max_force_component`: The maximum allowed value for any single component (x, y, or z) of the force vector on any atom (in eV/Å)[cite: 1].

#### Selection (`selection_method`)
- `total_structures_to_select`: The target number of structures to select from each seed's candidate pool[cite: 4].
- **`'fps'` (Farthest Point Sampling)**[cite: 3]:
  - `distance_metric`: `'euclidean'` or `'cosine'` for comparing SOAP fingerprints[cite: 3].
- **`'pca_grid'` (SOAP+PCA+Gridding)**[cite: 5]:
  - `n_components`: The number of principal components to use (typically 2 or 3)[cite: 5].
  - `grid_size`: A list defining the number of bins for each principal component (e.g., `[10, 10]` for a 10x10 grid in 2D)[cite: 5].

#### SOAP Settings (`soap_params`)
- These parameters define the SOAP "fingerprint" used to compare structures.
- `species`: A list of all chemical element symbols present in your system.
- `rcut`: The cutoff radius for atomic environments.
- `nmax`, `lmax`: The number of radial and angular basis functions.

