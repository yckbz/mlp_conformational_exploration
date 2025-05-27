# Conformation Explorer for MLP Training Sets

## Overview

This project provides a modular and configurable Python workflow designed to explore the potential energy surface (PES) of a given molecular system. The core goal is to efficiently sample and select a diverse set of conformations for use in training high-quality Machine Learning Potentials (MLPs) like DeepMD.

Starting with an initial potential (`.pb` file) and one or more seed structures, the workflow generates thousands of candidate structures, filters out unrealistic ones in real-time, and then uses sophisticated selection algorithms to create a final, information-rich, and minimally sized training set.

## Features

- **Multiple Input Formats**: Supports reading initial structures from `POSCAR`, extended `XYZ`, and `CIF` files.
- **Advanced Sampling Engines**:
    - **Molecular Dynamics (MD)**: Generates non-equilibrium conformations by simulating system dynamics at various temperatures and pressures.
    - **Metropolis Monte Carlo (MC)**: Samples equilibrium conformations through random trial moves.
    - **Potential Energy Surface Random Walk**: Explores local minima by applying large perturbations followed by energy minimization.
- **NVT & NPT Ensemble Support**: Both MD and MC samplers support simulations in the canonical (NVT) and isothermal-isobaric (NPT) ensembles for comprehensive thermodynamic exploration.
- **Real-time Sanity Filters**:
    - **Minimum Atomic Distance Check**: Rejects structures where atoms are unrealistically close.
    - **Maximum Energy Check**: Prevents exploration of excessively high-energy regions.
    - **Maximum Force Check**: Avoids unstable structures where forces are too high.
- **Intelligent Selection Algorithms**:
    - **Farthest Point Sampling (FPS)**: Selects a subset of structures that maximizes geometric diversity.
    - **SOAP+PCA+Gridding**: Ensures uniform coverage across the primary dimensions of conformational space.
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
├── sampling.py             # MD, MC, and Random Walk samplers
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
    * Place your initial frozen model file (e.g., `graph.pb`) in the root directory.
    * Place your seed structure files (e.g., `seed1.cif`, `molecule.xyz`) inside the `seeds/` directory.

2.  **Configure the Workflow**:
    * Open `config.yaml` and adjust the parameters to suit your system and goals. See the detailed configuration section below.

3.  **Execute the Script**:
    * Run the main script from the terminal:
    ```bash
    python main.py
    ```

4.  **Check the Output**:
    * The final selected structures will be combined and saved to the file specified by `output_file` in the configuration (e.g., `selected_structures.xyz`).

## Configuration Details (`config.yaml`)

#### General Settings
- `initial_potential`: Path to your `.pb` model file.
- `seed_files`: A list of paths to your initial structure files.
- `output_file`: Name of the final extended XYZ file for the selected structures.

---
#### Sampling Settings

This section controls the generation of candidate structures.

-   `sampling_mode`: The main switch for the sampling engine. Options are:
    -   `'molecular_dynamics'`
    -   `'monte_carlo'`
    -   `'random_walk'`

##### Molecular Dynamics (`md_params`)
Generates non-equilibrium structures through direct simulation. Ideal for creating robust training data.
-   `ensemble`: `'NVT'` for constant volume or `'NPT'` for constant pressure simulations.
-   `temperatures`: A single temperature or a list of temperatures (in Kelvin) to simulate.
-   `pressures`: (For NPT only) A single pressure or a list of pressures (in bar). The script will run a simulation for every combination of temperature and pressure.
-   `n_steps`: Total number of simulation steps to run for each (T, P) condition.
-   `timestep`: Timestep for the simulation in femtoseconds.
-   `save_interval`: The interval at which to save a structure. A snapshot is saved every `save_interval` steps.

##### Monte Carlo (`mc_params`)
Samples equilibrium configurations using random trial moves.
-   `ensemble`: `'NVT'` or `'NPT'`.
-   `temperatures`: A single temperature or a list of temperatures (in Kelvin).
-   `pressures`: (For NPT only) A single pressure or a list of pressures (in bar). The script runs a simulation for every T/P combination.
-   `num_steps`: Number of trial moves to attempt for each (T, P) condition.
-   `max_displacement`: The maximum distance (in Å) an atom can be moved in a single step.
-   `max_volume_change_fraction`: (For NPT only) The maximum fractional change in volume for a volume-change move.

##### Random Walk (`rw_params`)
Explores local minima on the potential energy surface.
-   `perturbation_magnitude`: The strength of the random displacement (in Å) applied to all atoms before minimization.
-   `optimization_fmax`: The force convergence criterion (in eV/Å) for the local energy minimization.
-   `num_walks`: The total number of perturbation/relaxation cycles to perform.

---
#### Filter Settings (`filters`)
These checks are applied in real-time to discard physically unrealistic structures during sampling.

-   `min_distance_check`:
    -   `enabled`: `true` or `false`.
    -   `min_dist_matrix`: A dictionary defining the minimum allowed distance (in Å) between pairs of elements (e.g., `'Au-Au': 2.0`).
-   `max_energy_check`:
    -   `enabled`: `true` or `false`.
    -   `energy_tolerance`: The maximum allowed energy *per atom* (in eV) above the initial relaxed seed's energy.
-   `max_force_check`:
    -   `enabled`: `true` or `false`.
    -   `max_force_component`: The maximum allowed value for any single force component (x, y, or z) on any atom (in eV/Å).

---
#### Selection Settings (`selection_method`)
After a large pool of candidates is generated, these algorithms select a smaller, diverse subset for the final output.

-   `selection_method`: The algorithm to use. Options are `'fps'` or `'pca_grid'`.
-   `total_structures_to_select`: The target number of structures to save in the final output file.

##### Farthest Point Sampling (`fps_params`)
-   `distance_metric`: `'euclidean'` or `'cosine'` for comparing structure fingerprints.

##### SOAP+PCA+Gridding (`pca_grid_params`)
-   `n_components`: The number of principal components to use (typically 2 or 3).
-   `grid_size`: A list defining the number of bins for each principal component (e.g., `[10, 10]` for a 10x10 grid).

---
#### SOAP Descriptor Settings (`soap_params`)
These parameters define the SOAP "fingerprint" used by the selection algorithms to compare structures.

-   `species`: A list of all chemical element symbols present in your system.
-   `r_cut`: The cutoff radius for atomic environments.
-   `n_max`, `l_max`: The number of radial and angular basis functions.
