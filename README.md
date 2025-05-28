# Conformation Explorer for MLP Training Sets

## Overview

This project provides a modular Python workflow for exploring the potential energy surface (PES) of molecular systems. It efficiently samples and selects diverse conformations for training Machine Learning Potentials (MLPs), such as those used in DeepMD. The process starts with an initial potential and seed structures, generates numerous candidate structures, applies real-time filters, and uses advanced selection algorithms to produce an information-rich, minimally sized training set.

## Features

- **Versatile Input**: Reads initial structures from `POSCAR`, extended `XYZ`, and `CIF` formats.
- **Sampling Engines**:
    - **Molecular Dynamics (MD)**: Simulates system dynamics in NVT or NPT ensembles to generate non-equilibrium conformations.
    - **Metropolis Monte Carlo (MC)**: Samples equilibrium conformations in NVT or NPT ensembles.
    - **Potential Energy Surface Random Walk**: Explores local minima through perturbation and minimization.
- **Sanity Filters**:
    - **Minimum Atomic Distance**: Rejects structures with unrealistically close atoms.
    - **Maximum Energy**: Avoids excessively high-energy regions.
    - **Maximum Force**: Discards unstable, high-force structures.
- **Structure Selection**:
    - **Farthest Point Sampling (FPS)**: Maximizes geometric diversity in the selected set.
    - **SOAP+PCA+Gridding**: Ensures uniform coverage across the conformational space defined by SOAP descriptors and Principal Component Analysis.
- **Flexible Selection Scope**:
    - **Per-Seed Selection**: Selects a target number of structures independently from each seed's candidate pool.
    - **Global Selection**: Pools all candidates from all seeds and selects a single diverse set.
- **Configurable Workflow**: All parameters are managed through a single `config.yaml` file.

## Project Structure

```
mlp_conformational_exploration/
├── seeds/                  # Directory for initial seed structures (e.g., seed1.cif)
│   ├── seed1.cif
│   └── ...
├── graph.pb                # Input DeepMD potential file
├── config.yaml             # Main configuration file
├── main.py                 # Main execution script
├── potential.py            # ASE wrapper for deepmd-kit
├── sampling.py             # MD, MC, and Random Walk samplers
├── filters.py              # Sanity check functions
├── selection.py            # FPS and PCA-Grid selection algorithms
├── utils.py                # Helper functions for I/O
└── README.md               # This documentation
```

## Setup and Installation

1.  **Clone the repository.**
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install "deepmd-kit[gpu]"  # or "deepmd-kit" for CPU-only
    pip install ase numpy scikit-learn dscribe pyyaml tqdm
    ```

## How to Run

1.  **Prepare Inputs**:
    * Place your initial model file (e.g., `graph.pb`) in the project root.
    * Add seed structure files (e.g., `seed1.cif`) to the `seeds/` directory.
2.  **Configure**: Edit `config.yaml` to define the workflow parameters.
3.  **Execute**: Run `python main.py` from the terminal.
4.  **Output**: Selected structures are saved to the file specified by `output_file` in `config.yaml`.

## Configuration Details (`config.yaml`)

#### General Settings
- `initial_potential`: Path to the `.pb` model file.
- `seed_files`: List of paths to initial structure files (e.g., `['seeds/seed1.cif', 'seeds/seed2.xyz']`).
- `output_file`: Name for the final XYZ file containing selected structures.

---
#### Sampling Settings

-   `sampling_mode`: Choose the sampling engine: `'molecular_dynamics'`, `'monte_carlo'`, or `'random_walk'`.

##### Molecular Dynamics (`md_params`)
For generating non-equilibrium structures.
-   `ensemble`: `'NVT'` (constant volume) or `'NPT'` (constant pressure).
-   `temperatures`: Single temperature or a list of temperatures (Kelvin).
-   `pressures`: (NPT only) Single pressure or a list of pressures (bar). Runs simulations for all T/P combinations.
-   `n_steps`: Total simulation steps per (T, P) condition.
-   `timestep`: Simulation timestep (femtoseconds).
-   `save_interval`: Interval to save structures (e.g., save every 20 steps).

##### Monte Carlo (`mc_params`)
For sampling equilibrium configurations.
-   `ensemble`: `'NVT'` or `'NPT'`.
-   `temperatures`: Single or list of temperatures (Kelvin).
-   `pressures`: (NPT only) Single or list of pressures (bar). Runs simulations for all T/P combinations.
-   `num_steps`: Number of MC trial moves per (T, P) condition.
-   `max_displacement`: Maximum atomic displacement per trial move (Å).
-   `max_volume_change_fraction`: (NPT only) Maximum fractional volume change per trial move.

##### Random Walk (`rw_params`)
For exploring local minima.
-   `perturbation_magnitude`: Strength of random atomic displacements before minimization (Å).
-   `optimization_fmax`: Force convergence criterion for local energy minimization (eV/Å).
-   `num_walks`: Number of perturbation/relaxation cycles.

---
#### Filter Settings (`filters`)
Applied during sampling to discard unrealistic structures.

-   `min_distance_check`:
    -   `enabled`: `true` or `false`.
    -   `min_dist_matrix`: Dictionary of minimum allowed distances between element pairs (e.g., `'Au-Au': 2.0`).
-   `max_energy_check`:
    -   `enabled`: `true` or `false`.
    -   `energy_tolerance`: Max allowed energy per atom (eV) above the relaxed seed's energy.
-   `max_force_check`:
    -   `enabled`: `true` or `false`.
    -   `max_force_component`: Max allowed force component on any atom (eV/Å).

---
#### Selection Settings
Controls how the final diverse subset is chosen from generated candidates.

-   `selection_method`: Algorithm: `'fps'` (Farthest Point Sampling) or `'pca_grid'` (SOAP+PCA+Gridding).
-   `selection_scope`: Defines how structures are selected:
    -   `'per_seed'`: `total_structures_to_select` are chosen from *each seed's* candidate pool independently.
    -   `'global'`: All candidates from all seeds are pooled, and then `total_structures_to_select` are chosen from this global pool.
-   `total_structures_to_select`: Target number of structures for the final output. Its interpretation depends on `selection_scope`.

##### Farthest Point Sampling (`fps_params`)
-   `distance_metric`: `'euclidean'` or `'cosine'` for comparing SOAP fingerprints.

##### SOAP+PCA+Gridding (`pca_grid_params`)
-   `n_components`: Number of principal components for dimensionality reduction (typically 2 or 3).
-   `grid_size`: List defining grid bins per principal component (e.g., `[10, 10]` for a 2D 10x10 grid).

---
#### SOAP Descriptor Settings (`soap_params`)
Parameters for the SOAP "fingerprint" used in selection algorithms.

-   `species`: Comprehensive list of all chemical element symbols present across *all* seed structures, especially if using `global` selection scope.
-   `r_cut`: Cutoff radius for local atomic environments (Å).
-   `n_max`, `l_max`: Number of radial and angular basis functions for SOAP.