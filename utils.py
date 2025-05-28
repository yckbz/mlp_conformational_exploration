# utils.py

from ase.io import read, write
import yaml

def load_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_seeds(file_paths: list):
    """
    Loads initial seed structures from one or more files.
    Supports various common atomic structure file formats (e.g., cif, xyz, poscar).

    Args:
        file_paths (list): A list of string paths to the structure files.

    Returns:
        list: A list of ASE (Atomic Simulation Environment) Atoms objects.
              Returns an empty list if no valid files are found or files cannot be read.
    """
    seeds = []
    if not file_paths: # Handle case where no file paths are provided
        print("Warning: No seed file paths provided.")
        return seeds
        
    for fpath in file_paths:
        try:
            # ASE's read function automatically determines the file format.
            atoms = read(fpath)
            seeds.append(atoms)
            print(f"Successfully loaded seed from: {fpath}")
        except Exception as e:
            # Catch any errors during file reading (e.g., file not found, invalid format).
            print(f"Warning: Could not read seed file {fpath}. Error: {e}")
    return seeds

def save_structures(structures: list, filename: str, append: bool = False):
    """
    Saves a list of ASE Atoms objects to a single file, typically in extended XYZ format.

    Args:
        structures (list): List of ASE Atoms objects to be saved.
        filename (str): The name of the output file.
        append (bool): If True, appends structures to the file if it exists.
                       If False (default), overwrites the file if it exists.
    """
    if not structures:
        print("Warning: No structures to save.")
        return
    
    # The 'extxyz' format is commonly used for trajectories and multiple structures.
    write(filename, structures, format='extxyz', append=append)
    print(f"Saved {len(structures)} structures to {filename}.")