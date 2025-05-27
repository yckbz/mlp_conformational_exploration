# utils.py

from ase.io import read, write
import yaml

def load_config(path: str) -> dict:
    """Loads the YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_seeds(file_paths: list):
    """
    Loads initial seed structures from various file formats.
    
    Args:
        file_paths (list): A list of paths to structure files (cif, xyz, poscar).

    Returns:
        list: A list of ASE Atoms objects.
    """
    seeds = []
    for fpath in file_paths:
        try:
            atoms = read(fpath)
            seeds.append(atoms)
            print(f"Successfully loaded seed from: {fpath}")
        except Exception as e:
            print(f"Warning: Could not read seed file {fpath}. Error: {e}")
    return seeds

def save_structures(structures: list, filename: str, append: bool = False):
    """
    Saves a list of ASE Atoms objects to an extended XYZ file.

    Args:
        structures (list): List of ASE Atoms objects to save.
        filename (str): The output filename.
        append (bool): If True, appends to the file. Otherwise, overwrites.
    """
    if not structures:
        print("Warning: No structures to save.")
        return
    
    write(filename, structures, format='extxyz', append=append)
    print(f"Saved {len(structures)} structures to {filename}.")
