# potential.py

import numpy as np
from deepmd.infer import DeepPot
from ase.calculators.calculator import Calculator, all_changes

class DeepMDPotential(Calculator):
    """
    ASE-compatible calculator for evaluating energies, forces, and stresses
    using a deepmd-kit potential model.
    """
    # Properties that this calculator can compute.
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_path: str, **kwargs):
        """
        Initializes the DeepMD ASE calculator.

        Args:
            model_path (str): Path to the frozen deepmd-kit model file (.pb).
        """
        super().__init__(**kwargs) # Initialize base ASE Calculator

        try:
            self.dp = DeepPot(model_path) # Load the DeepMD model.
            self.type_map = self.dp.get_type_map() # Get the mapping of chemical symbols to model atom types.
            print(f"Model loaded successfully. Type map: {self.type_map}")
        except Exception as e:
            print(f"Error loading DeepMD model from {model_path}: {e}")
            raise
        
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Core ASE method to perform the calculation of energy, forces, and/or stress.
        This method is called by ASE optimizers and dynamics engines.
        """
        super().calculate(atoms, properties, system_changes)

        # Prepare inputs for the deepmd-kit model.
        coords = np.ascontiguousarray(self.atoms.get_positions(), dtype=np.float32).reshape([1, -1])
        symbols = self.atoms.get_chemical_symbols()
        atom_types = np.array([self.type_map.index(s) for s in symbols], dtype=np.int32)
        
        # Handle periodic boundary conditions and cell representation.
        # If cell lengths are all zero, assume a large vacuum box for non-periodic systems.
        if np.all(self.atoms.get_cell().lengths() == 0):
            # Define a large box for isolated systems, as required by some models.
            box = np.ascontiguousarray(np.eye(3) * 100.0, dtype=np.float32).reshape([1, 9])
        else:
            box = np.ascontiguousarray(self.atoms.get_cell(), dtype=np.float32).reshape([1, 9])

        # Evaluate the potential using the loaded deepmd-kit model.
        # e: energy, f: forces, v: virial tensor (9 components, 3x3 matrix)
        e, f, v = self.dp.eval(coords, box, atom_types)

        # Store results in the format expected by ASE.
        self.results['energy'] = e[0]
        self.results['forces'] = f[0] # Forces are per atom.
        
        # Calculate and store stress if requested and the system has volume.
        if 'stress' in properties and self.atoms.get_volume() > 0:
            # deepmd-kit returns virial tensor (v).
            # Stress = -Virial / Volume.
            virial_tensor_3x3 = v[0].reshape((3, 3))
            stress_tensor_3x3 = -virial_tensor_3x3 / self.atoms.get_volume()

            # Convert 3x3 stress tensor to ASE's 6-component Voigt notation:
            # [xx, yy, zz, yz, xz, xy]
            stress_voigt = np.array([
                stress_tensor_3x3[0, 0], stress_tensor_3x3[1, 1],
                stress_tensor_3x3[2, 2], stress_tensor_3x3[1, 2],
                stress_tensor_3x3[0, 2], stress_tensor_3x3[0, 1]
            ])
            self.results['stress'] = stress_voigt