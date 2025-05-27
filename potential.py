# potential.py

import numpy as np
from deepmd.infer import DeepPot
from ase.calculators.calculator import Calculator, all_changes

class DeepMDPotential(Calculator):
    """
    A deepmd-kit calculator that is fully compliant with the ASE Calculator interface.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, model_path: str, **kwargs):
        """
        Initializes the DeepMD ASE calculator.

        Args:
            model_path (str): Path to the frozen model file (.pb).
        """
        # Call the parent class's __init__ method
        Calculator.__init__(self, **kwargs)

        try:
            self.dp = DeepPot(model_path)
            self.type_map = self.dp.get_type_map()
            print(f"Model loaded successfully. Type map: {self.type_map}")
        except Exception as e:
            print(f"Error loading DeepMD model from {model_path}: {e}")
            raise
        
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        This is the core method that ASE optimizers and other tools will call.
        It performs the calculation of energy and forces.
        """
        # Call the parent class's calculate method to handle system changes
        Calculator.calculate(self, atoms, properties, system_changes)

        # Get atomic coordinates and types
        coords = np.ascontiguousarray(self.atoms.get_positions(), dtype=np.float32).reshape([1, -1])
        symbols = self.atoms.get_chemical_symbols()
        atom_types = np.array([self.type_map.index(s) for s in symbols], dtype=np.int32)
        
        # Get cell, handling non-periodic systems
        if np.all(self.atoms.get_cell().lengths() == 0):
            box = np.ascontiguousarray(np.eye(3) * 100.0, dtype=np.float32).reshape([1, 9])
        else:
            box = np.ascontiguousarray(self.atoms.get_cell(), dtype=np.float32).reshape([1, 9])

        # Evaluate the potential
        e, f, v = self.dp.eval(coords, box, atom_types)

        # Store the results in the format ASE expects
        self.results['energy'] = e[0]
        self.results['forces'] = f[0]