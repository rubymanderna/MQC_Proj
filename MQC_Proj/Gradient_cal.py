# ==> Import Psi4, NumPy, and helper_CQED_RHF <==
import psi4
import numpy as np
from helper_CQED_RHF_new import *

# Set Psi4 & NumPy Memory Options
psi4.set_memory("2 GB")
psi4.core.set_output_file("output.dat", False)

numpy_memory = 2

# options for H2O
h2o_options_dict = {
    "basis": "cc-pVDZ",
    "symmetry": 'c1',
    "save_jk": True,
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}


# molecule string for H2O
h2o_string = """

0 1
    O      0.000000000000   0.000000000000  -0.068516219320
    H      0.000000000000  -0.790689573744   0.543701060715
    H      0.000000000000   0.790689573744   0.543701060715
no_reorient
symmetry c1
"""

# energy for H2O from hilbert package described in [DePrince:2021:094112]
expected_h2o_e = -76.016355284146

# electric field for H2O - polarized along z-axis with mangitude 0.05 atomic units
lam_h2o = np.array([0.0, 0.0, 0.05])

print("NUCLEAR GRADIENTS CALCULATION STARTED")

gradient_calculator = CQED_RHF_NuclearGradient(lam_h2o, h2o_string, h2o_options_dict)
gradient = gradient_calculator.compute_gradient()
print("Computed gradient:", gradient)

# parse dictionary for ordinary RHF and CQED-RHF energy
# h2o_cqed_rhf_e = a3["CQED-RHF ENERGY"]
# h2o_rhf_e = a3["RHF ENERGY"]