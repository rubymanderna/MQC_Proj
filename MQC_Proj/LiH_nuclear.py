"""
Simple demonstration of CQED-RHF method on the water molecule
coupled to a strong photon field with comparison to results from 
code in the hilbert package described in [DePrince:2021:094112] and available
at https://github.com/edeprince3/hilbert

"""

__authors__ = ["Jon McTague", "Jonathan Foley"]
__credits__ = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2021-08-19"

# ==> Import Psi4, NumPy, and helper_CQED_RHF <==
import psi4
import numpy as np
from helper_CQED_RHF_new import CQED_RHF_Calculation

# Set Psi4 & NumPy Memory Options
psi4.set_memory("2 GB")
psi4.core.set_output_file("output.dat", False)

numpy_memory = 2

# options for H2O
psi4_options_dict = {
    "basis": "6-31G",
    "save_jk": True,
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}


# molecule string for H2O
mol_str = """
0  1
Li   0.0    0.0    -0.276328822272
H    0.0   0.0    1.923671177728
symmetry c1
"""
#convert the z-matrix to xyz coordinates
# mol = psi4.geometry(mol_str)
# mol.update_geometry()
# mol.print_out_in_angstrom()
# print(mol.save_string_xyz()) 


# electric field for H2O - polarized along z-axis with mangitude 0.05 atomic units
lam_vec = np.array([0.0, 0.0, 0.05])

# Instantiate the Psi4Calculator
cal_energy = CQED_RHF_Calculation(lam_vec, mol_str, psi4_options_dict)

# run cqed_rhf on H2O
a1 = cal_energy.cal_Integrals()
a2 = cal_energy.cal_quadrapole_moments()
h2o_dict = cal_energy.cal_H_core()
a3 = cal_energy.cqed_rhf()
print("CQED-RHF Energy for LiH is ", a3["CQED-RHF ENERGY"])

# parse dictionary for ordinary RHF and CQED-RHF energy
h2o_cqed_rhf_e = a3["CQED-RHF ENERGY"]
# h2o_rhf_e = a3["RHF ENERGY"]

# parse dictionary for ordinary RHF and CQED-RHF energy
# h2o_cqed_rhf_e = h2o_dict["CQED-RHF ENERGY"]
# h2o_rhf_e = h2o_dict["RHF ENERGY"]

print("    CQED-RHF Energy:           %.8f" % h2o_cqed_rhf_e)



