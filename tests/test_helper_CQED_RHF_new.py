import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os
# Add the folder path where the file is located
sys.path.append(os.path.abspath('../MQC_Proj'))
# from helper_CQED_RHF_new import CQED_RHF_Calculation
from MQC_Proj.helper_CQED_RHF_new import *

class Test_CQED_RHF_Calculation(unittest.TestCase):
    
    def setUp(self):
        """Set up the class instance with example data"""
        # Example test data for lambda_vector and molecule_string
        lambda_vector = [0.0, 0.0, 0.5]

        # molecule string for MgH
        molecule_string = """
        1 1
        Mg 
        F    1  1.0
        no_reorient
        symmetry c1
        """
        #energy from hilbert(scf_type = cd,cholesky_tolerance 1e-12)
        expected_mgf_e = -297.621331340683
        psi4_options_dict = {"basis": "sto-3g"}
        
        # Create the CQED_RHF_Calculation instance
        self.cal = CQED_RHF_Calculation(lambda_vector, molecule_string, psi4_options_dict)
        
        # h2o_dict_origin = cqed_rhf(lam_h2o, h2o_string, h2o_options_dict)
        # h2o_cqed_rhf_e = h2o_dict_origin["CQED-RHF ENERGY"]
        # assert psi4.compare_values(h2o_cqed_rhf_e,expected_mgf_e)

        # Mock MintsHelper's functions for controlled testing
        self.cal.mints = MagicMock()
        self.cal.mints.ao_potential.return_value = np.array([[0.5, 0.1], [0.1, 0.5]])
        self.cal.mints.ao_kinetic.return_value = np.array([[1.0, 0.2], [0.2, 1.0]])
        self.cal.mints.ao_eri.return_value = np.array([[[[0.8]]]])
        self.cal.mints.ao_dipole.return_value = [np.array([[0.3, 0.1], [0.1, 0.3]])] * 3
        
        # Mock nuclear dipole components
        self.cal.mol = MagicMock()
        self.cal.mol.nuclear_dipole.return_value = [0.1, 0.2, 0.3]
        
        # Manually set the density matrix (D) to avoid Psi4 dependency
        self.cal.D = np.array([[0.6, 0.2], [0.2, 0.6]])

    def test_initialization(self):
        """Test that the class initializes correctly and sets attributes."""
        self.assertEqual(self.cal.lambda_vector, [0.0, 0.0, 0.5], "lambda_vector initialization is incorrect")
        self.assertIsInstance(self.cal.molecule_string, str, "molecule_string should be a string")
        self.assertIsInstance(self.cal.psi4_options_dict, dict, "psi4_options_dict should be a dictionary")

    def test_cal_integrals(self):
        """Test the cal_Integrals method and its calculations."""
        self.cal.cal_Integrals()

        # Check the core integrals
        np.testing.assert_array_equal(self.cal.V, np.array([[0.5, 0.1], [0.1, 0.5]]), "ao_potential is incorrect")
        np.testing.assert_array_equal(self.cal.T, np.array([[1.0, 0.2], [0.2, 1.0]]), "ao_kinetic is incorrect")
        np.testing.assert_array_equal(self.cal.I, np.array([[[[0.8]]]]), "ao_eri is incorrect")

        # Check computed dipole moment integrals
        expected_mu_ao = [np.array([[0.3, 0.1], [0.1, 0.3]])] * 3
        np.testing.assert_array_equal(self.cal.mu_ao_x, expected_mu_ao[0], "mu_ao_x is incorrect")
        np.testing.assert_array_equal(self.cal.mu_ao_y, expected_mu_ao[1], "mu_ao_y is incorrect")
        np.testing.assert_array_equal(self.cal.mu_ao_z, expected_mu_ao[2], "mu_ao_z is incorrect")

        # Check Pauli-Fierz components
        expected_l_dot_mu_el = (
            self.cal.lambda_vector[0] * self.cal.mu_ao_x +
            self.cal.lambda_vector[1] * self.cal.mu_ao_y +
            self.cal.lambda_vector[2] * self.cal.mu_ao_z
        )
        np.testing.assert_array_equal(self.cal.l_dot_mu_el, expected_l_dot_mu_el, "l_dot_mu_el is incorrect")

        # Check rhf_dipole_moment calculations
        expected_rhf_dipole_moment = [
            np.einsum("pq,pq->", 2 * expected_mu_ao[0], self.cal.D) + self.cal.mu_nuc_x,
            np.einsum("pq,pq->", 2 * expected_mu_ao[1], self.cal.D) + self.cal.mu_nuc_y,
            np.einsum("pq,pq->", 2 * expected_mu_ao[2], self.cal.D) + self.cal.mu_nuc_z,
        ]
        np.testing.assert_almost_equal(self.cal.rhf_dipole_moment, expected_rhf_dipole_moment, decimal=5, err_msg="RHF dipole moment is incorrect")

    def test_dipole_expectation(self):
        """Test dipole expectation calculation for accuracy."""
        self.cal.cal_Integrals()
        
        # Check for electric field dotted into nuclear and electronic dipole expectation values
        expected_l_dot_mu_nuc = (
            self.cal.lambda_vector[0] * self.cal.mu_nuc_x +
            self.cal.lambda_vector[1] * self.cal.mu_nuc_y +
            self.cal.lambda_vector[2] * self.cal.mu_nuc_z
        )
        self.assertAlmostEqual(self.cal.l_dot_mu_nuc, expected_l_dot_mu_nuc, places=5, msg="l_dot_mu_nuc calculation is incorrect")

        expected_l_dot_mu_exp = (
            self.cal.lambda_vector[0] * self.cal.rhf_dipole_moment[0] +
            self.cal.lambda_vector[1] * self.cal.rhf_dipole_moment[1] +
            self.cal.lambda_vector[2] * self.cal.rhf_dipole_moment[2]
        )
        self.assertAlmostEqual(self.cal.l_dot_mu_exp, expected_l_dot_mu_exp, places=5, msg="l_dot_mu_exp calculation is incorrect")

    def test_pauli_fierz_term(self):
        """Test Pauli-Fierz term calculations in d_PF."""
        self.cal.cal_Integrals()
        
        expected_d_PF = (self.cal.l_dot_mu_nuc - self.cal.l_dot_mu_exp) * self.cal.l_dot_mu_el
        np.testing.assert_array_almost_equal(self.cal.d_PF, expected_d_PF, decimal=5, err_msg="d_PF calculation is incorrect")

        # Mock the mints and ao_quadrupole output
        self.cal.mints = MagicMock()
        self.cal.mints.ao_quadrupole.return_value = [
            np.array([[1.0, 0.1], [0.1, 1.0]]),  # Q_ao_xx
            np.array([[0.2, 0.1], [0.1, 0.2]]),  # Q_ao_xy
            np.array([[0.3, 0.1], [0.1, 0.3]]),  # Q_ao_xz
            np.array([[1.1, 0.1], [0.1, 1.1]]),  # Q_ao_yy
            np.array([[0.4, 0.1], [0.1, 0.4]]),  # Q_ao_yz
            np.array([[1.2, 0.1], [0.1, 1.2]])   # Q_ao_zz
        ]

    def test_cal_quadrapole_moments(self):
        # Expected Q_PF calculation based on mocked values
        lambda_vector = self.cal.lambda_vector

        # Manually compute expected Q_PF based on the lambda_vector and mocked quadrupole moments
        Q_ao_xx = np.array([[1.0, 0.1], [0.1, 1.0]])
        Q_ao_xy = np.array([[0.2, 0.1], [0.1, 0.2]])
        Q_ao_xz = np.array([[0.3, 0.1], [0.1, 0.3]])
        Q_ao_yy = np.array([[1.1, 0.1], [0.1, 1.1]])
        Q_ao_yz = np.array([[0.4, 0.1], [0.1, 0.4]])
        Q_ao_zz = np.array([[1.2, 0.1], [0.1, 1.2]])

        # Calculating expected Q_PF value as per the formula
        expected_Q_PF = -0.5 * lambda_vector[0] ** 2 * Q_ao_xx
        expected_Q_PF -= 0.5 * lambda_vector[1] ** 2 * Q_ao_yy
        expected_Q_PF -= 0.5 * lambda_vector[2] ** 2 * Q_ao_zz
        expected_Q_PF -= lambda_vector[0] * lambda_vector[1] * Q_ao_xy
        expected_Q_PF -= lambda_vector[0] * lambda_vector[2] * Q_ao_xz
        expected_Q_PF -= lambda_vector[1] * lambda_vector[2] * Q_ao_yz

        # Run the method under test
        actual_Q_PF = self.cal.cal_quadrapole_moments()

        # Check that the calculated Q_PF is as expected
        np.testing.assert_array_almost_equal(actual_Q_PF, expected_Q_PF, decimal=5,
                                             err_msg="Quadrupole moment Q_PF calculation is incorrect")

if __name__ == "__main__":
    unittest.main()

