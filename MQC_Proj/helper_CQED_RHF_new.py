"""
Helper function for CQED_RHF

References:
    Equations and algorithms from 
    [Haugland:2020:041043], [DePrince:2021:094112], and [McTague:2021:ChemRxiv] 

"""

__authors__ = ["Jon McTague", "Jonathan Foley"]
__credits__ = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2021-08-19"

# ==> Import Psi4, NumPy, & SciPy <==
import psi4
import numpy as np
import time

class CQED_RHF_Calculation:
    """" 
    Arguments
    ---------
    lambda_vector : 1 x 3 array of floats
        the electric field vector, see e.g. Eq. (1) in [DePrince:2021:094112]
        and (15) in [Haugland:2020:041043]

    molecule_string : string
        specifies the molecular geometry """

    def __init__(self, lambda_vector, molecule_string, psi4_options_dict):
        # Set the geometry and options for Psi4
        self.molecule_string = molecule_string
        self.psi4_options_dict = psi4_options_dict

        self.mol = psi4.geometry(molecule_string)
        psi4.set_options(psi4_options_dict)

        # Initialize lambda_vector with a default or passed value
        if lambda_vector is None:
        # Default value for lambda_vector if not provided
            self.lambda_vector = [0, 0, 0]
        else:
            self.lambda_vector = lambda_vector

        print("Electric field vector Lambda: ", self.lambda_vector)

        # Calculate the RHF energy and wavefunction
        self.psi4_rhf_energy, self.wfn = psi4.energy("scf", return_wfn=True)
        
        # Create a MintsHelper object to access basis set information
        self.mints = psi4.core.MintsHelper(self.wfn.basisset())

        """ Grab the number of doubly occupied orbitals(alpha electrons)"""
        self.ndocc = self.wfn.nalpha()
        print("Number of doubly occupied orbitals: ", self.ndocc)
        
        # grab all transformation vectors and store to a numpy array
        # C is the transformation matrix from AO to MO basis with columns as MOs and rows as AOs with shape(n_basis_functions, n_molecular_orbitals)
        self.C = np.asarray(self.wfn.Ca())

        # use canonical RHF orbitals for guess CQED-RHF orbitals, Cocc is the occupied orbitals with shape (n_basis_functions, ndocc)
        self.Cocc = self.C[:, :self.ndocc]

        """Computes the density matrix from the occupied orbitals
        Form guess density matrix (C_muv*C^*_nuv) for occupied orbitals, [Szabo:1996] Eqn. 3.145, pp. 139 """

        # Compute the density matrix D using einsum
        self.D = np.einsum("pi,qi->pq", self.Cocc, self.Cocc)
        # print("Density Matrix D", self.D)
    
    def cal_Integrals(self):
        """Computes the integrals required for CQED-RHF calculation, Ordinary integrals first, ao_kinetic(),
            ao_potential() and ao_eri() are functions of the MintsHelper class in Psi4
            Similarly, ao_dipole() is a function of the MintsHelper class computes the dipole moment integrals in the atomic orbital (AO) basis,
            gives a list of matrixes in psi4.core.Matrix form in Psi4"""
        
        self.V = np.asarray(self.mints.ao_potential())
        self.T = np.asarray(self.mints.ao_kinetic())
        self.I = np.asarray(self.mints.ao_eri())
        print("Ordinary Integrals Computation", self.V, self.T)
        # Extra terms for Pauli-Fierz Hamiltonian
        # nuclear dipole
        self.mu_nuc_x = self.mol.nuclear_dipole()[0] 
        self.mu_nuc_y = self.mol.nuclear_dipole()[1]
        self.mu_nuc_z = self.mol.nuclear_dipole()[2]

        self.mu_ao_x = np.asarray(self.mints.ao_dipole()[0]) #first element of the returned list (the x-component)
        self.mu_ao_y = np.asarray(self.mints.ao_dipole()[1])  #second element of the returned list (the y-component)
        self.mu_ao_z = np.asarray(self.mints.ao_dipole()[2])  #third element of the returned list (the z-component)

        # \lambda \cdot \mu_el (see within the sum of line 3 of Appendix Eq. (A3) in [https://doi.org/10.1063/5.0091953])
        l_dot_mu_el = self.lambda_vector[0] * self.mu_ao_x
        l_dot_mu_el += self.lambda_vector[1] * self.mu_ao_y
        l_dot_mu_el += self.lambda_vector[2] * self.mu_ao_z
        # Store the final result in the instance variable
        self.l_dot_mu_el = l_dot_mu_el

        # compute electronic dipole expectation value with canonincal RHF density
        mu_exp_x = np.einsum("pq,pq->", 2 * self.mu_ao_x, self.D)
        mu_exp_y = np.einsum("pq,pq->", 2 * self.mu_ao_y, self.D)
        mu_exp_z = np.einsum("pq,pq->", 2 * self.mu_ao_z, self.D)

        # need to add the nuclear term to the sum over the electronic dipole integrals
        mu_exp_x += self.mu_nuc_x
        mu_exp_y += self.mu_nuc_y
        mu_exp_z += self.mu_nuc_z

        #total dipole moment
        self.rhf_dipole_moment = np.array([mu_exp_x, mu_exp_y, mu_exp_z])

        # Calculating the electric field dotted into the nuclear dipole moment and  RHF electronic dipole expectation value(see within the sum of line 3 of Appendix Eq. (A3) in [https://doi.org/10.1063/5.0091953])
        # \lambda_vector \cdot \mu_{nuc}
        l_dot_mu_nuc = ( 
            self.lambda_vector[0] * self.mu_nuc_x 
            + self.lambda_vector[1] * self.mu_nuc_y 
            + self.lambda_vector[2] * self.mu_nuc_z
            )
        
        # Store the final result in the instance variable
        self.l_dot_mu_nuc = l_dot_mu_nuc

        # \lambda_vecto \cdot < \mu > where <\mu> contains electronic and nuclear contributions
        l_dot_mu_exp = (
            self.lambda_vector[0] * mu_exp_x 
            + self.lambda_vector[1] * mu_exp_y 
            + self.lambda_vector[2] * mu_exp_z
            )

        # Store the final result in the instance variable
        self.l_dot_mu_exp = l_dot_mu_exp

        #last line of eq A3
        self.d_c = (0.5 * l_dot_mu_nuc ** 2 - l_dot_mu_nuc * l_dot_mu_exp + 0.5 * l_dot_mu_exp ** 2)

        # Pauli-Fierz 1-e dipole terms scaled by (\lambda_vector \cdot \mu_{nuc} - \lambda_vector \cdot <\mu>) 
        # Line 2 of Eq. (A3) in [https://doi.org/10.1063/5.0091953]
        self.d_PF = (l_dot_mu_nuc - l_dot_mu_exp) * l_dot_mu_el
        print("1-e Dipole term d_PF Calulation")

        # return self.V, self.T, self.I, self.rhf_dipole_moment,self.l_dot_mu_nuc, self.l_dot_mu_exp, self.d_c, self.d_PF
    
    def cal_quadrapole_moments(self):
        """Computes the quadrupole moments of the molecule, Quadrupole moments are important for describing the charge distribution in molecules,
            especially those without a dipole moment"""
        
        Q_ao_xx = np.asarray(self.mints.ao_quadrupole()[0])   #The xx component represents how the x-component of the electric field changes with x-direction
        Q_ao_xy = np.asarray(self.mints.ao_quadrupole()[1])
        Q_ao_xz = np.asarray(self.mints.ao_quadrupole()[2])
        Q_ao_yy = np.asarray(self.mints.ao_quadrupole()[3])
        Q_ao_yz = np.asarray(self.mints.ao_quadrupole()[4])
        Q_ao_zz = np.asarray(self.mints.ao_quadrupole()[5])

        # Pauli-Fierz 1-e quadrupole terms, Line 1 of Eq. (A3) in [https://doi.org/10.1063/5.0091953]
        Q_PF = -0.5 * self.lambda_vector[0] * self.lambda_vector[0] * Q_ao_xx
        Q_PF -= 0.5 * self.lambda_vector[1] * self.lambda_vector[1] * Q_ao_yy
        Q_PF -= 0.5 * self.lambda_vector[2] * self.lambda_vector[2] * Q_ao_zz

        # accounting for the fact that Q_ij = Q_ji by weighting Q_ij x 2 which cancels factor of 1/2
        Q_PF -= self.lambda_vector[0] * self.lambda_vector[1] * Q_ao_xy
        Q_PF -= self.lambda_vector[0] * self.lambda_vector[2] * Q_ao_xz
        Q_PF -= self.lambda_vector[1] * self.lambda_vector[2] * Q_ao_yz
        print("1-e Quadrupole term Q_PF", Q_PF)
        
        self.Q_PF = Q_PF
        print("Quadrupole moment Q_PF Calculation")

        return self.Q_PF

    def cal_H_core(self):
        """Computes the core Hamiltonian and adds the Pauli-Fierz terms to the Hamiltonian"""
    
        # ordinary H_core
        self.H_0 = self.T + self.V

        # Add Pauli-Fierz terms to H_core equation A5
        self.H = self.H_0 + self.Q_PF + self.d_PF
        # print("Core Hamiltonian H Calculation", self.H)

        """ Overlap for DIIS
        DIIS is an extrapolation technique used to accelerate convergence in iterative procedures, such as the Self-Consistent Field (SCF) method.
        S represents the overlap between atomic orbitals and elements S_ij measure how much orbital i overlaps with orbital j."""
        self.S = self.mints.ao_overlap()      #S is symmetric (S_ij = S_ji) and its diagonal elements are always 1 (S_ii = 1) 

        # Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
        # A is used to transform the basis from a non-orthogonal set (like typical atomic orbitals) to an orthogonal set.
        A = self.mints.ao_overlap()
        A.power(-0.5, 1.0e-16)
        self.A = np.asarray(A)
    
        # return self.H_0, self.H, self.S, self.A
    
    """All these quantities are Computed: kinetic, nuclear attraction, electron repulsion, dipole, and quadrupole integrals in AO basis.
        Performing canonical RHF calculation """
    
    def cqed_rhf(self):
        """Computes the QED-RHF energy

        options_dict : dictionary
        specifies the psi4 options to be used in running the canonical RHF

        Returns
        -------
        cqed_rhf_dictionary : dictionary
        Contains important quantities from the cqed_rhf calculation, with keys including:
            'RHF ENERGY' -> result of canonical RHF calculation using psi4 defined by molecule_string and psi4_options_dict
            'CQED-RHF ENERGY' -> result of CQED-RHF calculation, see Eq. (13) of [McTague:2021:ChemRxiv]
            'CQED-RHF C' -> orbitals resulting from CQED-RHF calculation
            'CQED-RHF DENSITY MATRIX' -> density matrix resulting from CQED-RHF calculation
            'CQED-RHF EPS'  -> orbital energies from CQED-RHF calculation
            'PSI4 WFN' -> wavefunction object from psi4 canonical RHF calcluation
            'CQED-RHF DIPOLE MOMENT' -> total dipole moment from CQED-RHF calculation (1x3 numpy array)
            'NUCLEAR DIPOLE MOMENT' -> nuclear dipole moment (1x3 numpy array)
            'DIPOLE ENERGY' -> See Eq. (14) of [McTague:2021:ChemRxiv]
            'NUCLEAR REPULSION ENERGY' -> Total nuclear repulsion energy

        Example
        -------
        >>> cqed_rhf_dictionary = cqed_rhf([0., 0., 1e-2], '''\nMg\nH 1 1.7\nsymmetry c1\n1 1\n''', psi4_options_dictionary)

        """
        # Defining varaibles for SCF iterations with initial values
        H = self.H
        d_c = self.d_c
        D = self.D

        print("\nStart SCF iterations:\n")
        t = time.time()
        E = 0.0
        Enuc = self.mol.nuclear_repulsion_energy()
        Eold = 0.0

        E_1el_crhf = np.einsum("pq,pq->", self.H_0 + self.H_0, self.D)
        E_1el = np.einsum("pq,pq->", H + H, self.D)
        print("Done till here")
        print("Canonical RHF One-electron energy = %4.16f" % E_1el_crhf)
        print("CQED-RHF One-electron energy      = %4.16f" % E_1el)
        print("Nuclear repulsion energy          = %4.16f" % Enuc)
        print("Dipole energy                     = %4.16f" % d_c)

        # Set convergence criteria from psi4_options_dict
        if "e_convergence" in self.psi4_options_dict:
            E_conv = self.psi4_options_dict["e_convergence"]
        else:
            E_conv = 1.0e-7
        if "d_convergence" in self.psi4_options_dict:
            D_conv = self.psi4_options_dict["d_convergence"]
        else:
            D_conv = 1.0e-5

        t = time.time()

        # maxiter
        maxiter = 500
        for SCF_ITER in range(1, maxiter + 1):

            # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
            J = np.einsum("pqrs,rs->pq", self.I, D)
            K = np.einsum("prqs,rs->pq", self.I, D)

            # Pauli-Fierz 2-e dipole-dipole terms, line 2 of Eq. (12) in [McTague:2021:ChemRxiv]
            M = np.einsum("pq,rs,rs->pq", self.l_dot_mu_el, self.l_dot_mu_el, D)
            N = np.einsum("pr,qs,rs->pq", self.l_dot_mu_el, self.l_dot_mu_el, D)

            # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
            # plus Pauli-Fierz terms Eq. (12) in [McTague:2021:ChemRxiv]
            
            F = H + J * 2 - K + 2 * M - N

            diis_e = np.einsum("ij,jk,kl->il", F, D, self.S) - np.einsum("ij,jk,kl->il", self.S, D, F)
            diis_e = self.A.dot(diis_e).dot(self.A)
            dRMS = np.mean(diis_e ** 2) ** 0.5

            # SCF energy and update: [Szabo:1996], Eqn. 3.184, pp. 150
            # Pauli-Fierz terms Eq. 13 of [McTague:2021:ChemRxiv]
            SCF_E = np.einsum("pq,pq->", F + H, D) + Enuc + d_c

            print(
                "SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E"
                % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS)
            )
            if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
                break

            Eold = SCF_E

            # Diagonalize Fock matrix: [Szabo:1996] pp. 145
            Fp = self.A.dot(F).dot(self.A)  # Eqn. 3.177
            e, C2 = np.linalg.eigh(Fp)  # Solving Eqn. 1.178
            C = self.A.dot(C2)  # Back transform, Eqn. 3.174
            Cocc = C[:, :self.ndocc]
            D = np.einsum("pi,qi->pq", Cocc, Cocc)  # [Szabo:1996] Eqn. 3.145, pp. 139

            # update electronic dipole expectation value
            mu_exp_x = np.einsum("pq,pq->", 2 * self.mu_ao_x, D)
            mu_exp_y = np.einsum("pq,pq->", 2 * self.mu_ao_y, D)
            mu_exp_z = np.einsum("pq,pq->", 2 * self.mu_ao_z, D)

            mu_exp_x += self.mu_nuc_x
            mu_exp_y += self.mu_nuc_y
            mu_exp_z += self.mu_nuc_z

            # update \lambda \cdot <\mu>
            l_dot_mu_exp = (
                self.lambda_vector[0] * mu_exp_x
                + self.lambda_vector[1] * mu_exp_y
                + self.lambda_vector[2] * mu_exp_z
            )
            # Line 3 in full of Eq. (9) in [McTague:2021:ChemRxiv]
            d_PF = (self.l_dot_mu_nuc - l_dot_mu_exp) * self.l_dot_mu_el

            # update Core Hamiltonian
            H = self.H_0 + self.Q_PF + d_PF
            
            # update dipole energetic contribution, Eq. (14) in [McTague:2021:ChemRxiv]
            d_c = (
                0.5 * self.l_dot_mu_nuc ** 2
                - self.l_dot_mu_nuc * l_dot_mu_exp
                + 0.5 * l_dot_mu_exp ** 2
            )

            if SCF_ITER == maxiter:
                psi4.core.clean()
                raise Exception("Maximum number of SCF cycles exceeded.")

        print("Total time for SCF iterations: %.3f seconds \n" % (time.time() - t))
        print("QED-RHF   energy: %.8f hartree" % SCF_E)
        print("Psi4  SCF energy: %.8f hartree" % self.psi4_rhf_energy)

        rhf_one_e_cont = (2 * self.H_0)  # note using H_0 which is just T + V, and does not include Q_PF and d_PF
        rhf_two_e_cont = (J * 2 - K)  # note using just J and K that would contribute to ordinary RHF 2-electron energy
        pf_two_e_cont = 2 * M - N

        SCF_E_One = np.einsum("pq,pq->", rhf_one_e_cont, D)
        SCF_E_Two = np.einsum("pq,pq->", rhf_two_e_cont, D)
        CQED_SCF_E_Two = np.einsum("pq,pq->", pf_two_e_cont, D)

        CQED_SCF_E_D_PF = np.einsum("pq,pq->", 2 * d_PF, D)
        CQED_SCF_E_Q_PF = np.einsum("pq,pq->", 2 * self.Q_PF, D)

        assert np.isclose(
            SCF_E_One + SCF_E_Two + CQED_SCF_E_D_PF + CQED_SCF_E_Q_PF + CQED_SCF_E_Two,
            SCF_E - d_c - Enuc,)

        cqed_rhf_dict = {
            "RHF ENERGY": self.psi4_rhf_energy,
            "CQED-RHF ENERGY": SCF_E,
            "1E ENERGY": SCF_E_One,
            "2E ENERGY": SCF_E_Two,
            "1E DIPOLE ENERGY": CQED_SCF_E_D_PF,
            "1E QUADRUPOLE ENERGY": CQED_SCF_E_Q_PF,
            "2E DIPOLE ENERGY": CQED_SCF_E_Two,
            "CQED-RHF C": C,
           "CQED-RHF DENSITY MATRIX": D,
            "CQED-RHF EPS": e,
            "PSI4 WFN": self.wfn,
           "RHF DIPOLE MOMENT": self.rhf_dipole_moment,
           "CQED-RHF DIPOLE MOMENT": np.array([mu_exp_x, mu_exp_y, mu_exp_z]),
        #    "NUCLEAR DIPOLE MOMENT": np.array([mu_nuc_x, mu_nuc_y, mu_nuc_z]),
           "DIPOLE ENERGY": d_c,
           "NUCLEAR REPULSION ENERGY": Enuc,
        }
    
        return cqed_rhf_dict


class CQED_RHF_NuclearGradient:
    def __init__(self, lambda_vector, molecule_string, psi4_options_dict, step_size=1e-5):
        """
        Initialize the nuclear gradient calculation for CQED-RHF.

        Parameters:
        - lambda_vector: 1x3 array of floats representing the electric field vector
        - molecule_string: str, specifies the molecular geometry
        - psi4_options_dict: dict, options for the Psi4 calculation
        - step_size: float, the finite difference step size (default: 1e-5 bohr)
        """
        self.lambda_vector = lambda_vector
        self.molecule_string = molecule_string
        self.psi4_options_dict = psi4_options_dict
        self.step_size = step_size
        self.mol = psi4.geometry(molecule_string)
        self.n_atoms = self.mol.natom()
        print("Number of atoms in the molecule:", self.n_atoms)

        # Store the initial Cartesian coordinates
        self.cartesian_coords = self.mol.geometry().np
        print("Initial Cartesian coordinates:", self.cartesian_coords)
        
        # Initialize a CQED_RHF_Calculation class instance
        self.cqed_calculation = CQED_RHF_Calculation(self.lambda_vector, self.molecule_string, self.psi4_options_dict)

    # Function to compute the initial CQED-RHF energy

    def compute_initial_energy(self):
        """Compute the initial CQED-RHF energy."""
        self.cqed_calculation.cal_Integrals()
        self.cqed_calculation.cal_quadrapole_moments()
        self.cqed_calculation.cal_H_core()
        return self.cqed_calculation.cqed_rhf()["CQED-RHF ENERGY"]

     # Function to compute the displaced molecule's CQED-RHF energy

    def compute_displaced_energy(self, displaced_mol):
        """Compute the CQED-RHF energy for a displaced molecule
            Initialize a CQED_RHF_Calculation class instance for the displaced molecule"""
        # displaced_mol_string = displaced_mol.save_string_xyz()
        displaced_calculation = CQED_RHF_Calculation(self.lambda_vector, displaced_mol, self.psi4_options_dict)
        displaced_calculation.cal_Integrals()
        displaced_calculation.cal_quadrapole_moments()
        displaced_calculation.cal_H_core()
        return displaced_calculation.cqed_rhf()["CQED-RHF ENERGY"]

    # def compute_gradient(self):
    #     """Compute the nuclear gradient using finite differences of the QED-RHF energy."""
    #     # Initialize gradient array
    #     gradient = np.zeros_like(self.cartesian_coords)

    #     # Compute the initial energy
    #     initial_energy = self.compute_initial_energy()
    #     print("Initial QED-RHF energy:", initial_energy)

    #     # Loop over all atoms and their x, y, z coordinates
    #     for atom in range(self.n_atoms):
    #         for coord in range(3):
    #             # Create copies of the molecule for positive and negative displacements
    #             mol_plus = self.mol.clone()
    #             mol_minus = self.mol.clone()

    #             # Displace the coordinates
    #             coords_plus = mol_plus.geometry().np
    #             coords_minus = mol_minus.geometry().np

    #             coords_plus[atom, coord] += self.step_size
    #             coords_minus[atom, coord] -= self.step_size
    #             print("Displaced coordinates:", coords_plus, coords_minus)  

    #             # Set the displaced geometries
    #             mol_plus.set_geometry(psi4.core.Matrix.from_array(coords_plus))
    #             mol_minus.set_geometry(psi4.core.Matrix.from_array(coords_minus))

    #             #converting the coordinates to string(z matriz)
    #             mol_plus_string = mol_plus.save_string_xyz()
    #             mol_minus_string = mol_minus.save_string_xyz()
    #             # Print mol_plus geometry
    #             print("mol_plus geometry:", mol_plus_string)
    #             print("molecule_string type:", type(mol_plus_string))
    #             # print("molecule_string:", molecule_string)  
    #             # print(mol_plus.geometry().to_array())

    #             # Print mol_minus geometry
    #             print("mol_minus geometry:", mol_minus_string)
    #             # print(mol_minus.geometry().to_array())

    #             # some error in psi4 geometry while calling this function
    #             # Calculate energies for displaced geometries
    #             qed_rhf_energy_plus = self.compute_displaced_energy(mol_plus)
    #             qed_rhf_energy_minus = self.compute_displaced_energy(mol_minus)

    #             # Compute the gradient using central difference
    #             gradient[atom, coord] = (qed_rhf_energy_plus - qed_rhf_energy_minus) / (2 * self.step_size)
    #             print("Computed gradient")

    #     return gradient

