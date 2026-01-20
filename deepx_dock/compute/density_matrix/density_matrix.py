import numpy as np
import libtetrabz

from deepx_dock.compute.eigen.fermi_dos import FermiEnergyAndDOSGenerator
from deepx_dock.compute.eigen.hamiltonian import AOMatrixR, AOMatrixK


class DensityMatrixObj:
    def __init__(self, info_dir_path, H_file_path=None):
        self._get_necessary_data_path(info_dir_path, H_file_path)
        #
        self.mat_DM = None
        #
        self.parse_data()

class DensityMatrixGenerator(FermiEnergyAndDOSGenerator):
    
    def calc_dm_K(self, method='smearing', sigma=0.02, k_process_num=1):
        if self.fermi_energy is None:
            raise ValueError("Fermi energy not determined. Run find_fermi_energy first.")

        if self.eigvals is None:
            self._calc_eigvals_on_mesh(dk=0.02, k_process_num=k_process_num)

        # 1. Calculate occupation weights w_{nk}
        # eigvals: [nband, nktot]
        # weights: [nktot, nband]
        if not self._is_metal:
            # Insulator / Semiconductor
            # Weights are 1 for E < Ef, 0 otherwise (T=0)
            weights = (self.eigvals < self.fermi_energy).T.astype(float)
        else:
            # Metal
            if method == 'tetrahedron':
                # Use libtetrabz.occ
                # Input: bvec, eig. (Ef assumed 0, so pass eig - Ef)
                # eigvals shape needed: [nk0, nk1, nk2, nband]
                eigvals_reshaped = self.eigvals.T.reshape(
                    *self.kpoint_density, self.band_quantity
                )
                weights = libtetrabz.occ(
                    self.reciprocal_lattice, eigvals_reshaped - self.fermi_energy
                ) 
                # weights from libtetrabz are integration weights (sum ~ N_elec).
                # k2r assumes MKs is intensive (O(1)) and applies 1/N_k factor.
                # So we multiply weights by N_k to convert to occupation numbers (sum ~ N_elec * N_k).
                weights = weights.reshape(self.nktot, self.band_quantity) * self.nktot
            elif method == 'smearing': # Fermi-Dirac smearing
                # f = 1 / (exp((E-Ef)/sigma) + 1)
                # Avoid overflow in exp
                x = (self.eigvals - self.fermi_energy) / sigma
                # x large positive -> exp(x) large -> f -> 0
                # x large negative -> exp(x) small -> f -> 1
                # Clip x to avoid overflow
                x = np.clip(x, -100, 100)
                weights = (1.0 / (np.exp(x) + 1.0)).T
            else:
                raise ValueError(f"Unknown method matrix calculation: {method}")

        # 2. Get Eigenvectors (Wave functions)
        # We need full diagonalization here to get eigenvectors
        # Use obj_H.diag. Note: We need to use the SAME k-mesh as used for eigenvalues
        # ks: [nktot, 3] from self.ks
        print(f"Calculating eigenvectors for Density Matrix with k_process_num={k_process_num} ...")
        # diag returns (eigvals, eigvecs)
        # eigvecs: [Norb, Nband, Nk]
        _, eigvecs = self.obj_H.diag(
            self.ks, k_process_num=k_process_num,
            sparse_calc=False, bands_only=False
        )
        # Transpose eigvecs to [Nk, Nband, Norb] for easier broadcasting
        # eigvecs from diag: (Norb, Nband, Nk) -> (Nk, Nband, Norb)
        eigvecs = eigvecs.transpose(2, 1, 0) # [Nk, Nband, Norb]
        U = eigvecs.transpose(0, 2, 1) 
        
        # (Nk, Nband) -> (Nk, 1, Nband)
        W = weights[:, None, :] 
        U_weighted = U * W
        MKs = np.matmul(U_weighted, U.conj().transpose(0, 2, 1))
        
        return AOMatrixK(self.ks, MKs)

    def calc_dm_R(self, method='tetrahedron', sigma=0.02, k_process_num=1):
        dm_K = self.calc_dm_K(method=method, sigma=sigma, k_process_num=k_process_num)
        
        Rs = self.obj_H.mat_S.Rs # Get Lattice displacements from overlap matrix
        return AOMatrixR(Rs, dm_K.k2r(Rs))
