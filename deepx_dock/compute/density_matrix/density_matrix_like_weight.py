import numpy as np
import h5py
from tqdm import tqdm
from copy import copy

from deepx_dock.compute.eigen.fermi_dos import FermiEnergyAndDOSGenerator
from deepx_dock.misc import set_num_threads


class DensityMatrixLikeWeightGenerator(FermiEnergyAndDOSGenerator):
    def calc_weights_mat(self, k_mesh=(2, 2, 2), emin=-3.0, emax=3.0, sigma=0.1, n_jobs=1, parallel_k=False):
        # Pre-compute k-mesh and k-points for diagonalization
        self.kpoint_density = self._decide_k_mesh(k_mesh, dk=0.02)
        self.nktot = np.prod(self.kpoint_density)
        self.ks = np.stack(
            [
                ki.reshape(-1)
                for ki in np.meshgrid(*[np.arange(nk) * 1.0 / nk for nk in self.kpoint_density], indexing="ij")
            ],
            axis=1,
        )

        # Diagonalization for both eigvals and eigvecs
        print(f"Calculating eigvals and eigvecs with n_jobs={n_jobs}, parallel_k={parallel_k} ...")
        eigvals, eigvecs = self.obj_H.diag(
            self.ks, n_jobs=n_jobs, sparse_calc=False, bands_only=False, parallel_k=parallel_k
        )

        # Set self.eigvals to avoid recalculate them in self.find_fermi_energy
        self.eigvals = eigvals
        self.find_fermi_energy(
            k_mesh=k_mesh, n_jobs=n_jobs, parallel_k=parallel_k, method="counting", force_recalc=True
        )
        print(f"Fermi energy: {self.fermi_energy} eV")

        # eigvals: [Nband, Nktot] -> [Nktot, Nband]
        eigvals_T = self.eigvals.T

        # 1. Calculate window function
        x_1 = (eigvals_T - self.fermi_energy - emin) / sigma
        x_2 = (eigvals_T - self.fermi_energy - emax) / sigma
        mask_kb = np.logical_and(x_1 > -10.0, x_2 < 10.0)
        mask_b = np.any(mask_kb, axis=0)
        eigvals_T = eigvals_T[:, mask_b]
        x_1 = x_1[:, mask_b]
        x_2 = x_2[:, mask_b]
        weights_kb = 1.0 / (np.exp(-x_1) + 1.0) / (np.exp(x_2) + 1.0)

        # 2. Select eigenstates
        eigvecs = eigvecs[:, mask_b, :].transpose(2, 1, 0)  # [Nk, Nband_cut, Norb]
        Nk, Nband_cut, Norb = eigvecs.shape
        Norb_sq = Norb * Norb

        # 3. Determine chunk size (~512 MB peak memory per chunk)
        if parallel_k:
            batch_size = Nk
            set_num_threads(n_jobs)
        else:
            bytes_per_k = Nband_cut * Norb_sq * 40  # prod_flat(c128) + result_flat(c128) + abs2_flat(f64)
            max_chunk_bytes = 512 * 1024 * 1024
            batch_size = max(1, min(Nk, max_chunk_bytes // max(1, bytes_per_k)))

        # 4. Accumulate in k-point chunks
        degenerate_thres = sigma
        weights_total = np.sum(weights_kb)
        mat = np.zeros((Norb, Norb), dtype=np.float64)
        print(f"Processing {Nk} k-points in {(Nk-1)//batch_size+1} batches ...")

        for k_start in tqdm(range(0, Nk, batch_size), desc="Processing"):
            k_end = min(k_start + batch_size, Nk)
            ks_slice = slice(k_start, k_end)
            nk_chunk = k_end - k_start

            # Compute convolution kernel to merge degenerate states
            eigvals_chunk = eigvals_T[ks_slice]
            conv_band = np.exp(
                -(eigvals_chunk[:, :, None] - eigvals_chunk[:, None, :])**2 / degenerate_thres**2
            )
            conv_band /= np.sqrt(np.sum(conv_band**2, axis=2, keepdims=True))

            # Outer product of eigenstates -> flatten: [nk_chunk, Nband_cut, Norb^2]
            eigvecs_chunk = eigvecs[ks_slice]
            prod_flat = (eigvecs_chunk[:, :, :, None].conj() * eigvecs_chunk[:, :, None, :]).reshape(
                nk_chunk, Nband_cut, Norb_sq
            )

            # Convolution (matmul, BLAS-optimized)
            result_flat = np.matmul(conv_band, prod_flat)

            # Weighted accumulation: sum_{k,c} weights * |result|^2 -> [Norb, Norb]
            abs2_flat = np.abs(result_flat) ** 2
            mat += (abs2_flat * weights_kb[ks_slice, :, None]).sum(axis=(0, 1)).reshape(Norb, Norb)

        mat = np.sqrt(mat / weights_total * Norb_sq + 1.0e-8)
        self.weights_mat = mat

    def dump_h5(self, new_h5_path):
        self.obj_new = copy(self.obj_H)
        self.obj_new.SR = None
        self.obj_new.mats = self.weights_mat.reshape(1, *self.weights_mat.shape)
        R_to_index = {tuple(R): 0 for R in self.obj_new.Rijk_list}
        entries = self.obj_new._build_entries_H_like(R_to_index=R_to_index, dtype=np.float64)
        print("[info] Mean value:", np.mean(entries), "Max value:", np.max(entries), "Min value:", np.min(entries))

        with h5py.File(new_h5_path, "w") as f_new_h5:
            f_new_h5.create_dataset("atom_pairs", data=self.obj_new.atom_pairs)
            f_new_h5.create_dataset("chunk_boundaries", data=self.obj_new.bounds)
            f_new_h5.create_dataset("chunk_shapes", data=self.obj_new.shapes)
            f_new_h5.create_dataset("entries", data=entries)

