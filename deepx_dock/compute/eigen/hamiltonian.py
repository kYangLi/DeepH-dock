"""
Hamiltonian Object

This module provides the HamiltonianObj class for handling Hamiltonian and
overlap matrices, with support for diagonalization and ill-conditioned
eigenvalue handling.
"""

import os
from typing import List, Optional

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
import threadpoolctl

from deepx_dock.parallel import parallel_map
from deepx_dock.compute.eigen.matrix_obj import AOMatrixObj


class HamiltonianObj(AOMatrixObj):
    """
    Hamiltonian object containing both H and S matrices.

    This class extends AOMatrixObj to include the overlap matrix and provides
    methods for diagonalization with support for ill-conditioned eigenvalue
    handling.

    Parameters
    ----------
    data_path : str or Path
        Path to the directory containing DeepH format files.
    H_file_path : str or Path, optional
        Path to the Hamiltonian file. Default: hamiltonian.h5 under data_path.

    Properties
    ----------
    HR : np.ndarray
        Hamiltonian matrix in real space.
    SR : np.ndarray
        Overlap matrix in real space.
    """

    def __init__(self, data_path, H_file_path=None):
        super().__init__(data_path, H_file_path)
        overlap_obj = AOMatrixObj(data_path, matrix_type="overlap")
        self.assert_compatible(overlap_obj)
        self.SR = overlap_obj.mats

    @property
    def HR(self):
        """Hamiltonian matrix in real space."""
        return self.mats

    @staticmethod
    def _r2k(ks, Rijk_list, mats):
        """
        Fourier transform from real space to reciprocal space.

        Parameters
        ----------
        ks : np.ndarray, shape (Nk, 3) or (3,)
            k-points in fractional coordinates.
        Rijk_list : np.ndarray, shape (NR, 3)
            R-vectors in fractional coordinates.
        mats : np.ndarray, shape (NR, Nb, Nb)
            Matrices in real space.

        Returns
        -------
        MKs : np.ndarray, shape (Nk, Nb, Nb)
            Matrices in reciprocal space.
        """
        phase = np.exp(2j * np.pi * np.matmul(ks, Rijk_list.T))
        MRs_flat = mats.reshape(len(Rijk_list), -1)
        Mks_flat = np.matmul(phase, MRs_flat)
        return Mks_flat.reshape(len(ks), *mats.shape[1:])

    def Sk_and_Hk(self, k):
        """
        Get overlap and Hamiltonian matrices at given k-point(s).

        Parameters
        ----------
        k : np.ndarray
            k-point(s) in fractional coordinates. Shape (3,) for single k-point
            or (Nk, 3) for multiple k-points.

        Returns
        -------
        Sk, Hk : np.ndarray
            Overlap and Hamiltonian matrices at the k-point(s).
            Shape (Nb, Nb) for single k-point or (Nk, Nb, Nb) for multiple.
        """
        if k.ndim == 1:
            ks = k[None, :]
            squeeze = True
        else:
            ks = k
            squeeze = False

        Sk = self._r2k(ks, self.Rijk_list, self.SR)
        Hk = self._r2k(ks, self.Rijk_list, self.HR)

        if squeeze:
            return Sk[0], Hk[0]
        return Sk, Hk

    def diag(
        self,
        ks,
        n_jobs: int = -1,
        parallel_k: bool = True,
        sparse_calc: bool = False,
        bands_only: bool = True,
        ill_handler=None,
        kept_orbitals: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Diagonalize the Hamiltonian at specified k-points.

        Parameters
        ----------
        ks : array_like, shape (Nk, 3)
            k-points in reduced coordinates (fractional).
        n_jobs : int, optional
            Number of parallel workers. Default is -1 (auto-detect CPU cores).
            - If parallel_k=True: number of k-points to process in parallel.
            - If parallel_k=False: number of BLAS threads for diagonalization.
        parallel_k : bool, optional
            Parallelization strategy. Default is True.
            - True: Multiple k-points in parallel, each with 1 BLAS thread.
            - False: K-points processed sequentially, each with n_jobs BLAS threads.
        sparse_calc : bool, optional
            If True, use sparse solver (eigsh). Default is False.
        bands_only : bool, optional
            If True, only compute eigenvalues. Default is True.
        ill_handler : IllConditionedHandler, optional
            Handler for ill-conditioned eigenvalues.
        kept_orbitals : list of int, optional
            Indices of orbitals to keep (for orbital removal mode).
        **kwargs : dict
            Additional arguments passed to the solver.

        Returns
        -------
        eigvals : np.ndarray, shape (Nband, Nk)
            Eigenvalues (band energies).
        eigvecs : np.ndarray, shape (Norb, Nband, Nk), optional
            Eigenvectors (only if bands_only is False).

        Notes
        -----
        The parallel_k parameter controls the parallelization strategy:

        - parallel_k=True (default): Best for many small k-point calculations.
          Multiple k-points are processed concurrently, each using single-threaded
          BLAS. This avoids thread oversubscription and cache contention.

        - parallel_k=False: Best for few large matrix diagonalizations.
          K-points are processed one at a time, but each diagonalization uses
          multi-threaded BLAS for faster matrix operations.
        """
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1

        HR = self.HR
        SR = self.SR

        def process_k(k):
            Sk = self._r2k(k[None, :], self.Rijk_list, SR)[0]
            Hk = self._r2k(k[None, :], self.Rijk_list, HR)[0]

            if ill_handler is not None:
                return ill_handler.process_k(Hk, Sk, return_vecs=not bands_only)

            if kept_orbitals is not None:
                from deepx_dock.compute.eigen.ill_conditioned import eig_with_orbital_mask

                return eig_with_orbital_mask(Hk, Sk, kept_orbitals, return_vecs=not bands_only)

            if sparse_calc:
                if bands_only:
                    vals = eigsh(Hk, M=Sk, return_eigenvectors=False, **kwargs)
                    return np.sort(vals)
                else:
                    vals, vecs = eigsh(Hk, M=Sk, **kwargs)
                    idx = np.argsort(vals)
                    return vals[idx], vecs[:, idx]
            else:
                if bands_only:
                    vals = eigh(Hk, Sk, eigvals_only=True)
                    return vals
                else:
                    vals, vecs = eigh(Hk, Sk)
                    return vals, vecs

        if parallel_k:
            n_blas_threads = 1
            with threadpoolctl.threadpool_limits(limits=n_blas_threads, user_api="blas"):
                if n_jobs == 1:
                    results = [process_k(k) for k in tqdm(ks, leave=False, desc="Diagonalizing")]
                else:
                    results = parallel_map(process_k, ks, n_jobs=n_jobs, desc="Diagonalizing")
        else:
            n_blas_threads = n_jobs
            with threadpoolctl.threadpool_limits(limits=n_blas_threads, user_api="blas"):
                results = [process_k(k) for k in tqdm(ks, leave=False, desc="Diagonalizing")]

        if bands_only:
            return np.stack(results, axis=1)
        else:
            eigvals = np.stack([res[0] for res in results], axis=1)
            eigvecs = np.stack([res[1] for res in results], axis=2)
            return eigvals, eigvecs

    def get_all_Sk(self, ks, n_jobs: int = -1, parallel_k: bool = True):
        """
        Get overlap matrices for all k-points.

        This method is useful for the orbital removal algorithm which needs
        all Sk matrices to determine which orbitals to remove globally.

        Parameters
        ----------
        ks : np.ndarray, shape (Nk, 3)
            k-points in fractional coordinates.
        n_jobs : int, optional
            Number of parallel workers. Default is -1 (auto-detect CPU cores).
        parallel_k : bool, optional
            Parallelization strategy. Default is True.
            See diag() method for details.

        Returns
        -------
        Sk_list : list of np.ndarray
            List of overlap matrices, each with shape (Nb, Nb).
        """
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1

        def get_Sk(k):
            return self._r2k(k[None, :], self.Rijk_list, self.SR)[0]

        if parallel_k:
            n_blas_threads = 1
            with threadpoolctl.threadpool_limits(limits=n_blas_threads, user_api="blas"):
                if n_jobs == 1:
                    Sk_list = [get_Sk(k) for k in tqdm(ks, leave=False, desc="Getting Sk")]
                else:
                    Sk_list = parallel_map(get_Sk, ks, n_jobs=n_jobs, desc="Getting Sk")
        else:
            n_blas_threads = n_jobs
            with threadpoolctl.threadpool_limits(limits=n_blas_threads, user_api="blas"):
                Sk_list = [get_Sk(k) for k in tqdm(ks, leave=False, desc="Getting Sk")]

        return Sk_list
