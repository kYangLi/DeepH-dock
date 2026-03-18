"""
Ill-Conditioned Eigenvalue Handler

This module provides algorithms for handling ill-conditioned eigenvalues
in generalized eigenvalue problems arising from non-orthogonal basis sets.

Two methods are implemented:
1. Window Regularization: Projects out ill-conditioned components within
   a specified energy window based on the quality matrix eigenanalysis.
2. Orbital Removal: Globally identifies and removes orbitals that cause
   ill-conditioning across all k-points.

References
----------
The algorithms are based on the PostProcess scripts for handling
ill-conditioned Hamiltonian predictions from neural network models.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh


def window_regularization_eig(
    Hk: np.ndarray,
    Sk: np.ndarray,
    emin: float,
    emax: float,
    ill_threshold: float,
    fill_value: float = 1e4,
    return_vecs: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Window regularization algorithm for ill-conditioned eigenvalues.

    This method projects out ill-conditioned eigenvalue components within
    a specified energy window by analyzing the quality matrix (mass matrix)
    constructed from the filtered eigenvectors.

    Algorithm
    ---------
    1. Solve the full generalized eigenvalue problem: H ψ = E S ψ
    2. Filter eigenvectors within energy window [emin, emax]
    3. Construct quality matrix: M = Ψ† Ψ (measures linear independence)
    4. Diagonalize M to find nearly linear-dependent combinations
    5. Project onto well-conditioned subspace (λ_M > threshold)
    6. Re-solve in the reduced subspace

    Parameters
    ----------
    Hk : np.ndarray, shape (Norb, Norb)
        Hamiltonian matrix at k-point.
    Sk : np.ndarray, shape (Norb, Norb)
        Overlap matrix at k-point.
    emin : float
        Minimum energy of the window (absolute, not relative to Fermi).
    emax : float
        Maximum energy of the window (absolute, not relative to Fermi).
    ill_threshold : float
        Threshold for quality matrix eigenvalues. Eigenvectors with
        quality eigenvalue below this threshold are considered ill-conditioned.
    fill_value : float, default=1e4
        Value used to fill removed eigenvalue slots.
    return_vecs : bool, default=False
        If True, also return eigenvectors.

    Returns
    -------
    eigvals : np.ndarray, shape (Norb,)
        Processed eigenvalues (sorted in ascending order).
    eigvecs : np.ndarray, shape (Norb, Norb), optional
        Processed eigenvectors (only if return_vecs=True).
    """
    norb = Hk.shape[0]

    eigvals, eigvecs = eigh(Hk, Sk)

    mask = (eigvals >= emin) & (eigvals <= emax)
    n_window = np.sum(mask)

    if n_window == 0:
        if return_vecs:
            return eigvals, eigvecs
        return eigvals, None

    filtered_vals = eigvals[mask]
    filtered_vecs = eigvecs[:, mask]

    mass_matrix = filtered_vecs.T.conj() @ filtered_vecs

    mass_eigvals, mass_eigvecs = eigh(mass_matrix)

    project_mask = mass_eigvals > ill_threshold
    n_good = np.sum(project_mask)

    if n_good < n_window:
        good_mass_vecs = mass_eigvecs[:, project_mask]

        H_sub = np.diag(filtered_vals)
        M_sub = good_mass_vecs.T.conj() @ good_mass_vecs

        try:
            sub_vals, sub_vecs = eigh(H_sub, M_sub)
        except np.linalg.LinAlgError:
            if return_vecs:
                return eigvals, eigvecs
            return eigvals, None

        final_vecs_subspace = good_mass_vecs @ sub_vecs
        final_vecs = filtered_vecs @ final_vecs_subspace

        final_vals = np.full(norb, fill_value)
        final_vals[:n_good] = sub_vals

        sort_idx = np.argsort(final_vals)
        final_vals = final_vals[sort_idx]
        final_vecs = final_vecs[:, sort_idx]

        if return_vecs:
            return final_vals, final_vecs
        return final_vals, None

    if return_vecs:
        return eigvals, eigvecs
    return eigvals, None


def global_orbital_truncation(Sk_list: List[np.ndarray], ill_threshold: float, verbose: bool = True) -> List[int]:
    """
    Global orbital truncation algorithm.

    This method iteratively removes orbitals that cause ill-conditioning
    in the overlap matrix. The removal is done globally across all k-points
    to maintain consistency.

    Algorithm
    ---------
    1. For each k-point, find the smallest eigenvalue of S(k) and its eigenvector
    2. Find the k-point with the globally smallest S eigenvalue
    3. Identify the orbital with largest amplitude in that eigenvector
    4. Remove that orbital from all k-points
    5. Repeat until all S eigenvalues are above threshold

    Parameters
    ----------
    Sk_list : list of np.ndarray
        List of overlap matrices for all k-points. Each has shape (Norb, Norb).
    ill_threshold : float
        Threshold for minimum S eigenvalue. Orbitals are removed until
        all S(k) have minimum eigenvalue above this threshold.
    verbose : bool, default=True
        If True, print progress information.

    Returns
    -------
    kept_orbitals : list of int
        Indices of orbitals that are kept (in original numbering).
    """
    norb = Sk_list[0].shape[0]
    remaining = list(range(norb))

    Sk_work = [Sk.copy() for Sk in Sk_list]

    iteration = 0
    while True:
        nk = len(Sk_work)
        min_eigvals = np.zeros(nk)
        min_eigvecs = [None] * nk

        for ik, Sk in enumerate(Sk_work):
            vals, vecs = eigh(Sk)
            min_eigvals[ik] = vals[0]
            min_eigvecs[ik] = vecs[:, 0]

        k_worst = int(np.argmin(min_eigvals))
        global_min = min_eigvals[k_worst]

        if global_min > ill_threshold:
            break

        v = min_eigvecs[k_worst]
        worst_local_idx = int(np.argmax(np.abs(v)))
        orbital_to_remove = remaining[worst_local_idx]

        if verbose:
            print(f"[iter {iteration}] ill_value = {global_min:.2e}, removing orbital {orbital_to_remove}")

        for ik in range(len(Sk_work)):
            n_curr = Sk_work[ik].shape[0]
            keep = [j for j in range(n_curr) if j != worst_local_idx]
            if len(keep) > 0:
                Sk_work[ik] = Sk_work[ik][np.ix_(keep, keep)]
            else:
                Sk_work[ik] = np.zeros((0, 0), dtype=Sk_work[ik].dtype)

        remaining.remove(orbital_to_remove)

        if len(remaining) == 0:
            if verbose:
                print("[warning] All orbitals removed, stopping")
            break

        iteration += 1

    if verbose:
        print(f"[done] Kept {len(remaining)}/{norb} orbitals")

    return remaining


def eig_with_orbital_mask(
    Hk: np.ndarray, Sk: np.ndarray, kept_orbitals: List[int], fill_value: float = 1e4, return_vecs: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Solve eigenvalue problem with orbital mask.

    This function diagonalizes the Hamiltonian in a reduced orbital space
    defined by the kept_orbitals list. Removed orbitals are represented
    by fill_value in the output.

    Parameters
    ----------
    Hk : np.ndarray, shape (Norb, Norb)
        Hamiltonian matrix at k-point.
    Sk : np.ndarray, shape (Norb, Norb)
        Overlap matrix at k-point.
    kept_orbitals : list of int
        Indices of orbitals to keep.
    fill_value : float, default=1e4
        Value used to fill removed orbital slots.
    return_vecs : bool, default=False
        If True, also return eigenvectors.

    Returns
    -------
    eigvals : np.ndarray, shape (Norb,)
        Eigenvalues with removed slots filled.
    eigvecs : np.ndarray, shape (Norb, N_kept), optional
        Eigenvectors (only if return_vecs=True). Note that the shape
        is (Norb, N_kept) not (Norb, Norb) because removed orbitals
        have no meaningful components.
    """
    norb_total = Hk.shape[0]
    n_kept = len(kept_orbitals)

    if n_kept == 0:
        eigvals = np.full(norb_total, fill_value)
        if return_vecs:
            return eigvals, np.zeros((norb_total, 0), dtype=Hk.dtype)
        return eigvals, None

    idx = np.ix_(kept_orbitals, kept_orbitals)
    Hk_sub = Hk[idx]
    Sk_sub = Sk[idx]

    try:
        if return_vecs:
            vals, vecs_sub = eigh(Hk_sub, Sk_sub)

            full_vecs = np.zeros((norb_total, n_kept), dtype=vecs_sub.dtype)
            full_vecs[kept_orbitals, :] = vecs_sub

            full_vals = np.full(norb_total, fill_value)
            full_vals[:n_kept] = vals

            return np.sort(full_vals), full_vecs[:, np.argsort(vals)]
        else:
            vals = eigh(Hk_sub, Sk_sub, eigvals_only=True)

            full_vals = np.full(norb_total, fill_value)
            full_vals[:n_kept] = vals

            return np.sort(full_vals), None

    except np.linalg.LinAlgError:
        eigvals = np.full(norb_total, fill_value)
        if return_vecs:
            return eigvals, np.zeros((norb_total, 0), dtype=Hk.dtype)
        return eigvals, None


class IllConditionedHandler:
    """
    High-level interface for handling ill-conditioned eigenvalues.

    This class provides a unified interface for both window regularization
    and orbital removal methods, managing the necessary preprocessing
    and state for each method.

    Parameters
    ----------
    method : str
        Method to use: 'window_regularization' or 'orbital_removal'.
    ill_threshold : float, default=1e-3
        Threshold for ill-conditioning detection.
    window_emin : float, default=-1000.0
        Minimum energy for window (relative to Fermi energy, in eV).
    window_emax : float, default=6.0
        Maximum energy for window (relative to Fermi energy, in eV).
    fermi_energy : float, default=0.0
        Fermi energy in eV. Window boundaries are shifted by this value.
    fill_value : float, default=1e4
        Value used to fill removed eigenvalue slots.
    verbose : bool, default=True
        If True, print progress information.

    Attributes
    ----------
    kept_orbitals : list of int or None
        For orbital removal mode, stores the indices of kept orbitals
        after preprocessing.
    """

    METHOD_WINDOW = "window_regularization"
    METHOD_ORBITAL = "orbital_removal"

    def __init__(
        self,
        method: str,
        ill_threshold: float = 1e-3,
        window_emin: float = -1000.0,
        window_emax: float = 6.0,
        fermi_energy: float = 0.0,
        fill_value: float = 1e4,
        verbose: bool = True,
    ):
        if method not in (self.METHOD_WINDOW, self.METHOD_ORBITAL):
            raise ValueError(f"Unknown method: {method}. Choose from '{self.METHOD_WINDOW}' or '{self.METHOD_ORBITAL}'")

        self.method = method
        self.ill_threshold = ill_threshold
        self.window_emin = window_emin + fermi_energy
        self.window_emax = window_emax + fermi_energy
        self.fill_value = fill_value
        self.verbose = verbose

        self._kept_orbitals: Optional[List[int]] = None

    @property
    def kept_orbitals(self) -> Optional[List[int]]:
        """Indices of orbitals kept after orbital removal preprocessing."""
        return self._kept_orbitals

    def prepare_orbital_truncation(self, Sk_list: List[np.ndarray]) -> List[int]:
        """
        Precompute orbital truncation for orbital removal mode.

        This method must be called before `process_k` when using the
        orbital removal method. It computes which orbitals to keep
        based on global analysis of all overlap matrices.

        Parameters
        ----------
        Sk_list : list of np.ndarray
            List of overlap matrices for all k-points.

        Returns
        -------
        kept_orbitals : list of int
            Indices of orbitals to keep.

        Raises
        ------
        RuntimeError
            If the handler is not configured for orbital removal mode.
        """
        if self.method != self.METHOD_ORBITAL:
            raise RuntimeError(
                f"prepare_orbital_truncation() is only valid for orbital removal mode. Current method: {self.method}"
            )
        self._kept_orbitals = global_orbital_truncation(Sk_list, self.ill_threshold, self.verbose)
        return self._kept_orbitals

    def process_k(self, Hk: np.ndarray, Sk: np.ndarray, return_vecs: bool = False):
        """
        Process a single k-point.

        Parameters
        ----------
        Hk : np.ndarray, shape (Norb, Norb)
            Hamiltonian matrix at k-point.
        Sk : np.ndarray, shape (Norb, Norb)
            Overlap matrix at k-point.
        return_vecs : bool, default=False
            If True, also return eigenvectors.

        Returns
        -------
        eigvals : np.ndarray, shape (Norb,)
            Processed eigenvalues.
        eigvecs : np.ndarray or None (only if return_vecs=True)
            Processed eigenvectors.

        Raises
        ------
        RuntimeError
            If orbital removal mode is used but prepare_orbital_truncation
            has not been called.
        """
        if self.method == self.METHOD_WINDOW:
            eigvals, eigvecs = window_regularization_eig(
                Hk, Sk, self.window_emin, self.window_emax, self.ill_threshold, self.fill_value, return_vecs
            )
        elif self.method == self.METHOD_ORBITAL:
            if self._kept_orbitals is None:
                raise RuntimeError("Orbital removal mode requires calling prepare_orbital_truncation() first")
            eigvals, eigvecs = eig_with_orbital_mask(Hk, Sk, self._kept_orbitals, self.fill_value, return_vecs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if return_vecs:
            return eigvals, eigvecs
        else:
            return eigvals

    def get_info(self) -> dict:
        """
        Get information about the handler configuration.

        Returns
        -------
        info : dict
            Dictionary containing method parameters and state.
        """
        info = {
            "method": self.method,
            "ill_threshold": self.ill_threshold,
            "window_emin": self.window_emin,
            "window_emax": self.window_emax,
            "fill_value": self.fill_value,
        }
        if self._kept_orbitals is not None:
            info["n_kept_orbitals"] = len(self._kept_orbitals)
            info["kept_orbitals"] = self._kept_orbitals
        return info
