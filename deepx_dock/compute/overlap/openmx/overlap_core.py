"""
Core overlap matrix calculation using k-space method.

This module implements the OpenMX-style overlap calculation algorithm.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
from scipy.special import sph_harm, factorial
from scipy.integrate import trapezoid
from scipy.special import spherical_jn
import warnings

from .basis import ElementBasis, BasisSet


def spherical_bessel(l: int, x: np.ndarray) -> np.ndarray:
    """
    Spherical Bessel function j_l(x).
    """
    result = np.zeros_like(x, dtype=np.float64)
    nonzero = x != 0

    if l == 0:
        result[nonzero] = np.sin(x[nonzero]) / x[nonzero]
        result[~nonzero] = 1.0
    else:
        result[nonzero] = spherical_jn(l, x[nonzero])
        result[~nonzero] = 0.0

    return result


def compute_k_space_radial_function(
    r_grid: np.ndarray,
    radial_wf: np.ndarray,
    k_grid: np.ndarray,
    l: int,
) -> np.ndarray:
    """
    Compute k-space radial function via Fourier transform.

    R̃_l(k) = ∫ j_l(kr) R_l(r) r² dr
    """
    R_tilde = np.zeros(len(k_grid))

    for i, k in enumerate(k_grid):
        kr = k * r_grid
        jl = spherical_bessel(l, kr)
        integrand = radial_wf * jl * r_grid**2
        R_tilde[i] = trapezoid(integrand, r_grid)

    return R_tilde


def compute_radial_integral(
    R1_tilde: np.ndarray,
    R2_tilde: np.ndarray,
    k_grid: np.ndarray,
    R: float,
    l: int,
) -> float:
    """
    Compute radial integral in k-space.

    I_l(R) = ∫ R̃₁(k) R̃₂(k) j_l(kR) k² dk
    """
    kR = k_grid * R
    jl = spherical_bessel(l, kR)
    integrand = R1_tilde * R2_tilde * jl * k_grid**2

    return trapezoid(integrand, k_grid)


def real_sph_harm_openmx(l: int, m: int, theta: float, phi: float) -> float:
    """
    Compute real (tesseral) spherical harmonic in OpenMX convention.

    OpenMX m-ordering: m = 0, 1, -1, 2, -2, ...
    Real SH definition:
    - m = 0: Y_l^0 (same as complex)
    - m > 0: (Y_l^{-m} + (-1)^m Y_l^m) / sqrt(2)
    - m < 0: i(Y_l^{-|m|} - (-1)^{|m|} Y_l^{|m|}) / sqrt(2)
    """
    if m == 0:
        return np.real(sph_harm(0, l, phi, theta))
    elif m > 0:
        Y_m = sph_harm(m, l, phi, theta)
        Y_neg_m = sph_harm(-m, l, phi, theta)
        return np.real((Y_neg_m + (-1) ** m * Y_m) / np.sqrt(2))
    else:
        Y_pos_m = sph_harm(-m, l, phi, theta)
        Y_m = sph_harm(m, l, phi, theta)
        return np.imag((Y_pos_m - (-1) ** (-m) * Y_m) / np.sqrt(2))


def gaunt_coefficient_real(l1: int, m1: int, l2: int, m2: int, l: int, m: int) -> float:
    """
    Compute Gaunt coefficient for real spherical harmonics in OpenMX convention.

    C = ∫ Y_{l1}^{m1} Y_{l2}^{m2} Y_l^m dΩ

    Uses numerical integration.
    """
    if m != m1 + m2:
        return 0.0

    if abs(l1 - l2) > l or l > l1 + l2:
        return 0.0

    if abs(m) > l:
        return 0.0

    if (l1 + l2 + l) % 2 != 0:
        return 0.0

    n_theta = 30
    n_phi = 60

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    total = 0.0
    for th in theta:
        for ph in phi:
            Y1 = real_sph_harm_openmx(l1, m1, th, ph)
            Y2 = real_sph_harm_openmx(l2, m2, th, ph)
            Y3 = real_sph_harm_openmx(l, m, th, ph)
            integrand = Y1 * Y2 * Y3 * np.sin(th)
            total += integrand

    return total * dtheta * dphi


def precompute_gaunt_coefficients(lmax: int) -> Dict[Tuple[int, int, int, int, int, int], float]:
    """
    Precompute all Gaunt coefficients up to lmax.

    Uses vectorized numerical integration for efficiency.
    """
    n_theta = 30
    n_phi = 60

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    sin_theta = np.sin(THETA)

    Y_cache = {}
    for l in range(2 * lmax + 1):
        for m in _get_openmx_m_values(l):
            Y_cache[(l, m)] = real_sph_harm_openmx(l, m, THETA, PHI)

    gaunt_dict = {}
    l_int_max = 2 * lmax

    for l1 in range(lmax + 1):
        for m1 in _get_openmx_m_values(l1):
            Y1 = Y_cache[(l1, m1)]
            for l2 in range(lmax + 1):
                for m2 in _get_openmx_m_values(l2):
                    Y2 = Y_cache[(l2, m2)]
                    for l_int in range(abs(l1 - l2), min(l1 + l2, l_int_max) + 1):
                        m = m1 + m2
                        if abs(m) > l_int:
                            continue
                        if (l1 + l2 + l_int) % 2 != 0:
                            continue

                        if (l_int, m) not in Y_cache:
                            Y_cache[(l_int, m)] = real_sph_harm_openmx(l_int, m, THETA, PHI)
                        Y3 = Y_cache[(l_int, m)]

                        integrand = Y1 * Y2 * Y3 * sin_theta
                        integral = np.sum(integrand) * dtheta * dphi

                        if abs(integral) > 1e-15:
                            gaunt_dict[(l1, m1, l2, m2, l_int, m)] = float(integral)

    return gaunt_dict


def _get_openmx_m_values(l: int) -> List[int]:
    """Get m values in OpenMX order: 0, 1, -1, 2, -2, ..."""
    return [0] + [m for s in range(1, l + 1) for m in [s, -s]]


class OverlapCore:
    """
    Core overlap matrix calculator using k-space method.
    """

    def __init__(self, lmax_gaunt: int = 6):
        self.lmax_gaunt = lmax_gaunt
        self._gaunt_cache: Dict[Tuple[int, int, int, int, int, int], float] = {}

    def precompute_gaunt(self, lmax: int):
        self._gaunt_cache = precompute_gaunt_coefficients(lmax)

    def get_gaunt(self, l1: int, m1: int, l2: int, m2: int, l: int, m: int) -> float:
        key = (l1, m1, l2, m2, l, m)
        if key not in self._gaunt_cache:
            self._gaunt_cache[key] = gaunt_coefficient_real(l1, m1, l2, m2, l, m)
        return self._gaunt_cache[key]

    def compute_atom_pair_overlap(
        self,
        basis1: BasisSet,
        basis2: BasisSet,
        R_vec: np.ndarray,
        k_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Compute overlap matrix between two atoms.
        """
        R = float(np.linalg.norm(R_vec))
        if R < 1e-10:
            raise ValueError("Atoms are at the same position")

        theta = float(np.arccos(R_vec[2] / R))
        phi = float(np.arctan2(R_vec[1], R_vec[0]))

        lmax1 = basis1.metadata.lmax
        lmax2 = basis2.metadata.lmax
        num_mu1 = basis1.metadata.num_mu
        num_mu2 = basis2.metadata.num_mu

        r_grid1 = basis1.radial_grid.r
        r_grid2 = basis2.radial_grid.r

        R1_tilde = {}
        for L in range(lmax1 + 1):
            R1_tilde[L] = {}
            for mu in range(num_mu1):
                radial_wf = basis1.get_radial_wf(L, mu)
                R1_tilde[L][mu] = compute_k_space_radial_function(r_grid1, radial_wf, k_grid, L)

        R2_tilde = {}
        for L in range(lmax2 + 1):
            R2_tilde[L] = {}
            for mu in range(num_mu2):
                radial_wf = basis2.get_radial_wf(L, mu)
                R2_tilde[L][mu] = compute_k_space_radial_function(r_grid2, radial_wf, k_grid, L)

        l_int_max = 2 * max(lmax1, lmax2)

        radial_integrals = {}
        for L1 in range(lmax1 + 1):
            for mu1 in range(num_mu1):
                for L2 in range(lmax2 + 1):
                    for mu2 in range(num_mu2):
                        for l_int in range(abs(L1 - L2), min(L1 + L2, l_int_max) + 1):
                            if (L1 + L2 + l_int) % 2 != 0:
                                continue
                            key = (L1, mu1, L2, mu2, l_int)
                            if key not in radial_integrals:
                                radial_integrals[key] = compute_radial_integral(
                                    R1_tilde[L1][mu1], R2_tilde[L2][mu2], k_grid, R, l_int
                                )

        def get_basis_size(basis):
            size = 0
            for L in range(basis.metadata.lmax + 1):
                size += (2 * L + 1) * basis.metadata.num_mu
            return size

        n_basis1 = get_basis_size(basis1)
        n_basis2 = get_basis_size(basis2)
        S_block = np.zeros((n_basis1, n_basis2), dtype=np.float64)

        idx1 = 0
        for L1 in range(lmax1 + 1):
            for mu1 in range(num_mu1):
                for m1 in _get_openmx_m_values(L1):
                    idx2 = 0
                    for L2 in range(lmax2 + 1):
                        for mu2 in range(num_mu2):
                            for m2 in _get_openmx_m_values(L2):
                                S_val = 0.0

                                for l_int in range(abs(L1 - L2), min(L1 + L2, l_int_max) + 1):
                                    if (L1 + L2 + l_int) % 2 != 0:
                                        continue

                                    m = m1 + m2
                                    if abs(m) > l_int:
                                        continue

                                    C = self.get_gaunt(L1, m1, L2, m2, l_int, m)
                                    if abs(C) < 1e-15:
                                        continue

                                    I_radial = radial_integrals.get((L1, mu1, L2, mu2, l_int), 0.0)
                                    if I_radial is None:
                                        continue

                                    Y_lm = real_sph_harm_openmx(l_int, m, theta, phi)

                                    Ls = -L1 + L2 + l_int
                                    phase = (-1.0j) ** Ls

                                    S_val += float(np.real(phase * C * Y_lm * I_radial))

                                S_block[idx1, idx2] = S_val
                                idx2 += 1
                    idx1 += 1

        return S_block
