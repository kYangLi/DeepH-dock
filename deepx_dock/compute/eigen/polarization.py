"""
Electric polarization via Berry phase (modern theory of polarization).

This module implements the bulk electronic polarization following the
algorithm of OpenMX's ``polB.c`` (first-order, symmetrized formulation).
Both spinless and spinful (non-collinear spinor) systems are supported.

The first-order (k-mesh convergent) formulation requires the position
operator matrix ``<chi_i | r | chi_j>`` in addition to ``H`` and ``S``.
The position matrix is supplied as a DeepH-format dict with keys
``atom_pairs``, ``chunk_shapes``, ``chunk_boundaries`` and ``entries``,
where ``entries`` has shape ``(3, n_total)`` ordered x, y, z.  Its
coordinates are referenced to the **lattice origin**, in Angstrom.
"""

from __future__ import annotations

import numpy as np
import os
import threadpoolctl
from typing import Dict, List, Optional, Tuple

from deepx_dock.compute.eigen.matrix_obj import AOMatrixObj
from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
from deepx_dock.CONSTANT import ANGSTROM_TO_BOHR
from deepx_dock.parallel import parallel_map
from deepx_dock.misc import set_num_threads

# 1 Debye per e*bohr and muC/cm^2 per e/bohr/bohr (atomic-unit based, see polB.c)
AU2DEBYE = 2.54174776
AU2MUCM = 5721.52891433


def assemble_position_matrix(position_matrix, obj_H):
    """
    Assemble the position operator matrix ``<chi_i | r | chi_j>``.

    ``position_matrix["entries"]`` has shape ``(3, n_total)`` ordered x, y, z
    (absolute coordinates, Angstrom).  Each component is a standard DeepH
    scalar matrix, so ``AOMatrixObj._assemble_matrix_from_deeph_data`` is
    reused per component (as ``matrix_type="overlap"``, spinless).

    Parameters
    ----------
    position_matrix : dict
        DeepH-format dict with ``atom_pairs`` (N, 5), ``chunk_shapes`` (N, 2),
        ``chunk_boundaries`` (N+1,) and ``entries`` (3, n_total).
    obj_H : HamiltonianObj
        Used for ``atom_num_orbits`` (spinless orbital count) and
        ``Rijk_list`` ordering.

    Returns
    -------
    r_abs : np.ndarray, shape (N_R, 3, Norb, Norb) or (N_R, 3, 2*Norb, 2*Norb)
        Position matrix in real space, aligned to ``obj_H.Rijk_list``.
        For spinful systems it is expanded to a block-diagonal
        ``[[r, 0], [0, r]]`` spinor structure (the position operator is
        spin-diagonal), with ``Norb = obj_H.orbits_quantity``.
    """
    atom_pairs = np.asarray(position_matrix["atom_pairs"])
    chunk_boundaries = np.asarray(position_matrix["chunk_boundaries"])
    chunk_shapes = np.asarray(position_matrix["chunk_shapes"])
    entries = np.asarray(position_matrix["entries"])

    if entries.ndim != 2 or entries.shape[0] != 3:
        raise ValueError(
            f"position matrix entries must have shape (3, n_total), got {entries.shape}"
        )

    components = []
    Rijk = None
    for a in range(3):
        R, mat = AOMatrixObj._assemble_matrix_from_deeph_data(
            atom_pairs,
            chunk_boundaries,
            chunk_shapes,
            entries[a],
            obj_H.atom_num_orbits,
            obj_H.spinful,
            matrix_type="overlap",
        )
        if Rijk is None:
            Rijk = R
        elif not np.array_equal(R, Rijk):
            raise ValueError("position matrix Rijk_list inconsistent across x/y/z components")
        components.append(mat)

    r_abs = np.stack(components, axis=0)  # (3, N_R, Norb, Norb)
    r_abs = r_abs.transpose(1, 0, 2, 3)  # (N_R, 3, Norb, Norb)

    # Align to obj_H.Rijk_list ordering.
    target = obj_H.Rijk_list
    idx = []
    for R in target:
        match = np.where((Rijk == R).all(axis=1))[0]
        if len(match) != 1:
            raise ValueError(f"R-vector {R} not found (or duplicated) in position matrix")
        idx.append(match[0])
    return r_abs[np.array(idx)] # (N_R, 3, Norb, Norb)


class PolCalc:
    """
    Electric polarization calculator (Berry phase).

    Parameters
    ----------
    obj_H : HamiltonianObj
        Hamiltonian object with real-space ``HR``/``SR`` and diagonalization
        support.
    position_matrix : dict
        DeepH-format position operator matrix dict with keys:
        ``atom_pairs`` (N, 5),
        ``chunk_shapes`` (N, 2),
        ``chunk_boundaries`` (N+1,),
        ``entries`` (3, N_entries).
    element_ion_charge : dict[str, float]
        Map from element symbol to the ionic (pseudo) charge.
    occupation : int, optional
        Total number of electrons.  The number of occupied bands is derived
        as ``occupation / (2 - spinful)`` (spinless: two electrons per band;
        spinful: one electron per spinor band).  If ``None``, it is read from
        ``obj_H.occupation``.  Provide it explicitly when ``obj_H.occupation``
        is unavailable.
    """

    def __init__(
        self,
        obj_H: HamiltonianObj,
        position_matrix: Dict,
        element_ion_charge: Dict[str, float],
        occupation: Optional[int] = None,
    ):
        self.spinful = obj_H.spinful

        self.obj_H = obj_H
        self.lattice = np.asarray(obj_H.lattice, dtype=float)
        self.rlv = np.asarray(obj_H.reciprocal_lattice, dtype=float)
        self.frac_coords = np.asarray(obj_H.frac_coords, dtype=float)
        self.orbits_quantity = obj_H.orbits_quantity
        self.n_bands = self.orbits_quantity * (1 + self.spinful)
        self.atoms_quantity = obj_H.atoms_quantity
        self.volume = abs(np.linalg.det(self.lattice))

        self._build_tau_cart()
        self.r_abs = assemble_position_matrix(position_matrix, obj_H) # (N_R, 3, Norb or 2Norb, ...)

        self.z_ion = np.array([element_ion_charge[el] for el in obj_H.elements], dtype=float)  # (Natom,)
        self.cart_coords = self.frac_coords @ self.lattice  # (Natom, 3)

        if occupation is None:
            occupation = obj_H.occupation
        if occupation is None:
            raise ValueError(
                "occupation is None; provide the total number of electrons explicitly."
            )
        self.n_occ = int(round(float(occupation) / (2 - self.spinful)))
        if self.n_occ <= 0 or self.n_occ > self.n_bands:
            raise ValueError(f"Invalid number of occupied bands n_occ={self.n_occ}")

    # -------------------------------- setup ----------------------------------
    def _build_tau_cart(self):
        """Cartesian (absolute, Angstrom) positions of every orbital's atom."""
        cumsum = self.obj_H.atom_num_orbits_cumsum
        atom_of_orbital = np.zeros(self.orbits_quantity, dtype=int)
        for a in range(self.atoms_quantity):
            atom_of_orbital[cumsum[a]:cumsum[a + 1]] = a
        self.atom_of_orbital = atom_of_orbital
        self.tau_cart = self.frac_coords[atom_of_orbital] @ self.lattice  # (Norb, 3)
        if self.spinful:
            # Duplicate the spinless orbital positions for the two spinor blocks.
            self.tau_cart = np.concatenate([self.tau_cart, self.tau_cart], axis=0)  # (2Norb, 3)

    # ------------------------------- kpoints ---------------------------------
    @staticmethod
    def _ks(k_mesh):
        n1, n2, n3 = k_mesh
        ks = np.array(
            [
                [i1 / n1, i2 / n2, i3 / n3] # gamma center
                for i3 in range(n3)
                for i2 in range(n2)
                for i1 in range(n1) # i1 runs fastest
            ]
        )
        return ks  # flat index i1 + n1*(i2 + n2*i3)

    @staticmethod
    def _inverse_k_idx_map(k_mesh):
        """Map k to -k (mod 1). Note that 0 -> 0, 1/2 -> 1/2."""
        n1, n2, n3 = k_mesh
        neg = np.zeros(n1 * n2 * n3, dtype=int)
        for i1 in range(n1):
            for i2 in range(n2):
                for i3 in range(n3):
                    f = i1 + n1 * (i2 + n2 * i3)
                    nf = (n1 - i1) % n1 + n1 * ((n2 - i2) % n2 + n2 * ((n3 - i3) % n3))
                    neg[f] = nf
        return neg

    @staticmethod
    def _string_k_idx(k_mesh, i, ia, ib, direction, trans):
        """Flat mesh index of the point with string index i and transverse (ia, ib)."""
        idx = [0, 0, 0]
        idx[direction] = i
        idx[trans[0]] = ia
        idx[trans[1]] = ib
        return idx[0] + k_mesh[0] * (idx[1] + k_mesh[1] * idx[2])

    # ------------------------------ r2k / k2r --------------------------------
    @staticmethod
    def _r2k_complex(mats_R, ks, Rijk_list):
        """Fourier transform a complex (N_R, ...) matrix to k-space (N_k, ...)."""
        phase = np.exp(2j * np.pi * (ks @ Rijk_list.T))  # (Nk, N_R)
        flat = mats_R.reshape(mats_R.shape[0], -1)  # (N_R, -1)
        out = phase @ flat  # (N_k, -1)
        return out.reshape(len(ks), *mats_R.shape[1:])

    def _build_tR(self, k_mesh, direction, sign=1.0):
        """
        Real-space ``t(R)`` for one direction.

        ``t(R) = D @ ( S(R) - i * sum_a dk_cart[a] * P1[a](R) )`` with
        ``D = diag(exp(-i dk_cart . tau_i))`` and ``P1[a]`` the position
        matrix relative to the bra atom ``i``.

        Parameters
        ----------
        k_mesh : tuple[int, int, int]
            Gamma-centered k-mesh ``(n1, n2, n3)``.
        direction : int
            Polarization direction (0, 1, 2).
        sign : float, optional
            Sign of ``dk`` (``+1`` for k1->k2, ``-1`` for the reversed
            direction).  Used to test the Hermiticity identity.
        """
        n = k_mesh[direction]
        dk_frac = np.zeros(3)
        dk_frac[direction] = sign / n
        dk_cart = dk_frac @ self.rlv  # (3,)

        tau_cart = self.tau_cart
        phase = np.exp(-1j * (tau_cart @ dk_cart))  # (Norb,)

        SR = self.obj_H.SR  # (N_R, Norb, Norb)
        r1 = self.r_abs - tau_cart.T[None, :, :, None] * SR[:, None, :, :] # (N_R, 3, Norb, Norb)
        r1sum = np.einsum("a,Raij->Rij", dk_cart, r1)
        tR = np.einsum("i,Rij->Rij", phase, SR - 1j * r1sum) # (N_R, Norb, Norb) complex

        return tR

    # ----------------------------- core funcs --------------------------------
    def electronic_polarization(self, k_mesh: List | Tuple, **diag_kwargs):
        """
        Compute the electronic polarization (Berry phase).

        Parameters
        ----------
        k_mesh : tuple[int, int, int]
            Gamma-centered k-mesh ``(n1, n2, n3)``.
        **diag_kwargs : dict
            Extra keyword arguments forwarded to ``obj_H.diag``
            (e.g. ``n_jobs``, ``parallel_k``).

        Returns
        -------
        dipole_frac : np.ndarray, shape (3,)
            Electronic dipole in fractional coordinates, with the electron
            charge (-e) folded in, so that
            ``dipole_elec = dipole_frac @ lattice``.
        """
        _k_mesh = tuple(int(n) for n in k_mesh)
        ks = self._ks(_k_mesh)
        ## eigvals: (Nband, Nk); eigvecs: (Norb, Nband, Nk)
        self.eigvals, eigvecs = self.obj_H.diag(ks, bands_only=False, **diag_kwargs)

        # Metal detection: overlap between HOMO (n_occ-1) and LUMO (n_occ) bands.
        if self.n_occ >= self.n_bands:
            # No empty band to define a gap; treat as metallic.
            self.band_gap = float("nan")
            self._warn_metal(self.band_gap)
        else:
            E_homo = self.eigvals[self.n_occ - 1]  # (Nk,)
            E_lumo = self.eigvals[self.n_occ]  # (Nk,)
            gap = float(E_lumo.min() - E_homo.max())
            self.band_gap = gap
            if gap <= 1e-4: # LUMO <= HOMO means metal or semi-metal
                self._warn_metal(gap)

        eigvecs_occ = eigvecs[:, :self.n_occ, :]  # (Norb, n_occ, Nk)
        inv_k_idx = self._inverse_k_idx_map(_k_mesh)

        ## heavy calculations. needs parallel computing
        n_jobs = diag_kwargs.get("n_jobs", -1)
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        parallel_k = diag_kwargs.get("parallel_k", True)
        n_blas_threads = 1 if parallel_k else n_jobs
        n_jobs = n_jobs if parallel_k else 1
        set_num_threads(n_blas_threads)

        wcc_frac = np.zeros(3)
        for direction in range(3): # calc polarization along this direction
            tR = self._build_tR(_k_mesh, direction)
            tk = self._r2k_complex(tR, ks, self.obj_H.Rijk_list)  # (Nk, Norb, Norb)

            trans = [a for a in range(3) if a != direction]
            na, nb = _k_mesh[trans[0]], _k_mesh[trans[1]]

            with threadpoolctl.threadpool_limits(limits=n_blas_threads, user_api="blas"):
                if n_jobs == 1:
                    berry_phases = np.empty((na, nb), dtype=np.float64)
                    for ia in range(na):
                        for ib in range(nb):
                            inputs = self._berry_phase_inputs(
                                _k_mesh, ia, ib, direction, trans, eigvecs_occ, tk, inv_k_idx
                            )
                            berry_phases[ia, ib] = self._berry_phase_1d(inputs)
                else:
                    worker_inputs = [self._berry_phase_inputs(
                        _k_mesh, ia, ib, direction, trans, eigvecs_occ, tk, inv_k_idx
                    ) for ia in range(na) for ib in range(nb)]
                    berry_phases = parallel_map(
                        self._berry_phase_1d, worker_inputs, n_jobs=n_jobs,
                        desc=f"Berry phase along k{direction+1}",
                    )
                    berry_phases = np.asarray(berry_phases, dtype=np.float64).reshape((na, nb))

            ## align the wrapped berry phases into a continuous function on
            ## the transverse torus, so that the average has no branch jumps
            berry_phases, winding = self._align_berry_phases(berry_phases)
            if winding != (0, 0):
                self._warn_topological(winding, direction, trans)
            ## integrate on the transverse directions (ka and kb)
            ## the minus sign comes from the definition of berry phase and wannier charge center
            ## berry_phase ~= \int <u(k)|u(k+dk)>dk ~= -1 * wannier_charge_center
            wcc_frac[direction] = - (2 - self.spinful) * np.mean(berry_phases)

        ## the minus sign comes from the negative charge of electrons
        self.dipole_frac_elec = -wcc_frac
        self.dipole_elec = -wcc_frac @ self.lattice  # e*Angstrom
        return -wcc_frac

    @staticmethod
    def _berry_phase_inputs(
        _k_mesh, ia, ib, direction, trans, eigvecs_occ, tk, inv_k_idx
    ):
        n = _k_mesh[direction]
        k1_idxs = np.array([
            PolCalc._string_k_idx(_k_mesh, i, ia, ib, direction, trans)
            for i in range(n)
        ])
        k2_idxs = np.array([
            PolCalc._string_k_idx(_k_mesh, (i + 1) % n, ia, ib, direction, trans)
            for i in range(n)
        ])
        c1 = eigvecs_occ[:, :, k1_idxs]  # (Norb, n_occ, n)
        c2 = eigvecs_occ[:, :, k2_idxs]
        t12 = (tk[k2_idxs] + tk[inv_k_idx[k1_idxs]].transpose(0, 2, 1)) / 2
        return c1, c2, t12

    @staticmethod
    def _berry_phase_1d(inputs):
        c1, c2, t12 = inputs
        ## <u_k1|u_k2> = c_k1^* <φ1|exp(-i(k2-k1)r)|φ2> c_k2 = c_k1^* t_k1k2 c_k2
        Sop = np.einsum("imb,bij,jnb->bmn", c1.conj(), t12, c2)
        dets = np.linalg.det(Sop) # operates on the last two dims
        ## berry phase of this string (in units of 2π)
        return np.angle(dets.prod()) / (2.0 * np.pi)

    def _warn_metal(self, gap):
        import warnings

        warnings.warn(
            f"HOMO-LUMO gap = {gap:.6f} eV <= 1e-4. "
            "The system may be metallic; the Berry-phase polarization may be ill-defined."
        )

    def _warn_topological(self, winding, direction, trans):
        import warnings

        warnings.warn(
            f"Nonzero Chern winding {winding} detected in the k{trans[0]+1}-k{trans[1]+1} torus when calculating the polarization along k{direction+1}! The Berry phase is multi-valued on the torus and the polarization is ill-defined!"
        )

    @staticmethod
    def _align_berry_phases(phase):
        """Align Berry-phase field into a continuous single-valued function
        on the 2D torus by adding integers.

        Parameters
        ----------
        phase : np.ndarray, shape (na, nb)
            Wrapped Berry phase (in units of 2*pi), each entry in [-0.5, 0.5).

        Returns
        -------
        aligned : np.ndarray, shape (na, nb)
            Continuous phase such that ``aligned - phase`` is integer-valued
            (``aligned == phase + n``).  Falls back to ``phase`` unchanged
            when unwrapping is unreliable (nonzero residue or winding).
        winding : tuple[int, int]
            Winding numbers (Hx, Hy) around the two non-contractible loops.
            ``(0, 0)`` means a single-valued lift exists; nonzero means a
            Chern obstruction and an ill-defined polarization.
        """
        import warnings

        na, nb = phase.shape
        wrap = lambda x: (x + 0.5) % 1.0 - 0.5

        # Wrapped nearest-neighbor differences (periodic boundaries).
        Dx = wrap(np.roll(phase, -1, axis=0) - phase)  # along ia
        Dy = wrap(np.roll(phase, -1, axis=1) - phase)  # along ib

        # Residue check (coarse mesh / phase singularity): curl must vanish.
        curl = Dx + np.roll(Dy, -1, axis=0) - np.roll(Dx, -1, axis=1) - Dy
        if np.any(np.abs(curl) > 1e-6):
            warnings.warn("Berry phase has nonzero residues. The k-mesh maybe too coarse!")
            return phase, (0, 0)

        # Winding numbers around the two non-contractible loops.  With zero
        # residue these are independent of the reference row/column.
        Hx = int(round(Dx[:, 0].sum()))
        Hy = int(round(Dy[0, :].sum()))
        if Hx != 0 or Hy != 0:
            return phase, (Hx, Hy)

        # Comb-path integration: curl-free so the result is path-independent.
        aligned = np.empty((na, nb))
        aligned[0, 0] = phase[0, 0]
        aligned[0, 1:] = phase[0, 0] + np.cumsum(Dy[0, :-1])
        aligned[1:, :] = aligned[0, :] + np.cumsum(Dx[:-1, :], axis=0)
        return aligned, (0, 0)

    def ionic_polarization(self):
        """Ionic (nuclear/pseudo) dipole contribution.

        Returns
        -------
        dipole_frac : np.ndarray, shape (3,)
            Ionic dipole in fractional coordinates, with the ionic charges (+Z)
            folded in, so that
            ``dipole_ion = dipole_frac @ lattice``.
        """
        self.dipole_ion = np.einsum("i,ij->j", self.z_ion, self.cart_coords)  # e*Angstrom
        self.dipole_frac_ion = np.einsum("i,ij->j", self.z_ion, self.frac_coords)  # dimensionless
        return self.dipole_frac_ion

    def calc(self, k_mesh: List | Tuple, **diag_kwargs):
        """
        Compute electronic + ionic polarization and return a summary dict.

        Parameters
        ----------
        k_mesh : tuple[int, int, int]
            Gamma-centered k-mesh ``(n1, n2, n3)``.
        **diag_kwargs : dict
            Extra keyword arguments forwarded to ``obj_H.diag``.

        Returns
        -------
        dict with keys:
            dipole_frac_elec, dipole_frac_ion, dipole_frac_total : np.ndarray (3,)
                Dipoles in fractional coordinates (only reasonable under
                ``mod 1``), with charges folded in (-e for electronic, +Z for
                ionic), so that ``dipole_X = dipole_frac_X @ lattice``.
            dipole_elec, dipole_ion, dipole_total : np.ndarray (3,)  (e*Angstrom)
            dipole_debye, dipole_elec_debye, dipole_ion_debye : np.ndarray (3,)
            polarization_mucm2, polarization_elec_mucm2, polarization_ion_mucm2
            band_gap : float  (HOMO-LUMO gap in eV)
        """
        dipole_frac_elec = self.electronic_polarization(k_mesh, **diag_kwargs)
        dipole_frac_ion = self.ionic_polarization()
        dipole_frac_total = dipole_frac_elec + dipole_frac_ion

        dipole_elec = self.dipole_elec  # e*Angstrom
        dipole_ion = self.dipole_ion
        dipole_total = dipole_elec + dipole_ion  # e*Angstrom

        # e*Angstrom -> Debye  (1 e*Ang = ANGSTROM_TO_BOHR * AU2DEBYE Debye)
        debye_elec = dipole_elec * ANGSTROM_TO_BOHR * AU2DEBYE
        debye_ion = dipole_ion * ANGSTROM_TO_BOHR * AU2DEBYE
        debye_total = dipole_total * ANGSTROM_TO_BOHR * AU2DEBYE

        # e*Angstrom -> muC/cm^2  (muC/cm^2 = AU2MUCM * e*bohr / bohr^3)
        mucm_elec = AU2MUCM * dipole_elec / (self.volume * ANGSTROM_TO_BOHR**2)
        mucm_ion = AU2MUCM * dipole_ion / (self.volume * ANGSTROM_TO_BOHR**2)
        mucm_total = AU2MUCM * dipole_total / (self.volume * ANGSTROM_TO_BOHR**2)

        result = {
            "dipole_frac_elec": dipole_frac_elec,
            "dipole_frac_ion": dipole_frac_ion,
            "dipole_frac_total": dipole_frac_total,
            "dipole_elec": dipole_elec,
            "dipole_ion": dipole_ion,
            "dipole_total": dipole_total,
            "dipole_debye": debye_total,
            "dipole_elec_debye": debye_elec,
            "dipole_ion_debye": debye_ion,
            "polarization_mucm2": mucm_total,
            "polarization_elec_mucm2": mucm_elec,
            "polarization_ion_mucm2": mucm_ion,
            "band_gap": self.band_gap,
        }

        labels = ["x", "y", "z"]
        print(f"Band gap = {self.band_gap:.6f} eV")
        print("\nDipole moment (fractional):")
        print("          Ionic          Electronic     Total")
        for a in range(3):
            print(
                f"  {labels[a]}   {dipole_frac_ion[a]:+12.6f}   {dipole_frac_elec[a]:+12.6f}   {dipole_frac_total[a]:+12.6f}"
            )
        print("\nDipole moment (eA):")
        print("          Ionic          Electronic     Total")
        for a in range(3):
            print(
                f"  {labels[a]}   {dipole_ion[a]:+12.6f}   {dipole_elec[a]:+12.6f}   {dipole_total[a]:+12.6f}"
            )
        print("\nDipole moment (Debye):")
        print("          Ionic          Electronic     Total")
        for a in range(3):
            print(
                f"  {labels[a]}   {debye_ion[a]:+12.6f}   {debye_elec[a]:+12.6f}   {debye_total[a]:+12.6f}"
            )
        print("\nPolarization (muC/cm^2):")
        print("          Ionic          Electronic     Total")
        for a in range(3):
            print(
                f"  {labels[a]}   {mucm_ion[a]:+12.6f}   {mucm_elec[a]:+12.6f}   {mucm_total[a]:+12.6f}"
            )
        print(f"  |P| = {np.linalg.norm(mucm_total):.6f} muC/cm^2\n")

        return result
