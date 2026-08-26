"""
MPI-parallel Hamiltonian diagonalization based on PETSc/SLEPc.

This module provides :class:`PETScHamiltonianObj`, an MPI-parallel sibling of
:class:`~deepx_dock.compute.eigen.hamiltonian.HamiltonianObj`. It solves the
generalized eigenvalue problem ``H(k) v = e S(k) v`` (with Hermitian ``H(k)``
and positive-definite ``S(k)``) with SLEPc
(Krylov-Schur + shift-invert spectral transform), distributing the matrix
across all MPI ranks by PETSc.

Only rank 0 parses the DeepH input files and holds the real-space matrices
(assembled directly into per-R CSR sparse form, so the memory footprint scales
with the number of non-zeros instead of ``N_R * N_b^2``). Scalar metadata is
broadcast to the other ranks.

Requirements
------------
- ``mpi4py``, ``petsc4py`` and ``slepc4py`` (install extra:
  ``pip install deepx-dock[petsc,slepc]``).
- PETSc must be compiled with **complex** scalar type, because the spin-orbit
  coupled Hamiltonian is complex.

Usage
-----
Run the CLI under ``mpirun`` to parallelize one k-point over all ranks::

    mpirun -np 4 dock compute eigen <job_name>-petsc <data_path>
"""

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

try:
    from mpi4py import MPI
except Exception as e:
    raise ImportError(
        f"{e}\n[error] mpi4py is not installed. Install it with `pip install deepx-dock[mpi]`."
    ) from e

try:
    from petsc4py import PETSc
except Exception as e:
    raise ImportError(
        f"{e}\n[error] petsc4py is not installed. Install it with `pip install deepx-dock[petsc]`."
    ) from e

try:
    from slepc4py import SLEPc
except Exception as e:
    raise ImportError(
        f"{e}\n[error] slepc4py is not installed. Install it with `pip install deepx-dock[slepc]`."
    ) from e

from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME, EXTREMELY_SMALL_FLOAT


class PETScHamiltonianObj(HamiltonianObj):
    """
    MPI-parallel Hamiltonian object based on PETSc/SLEPc sparse eigensolvers.

    Parameters
    ----------
    data_path : str or Path
        Path to the directory containing the DeepH format files
        (POSCAR, info.json, overlap.h5 and hamiltonian.h5).
    H_file_path : str or Path, optional
        Path to the Hamiltonian file. Default: hamiltonian.h5 under
        `data_path`.
    comm : mpi4py.MPI.Comm, optional
        MPI communicator over which the k-point matrices are distributed.
        Default: ``MPI.COMM_WORLD``.

    Notes
    -----
    Only rank 0 reads the input files and stores the real-space Hamiltonian
    and overlap as per-R ``scipy.sparse.csr_matrix`` dictionaries
    (:attr:`SR_csr`, :attr:`HR_csr`). Other ranks only receive scalar
    metadata, therefore structure attributes such as :attr:`lattice` or
    :attr:`elements` are only available on rank 0.

    Unlike the dense :class:`HamiltonianObj`, only a partial spectrum around
    a target energy is computed, so a Fermi energy determined by electron
    counting is not available from this solver.

    The problem is solved in the non-Hermitian mode (GNHEP) for robustness
    when the shift lands near the spectrum; eigenvectors are B-normalized
    individually (``v / sqrt(v^H S v)``) since GNHEP does not guarantee
    B-orthonormality.
    """

    def __init__(self, data_path: str | Path, H_file_path: str | Path | None = None, comm=MPI.COMM_WORLD):
        if not np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
            raise RuntimeError("PETSc must be compiled with complex scalar type (--with-scalar-type=complex).")

        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        petsc_sys = PETSc.Sys()
        petsc_sys.setDefaultComm(self.comm)
        PETSc.Sys.Print(f"Using {self.size} processes.", flush=True)

        self._get_necessary_data_path(data_path, H_file_path, matrix_type="hamiltonian")

        self.mats = None
        self.Rijk_list = None
        self.SR = None
        self.SR_csr: dict[tuple[int, int, int], csr_matrix] | None = None
        self.HR_csr: dict[tuple[int, int, int], csr_matrix] | None = None
        self.vecs_empty = None
        self._init_space_vecs: list = []

        if self.rank == 0:
            self._parse_info()
            self._parse_poscar()
            self._parse_orbit_types()
            self.SR_csr = self._construct_csr_matrix(
                self._read_h5(self.info_dir_path / DEEPX_OVERLAP_FILENAME, dtype=np.float64),
                matrix_type="overlap"
            )
            self.HR_csr = self._construct_csr_matrix(
                self._read_h5(self.matrix_path, dtype=PETSc.ScalarType if self.spinful else np.float64),
                matrix_type="hamiltonian",
            )
            assert set(self.SR_csr) == set(self.HR_csr), "The R sets of overlap.h5 and hamiltonian.h5 do not match."
            self.nrows = self.orbits_quantity * (1 + self.spinful)
            self.ncols = self.nrows
        else:
            self.reciprocal_lattice = None
            self.spinful = None
            self.fermi_energy = None
            self.occupation = None
            self.orbits_quantity = None
            self.nrows = None
            self.ncols = None

        self.reciprocal_lattice = self.comm.bcast(self.reciprocal_lattice, root=0)
        self.spinful = self.comm.bcast(self.spinful, root=0)
        self.fermi_energy = self.comm.bcast(self.fermi_energy, root=0)
        self.occupation = self.comm.bcast(self.occupation, root=0)
        self.orbits_quantity = self.comm.bcast(self.orbits_quantity, root=0)
        self.nrows = self.comm.bcast(self.nrows, root=0)
        self.ncols = self.comm.bcast(self.ncols, root=0)
        self.comm.barrier()

    def _construct_csr_matrix(
        self, obs_tuple: tuple[np.ndarray, ...], matrix_type: str
    ) -> dict[tuple[int, int, int], csr_matrix]:
        """
        Assemble a DeepH-format observable into per-R CSR sparse matrices.

        The pair-wise blocks are scattered into sparse matrices, so the memory
        footprint scales with the number of non-zeros. Elements smaller than
        ``10 * EXTREMELY_SMALL_FLOAT`` are treated as zeros.

        Parameters
        ----------
        obs_tuple : tuple of np.ndarray
            Raw DeepH matrix arrays ``(atom_pairs, chunk_boundaries,
            chunk_shapes, entries)`` as returned by
            :meth:`~deepx_dock.compute.eigen.matrix_obj.AOMatrixObj._read_h5`.
        matrix_type : str
            Type of the matrix. For "overlap", the spinless blocks are
            expanded into [[S,0],[0,S]] when the system is spinful. For
            "hamiltonian" of a spinful system, the stored blocks contain the
            full spin structure (shape ``(2*n_i, 2*n_j)``) and are scattered
            into the four spin quadrants.

        Returns
        -------
        value_R : dict
            Mapping ``(Rx, Ry, Rz) -> csr_matrix`` with data dtype matching
            ``PETSc.ScalarType``.
        """

        atom_pairs, chunk_boundaries, chunk_shapes, entries = obs_tuple
        need_expand_spin = (matrix_type in ["overlap",]) and self.spinful

        n_orb = self.orbits_quantity
        matrix_dim = n_orb * (1 + self.spinful)
        small_value = 10 * EXTREMELY_SMALL_FLOAT
        R_set = {tuple(int(v) for v in R) for R in np.unique(atom_pairs[:, :3], axis=0)}
        rows: dict[tuple[int, int, int], list] = {R: [] for R in R_set}
        cols: dict[tuple[int, int, int], list] = {R: [] for R in R_set}
        data: dict[tuple[int, int, int], list] = {R: [] for R in R_set}

        for i_ap in range(atom_pairs.shape[0]):
            R = tuple(int(v) for v in atom_pairs[i_ap, :3])
            ia, ja = int(atom_pairs[i_ap, 3]), int(atom_pairs[i_ap, 4])
            block = entries[chunk_boundaries[i_ap] : chunk_boundaries[i_ap + 1]].reshape(chunk_shapes[i_ap])
            nz_r, nz_c = np.nonzero(np.abs(block) > small_value)
            if nz_r.size == 0:
                continue
            vals = block[nz_r, nz_c].astype(PETSc.ScalarType)

            i0 = int(self.atom_num_orbits_cumsum[ia])
            j0 = int(self.atom_num_orbits_cumsum[ja])
            n_i = int(self.atom_num_orbits[ia])
            n_j = int(self.atom_num_orbits[ja])

            if not self.spinful:  # mat -> mat
                global_rows = i0 + nz_r
                global_cols = j0 + nz_c
            elif need_expand_spin:  # mat -> [[mat,0],[0,mat]]
                global_rows = np.concatenate([i0 + nz_r, i0 + n_orb + nz_r])
                global_cols = np.concatenate([j0 + nz_c, j0 + n_orb + nz_c])
                vals = np.concatenate([vals, vals])
            else:  # [[uu,ud],[du,dd]] -> [[uu,ud],[du,dd]]
                up_r, up_c = nz_r < n_i, nz_c < n_j
                uu, ud = up_r & up_c, up_r & ~up_c
                du, dd = ~up_r & up_c, ~up_r & ~up_c
                global_rows = np.concatenate(
                    [i0 + nz_r[uu], i0 + nz_r[ud], i0 + n_orb + nz_r[du] - n_i, i0 + n_orb + nz_r[dd] - n_i]
                )
                global_cols = np.concatenate(
                    [j0 + nz_c[uu], j0 + n_orb + nz_c[ud] - n_j, j0 + nz_c[du], j0 + n_orb + nz_c[dd] - n_j]
                )
                vals = np.concatenate([vals[uu], vals[ud], vals[du], vals[dd]])

            rows[R].append(global_rows)
            cols[R].append(global_cols)
            data[R].append(vals)

        value_R: dict[tuple[int, int, int], csr_matrix] = {}
        for R, row_list in rows.items():
            if row_list:
                value_R[R] = csr_matrix(
                    (np.concatenate(data[R]), (np.concatenate(row_list), np.concatenate(cols[R]))),
                    shape=(matrix_dim, matrix_dim),
                    dtype=PETSc.ScalarType,
                )
            else:
                value_R[R] = csr_matrix((matrix_dim, matrix_dim), dtype=PETSc.ScalarType)
        return value_R

    @staticmethod
    def _ft(k, MRs: dict[tuple[int, int, int], csr_matrix]) -> csr_matrix:
        """Fourier transform ``M(k) = sum_R M(R) exp(2*pi*i k R)`` for sparse per-R matrices."""
        Mk = csr_matrix(MRs[(0, 0, 0)].shape, dtype=PETSc.ScalarType)
        for R, MR in MRs.items():
            phase = np.exp(2j * np.pi * (R[0] * k[0] + R[1] * k[1] + R[2] * k[2]))
            Mk += MR * phase
        return Mk

    def Sk_and_Hk(self, k) -> tuple[csr_matrix, csr_matrix]:
        """Get the sparse overlap and Hamiltonian matrices at a given k-point (rank 0 only)."""
        return self._ft(k, self.SR_csr), self._ft(k, self.HR_csr)

    def parse_diag_kwargs(self, kwargs: dict) -> dict:
        """
        Translate user-facing diagonalization options into SLEPc solver options.

        Accepted keys: ``num_band`` (default 50), ``dim_subspace`` (default
        ``1.5 * num_band``), ``fermi_energy_eV`` (default from info.json),
        ``target_band_energy`` (default -0.5, relative to the Fermi level),
        ``maxiter`` (default 300), ``tol`` (default 1e-5), ``purify``
        (default True), ``init_space`` (default False) and
        ``same_nonzero_pattern`` (default True).

        Note that the dense Rayleigh-Ritz workspace of the solver scales as
        ``ncv^2`` (replicated on every rank) and the Krylov basis as ``ncv``,
        so for large ``nev`` keep ``dim_subspace`` small (1.2~1.5 x nev is a
        good compromise).

        ``same_nonzero_pattern`` is on by default: the sparsity patterns of
        H(k) and S(k) are unified (:meth:`_unify_sparse_pattern`) so that the
        symbolic factorization and the MUMPS workspace are reused across
        k-points. Without it, the per-k destroy-and-rebuild of the
        factorization ratchets up memory and eventually fails with
        ``MUMPS INFOG(1)=-9, INFO(2)=0`` on large systems.

        Returns
        -------
        kwargs_now : dict
            Keys: ``nev``, ``ncv``, ``target``, ``max_it``, ``tol``,
            ``purify``, ``init_space``, ``same_nonzero_pattern``.
        """
        _num_band = kwargs.get("num_band", 50)
        _ncv = kwargs.get("dim_subspace", int(1.5 * _num_band))
        _fermi_energy = kwargs.get("fermi_energy_eV", self.fermi_energy)
        _target_band_energy = kwargs.get("target_band_energy", -0.5)
        _maxiter = kwargs.get("maxiter", 300)
        _tol = kwargs.get("tol", 1e-5)
        _purify = kwargs.get("purify", True)
        _init_space = kwargs.get("init_space", False)
        _same_nonzero_pattern = kwargs.get("same_nonzero_pattern", True)
        PETSc.Sys.Print(
            f"Calculation that utilizes PETSc and SLEPc with num_band={_num_band}, dim_subspace={_ncv}, "
            f"target_band_energy={_target_band_energy}, maxiter={_maxiter}, tol={_tol}, purify={_purify}, "
            f"init_space={_init_space} ...",
            flush=True,
        )
        kwargs_now = {
            "nev": min(_num_band, self.nrows),
            "ncv": min(max(_ncv, min(_num_band, self.nrows) + 2), self.nrows),
            "target": _fermi_energy + _target_band_energy,
            "max_it": _maxiter,
            "tol": _tol,
            "purify": _purify,
            "init_space": _init_space,
            "same_nonzero_pattern": _same_nonzero_pattern,
        }
        return kwargs_now

    def diag(self, ks, bands_only: bool = True, **kwargs) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the Hamiltonian at the given k-points with SLEPc.

        Parameters
        ----------
        ks : array_like, shape (Nk, 3)
            k-points in fractional coordinates.
        bands_only : bool, optional
            If True, only return eigenvalues. If False, eigenvectors are
            returned as rank-local blocks (see below). Default: True.
        **kwargs : dict
            Solver options, see :meth:`parse_diag_kwargs`.

        Returns
        -------
        eigvals : np.ndarray, shape (Nband, Nk)
            Eigenvalues (band energies in eV).
        eigvecs : np.ndarray, shape (Norb_local, Nband, Nk)
            Rank-local eigenvector row blocks: each rank receives only the
            rows it owns (PETSc ownership ranges, concatenated in rank order
            they form the full ``[Norb, Nband, Nk]`` array). Use
            :meth:`gather_vec_to_rank0` to assemble the full array on rank 0
            (e.g. before dumping to disk). Returned only if
            ``bands_only=False``.
        """
        kwargs_now = self.parse_diag_kwargs(kwargs)
        eps = self._initialize_solver(**kwargs_now)
        eigvals_list = []
        eigvecs = None
        for ik, k in enumerate(ks):
            PETSc.Sys.Print(f"\n[do] k point {ik + 1}/{len(ks)} ...", flush=True)
            t1 = time.perf_counter()
            result_k = self.diag_one_k(eps, k, bands_only=bands_only, early_reset_ST=(ik == len(ks) - 1), **kwargs_now)
            if bands_only:
                eigvals_list.append(result_k)
            else:
                eigvals_list.append(result_k[0]) # [Nband]
                block_k = result_k[1] # [Norb_local, Nband]
                if eigvecs is None:
                    eigvecs = np.empty((block_k.shape[0], block_k.shape[1], len(ks)), dtype=block_k.dtype)
                assert eigvecs.shape[:2] == block_k.shape, (
                    f"Eigenvector block shape changed across k-points: {block_k.shape} vs {eigvecs.shape[:2]}"
                )
                eigvecs[:, :, ik] = block_k
                del block_k, result_k
            PETSc.Sys.Print(f"[done] k point {ik + 1}/{len(ks)}. Total Time: {time.perf_counter() - t1:.2f} sec", flush=True)
        eps.destroy()
        self._destroy_work_vecs()
        eigvals = np.stack(eigvals_list, axis=1) # [Nband, Nk]
        if bands_only:
            return eigvals
        return eigvals, eigvecs

    def diag_one_k(self, eps: SLEPc.EPS, k, bands_only: bool = True, early_reset_ST: bool = False, **kwargs):
        """
        Diagonalize the Hamiltonian at one k-point.

        Parameters
        ----------
        eps : SLEPc.EPS
            Configured eigenvalue solver (operators are set inside).
        k : array_like, shape (3,)
            k-point in fractional coordinates.
        bands_only : bool, optional
            See :meth:`diag`. Default: True.
        early_reset_ST : bool, optional
            Whether this is the last k-point of the :meth:`diag` loop. If set
            and eigenvectors are requested, the factorization is released
            before the collection loop (see :meth:`_solve`). Default: False.
        **kwargs : dict
            Solver options from :meth:`parse_diag_kwargs`.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            Sorted eigenvalues, or ``(eigenvalues, eigenvectors)`` where
            eigenvectors are the rank-local row block of shape
            ``(N_local, Nband)`` (see :meth:`diag`).
        """
        PETSc.Sys.Print(f"[info] k coord: {k}", flush=True)
        t1 = time.perf_counter()
        if self.rank == 0:
            Sk, Hk = self.Sk_and_Hk(k)
            nonher_Sk = abs((Sk - Sk.conj().transpose()) / 2.0).sum() / Sk.nnz
            nonher_Hk = abs((Hk - Hk.conj().transpose()) / 2.0).sum() / Hk.nnz
            Sk = (Sk + Sk.conj().transpose()) / 2.0
            Hk = (Hk + Hk.conj().transpose()) / 2.0
            if kwargs["same_nonzero_pattern"]:
                Sk, Hk = PETScHamiltonianObj._unify_sparse_pattern(Sk, Hk)
            PETSc.Sys.Print(f"[info] Non-hermitian part of Sk and Hk: {nonher_Sk:.2e} {nonher_Hk:.2e}", flush=True)
        else:
            Sk, Hk = None, None
        Sk = self._translate_scipy_to_petsc(Sk)
        Hk = self._translate_scipy_to_petsc(Hk)
        if self.vecs_empty is None:
            self.vecs_empty = Hk.createVecs()
        PETSc.Sys.Print(f"[time] Set up Sk and Hk: {time.perf_counter() - t1:.2f} sec", flush=True)
        t1 = time.perf_counter()
        ## Set the operators
        eps.setOperators(Hk, Sk)
        ## Setup the solver (spectral transform, Krylov solver, preconditioning, etc.)
        eps.setUp()
        PETSc.Sys.Print(f"[time] Set up the solver: {time.perf_counter() - t1:.2f} sec", flush=True)
        t1 = time.perf_counter()
        ## Solve the problem
        result_k = self._solve(eps, bands_only, kwargs["init_space"], kwargs["nev"], kwargs["max_it"], Sk, early_reset_ST=early_reset_ST)
        Sk.destroy()
        Hk.destroy()
        PETSc.garbage_cleanup(comm=self.comm)
        return result_k

    def _translate_scipy_to_petsc(self, sparse_matrix: csr_matrix) -> PETSc.Mat:
        """
        Distribute a rank-0 scipy CSR matrix into a parallel PETSc AIJ matrix.

        Row blocks follow ``PETSc.Mat.getOwnershipRanges()``; rank 0 sends each
        block to its owner rank with point-to-point communication.
        """
        petsc_matrix = PETSc.Mat().create(comm=self.comm)
        petsc_matrix.setSizes((self.nrows, self.ncols))
        petsc_matrix.setType(PETSc.Mat.Type.AIJ)
        petsc_matrix.setUp()
        if self.rank == 0:
            data = np.asarray(sparse_matrix.data, dtype=PETSc.ScalarType)
            indices = sparse_matrix.indices.astype(PETSc.IntType)
            indptr = sparse_matrix.indptr.astype(PETSc.IntType)
            ## data will be send to rank>0 processes
            ownership_ranges = petsc_matrix.getOwnershipRanges()
            for i in range(1, len(ownership_ranges) - 1):
                row_start = ownership_ranges[i]
                row_end = ownership_ranges[i + 1]
                ## indptr_i must minus the offset to make it start from 0
                data_i = data[indptr[row_start] : indptr[row_end]]
                indices_i = indices[indptr[row_start] : indptr[row_end]]
                indptr_i = indptr[row_start : row_end + 1] - indptr[row_start]
                self.comm.send(data_i, dest=i, tag=0)
                self.comm.send(indices_i, dest=i, tag=1)
                self.comm.send(indptr_i, dest=i, tag=2)
            ## data for rank=0 process
            data = data[indptr[ownership_ranges[0]] : indptr[ownership_ranges[1]]]
            indices = indices[indptr[ownership_ranges[0]] : indptr[ownership_ranges[1]]]
            indptr = indptr[ownership_ranges[0] : ownership_ranges[1] + 1]
        else:
            ## data for rank>0 process
            data = self.comm.recv(source=0, tag=0)
            indices = self.comm.recv(source=0, tag=1)
            indptr = self.comm.recv(source=0, tag=2)
        self.comm.barrier()
        ## Set values for the PETSc matrix
        petsc_matrix.setValuesCSR(indptr, indices, data)
        petsc_matrix.assemble()
        return petsc_matrix

    @staticmethod
    def _unify_sparse_pattern(A: csr_matrix, B: csr_matrix) -> tuple[csr_matrix, csr_matrix]:
        """
        Expand two CSR matrices to their union sparsity pattern.

        A temporary matrix with entries ``|A| + |B| + max(|A|+|B|) + 1`` guarantees
        that every entry of the union pattern stays positive, so subtracting
        it back after the addition preserves explicit zeros at positions that
        exist in only one of the input matrices. Equal patterns allow PETSc to
        reuse the symbolic factorization of `` SAME_NONZERO_PATTERN `` operators.
        """
        temp = abs(A) + abs(B)
        temp.data += np.max(temp.data) + 1.0
        A = csr_matrix(((A + temp).data - temp.data, temp.indices, temp.indptr), shape=temp.shape)
        B = csr_matrix(((B + temp).data - temp.data, temp.indices, temp.indptr), shape=temp.shape)
        return A, B

    def _initialize_solver(self, **kwargs) -> SLEPc.EPS:
        """
        Create and configure the SLEPc eigenvalue solver.

        Krylov-Schur iteration for a generalized Hermitian problem with a
        shift-and-invert spectral transform around the target energy.
        """
        eps = SLEPc.EPS().create(comm=self.comm)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)  # Krylov-Schur method
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)  # GNHEP (general LU path) is robust when the shift lands near the spectrum, where GHEP's symmetric pivoting on the nearly singular H - target*S may fail
        eps.setPurify(purify=kwargs["purify"])
        ## Set the options for partial eigenvalue solution
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eps.setTarget(kwargs["target"])
        eps.setTolerances(tol=kwargs["tol"], max_it=kwargs["max_it"])
        eps.setDimensions(nev=kwargs["nev"], ncv=kwargs["ncv"])
        # eps.setWhichEigenpairs(SLEPc.EPS.Which.ALL)
        # eps.setInterval(inta, intb)
        # eps.setKrylovSchurDimensions(nev=None, ncv=None, mpd=None)
        ## Set the Spectral Transform
        st = SLEPc.ST().create()
        st.setType(SLEPc.ST.Type.SINVERT)  # SINVERT: Shift-and-invert
        st.setShift(kwargs["target"])
        if kwargs["same_nonzero_pattern"]:
            st.setMatStructure(PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        eps.setST(st)
        st.destroy()
        return eps

    def _solve(
        self, eps: SLEPc.EPS, bands_only: bool, init_space: bool, nev: int, max_it: int, Sk: PETSc.Mat, early_reset_ST: bool = False
    ):
        """
        Solve the configured EPS problem and collect the results.

        Eigenvectors are B-normalized individually and written directly into a
        preallocated rank-local numpy block ``[N_local, Nband]`` (PETSc vector
        copies are only kept for the ``init_space`` mechanism).

        When ``early_reset_ST`` is set and eigenvectors are requested, the spectral
        transform's factorization is released (``STReset``) before the
        collection loop, lowering the peak memory of the extract-and-store
        phase. The eigenvectors live in the solver's DS/BV objects, which are
        unaffected by the ST reset.
        """
        t1 = time.perf_counter()
        ## Solve!
        eps.solve()
        PETSc.Sys.Print(f"[time] Solve: {time.perf_counter() - t1:.2f} sec", flush=True)
        t1 = time.perf_counter()
        ## Get the results
        if self.rank == 0:
            nconv = eps.getConverged()
            niter = eps.getIterationNumber()
            eigenvalues = np.array([eps.getEigenvalue(i) for i in range(nconv)])
            ## find the eigenvalues nearest to target
            sort_idx_1 = np.argsort(np.abs(eigenvalues - eps.getTarget()))[: min(nev, nconv)]
            ## sort the eigenvalues
            sort_idx_2 = np.argsort(eigenvalues[sort_idx_1])
            sort_idx = sort_idx_1[sort_idx_2]
            eigenvalues = eigenvalues[sort_idx].real
        else:
            nconv = None
            niter = None
            sort_idx = None
            eigenvalues = None
        nconv = self.comm.bcast(nconv, root=0)
        niter = self.comm.bcast(niter, root=0)
        sort_idx = self.comm.bcast(sort_idx, root=0)
        eigenvalues = self.comm.bcast(eigenvalues, root=0)

        eigenvectors: list = []
        local_block = None
        if init_space or (not bands_only):
            if not bands_only:
                rstart, rend = self.vecs_empty[0].getOwnershipRange()
                local_block = np.empty((rend - rstart, len(sort_idx)), dtype=PETSc.ScalarType)
            if early_reset_ST:
                ## release the factorization before the collection loop to lower
                ## the peak memory (eigenvectors live in DS/BV, not in the ST)
                eps.getST().reset()
            vec = self.vecs_empty[0].duplicate()
            for col, i in enumerate(sort_idx):
                ## PETSc is assumed to be compiled with complex dtype
                ## so that Vr stores the whole complex eigvec and Vi is emtpy
                eps.getEigenvector(i, Vr=vec)
                self._vec_normalize(vec, Sk, self.vecs_empty[1])
                if local_block is not None:
                    local_block[:, col] = vec.getArray(readonly=True)
                if init_space:
                    eigenvectors.append(vec.copy())
            vec.destroy()

        if init_space:
            ## destroy the old init_space_vecs and set them to new ones
            self._destroy_init_space_vecs()
            eps.setInitialSpace(eigenvectors)
            self._init_space_vecs = list(eigenvectors)

        PETSc.Sys.Print(f"[time] Postprocess: {time.perf_counter() - t1:.2f} sec", flush=True)
        if nconv < nev:
            PETSc.Sys.Print(
                f"[warning] The number of converged eigenvalues ({nconv}) is less than expected ({nev})!", flush=True
            )
        else:
            PETSc.Sys.Print(f"[info] The number of converged eigenvalues: {nconv}", flush=True)
        if niter > max_it - 2:
            PETSc.Sys.Print(
                f"[warning] The number of iterations reaches the max_iter ({max_it})! The results may be less accurate than expected!",
                flush=True,
            )
        else:
            PETSc.Sys.Print(f"[info] The number of iterations: {niter}", flush=True)

        if bands_only:
            return eigenvalues
        return eigenvalues, local_block

    def _vec_normalize(self, vec, Sk, tmp) -> None:
        """
        Normalize a vector with the overlap metric: ``v <- v / sqrt(v^H S v)``.

        GNHEP does not guarantee B-orthonormal eigenvectors, so each vector is
        normalized individually. All internal calls are collective: every rank
        must invoke this together.
        """
        Sk.mult(vec, tmp)
        norm_sq = vec.dot(tmp)
        if abs(norm_sq) > 0.0:
            vec.scale(1.0 / np.sqrt(norm_sq))

    def _destroy_work_vecs(self) -> None:
        """Release the cached work vectors and any initial-space vectors."""
        self._destroy_init_space_vecs()
        if self.vecs_empty is not None:
            self.vecs_empty[0].destroy()
            self.vecs_empty[1].destroy()
            self.vecs_empty = None

    def _destroy_init_space_vecs(self) -> None:
        for vec in self._init_space_vecs:
            vec.destroy()
        self._init_space_vecs = []

    def gather_vec_to_rank0(self, local_block: np.ndarray) -> np.ndarray | None:
        """
        Gather rank-local row blocks into the full array on rank 0.

        PETSc distributes rows contiguously in rank order (see
        ``Mat.getOwnershipRanges``), so concatenating the rank-ordered local
        blocks along axis 0 reproduces the global array.

        Parameters
        ----------
        local_block : np.ndarray
            Rank-local block whose axis 0 covers this rank's owned rows,
            e.g. ``[N_local, Nband]`` or ``[N_local, Nband, Nk]`` as returned
            by :meth:`diag` with ``bands_only=False``.

        Returns
        -------
        np.ndarray or None
            The full array on rank 0; ``None`` on every other rank.
        """
        parts = self.comm.gather(local_block, root=0)
        if self.rank == 0:
            return np.concatenate(parts, axis=0)
        return None

    @staticmethod
    def fA_mul_b(f: Callable[[np.ndarray], np.ndarray], A: PETSc.Mat, b: PETSc.Vec, m: int = 100) -> PETSc.Vec:
        """
        Compute ``y ~= f(A) b`` for a Hermitian matrix A via the m-step Lanczos method.

        The Krylov subspace ``K_m(A, b)`` is built with a three-term recurrence
        plus one full MGS reorthogonalization pass, giving the tridiagonal
        projection ``T_m = V_m^H A V_m``; then

            ``f(A) b ~= beta0 * V_m @ (T_m^e1)``,  ``f(T_m) e1 = Q (f(Lam) Q[0, :])``,

        with ``T_m = Q Lam Q^H`` the small dense eigendecomposition evaluated
        with numpy.

        Parameters
        ----------
        f : callable
            The matrix function applied to A, which is equivalent to the
            element-wise function applied to the Ritz values (a numpy array),
            e.g. ``lambda lam: np.sqrt(np.clip(lam, 0.0, None))`` for the
            matrix square root of A and ``lambda lam: lam**2`` for the matrix
            multiplication A @ A. Note that f is responsible for its own
            domain safety (e.g. clipping tiny negative Ritz values caused by
            round-off before taking a square root).
        A : PETSc.Mat
            Hermitian (distributed) matrix.
        b : PETSc.Vec
            Input vector (distributed with the same layout as A).
        m : int, optional
            Number of Lanczos steps. For ``f = sqrt`` the error decreases as
            ``rho^m`` with ``rho = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)``,
            e.g. m=100 reaches ~1e-10 for kappa ~ 1e3. Default: 100.

        Returns
        -------
        PETSc.Vec
            The distributed vector ``y ~= f(A) b`` (caller owns it).

        Notes
        -----
        All internal calls (Mat.mult, Vec.dot, ...) are collective: every
        rank must invoke this together. Note the petsc4py convention
        ``x.dot(y) == y^H x``.
        """
        beta0 = b.norm()
        if beta0 == 0.0 or m < 1:
            y = b.duplicate()
            y.zeroEntries()
            return y
        v = b.copy()
        v.scale(1.0 / beta0)
        basis = [v]
        alphas, betas = [], []
        w = b.duplicate()
        for j in range(m):
            A.mult(basis[j], w)
            ## petsc4py convention: x.dot(y) computes y^H x, so w.dot(basis[j]) = basis[j]^H w
            alpha = w.dot(basis[j])
            alphas.append(alpha)
            w.axpy(-alpha, basis[j])
            if j > 0:
                w.axpy(-betas[j - 1], basis[j - 1])
            for i in range(j + 1):  # full reorthogonalization (single MGS pass)
                coeff = w.dot(basis[i])
                w.axpy(-coeff, basis[i])
            beta = w.norm()
            betas.append(beta)
            if j == m - 1 or beta < 1e-13:
                break
            v_next = w.copy()
            v_next.scale(1.0 / beta)
            basis.append(v_next)
        k = len(alphas)
        T = np.zeros((k, k))
        np.fill_diagonal(T, np.array([a.real for a in alphas]))
        for i in range(k - 1):
            T[i + 1, i] = T[i, i + 1] = betas[i].real
        lam, Q = np.linalg.eigh(T)
        u = Q @ (f(lam) * Q[0, :])
        y = basis[0].duplicate()
        y.zeroEntries()
        for j in range(k):
            y.axpy(u[j], basis[j])
        y.scale(beta0)
        for vec in basis:
            vec.destroy()
        w.destroy()
        return y
