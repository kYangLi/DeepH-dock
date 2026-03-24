from pathlib import Path
import h5py
import numpy as np
from ase.io import read
from scipy.io import FortranFile
from functools import lru_cache
from typing import Optional

from .aodata import AOData_siesta
from HPRO.utils.structure import Structure
from HPRO.utils.supercell import SuperCell
from HPRO.utils.misc import atom_number2name
from HPRO.matao.findpairs import find_orb_pairs_proj
from HPRO.matao.findpairs import find_orb_pairs_direct
from HPRO.matao.matao import pairs_to_indices

from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.CONSTANT import DEEPX_PREDICT_HAMILTONIAN_FILENAME
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.CONSTANT import DEEPX_DENSITY_MATRIX_FILENAME
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME
from deepx_dock.CONSTANT import SIESTA_HSX_FILENAME, SIESTA_DM_FILENAME

HARTREE_TO_EV = 27.211386024367243
BOHR_TO_ANGSTROM = 0.529177210903 # siesta

# ---- helpers for dtype with chosen endianness ----
def _dtype_i4(endian: str) -> np.dtype:
    return np.dtype(endian + 'i4')

def _dtype_u4(endian: str) -> np.dtype:
    return np.dtype(endian + 'u4')

def _dtype_real(is_dp: bool, endian: str) -> np.dtype:
    return np.dtype(endian + ('f8' if is_dp else 'f4'))

def _as_bytes_record6(labelfis, zvalfis, nofis, lablen, real_dt, int_dt) -> np.ndarray:
    """
    Build Record-6 payload as raw bytes:
      [fixed-width label, zval (real), nofis (int)] × nspecies
    Return an uint8 array suitable for FortranFile.write_record.
    """
    parts = []
    nspecies = len(labelfis)
    for i in range(nspecies):
        lbl = str(labelfis[i]).encode('ascii', errors='ignore')[:lablen].ljust(lablen, b' ')
        parts.append(lbl)
        parts.append(np.asarray([zvalfis[i]], dtype=real_dt).tobytes())
        parts.append(np.asarray([nofis[i]],   dtype=int_dt ).tobytes())
    payload = b''.join(parts)
    return np.frombuffer(payload, dtype=np.uint8)

def _as_bytes_record7_per_species(ns, ls, zetas, cfglen, int_dt) -> np.ndarray:
    """
    Build Record-7 (per species) payload as raw bytes:
      (cfg_bytes, l:int, zeta:int) × nshell, where cfg_bytes[0] stores n (1 byte),
      and the remaining cfglen-1 bytes are zero.
    """
    parts = []
    nshell = len(ns)
    for j in range(nshell):
        n_val = int(ns[j])
        if not (0 <= n_val <= 255):
            raise ValueError(f"n exceeds byte range: n={n_val}")
        cfg_bytes = bytes([n_val]) + (b'\x00' * (cfglen - 1))
        parts.append(cfg_bytes)
        parts.append(np.asarray([ls[j]],   dtype=int_dt).tobytes())
        parts.append(np.asarray([zetas[j]], dtype=int_dt).tobytes())
    payload = b''.join(parts)
    return np.frombuffer(payload, dtype=np.uint8)

# ==========================
#   Record 1–7 + k-point
# ==========================
def write_hsx_r1_to_r7_kpt(
    fname: str,
    # ---- Record 1–3 ----
    version: int,
    is_dp: bool,
    na_u: int, nrows_g: int, nspin: int, nspecies: int, nsc: int,
    # ---- Record 4 ----
    ucell, Ef: float, qtot: float, temp: float,
    # ---- Record 5 ----
    isc_off, xa, isa, lasto,
    # ---- Record 6 ----
    labelfis, zvalfis, nofis,
    # ---- Record 7 ----
    cn_n_list_per_species, l_list_per_species, zeta_list_per_species,
    # ---- k-point ----
    k_cell, k_displ,
    # ---- options ----
    lablen: int = 20,
    cfglen: int = 4,
    endian: str = "<",
    k_displ_use_dp: bool | None = None
):
    """
    Write HSX records 1–7 and the k-point record.
    """
    # Dtypes with chosen endianness
    rec_hdr_dt = np.dtype(endian + 'u4')  # record marker
    int_dt     = _dtype_i4(endian)
    real_dt    = _dtype_real(is_dp, endian)
    k_real_dt  = _dtype_real(is_dp if k_displ_use_dp is None else bool(k_displ_use_dp), endian)

    # Open in write mode (truncates/creates)
    with FortranFile(fname, mode='w', header_dtype=rec_hdr_dt) as f:
        # R1: version
        f.write_record(np.asarray([version], dtype=int_dt))

        # R2: precision flag (1 if double)
        f.write_record(np.asarray([1 if is_dp else 0], dtype=int_dt))

        # R3: dimensions: [na_u, nrows_g, nspin, nspecies, nscx, nscy, nscz]
        nsc_arr = np.asarray(nsc, dtype=int_dt)  # length-3
        f.write_record(np.asarray([na_u, nrows_g, nspin, nspecies], dtype=int_dt), nsc_arr)

        # R4: ucell(3x3) in column-major + Ef, qtot, temp
        u = np.asarray(ucell, dtype=real_dt)     # shape (3,3)
        u_fortran = u.ravel()
        scalars = np.asarray([Ef, qtot, temp], dtype=real_dt)
        f.write_record(u_fortran, scalars)

        # R5: isc_off(:), xa(3,na_u) in column-major, isa(:), lasto(:)
        isc = np.asarray(isc_off, dtype=int_dt)
        xa_arr = np.asarray(xa, dtype=real_dt)   # shape (na_u, 3)
        xa_f = xa_arr.ravel()
        isa_arr  = np.asarray(isa,   dtype=int_dt)
        last_arr = np.asarray(lasto, dtype=int_dt)
        f.write_record(isc, xa_f, isa_arr, last_arr)

        # R6: species header (single record with interleaved fields)
        rec6_bytes = _as_bytes_record6(labelfis, zvalfis, nofis, lablen, real_dt, int_dt)
        f.write_record(rec6_bytes)

        # R7: one record per species, interleaving (cfg_bytes, l, zeta) for all shells
        for ispec in range(nspecies):
            ns  = cn_n_list_per_species[ispec]
            ls  = l_list_per_species[ispec]
            zts = zeta_list_per_species[ispec]
            if len(ns) == 0:
                f.write_record(np.array([], dtype=np.uint8))
            else:
                rec7_bytes = _as_bytes_record7_per_species(ns, ls, zts, cfglen, int_dt)
                f.write_record(rec7_bytes)

        if version > 1:
            # k-point record: k_cell(3x3 int) + k_displ(3 real)
            kcell = np.asarray(k_cell, dtype=int_dt).ravel()
            kdisp = np.asarray(k_displ, dtype=k_real_dt)
            f.write_record(kcell, kdisp)

# ==========================================
#   Append sparse pattern + matrices (H,S)
# ==========================================
def write_sparse_pattern_and_mats_append(
    fname: str,
    *,
    endian: str,
    is_dp: bool,
    nspin: int,
    lgncol,
    rows_by_col,
    H_vals,
    S_vals,
    H2_vals=None,
) -> None:
    """
    Append the sparse pattern and matrices to an existing HSX file:
      - pattern: lgncol(:) and per-column rows (1-based)
      - H (and optional H2): per column, contiguous by nonzero, with spin as the fastest index
      - S: per column, scalar per nonzero
    """
    int_dt  = _dtype_u4(endian)
    real_dt = _dtype_real(is_dp, endian)

    no = len(lgncol)
    if len(rows_by_col) != no or len(H_vals) != no or len(S_vals) != no:
        raise ValueError("Inconsistent number of columns among lgncol/rows_by_col/H_vals/S_vals")
    if any(len(rows_by_col[j]) != lgncol[j] for j in range(no)):
        raise ValueError("Some rows_by_col[j] length does not match lgncol[j]")
    if any(len(H_vals[j]) != lgncol[j] for j in range(no)):
        raise ValueError("Some H_vals[j] nonzero count does not match lgncol[j]")
    if any(len(S_vals[j]) != lgncol[j] for j in range(no)):
        raise ValueError("Some S_vals[j] nonzero count does not match lgncol[j]")
    if any(len(H_vals[j][k]) != nspin for j in range(no) for k in range(lgncol[j])):
        raise ValueError("Each H nonzero must have exactly nspin components")

    # Open the file at end-of-file and wrap with FortranFile in write mode.
    # FortranFile has no 'append' mode; using a file object opened in 'ab' is safe.
    with open(fname, 'ab') as raw:
        f = FortranFile(raw, mode='w', header_dtype=np.dtype(endian+'u4'))

        # pattern: lgncol(:)
        f.write_record(np.asarray(lgncol, dtype=int_dt))

        # per-column rows (1-based)
        for j in range(no):
            rows = np.asarray(rows_by_col[j], dtype=int_dt)
            if rows.size and rows.min() < 1:
                raise ValueError("rows_by_col must be 1-based (>=1)")
            f.write_record(rows)

        # H: per column; for each nonzero, contiguous nspin components
        for j in range(no):
            if lgncol[j] == 0:
                f.write_record(np.asarray([], dtype=real_dt))
            else:
                col = np.asarray(H_vals[j], dtype=real_dt).reshape(lgncol[j], nspin)
                f.write_record(col.ravel())

        # optional H2
        if H2_vals is not None:
            if len(H2_vals) != no or any(len(H2_vals[j]) != lgncol[j] for j in range(no)):
                raise ValueError("H2_vals shape mismatch vs lgncol")
            for j in range(no):
                if lgncol[j] == 0:
                    f.write_record(np.asarray([], dtype=real_dt))
                else:
                    col2 = np.asarray(H2_vals[j], dtype=real_dt).reshape(lgncol[j], nspin)
                    f.write_record(col2.ravel())

        # S: per column, scalar per nonzero
        for j in range(no):
            s = np.asarray(S_vals[j], dtype=real_dt)
            f.write_record(s)

# =====================
#   Full-file writer
# =====================
def write_hsx_full(
    fname: str,
    # ---- R1–3 ----
    version: int,
    is_dp: bool,
    na_u: int, nrows_g: int, nspin: int, nspecies: int, nsc: int,
    # ---- R4 ----
    ucell, Ef: float, qtot: float, temp: float,
    # ---- R5 ----
    isc_off, xa, isa, lasto,
    # ---- R6 ----
    labelfis, zvalfis, nofis,
    # ---- R7 ----
    cn_n_list_per_species, l_list_per_species, zeta_list_per_species,
    # ---- k-point ----
    k_cell, k_displ,
    # ---- sparse pattern + matrices ----
    lgncol, rows_by_col, H_vals, S_vals, H2_vals=None,
    *,
    lablen: int = 20, cfglen: int = 4, endian: str = "<", k_displ_use_dp: Optional[bool] = None
) -> None:
    """
    Convenience wrapper: write records 1–7 + k-point, then append the sparse pattern and matrices.
    """
    write_hsx_r1_to_r7_kpt(
        fname,
        version, is_dp,
        na_u, nrows_g, nspin, nspecies, nsc,
        ucell, Ef, qtot, temp,
        isc_off, xa, isa, lasto,
        labelfis, zvalfis, nofis,
        cn_n_list_per_species, l_list_per_species, zeta_list_per_species,
        k_cell, k_displ,
        lablen, cfglen, endian, k_displ_use_dp
    )
    write_sparse_pattern_and_mats_append(
        fname,
        endian=endian,
        is_dp=is_dp,
        nspin=nspin,
        lgncol=lgncol,
        rows_by_col=rows_by_col,
        H_vals=H_vals,
        S_vals=S_vals,
        H2_vals=H2_vals
    )

def write_dm_full(filename, dm_vals, rows_by_col, lgncol):
    """
    Write a SIESTA-style unformatted (binary) DM file. Only supports
    spinless case for now.

    Parameters
    ----------
    filename : str or path-like
        Output file path. The file will be created or overwritten and written
        in Fortran unformatted sequential format, compatible with SIESTA's
        iodm.F routine for density-matrix I/O.
    dm_vals : list of numpy.ndarray
        List of per-row density-matrix values.
        Length must be nbasistot. For each i in [0, nbasistot-1],
        dm_vals[i] is a 1D numpy array of dtype float64 with shape
        (lgncol[i],), containing all nonzero DM elements in row i
        (single-spin block, nspin = 1), ordered consistently with
        rows_by_col[i].
    rows_by_col : list of numpy.ndarray
        List of per-row column indices.
        Length must be nbasistot. For each i, rows_by_col[i] is a 1D numpy
        array of dtype int64 with shape (lgncol[i],), containing the
        1-based column indices of the nonzero elements stored in dm_vals[i].
        These indices follow SIESTA's global orbital numbering convention.
    lgncol : sequence of int
        numdg array (per-row nonzero counts).
        A 1D sequence of length nbasistot where lgncol[i] gives the number
        of nonzero entries in row i. This determines both the length of
        rows_by_col[i] and dm_vals[i] and is written as the numdg record
        in the DM file.
    """
    nbasistot = len(lgncol)
    if len(dm_vals) != nbasistot or len(rows_by_col) != nbasistot:
        raise ValueError("Lengths of dm_vals, rows_by_col, and lgncol must match.")
    nspin = 1
    endian = "<"
    int_dt = np.dtype(endian + "i4")
    real_dt = np.dtype(endian + "f8")
    hdr_dt = np.dtype(endian + "u4")
    with FortranFile(filename, mode="w", header_dtype=hdr_dt) as f:
        f.write_record(np.asarray([nbasistot, nspin], dtype=int_dt))
        f.write_record(np.asarray(lgncol, dtype=int_dt))
        for i in range(nbasistot):
            ncol = int(lgncol[i])
            cols = np.asarray(rows_by_col[i], dtype=int_dt)
            if cols.size != ncol:
                raise ValueError(f"Row {i}: len(rows_by_col[i]) != lgncol[i].")
            f.write_record(cols)
        for ispin in range(nspin):
            for i in range(nbasistot):
                ncol = int(lgncol[i])
                vals = np.asarray(dm_vals[i], dtype=real_dt)
                if vals.size != ncol:
                    raise ValueError(f"Row {i}: len(dm_vals[i]) != lgncol[i].")
                f.write_record(vals)


def _wrap_rank(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    av = np.abs(v)
    M  = av.max()
    rank = av.astype(np.int8)
    neg  = (v < 0)
    rank[neg] = (M + 1) + (M - av[neg])
    return rank

def siesta_sort_orbitals(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError("Input array must be of shape (N, 5)")

    tx = arr[:, 0]
    ty = arr[:, 1]
    tz = arr[:, 2]
    iat_uc = arr[:, 3]
    iorb = arr[:, 4]

    rx = _wrap_rank(tx).astype(np.uint64)
    ry = _wrap_rank(ty).astype(np.uint64)
    rz = _wrap_rank(tz).astype(np.uint64)

    iat_uc = iat_uc.astype(np.uint64)
    iorb   = iorb.astype(np.uint64)

    Bx = rx.max() + 1
    By = ry.max() + 1
    Bz = rz.max() + 1
    Bi = iat_uc.max() + 1
    Bo = iorb.max() + 1

    key = rz
    key = key * By + ry
    key = key * Bx + rx
    key = key * Bi + iat_uc
    key = key * Bo + iorb

    order = np.argsort(key, kind='stable')
    return arr[order], order

def build_dense_lut(
    A: np.ndarray,
    *,
    fill_value: int = -1,
    dtype=np.int64,
    max_volume: int | None = None,
    safe_query: bool = True,
):
    A = np.asarray(A)
    if A.ndim != 2 or not np.issubdtype(A.dtype, np.integer):
        raise ValueError("A must be a 2D integer array of shape (N, K).")
    N, K = A.shape

    mins = A.min(axis=0)
    maxs = A.max(axis=0)
    shape = (maxs - mins + 1).astype(np.int64)

    vol = int(np.prod(shape)) if shape.size else 0
    if max_volume is not None and vol > max_volume:
        raise MemoryError(
            f"LUT volume {vol} exceeds max_volume={max_volume}. "
            "Use a sparse/hash map instead."
        )

    # 1D LUT
    lut = np.full(vol, fill_value, dtype=dtype)

    # Flattened indices of A in the LUT
    P = (A - mins).astype(np.int64, copy=False)            # (N, K), all >=0
    flat_idx = np.ravel_multi_index(tuple(P.T), shape)     # (N,)
    lut[flat_idx] = np.arange(N, dtype=dtype)

    # Query functions
    def f1(x) -> int:
        x = np.asarray(x)
        if x.shape != (K,):
            raise ValueError(f"x must have shape ({K},), got {x.shape}.")
        if not np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.int64, copy=False)
        p = (x - mins).astype(np.int64, copy=False)
        if safe_query:
            if np.any((p < 0) | (p >= shape)):
                return int(fill_value)
        idx = np.ravel_multi_index(tuple(p), shape)
        return int(lut[idx])

    def fN(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != K:
            raise ValueError(f"X must have shape (M, {K}), got {X.shape}.")
        if not np.issubdtype(X.dtype, np.integer):
            X = X.astype(np.int64, copy=False)
        Pq = (X - mins).astype(np.int64, copy=False)       # (M, K)

        if safe_query:
            mask = np.all((Pq >= 0) & (Pq < shape), axis=1)
            out = np.full(X.shape[0], fill_value, dtype=dtype)
            if mask.any():
                idx = np.ravel_multi_index(tuple(Pq[mask].T), shape)
                out[mask] = lut[idx]
            return out
        else:
            idx = np.ravel_multi_index(tuple(Pq.T), shape) # 可能抛 IndexError
            return lut[idx]

    return f1, fN, mins, shape

def orb_slice_in_uc(aodata):
    structure = aodata.structure
    atom_spcs = structure.atomic_species
    order = np.argsort(atom_spcs)
    ispcs = order[np.searchsorted(atom_spcs, structure.atomic_numbers, sorter=order)]

    ls_orb_list = [aodata.ls_spc[spc] for spc in atom_spcs]
    max_len = max(len(x) for x in ls_orb_list)
    ls_orb = np.full((len(ls_orb_list), max_len), -1, dtype=int)
    for i, row in enumerate(ls_orb_list):
        ls_orb[i, :len(row)] = row
    mask_ls_orb = (ls_orb >= 0)
    dims_orb = (2 * ls_orb + 1) * mask_ls_orb
    nradial_ispc = np.array([aodata.nradial_spc[spc] for spc in atom_spcs], dtype=int)

    dims_orb_eachatom = dims_orb[ispcs]
    nradial_eachatom = nradial_ispc[ispcs]
    mask_orb_eachatom = mask_ls_orb[ispcs]

    dims_tot_eachatom = np.sum(dims_orb_eachatom, axis=1)
    chunk_boundaries_uc = np.cumsum(dims_orb_eachatom, axis=1)# * mask_orb_eachatom
    chunk_boundaries_atom = np.cumsum(dims_tot_eachatom)
    chunk_boundaries_uc = np.hstack((np.zeros((structure.natom, 1), dtype=int), chunk_boundaries_uc))
    chunk_boundaries_atom = np.hstack((0, chunk_boundaries_atom))
    chunk_boundaries_orb = chunk_boundaries_uc.copy()
    chunk_boundaries_uc += chunk_boundaries_atom[:-1, np.newaxis]
    chunk_boundaries_uc[:,1:] *= mask_orb_eachatom
    chunk_boundaries_orb[:,1:] *= mask_orb_eachatom

    last_row = chunk_boundaries_uc[-1]
    idx = np.nonzero(last_row)[0]
    num_orb_uc = last_row[idx[-1]]

    return chunk_boundaries_uc, chunk_boundaries_orb, chunk_boundaries_atom[1:], mask_orb_eachatom, num_orb_uc

def build_row_ranges(row_idxs_start, row_idxs_end,
                     cols_idxs_start, cols_idxs_end,
                     num_orb_uc: int):
    rs = np.asarray(row_idxs_start, dtype=np.int64)
    re = np.asarray(row_idxs_end,   dtype=np.int64)
    cs = np.asarray(cols_idxs_start, dtype=np.int64)
    ce = np.asarray(cols_idxs_end,   dtype=np.int64)

    R = re - rs
    L = ce - cs

    rows_all = np.concatenate([
        np.repeat(np.arange(rs[i], re[i], dtype=np.int64), L[i])
        for i in range(len(rs))
        if R[i] > 0 and L[i] > 0
    ], axis=0) if len(rs) else np.empty(0, dtype=np.int64)

    vals_all = np.concatenate([
        np.tile(np.arange(cs[i], ce[i], dtype=np.int64), R[i])
        for i in range(len(cs))
        if R[i] > 0 and L[i] > 0
    ], axis=0) if len(cs) else np.empty(0, dtype=np.int64)

    perm = np.argsort(rows_all, kind='mergesort')
    rows_sorted = rows_all[perm]
    vals_sorted = vals_all[perm]

    counts = np.bincount(rows_sorted, minlength=num_orb_uc)
    offsets = np.cumsum(np.r_[0, counts[:-1]])
    offsets = np.append(offsets, np.sum(counts))

    out = [vals_sorted[offsets[r]:offsets[r+1]] for r in range(num_orb_uc)]
    return counts, out

def batch_flat_segments(chunk_boundaries_orb: np.ndarray,
                        mask_orb_eachatom: np.ndarray,
                        Q: np.ndarray):
    chunk = np.asarray(chunk_boundaries_orb)
    mask  = np.asarray(mask_orb_eachatom, dtype=bool)
    Q = np.asarray(Q, dtype=np.int64) # M rows, each row: (iatom_i, iorb_i, iatom_j, iorb_j)

    i1 = Q[:, 0]
    i2 = Q[:, 1]
    j1 = Q[:, 2]
    j2 = Q[:, 3]

    s1 = chunk[i1, j1]
    e1_raw = chunk[i1, j1 + 1]
    valid1 = mask[i1, j1]
    e1 = np.where(valid1, e1_raw, s1)

    s2 = chunk[i2, j2]
    e2_raw = chunk[i2, j2 + 1]
    valid2 = mask[i2, j2]
    e2 = np.where(valid2, e2_raw, s2)
    n2_tot = chunk.max(axis=1)[i2]

    r = (e1 - s1).astype(np.int64)   # rows per query
    c = (e2 - s2).astype(np.int64)   # cols per query

    active = (r > 0) & (c > 0) & (n2_tot > 0)
    if not np.any(active):
        return (np.empty((0, 2), dtype=np.int64),
                np.zeros(len(Q), dtype=np.int64),
                np.zeros(len(Q) + 1, dtype=np.int64))

    i1 = i1[active]; i2 = i2[active]
    s1 = s1[active]; e1 = e1[active]
    s2 = s2[active]; e2 = e2[active]
    n2_tot = n2_tot[active]
    r = r[active]; c = c[active]

    M_act = len(r)
    qid = np.repeat(np.arange(M_act, dtype=np.int64), r)
    cs = np.cumsum(r)
    rel = np.arange(cs[-1], dtype=np.int64) - np.repeat(cs - r, r)

    o1 = s1[qid] + rel
    starts = o1 * n2_tot[qid] + s2[qid]
    stops  = o1 * n2_tot[qid] + e2[qid]
    segs = np.stack([starts, stops], axis=1)

    counts = np.zeros(len(Q), dtype=np.int64)
    counts[np.where(active)[0]] = r

    offsets = np.zeros(len(Q) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    return segs, counts, offsets

def map_and_mask(pair_indices_siesta: np.ndarray, pair_indices_deeph: np.ndarray):
    pos = np.searchsorted(pair_indices_deeph, pair_indices_siesta)
    valid = (pos < len(pair_indices_deeph)) & \
            (pair_indices_deeph[pos] == pair_indices_siesta)

    indices_in_deeph = np.where(valid, pos, 0)
    mask_deeph = np.isin(pair_indices_siesta, pair_indices_deeph)

    return indices_in_deeph, mask_deeph

def alternating_sign_rc(r: int, c: int):
    """
    phase difference in spherical harmonics between DeepH and SIESTA, (-1)**(m%2)
    """
    l1 = (r - 1) // 2
    l2 = (c - 1) // 2
    rows = (np.arange(r) + l1) & 1
    cols = (np.arange(c) + l2) & 1
    return (1 - 2 * (rows[:, None] ^ cols[None, :])).astype(np.int8)

@lru_cache(maxsize=128)
def _pattern_cached(r: int, c: int):
    return alternating_sign_rc(r, c)

def build_sparse_H_siesta_spinless(
    row_idxs_start, row_idxs_end, col_idxs_start, col_idxs_end, num_orb_uc,
    idx_siesta_in_deeph, mask_deeph, segs, counts, row_offsets,
    entries, chunk_boundaries
):
    num_entries = len(entries)
    assert num_entries == 1 or num_entries == 2 or num_entries == 5
    dtype = entries[0].dtype

    nrows_g = num_orb_uc
    H_vals = [[] for _ in range(nrows_g)]
    S_vals = [[] for _ in range(nrows_g)]
    cols_each_row = [[] for _ in range(nrows_g)]

    norb_pairs = len(row_idxs_start)
    for ientries in range(num_entries):
        entries_this = entries[ientries]
        nnz = 0
        for iorb_pair in range(norb_pairs):
            row_start, row_end = row_idxs_start[iorb_pair], row_idxs_end[iorb_pair]
            col_start, col_end = col_idxs_start[iorb_pair], col_idxs_end[iorb_pair]
            nrow = row_end - row_start
            ncol = col_end - col_start
            if not mask_deeph[iorb_pair]:
                matvals_this = np.zeros((nrow, ncol), dtype=entries_this.dtype)
            else:
                iblock = idx_siesta_in_deeph[iorb_pair]
                chunk_start, chunk_end = chunk_boundaries[iblock], chunk_boundaries[iblock+1]
                chunk_this = entries_this[chunk_start:chunk_end]
                segs_this = segs[row_offsets[iorb_pair]:row_offsets[iorb_pair+1]]

                idx = segs_this[:, 0, None] + np.arange(ncol)[None, :]
                matvals_this = np.where(np.arange(ncol)[None, :] < ncol , chunk_this[idx], np.nan)
                phase = _pattern_cached(nrow, ncol)
                matvals_this *= phase
                if ientries != num_entries - 1:
                    matvals_this *= 2 / HARTREE_TO_EV

            for irow in range(row_start, row_end):
                if ientries == 0:
                    cols_each_row[irow].append(np.arange(col_start, col_end, dtype=np.int64)+1)
                if ientries != num_entries - 1 or num_entries == 1:
                    H_vals[irow].append(matvals_this[irow - row_start, :])
                else:
                    S_vals[irow].append(matvals_this[irow - row_start, :])
            nnz += matvals_this.shape[0] * matvals_this.shape[1]
    if num_entries == 1:
        H_vals = [np.concatenate(H_vals[r], axis=0).astype(dtype, copy=False) for r in range(nrows_g)]
        S_vals = None
    else:
        H_vals = [np.concatenate(H_vals[r], axis=0).reshape(num_entries-1,-1).T.astype(dtype, copy=False) for r in range(nrows_g)]
        S_vals = [np.concatenate(S_vals[r], axis=0).astype(dtype, copy=False) for r in range(nrows_g)]
    cols_each_row = [np.concatenate(cols_each_row[r], axis=0) for r in range(nrows_g)]

    return H_vals, S_vals, cols_each_row

def dft_dict_to_siesta_HS_spinless(struc, aodata, projR, dft_dict_H, dft_dict_S):
    atom_pairs_H = dft_dict_H["atom_pairs"]
    chunk_boundaries_H = dft_dict_H["chunk_boundaries"]
    entries_H = dft_dict_H["entries"]

    atom_pairs_S = dft_dict_S["atom_pairs"]
    chunk_boundaries_S = dft_dict_S["chunk_boundaries"]
    entries_S = dft_dict_S["entries"]

    assert np.array_equal(atom_pairs_H, atom_pairs_S)
    assert np.array_equal(chunk_boundaries_H, chunk_boundaries_S)
    atom_pairs = atom_pairs_H
    chunk_boundaries = chunk_boundaries_H
    entries_list = [entries_H, entries_S]

    _, _, pairs_key_direct = find_orb_pairs_direct(struc, aodata.cutoffs_orb)
    pairs_key_proj = find_orb_pairs_proj(struc, projR.cutoffs, aodata.cutoffs_orb)
    pairs_key = np.unique(np.vstack((pairs_key_direct, pairs_key_proj)), axis=0)
    pair_indices_siesta = pairs_to_indices(struc, pairs_key[:,:3], pairs_key[:,3:5])
    argsort = np.argsort(pair_indices_siesta)
    pairs_key = pairs_key[argsort]
    pair_indices_siesta = pair_indices_siesta[argsort]
    pair_indices_deeph = pairs_to_indices(struc, atom_pairs[:,:3], atom_pairs[:,3:5]) # !!! assume sorted
    # which atom pair each orb pair belongs to
    idx_siesta_in_deeph, mask_deeph = map_and_mask(pair_indices_siesta, pair_indices_deeph)

    all_possible_tranlations = np.unique(pairs_key[:, :3], axis=0)
    print(f"Total {len(all_possible_tranlations)} unique translations in orbital pairs.")
    max_tx, min_tx = all_possible_tranlations[:,0].max(), all_possible_tranlations[:,0].min()
    max_ty, min_ty = all_possible_tranlations[:,1].max(), all_possible_tranlations[:,1].min()
    max_tz, min_tz = all_possible_tranlations[:,2].max(), all_possible_tranlations[:,2].min()
    sc_limits = np.array([[min_tx, max_tx+1], [min_ty, max_ty+1], [min_tz, max_tz+1]])
    supercell = SuperCell(struc, sc_limits, ordering='siesta')
    nsc = [max_tx - min_tx + 1, max_ty - min_ty + 1, max_tz - min_tz + 1]
    isc_off = supercell.translations_cuc.flatten().tolist()
    _, sc_to_idx, _, _ = build_dense_lut(supercell.translations_cuc)

    chunk_boundaries_uc, chunk_boundaries_orb, chunk_boundaries_atom, mask_orb_eachatom, num_orb_uc = orb_slice_in_uc(aodata)

    pairs_key_tmp = pairs_key
    lgncol = np.zeros(np.max(chunk_boundaries_uc[-1]), dtype=int)
    for key_idx in range(pairs_key_tmp.shape[0]):
        iatom1 = pairs_key_tmp[key_idx, 3]
        # iatom2 = pairs_key_tmp[key_idx, 4]
        iorb1 = pairs_key_tmp[key_idx, 5]
        iorb2 = pairs_key_tmp[key_idx, 6]
        # spc1 = struc.atomic_numbers[pairs_key_tmp[key_idx, 3]]
        spc2 = struc.atomic_numbers[pairs_key_tmp[key_idx, 4]]
        dim2 = aodata.ls_spc[spc2][iorb2] * 2 + 1

        igncol_start = chunk_boundaries_uc[iatom1, iorb1]
        igncol_end   = chunk_boundaries_uc[iatom1, iorb1+1]
        lgncol[igncol_start:igncol_end] += dim2
    nnz = np.sum(lgncol)
    print(f"Total {pairs_key_tmp.shape[0]} unique orbital pairs in orbital pairs, nnz={nnz}.")

    supercell_idxs = sc_to_idx(pairs_key[:, :3])
    row_idxs_start = chunk_boundaries_uc[pairs_key[:, 3], pairs_key[:, 5]]
    row_idxs_end   = chunk_boundaries_uc[pairs_key[:, 3], pairs_key[:, 5]+1]
    col_idxs_start = chunk_boundaries_uc[pairs_key[:, 4], pairs_key[:, 6]] + supercell_idxs * num_orb_uc
    col_idxs_end   = chunk_boundaries_uc[pairs_key[:, 4], pairs_key[:, 6]+1] + supercell_idxs * num_orb_uc

    segs, counts, row_offsets = batch_flat_segments(chunk_boundaries_orb, mask_orb_eachatom, pairs_key[:,3:7])

    H_vals, S_vals, cols_each_row = build_sparse_H_siesta_spinless(row_idxs_start, row_idxs_end, col_idxs_start, col_idxs_end, num_orb_uc,
                                idx_siesta_in_deeph, mask_deeph, segs, counts, row_offsets,
                                entries_list, chunk_boundaries)
    lgncol = [len(c) for c in cols_each_row]
    return H_vals, S_vals, cols_each_row, lgncol, nsc, isc_off, chunk_boundaries_atom

def dft_dict_to_siesta_dm_spinless(struc, aodata, projR, dft_dict_dm):
    atom_pairs = dft_dict_dm["atom_pairs"]
    chunk_boundaries = dft_dict_dm["chunk_boundaries"]
    entries_list = [dft_dict_dm["entries"]]

    _, _, pairs_key_direct = find_orb_pairs_direct(struc, aodata.cutoffs_orb)
    pairs_key_proj = find_orb_pairs_proj(struc, projR.cutoffs, aodata.cutoffs_orb)
    pairs_key = np.unique(np.vstack((pairs_key_direct, pairs_key_proj)), axis=0)
    pair_indices_siesta = pairs_to_indices(struc, pairs_key[:,:3], pairs_key[:,3:5])
    argsort = np.argsort(pair_indices_siesta)
    pairs_key = pairs_key[argsort]
    pair_indices_siesta = pair_indices_siesta[argsort]
    pair_indices_deeph = pairs_to_indices(struc, atom_pairs[:,:3], atom_pairs[:,3:5]) # !!! assume sorted
    # which atom pair each orb pair belongs to
    idx_siesta_in_deeph, mask_deeph = map_and_mask(pair_indices_siesta, pair_indices_deeph)

    all_possible_tranlations = np.unique(pairs_key[:, :3], axis=0)
    print(f"Total {len(all_possible_tranlations)} unique translations in orbital pairs.")
    max_tx, min_tx = all_possible_tranlations[:,0].max(), all_possible_tranlations[:,0].min()
    max_ty, min_ty = all_possible_tranlations[:,1].max(), all_possible_tranlations[:,1].min()
    max_tz, min_tz = all_possible_tranlations[:,2].max(), all_possible_tranlations[:,2].min()
    sc_limits = np.array([[min_tx, max_tx+1], [min_ty, max_ty+1], [min_tz, max_tz+1]])
    supercell = SuperCell(struc, sc_limits, ordering='siesta')
    # nsc = [max_tx - min_tx + 1, max_ty - min_ty + 1, max_tz - min_tz + 1]
    # isc_off = supercell.translations_cuc.flatten().tolist()
    _, sc_to_idx, _, _ = build_dense_lut(supercell.translations_cuc)

    chunk_boundaries_uc, chunk_boundaries_orb, chunk_boundaries_atom, mask_orb_eachatom, num_orb_uc = orb_slice_in_uc(aodata)

    pairs_key_tmp = pairs_key
    lgncol = np.zeros(np.max(chunk_boundaries_uc[-1]), dtype=int)
    for key_idx in range(pairs_key_tmp.shape[0]):
        iatom1 = pairs_key_tmp[key_idx, 3]
        # iatom2 = pairs_key_tmp[key_idx, 4]
        iorb1 = pairs_key_tmp[key_idx, 5]
        iorb2 = pairs_key_tmp[key_idx, 6]
        # spc1 = struc.atomic_numbers[pairs_key_tmp[key_idx, 3]]
        spc2 = struc.atomic_numbers[pairs_key_tmp[key_idx, 4]]
        dim2 = aodata.ls_spc[spc2][iorb2] * 2 + 1

        igncol_start = chunk_boundaries_uc[iatom1, iorb1]
        igncol_end   = chunk_boundaries_uc[iatom1, iorb1+1]
        lgncol[igncol_start:igncol_end] += dim2
    nnz = np.sum(lgncol)
    print(f"Total {pairs_key_tmp.shape[0]} unique orbital pairs in orbital pairs, nnz={nnz}.")

    supercell_idxs = sc_to_idx(pairs_key[:, :3])
    row_idxs_start = chunk_boundaries_uc[pairs_key[:, 3], pairs_key[:, 5]]
    row_idxs_end   = chunk_boundaries_uc[pairs_key[:, 3], pairs_key[:, 5]+1]
    col_idxs_start = chunk_boundaries_uc[pairs_key[:, 4], pairs_key[:, 6]] + supercell_idxs * num_orb_uc
    col_idxs_end   = chunk_boundaries_uc[pairs_key[:, 4], pairs_key[:, 6]+1] + supercell_idxs * num_orb_uc

    segs, counts, row_offsets = batch_flat_segments(chunk_boundaries_orb, mask_orb_eachatom, pairs_key[:,3:7])

    dm_vals, _, cols_each_row = build_sparse_H_siesta_spinless(row_idxs_start, row_idxs_end, col_idxs_start, col_idxs_end, num_orb_uc,
                                idx_siesta_in_deeph, mask_deeph, segs, counts, row_offsets,
                                entries_list, chunk_boundaries)
    lgncol = [len(c) for c in cols_each_row]
    return dm_vals, cols_each_row, lgncol

def encode_by_order(arr: np.ndarray):
    uniques, idx = np.unique(arr, return_index=True)
    order = np.argsort(idx)
    uniques_in_order = uniques[order]
    
    sorted_uniques = np.sort(uniques_in_order)
    sorter = np.argsort(uniques_in_order)
    
    pos_in_sorted = np.searchsorted(sorted_uniques, arr)
    codes = sorter[pos_in_sorted]

    counts = np.bincount(codes, minlength=len(uniques_in_order))
    return codes, uniques_in_order, counts

def _read_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        atom_pairs = np.array(f["atom_pairs"][:], dtype=np.int64)
        boundaries = np.array(f["chunk_boundaries"][:], dtype=np.int64)
        shapes = np.array(f["chunk_shapes"][:], dtype=np.int64)
        dset = f['entries']
        entries_dtype = np.complex128 if dset.dtype.kind == 'c' else np.float64
        entries = np.array(dset[:], dtype=entries_dtype)
    return {"atom_pairs": atom_pairs,
            "chunk_boundaries": boundaries,
            "chunk_shapes": shapes,
            "entries": entries}

def transfer_one_deeph_to_siesta(dir_name, siesta_path, deeph_path, basis_path, is_pred=False):
    """
    Convert DeepH DFT data to SIESTA HSX format for one directory.
    Args:
        dir_name (str): Name of the directory to process.
        siesta_path (str): Path to the SIESTA output directory.
        deeph_path (str): Path to the DeepH output directory.
        basis_path (str): The basis set root directory containing .ion files.
        is_pred (bool): When set to True, convert the file 'hamiltonian_pred.h5', otherwise convert 'hamiltonian.h5'.
    """
    try:
        dir_name = str(dir_name)
        siesta_path = Path(siesta_path)
        deeph_path = Path(deeph_path)
        deeph_dir_path = deeph_path / dir_name
        if not deeph_dir_path.is_dir():
            return
        siesta_dir_path = siesta_path / dir_name
        siesta_dir_path.mkdir(parents=True, exist_ok=True)
        
        poscar_path = deeph_dir_path / DEEPX_POSCAR_FILENAME
        if not is_pred:
            file_path_H = deeph_dir_path / DEEPX_HAMILTONIAN_FILENAME
        else:
            file_path_H = deeph_dir_path / DEEPX_PREDICT_HAMILTONIAN_FILENAME
        file_path_S = deeph_dir_path / DEEPX_OVERLAP_FILENAME

        atoms = read(poscar_path)
        struc = Structure(atoms.cell.array / BOHR_TO_ANGSTROM, atoms.numbers, atoms.positions / BOHR_TO_ANGSTROM, atomic_positions_is_cart=True)
        aodata = AOData_siesta(struc, basis_path_root=basis_path, mode='orb')
        projR = AOData_siesta(struc, basis_path_root=basis_path, mode='projR')

        dft_dict_H = _read_h5(file_path_H)
        dft_dict_S = _read_h5(file_path_S)
        H_vals, S_vals, rows_by_col, lgncol, nsc, isc_off, chunk_boundaries_atom = \
            dft_dict_to_siesta_HS_spinless(struc, aodata, projR, dft_dict_H, dft_dict_S)

        ############################ Prepare Header Info ############################
        version = 2
        is_dp = True

        na_u = struc.natom
        nrows_g = len(lgncol)
        nspin = 1
        nspecies = struc.nspc
        # nsc previously computed

        ucell = struc.rprim.tolist()
        # Ef should in principle be computed elsewhere. 
        # But in the band unfolding workflow, it is not read from the HSX file.
        Ef, temp = 0.0 * 2 / HARTREE_TO_EV, 0.0 
        # isc_off previously computed

        xa = struc.atomic_positions_cart.tolist()

        isa, unique_iat, counts = encode_by_order(struc.atomic_numbers)
        isa = (isa + 1).tolist()
        lasto = chunk_boundaries_atom.tolist()

        labelfis = atom_number2name(unique_iat)
        nofis    = [aodata.norbfull_spc[spc] for spc in unique_iat]

        cn_n_list_per_species = []
        l_list_per_species = []
        zeta_list_per_species = []
        zvalfis = []
        for spc in unique_iat:
            ls = aodata.ls_spc[spc]
            ns = aodata.ns_spc[spc]
            zs = aodata.zs_spc[spc]

            n_list_this = []
            l_list_this = []
            z_list_this = []
            for l, n, z in zip(ls, ns, zs):
                for _ in range((2 * l + 1)):
                    n_list_this.append(n)
                    l_list_this.append(l)
                    z_list_this.append(z)
            cn_n_list_per_species.append(n_list_this)
            l_list_per_species.append(l_list_this)
            zeta_list_per_species.append(z_list_this)
            zvalfis.append(aodata.charge_spc[spc])

        qtot = float(sum(zvalfis[i] * counts[i] for i in range(nspecies)))

        # k points
        k_cell  = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        k_displ = [0.0, 0.0, 0.0]
        ########################################################

        hsx_dest = siesta_dir_path / SIESTA_HSX_FILENAME
        write_hsx_full(
            hsx_dest,
            # ---- header and k points ----
            version=2, is_dp=True,
            na_u=na_u, nrows_g=nrows_g, nspin=nspin, nspecies=nspecies, nsc=nsc,
            ucell=ucell, Ef=Ef, qtot=qtot, temp=temp,
            isc_off=isc_off, xa=xa, isa=isa, lasto=lasto,
            labelfis=labelfis, zvalfis=zvalfis, nofis=nofis,
            cn_n_list_per_species=cn_n_list_per_species,
            l_list_per_species=l_list_per_species,
            zeta_list_per_species=zeta_list_per_species,
            k_cell=k_cell, k_displ=k_displ,
            # ---- sparsity info + matrix ----
            lgncol=lgncol,
            rows_by_col=rows_by_col,    # 1-based !!!
            H_vals=H_vals,
            S_vals=S_vals,
            # H2_vals=H2_vals,
            lablen=20, cfglen=4, endian="<", k_displ_use_dp=None
        )
        print(f"written to {hsx_dest}.")

    except Exception as e:
        print(f"Error in {dir_name}: {e}")
        
        
def transfer_one_dm_to_siesta(dir_name, siesta_path, deeph_path, basis_path):
    """
    Convert DeepH DFT data to SIESTA HSX format for one directory.
    Args:
        dir_name (str): Name of the directory to process.
        siesta_path (str): Path to the SIESTA output directory.
        deeph_path (str): Path to the DeepH output directory.
        basis_path (str): The basis set root directory containing .ion files.
    """
    try:
        dir_name = str(dir_name)
        siesta_path = Path(siesta_path)
        deeph_path = Path(deeph_path)
        basis_path = Path(basis_path)
        deeph_dir_path = deeph_path / dir_name
        if not deeph_dir_path.is_dir():
            return
        print('Processing directory:', dir_name)
        siesta_dir_path = siesta_path / dir_name
        siesta_dir_path.mkdir(parents=True, exist_ok=True)
        
        poscar_path = deeph_dir_path / DEEPX_POSCAR_FILENAME
        file_path_dm = deeph_dir_path / DEEPX_DENSITY_MATRIX_FILENAME

        atoms = read(poscar_path)
        struc = Structure(atoms.cell.array / BOHR_TO_ANGSTROM, atoms.numbers, atoms.positions / BOHR_TO_ANGSTROM, atomic_positions_is_cart=True)
        aodata = AOData_siesta(struc, basis_path_root=basis_path, mode='orb')
        projR = AOData_siesta(struc, basis_path_root=basis_path, mode='projR')

        dft_dict_dm = _read_h5(file_path_dm)
        dm_vals, rows_by_col, lgncol = \
            dft_dict_to_siesta_dm_spinless(struc, aodata, projR, dft_dict_dm)

        dm_dest = siesta_dir_path / SIESTA_DM_FILENAME
        write_dm_full(dm_dest, dm_vals, rows_by_col, lgncol)
        print(f"written to {dm_dest}.")
    
    except Exception as e:
        print(f"Error in {dir_name}: {e}")