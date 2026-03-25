import numpy as np

from deepx_dock.compute.overlap.findpairs import pairs_within_cutoff
from HPRO.v2h.twocenter import TwoCenterIntgSplines
from HPRO.utils.misc import slice_same
from HPRO.from_gpaw.gaunt import gaunt
from HPRO.matao.matao import MatAO
import HPRO.config as CFG


def calc_overlap(aodata1, aodata2=None, Ecut=1800, kdense=15):
    """
    Calculate overlap matrix S_ij = <φ_i | φ_j>.

    Uses uniform linear r-space grid for accurate Bessel transform,
    matching OpenMX's approach.

    Parameters
    ----------
    aodata1 : AOData
        AOData object containing basis set information.
    aodata2 : AOData, optional
        Second AOData for cross-structure overlap.
        If None, calculate self-overlap.
    Ecut : float
        Energy cutoff for k-space grid (Hartree).
        Controls kmax = sqrt(2*Ecut). Default: 1800 Ha.
    kdense : float
        k-space grid density. grid_nq = kmax * kdense.
        Default: 15 (OpenMX default).

    Returns
    -------
    MatAO
        Overlap matrices organized by atom pairs.

    Examples
    --------
    >>> overlap = calc_overlap(aodata)
    >>> overlap = calc_overlap(aodata, Ecut=50, kdense=91.6)
    """
    return _calc_matrix_impl(aodata1, aodata2, Ecut, kdense, kind=1)


def calc_kinetic(aodata1, aodata2=None, Ecut=1800, kdense=15):
    """
    Calculate kinetic energy matrix K_ij = <φ_i | -½∇² | φ_j>.

    Uses the same infrastructure as overlap calculation.

    Parameters
    ----------
    aodata1 : AOData
        AOData object containing basis set information.
    aodata2 : AOData, optional
        Second AOData for cross-structure kinetic matrix.
        If None, calculate self-kinetic matrix.
    Ecut : float
        Energy cutoff for k-space grid (Hartree).
        Default: 1800 Ha.
    kdense : float
        k-space grid density. Default: 15.

    Returns
    -------
    MatAO
        Kinetic matrices organized by atom pairs.
    """
    return _calc_matrix_impl(aodata1, aodata2, Ecut, kdense, kind=2)


def _calc_matrix_impl(aodata1, aodata2, Ecut, kdense, kind):
    """Internal implementation for overlap/kinetic matrix calculation."""
    original_kdense = CFG.AOFT_QGRID_DEN
    if kdense is not None:
        CFG.AOFT_QGRID_DEN = kdense

    try:
        return _calc_two_center_intg(aodata1, aodata2, Ecut, kind)
    finally:
        CFG.AOFT_QGRID_DEN = original_kdense


def _calc_two_center_intg(aodata1, aodata2, Ecut, kind):
    """Internal implementation of overlap calculation."""
    is_selfolp = aodata2 is None
    if is_selfolp:
        aodata2 = aodata1
    else:
        assert np.linalg.norm(aodata1.structure.rprim - aodata2.structure.rprim) < 1e-4
        assert np.sort(aodata1.structure.atomic_species) == np.sort(aodata2.structure.atomic_species)

    struc1 = aodata1.structure
    struc2 = aodata2.structure

    aodata1.calc_phiQ(Ecut)
    if not is_selfolp:
        aodata2.calc_phiQ(Ecut)

    # find lmax and GLLL
    lmax = 0
    for spc in struc1.atomic_species:
        l1max = max(aodata1.ls_spc[spc])
        l2max = 0 if is_selfolp else max(aodata2.ls_spc[spc])
        lmax = max(l1max, l2max, lmax)
    GLLL = gaunt(lmax)

    # initialize splines of two-center integral
    orbpairs = {}  # Dict[(int, int) -> List]
    for ispc in range(struc1.nspc):
        range2 = range(ispc, struc1.nspc) if is_selfolp else range(struc2.nspc)
        for jspc in range2:
            spc1 = struc1.atomic_species[ispc]
            spc2 = struc2.atomic_species[jspc]
            orbpairs_thisij = []
            for iorb in range(aodata1.nradial_spc[spc1]):
                # for Z1==Z2: only needs to calculate half of the splines
                istartj = iorb if (is_selfolp and (spc1 == spc2)) else 0
                for jorb in range(istartj, aodata2.nradial_spc[spc2]):
                    r1 = aodata1.phirgrids_spc[spc1][iorb].rcut
                    r2 = aodata2.phirgrids_spc[spc2][jorb].rcut
                    rcut = r1 + r2
                    tic_splines = TwoCenterIntgSplines(
                        aodata1.phiQlist_spc[spc1][iorb], aodata2.phiQlist_spc[spc2][jorb], rcut, kind=kind, GLLL=GLLL
                    )
                    orbpairs_thisij.append(tic_splines)
            orbpairs[(spc1, spc2)] = orbpairs_thisij

    if is_selfolp:
        pairs_ij = pairs_within_cutoff(struc1, aodata1.cutoffs)
        pairs_ij.sort()
        pairs_ij.remove_ji()
    else:
        pairs_ij = pairs_within_cutoff(struc1, aodata1.cutoffs, structure2=struc2, cutoffs2=aodata2.cutoffs)

    overlaps = MatAO.init_mats(pairs_ij, aodata1, aodata2=aodata2, filling_value=None)

    translations = overlaps.translations
    atom_pairs = overlaps.atom_pairs
    # spc_pairs = stru.atomic_numbers[atom_pairs]
    spc_pairs = np.column_stack([struc1.atomic_numbers[atom_pairs[:, 0]], struc2.atomic_numbers[atom_pairs[:, 1]]])

    # If self-olp:
    # Pairs are sorted first according to atomic numbers, then according to atomic species.
    # Therefore, pairs with the same atomic species are close to each other.
    # Furthermore, for each pair of atomic species (Z2, Z1) where Z2>Z1,
    # it must appear after the pair (Z1, Z2).

    slices_ij = slice_same(spc_pairs[:, 0] * 200 + spc_pairs[:, 1])

    for ix_ij in range(len(slices_ij) - 1):
        start_ij = slices_ij[ix_ij]
        end_ij = slices_ij[ix_ij + 1]
        nthisij = end_ij - start_ij
        thisij = slice(start_ij, end_ij)
        spc1, spc2 = spc_pairs[start_ij]

        # allocate S array
        size1 = aodata1.norbfull_spc[spc1]
        size2 = aodata2.norbfull_spc[spc2]
        S_thisij = np.empty((nthisij, size1, size2))

        # calculate overlap using splines
        pos_i = struc1.atomic_positions_cart[atom_pairs[thisij, 0]]
        pos_j = struc2.atomic_positions_cart[atom_pairs[thisij, 1]]
        Rs_thisij = translations[thisij, :] @ struc1.rprim + pos_j - pos_i  # struc1 and struc2 have the same rprim
        orbpairs_thisij = orbpairs[(spc1, spc2)]
        ix_orbpair = 0
        for iorb in range(aodata1.nradial_spc[spc1]):
            istartj = iorb if (is_selfolp and (spc1 == spc2)) else 0
            for jorb in range(istartj, aodata2.nradial_spc[spc2]):
                # print(iorb, jorb)
                # print(spc1, spc2, iorb, jorb)
                slice1 = slice(aodata1.orbslices_spc[spc1][iorb], aodata1.orbslices_spc[spc1][iorb + 1])
                slice2 = slice(aodata2.orbslices_spc[spc2][jorb], aodata2.orbslices_spc[spc2][jorb + 1])
                orbpair_splines = orbpairs_thisij[ix_orbpair]
                olp = orbpair_splines.calc(Rs_thisij)
                S_thisij[:, slice1, slice2] = olp
                if is_selfolp and (spc1 == spc2) and (jorb > iorb):
                    olp = orbpair_splines.calc(-Rs_thisij)
                    S_thisij[:, slice2, slice1] = olp.transpose(0, 2, 1)
                ix_orbpair += 1

        # send overlaps to their correct positions
        for ii in range(nthisij):
            # ipair = argsort_ijji[start_ij + ii]
            overlaps.mats[start_ij + ii] = S_thisij[ii]

    if is_selfolp:
        overlaps.unfold_with_hermiticity()

    return overlaps
