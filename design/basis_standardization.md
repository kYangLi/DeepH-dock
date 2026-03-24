# Basis Standardization Design

**Status**: ✅ Complete  
**Version**: 0.9.12  
**Last Updated**: 2025-03-24

---

## Overview

Standardize basis set formats for DeepH-dock, enabling:
1. Reusable basis sets across calculations
2. Easy integration with HPRO's GridFunc system
3. Support for machine learning on basis functions

---

## Architecture

```
Step 1: Basis Standardization
OpenMX PAO (.pao) → basis_convert.py → basis.h5

Step 2: Overlap Computation (requires MPI)
OpenMX input (.dat) + basis.h5 → HPRO → overlap.h5, POSCAR, info.json
```

---

## Standard Basis Format (basis.h5)

### Structure (Flat Format)

```
basis.h5
│
├── @element: str              # e.g., "Fe"
├── @basis_name: str           # e.g., "Fe6.0H"
├── @source: str               # e.g., "openmx"
├── @normalized: bool          # whether radial functions are normalized
├── @units_length: str         # "bohr"
│
├── radial_grid: [Nr]          # radial grid points (Bohr)
├── mul_list: [lmax+1] int     # number of orbitals per L
└── radial_basis: [total_orbitals, Nr]  # all radial functions stacked
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Flat structure | AI/tensor-friendly, no nested groups |
| Single matrix | Optimal I/O performance |
| Minimal attributes | Derive `lmax`, `total_orbitals`, `radial_cutoff` from data |
| Element as string | User readability |
| Bohr units | Native to OpenMX/HPRO, avoids conversion errors |

### Inferred Values

```python
lmax = len(mul_list) - 1
total_orbitals = sum(mul_list)
radial_cutoff = radial_grid.max()
offsets = np.cumsum([0] + mul_list.tolist())[:-1]
```

### Indexing Convention

| l | mul | flat index | formula |
|---|-----|------------|---------|
| 0 | 0   | 0          | `offsets[0] + 0` |
| 0 | 1   | 1          | |
| 1 | 0   | 2          | `offsets[1] + 0` |
| 1 | 1   | 3          | |
| 2 | 0   | 4          | `offsets[2] + 0` |

---

## Module Structure

```
deepx_dock/
├── convert/openmx/
│   ├── basis_convert.py    # PAO → HDF5 conversion
│   └── _cli.py             # CLI: convert-basis, convert-single
│
└── compute/overlap/openmx/
    ├── parse_input.py      # Parse openmx.in
    ├── loader.py           # BasisLoader class
    ├── calculator.py       # OpenMXOverlapCalculator
    └── _cli.py             # CLI: calc
```

---

## CLI Commands

### Basis Conversion

```bash
# Batch convert PAO files
dock convert openmx convert-basis pao_folder/ basis/ --pattern "*7.0.pao"

# Single file
dock convert openmx convert-single Fe6.0H.pao basis/Fe6.0H.h5
```

### Overlap Calculation

```bash
# With auto-conversion
dock compute overlap openmx calc openmx.in basis/ --raw-basis-dir pao/

# Using existing basis.h5
dock compute overlap openmx calc openmx.in basis/
```

---

## Orbital Selection

OpenMX input syntax: `Mo7.0-s3p2d2`

| Syntax | Meaning |
|--------|---------|
| `s2p2d2` | 2 s + 2 p + 2 d orbitals |
| `s3p2d1f1` | 3 s + 2 p + 1 d + 1 f |
| `Fe6.0H` | Use all orbitals |

---

## Bug Fix Summary

| Problem | Cause | Solution |
|---------|-------|----------|
| Grid type mismatch | LinearRGD for log grid | Use ExpRGD |
| Extra orbitals loaded | Default value for unselected L | Skip unselected L |
| Unit mismatch | Angstrom passed to HPRO | Convert to Bohr |
| FRAC coords not handled | Missing FRAC case | `positions @ lattice` |
| Lattice transpose | `rprim = lattice.T` | Use `rprim = lattice` |
| Atom order mismatch | DeepH requires species grouping | Auto-reorder atoms |

---

## Implementation Timeline

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | PAO parser, HDF5 conversion |
| 2 | ✅ | OpenMX input parser |
| 3 | ✅ | Overlap computation, HPRO integration |
| 4 | ✅ | Testing: 131 pairs, max error 2.5e-4 |
| 5 | ✅ | Documentation |

---

## References

| Resource | Path |
|----------|------|
| Usage Demo | `examples/compute/overlap/demo.ipynb` |
| Convert Demo | `examples/convert/openmx/demo.ipynb` |
| Test Cases | `tests/compute/overlap/openmx/test_basis_convert.py` |

---

**MPI Requirement**: HPRO requires MPI. Use:
```bash
source /path/to/intel/oneapi/setvars.sh
mpirun -np 1 dock compute overlap openmx calc ...
```

---

**Maintainer**: DeepH Team <deeph-pack@outlook.com>
