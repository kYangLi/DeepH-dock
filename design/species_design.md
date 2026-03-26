# Species File Design

**Status**: Production Ready  
**Version**: 2.0.0  
**Last Updated**: 2025-03-26

---

## 1. Overview

Unified species file format for DeepH-dock, enabling:

1. Reusable basis/pseudopotential data across calculations
2. Support for multiple DFT software (OpenMX, SIESTA, FHI-aims)
3. Easy integration with HPRO's GridFunc system
4. JAX/tensor-friendly storage format
5. Data provenance for validation

### File Naming Convention

```
species_{source}_{xc}.h5

Examples:
  species_openmx_pbe.h5    # OpenMX + PBE
  species_siesta_pbe.h5    # SIESTA + PBE
  species_fhiaims_pbe.h5   # FHI-aims + PBE
```

### Architecture

```
Step 1: Species File Preparation
┌─────────────────────────────────────────────────────────────┐
│ species_{source}_{xc}.h5                                    │
│                                                             │
│ OpenMX:   PAO/ + VPS/ → species_convert.py                  │
│ SIESTA:   *.ion     → siesta_parser.py (TODO)               │
│ FHI-aims: species/ → aims_parser.py (TODO)                  │
└─────────────────────────────────────────────────────────────┘

Step 2: Overlap Computation
┌─────────────────────────────────────────────────────────────┐
│ dock compute overlap {source} calc                          │
│   input.dat + species_{source}_{xc}.h5 → HPRO → overlap.h5  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. nljz_list Format

### Schema

```python
nljz_list = [[n, l, j, z], ...]  # shape: [M, 4]
```

| Column | Name | Description | Type | Missing Value |
|--------|------|-------------|------|---------------|
| 0 | n | Principal quantum number or index | int | 0 |
| 1 | l | Angular momentum | int | Required |
| 2 | j | Total angular momentum (stores 2j) | int | 0 |
| 3 | z | Zeta/multiplicity index | int | 1 |

### Core Principles

1. **Never Reorder** - Preserve original file parsing order from DFT software
2. **Maximum Commonality** - Use dummy values for missing fields

### j Column

- Stores `2j` as integer: j=1/2 → 1, j=3/2 → 3, j=5/2 → 5, j=7/2 → 7
- No SOC: fill with 0
- j-split expanded to multiple rows

### j-split Expansion Rules

| l value | j values | Rows |
|---------|----------|------|
| s (l=0) | j=1/2 | **1 row** |
| p (l=1) | j=1/2, 3/2 | 2 rows |
| d (l=2) | j=3/2, 5/2 | 2 rows |
| f (l=3) | j=5/2, 7/2 | 2 rows |

**Note**: s orbital has only one j value, expand to 1 row to avoid redundancy.

### Why No Magnetic Quantum Number m?

We store **radial functions R_nl(r)**, not angular parts.

```
Radial function R_nl(r) ────── stored in radius_data
         ×
Spherical harmonic Y_lm(θ,φ) ── not stored, generated on-the-fly
         =
Orbital wavefunction ψ_nlm(r,θ,φ)
```

Matrix dimension = Σ(2l+1) for each orbital.

---

## 3. Software Adaptation

### 3.1 OpenMX

#### Basis (PAO)

```
Mo7.0-s3p2d2 nljz_list:
[
  [0, 0, 0, 1],  # s-zeta1
  [0, 0, 0, 2],  # s-zeta2
  [0, 0, 0, 3],  # s-zeta3
  [0, 1, 0, 1],  # p-zeta1
  [0, 1, 0, 2],  # p-zeta2
  [0, 2, 0, 1],  # d-zeta1
  [0, 2, 0, 2],  # d-zeta2
]
```

- n: 0 (no principal quantum number concept)
- l: derived from mul_list
- j: 0 (no SOC)
- z: zeta index within l channel (starting from 1)

#### Nonlocal (VPS)

```
Mo_PBE19.vps <pseudo.NandL>:
index  n   l   cutoff
  0    4   0   1.10    # 4s
  1    4   1   1.20    # 4p (local)
  2    4   2   1.60    # 4d
  3    4   3   1.50    # 4f
  4    5   0   2.30    # 5s
  5    5   1   2.80    # 5p

nljz_list (after removing local, j-split expanded):
[
  [4, 0, 1, 1],  # 4s, j=1/2
  [4, 2, 3, 1],  # 4d, j=3/2
  [4, 2, 5, 1],  # 4d, j=5/2
  [4, 3, 5, 1],  # 4f, j=5/2
  [4, 3, 7, 1],  # 4f, j=7/2
  [5, 0, 1, 1],  # 5s, j=1/2
  [5, 1, 1, 1],  # 5p, j=1/2
  [5, 1, 3, 1],  # 5p, j=3/2
]
```

- n: principal quantum number (actual value)
- l: angular momentum
- j: 2j integer, ascending order (lower j first)
- z: 1 (only one per n,l,j combination)

**Key points**:
- `local_part_vps` specifies which index is local potential (not stored in nonlocal)
- Original OpenMX data: [j_plus, j_minus] = [l+1/2, l-1/2] (higher j first)
- We reorder to ascending j: [l-1/2, l+1/2]

### 3.2 SIESTA (TODO)

#### Basis (PAO)

```
Si.ion PAO format:
# orbital l, n, z, is_polarized, population
  0  3  1  0  2.0   # 3s-zeta1
  0  3  2  0  0.0   # 3s-zeta2
  1  3  1  1  2.0   # 3p-zeta1 (polarized)
  ...

nljz_list:
[
  [3, 0, 0, 1],  # 3s-zeta1
  [3, 0, 0, 2],  # 3s-zeta2
  [3, 1, 0, 1],  # 3p-zeta1
  ...
]
```

- n: principal quantum number (actual value)
- l: angular momentum
- j: 0 (no SOC)
- z: zeta index

#### Nonlocal (KB Projector)

```
L=0  Nkbl=1  # s channel, 1 projector
L=1  Nkbl=1  # p channel, 1 projector
L=2  Nkbl=1  # d channel, 1 projector

nljz_list:
[
  [1, 0, 0, 1],  # s projector (n=1 is index, not principal)
  [1, 1, 0, 1],  # p projector
  [1, 2, 0, 1],  # d projector
]
```

- n: sequence number (not principal quantum number)
- l: angular momentum
- j: 0 (no SOC)
- z: 1

### 3.3 FHI-aims (TODO)

#### Basis (NAO)

```
# valence basis states
valence  2  s   2.   # 2s, 2 electrons
valence  2  p   2.   # 2p, 2 electrons

# additional basis functions
hydro 2 p 1.7
hydro 3 d 6
hydro 2 s 4.9

nljz_list:
[
  [2, 0, 0, 1],  # 2s (valence)
  [2, 1, 0, 1],  # 2p (valence)
  [2, 1, 0, 2],  # 2p (hydro, 2nd p)
  [3, 2, 0, 1],  # 3d (hydro)
  [2, 0, 0, 2],  # 2s (hydro, 2nd s)
  ...
]
```

- n: principal quantum number
- l: angular momentum
- j: 0 (no SOC)
- z: index within same (n,l)

---

## 4. Storage Format

### 4.1 species_{source}_{xc}.h5 Structure

```
species_openmx_pbe.h5
├── @xc_functional = "PBE"
├── @global_nmax = 2000
├── @source = "openmx"
├── @units_length = "bohr"
│
├── /{species_name}            # e.g., /Mo7.0, /Fe5.5H
│   ├── @element = "Mo"
│   ├── @species_name = "Mo7.0"
│   ├── @valence_electrons = 14.0
│   ├── @basis_source = "Mo7.0.pao"
│   ├── @pseudo_source = "Mo_PBE19.vps"
│   ├── @xc_functional = "GGA"
│   │
│   ├── /basis
│   │   ├── nljz_list [M, 4]
│   │   ├── cutoff_radii [M]
│   │   ├── grid_length [M]
│   │   ├── radius_grid [M, N_max]
│   │   └── radius_data [M, N_max]
│   │
│   ├── /val_density
│   │   ├── nljz_list [1, 4]   # [[0, 0, 0, 1]]
│   │   ├── cutoff_radii [1]
│   │   ├── grid_length [1]
│   │   ├── radius_grid [1, N_max]
│   │   └── radius_data [1, N_max]
│   │
│   └── /pseudopotential
│       ├── /local
│       │   ├── nljz_list [1, 4]   # [[n, l, 0, 1]]
│       │   ├── cutoff_radii [1]
│       │   ├── grid_length [1]
│       │   ├── radius_grid [1, N_max]
│       │   └── radius_data [1, N_max]
│       │
│       ├── /nonlocal
│       │   ├── nljz_list [M_nlj, 4]  # j-split expanded
│       │   ├── cutoff_radii [M_nlj]
│       │   ├── grid_length [M_nlj]
│       │   ├── radius_grid [M_nlj, N_max]
│       │   └── radius_data [M_nlj, N_max]
│       │
│       └── /core_density      # If NLCC
│           ├── nljz_list [1, 4]
│           ├── cutoff_radii [1]
│           ├── grid_length [1]
│           ├── radius_grid [1, N_max]
│           └── radius_data [1, N_max]
```

### 4.2 Backward Compatibility

Old format fields (deprecated, but supported for reading):

| Old Field | New Field | Derivation |
|-----------|-----------|------------|
| `mul_list` | `nljz_list` | Count by l column |
| `l_list` | `nljz_list[:, 1]` | Direct extraction |

---

## 5. n Semantic Summary

| Software | Basis n | Nonlocal n |
|----------|---------|------------|
| OpenMX | 0 (no concept) | Principal quantum number |
| SIESTA | Principal quantum number | Sequence number |
| FHI-aims | Principal quantum number | N/A |

**Decision**: Keep different semantics per software, document clearly.

---

## 6. CLI Usage

### OpenMX

```bash
# Generate species file
dock convert openmx convert-species \
    /path/to/DFT_DATA19/PAO \
    /path/to/DFT_DATA19/VPS \
    species_openmx_pbe.h5

# Single file mode (default)
dock compute overlap openmx openmx_in.dat species_openmx_pbe.h5

# Auto-generate species file during calculation
dock compute overlap openmx openmx_in.dat species_openmx_pbe.h5 \
    --raw-species-dir /path/to/DFT_DATA19

# Batch mode (with --tier-num)
dock compute overlap openmx ./data species_openmx_pbe.h5 -t 0

# Batch mode with parallel jobs
dock compute overlap openmx ./data species_openmx_pbe.h5 -t 0 -j 4

# Process directory itself (tier=-1)
dock compute overlap openmx ./data species_openmx_pbe.h5 -t -1
```

### Mode Selection

| --tier-num | Mode | PATH meaning |
|------------|------|--------------|
| not provided | Single file | openmx_in.dat file path |
| -1 | Batch | Directory, process itself |
| 0 | Batch | Directory, process subdirectories |
| 1+ | Batch | Directory, process deeper tiers |

### Parameter Defaults

| OpenMX Parameter | Default | Our Parameter | Formula |
|------------------|---------|---------------|---------|
| `1DFFT.EnergyCutoff` | 3600 Ry | `ecut = 1800 Ha` | `kmax = sqrt(2*ecut)` |
| `1DFFT.NumGridK` | 900 | `kdense = 15` | `grid_nq = kmax * kdense` |
| `1DFFT.NumGridR` | 900 | `rdense = 100` | `grid_nr = rcut * rdense` |

---

## 7. Module Structure

```
deepx_dock/
├── convert/openmx/
│   ├── pao_parser.py      # Parse PAO files, generate nljz_list
│   ├── vps_parser.py      # Parse VPS files, expand j-split
│   ├── species_convert.py # PAO+VPS → species_openmx_pbe.h5
│   └── species_loader.py  # Load species file (backward compatible)
│
└── compute/overlap/
    ├── _cli.py                # CLI interface (unified openmx command)
    ├── batch_calculator.py    # Batch processing logic
    ├── overlap.py             # Core overlap calculation
    └── openmx/
        ├── parse_input.py     # Parse OpenMX input (spin, SOC)
        ├── loader.py          # AOData_from_species class
        └── calculator.py      # OpenMXOverlapCalculator
```

---

## 8. Validation Results

| Test Case | Atoms | Pairs | Max Error | Status |
|-----------|-------|-------|-----------|--------|
| MoTe2 | 3 | 131 | 3.89e-05 | ✓ |
| Bi2Se3_SOC | 5 | 273 | 2.21e-05 | ✓ |

---

## 9. Implementation Progress

### Completed
- [x] nljz_list schema design
- [x] OpenMX PAO parser with nljz_list
- [x] OpenMX VPS parser with j-split expansion
- [x] species_convert.py with new format
- [x] species_loader.py with backward compatibility
- [x] Overlap calculation validation
- [x] Rename to species_{source}_{xc}.h5 convention

### TODO
- [ ] SIESTA parser (siesta_parser.py)
- [ ] FHI-aims parser (aims_parser.py)

---

## 10. Bug Fix History

| Problem | Cause | Solution |
|---------|-------|----------|
| High-k noise | Logarithmic grid sparse at large r | Interpolate to linear grid |
| PAO parse error | Dividing r*phi by r | Store r*phi directly |
| Atom order mismatch | HPRO sort_atoms bug | Custom build_atom_reorder_mapping |
| spinful incorrect | Not parsed from input | Parse SOC/SpinPolarization |
| overlap.h5/POSCAR mismatch | HPRO save_mat_deeph bug | Custom save_overlap_deeph |
| VPS mismatch silent | No provenance stored | Add @pseudo_source attribute |
| Missing nonlocal data | Only local potential parsed | Parse j-split projectors |

---

**Maintainer**: DeepH Team <deeph-pack@outlook.com>
