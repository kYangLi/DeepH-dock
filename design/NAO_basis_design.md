# Unified NAO Basis Storage Architecture

**Status**: Production Ready  
**Version**: 0.9.12  
**Last Updated**: 2025-03-25

---

## 1. Overview

Standardize basis set formats for DeepH-dock, enabling:
1. Reusable basis sets across calculations
2. Easy integration with HPRO's GridFunc system
3. Support for heterogeneous cutoff radii (SIESTA, FHI-aims)
4. JAX/tensor-friendly storage format

### Architecture

```
Step 1: Basis Conversion
OpenMX PAO (.pao) → basis_convert.py → basis.h5
SIESTA .ion → basis_writer.py → basis.h5 (future)

Step 2: Overlap Computation
OpenMX input (.dat) + basis.h5 → HPRO → overlap.h5, POSCAR, info.json
```

---

## 2. Core Principles

1. **100% Physical Fidelity**: Preserve original grid coordinates exactly
2. **Static Tensor Alignment**: Fixed-shape `[M, N_max]` arrays, compatible with JAX/XLA
3. **Numerical Stability**: No `inf` or `NaN` values, smooth padding for neural network embeddings

---

## 3. `basis.h5` Format (v0.9.13)

```
basis.h5
├── @element: str              # e.g., "Fe"
├── @basis_name: str           # e.g., "Fe6.0H"
├── @source: str               # e.g., "openmx", "siesta"
├── @normalized: bool
├── @units_length: str         # "bohr"
│
├── mul_list: [lmax+1] int     # orbitals per angular momentum
├── cutoff_radii: [M] float    # per-orbital cutoff radius (Bohr)
├── grid_length: [M] int       # effective grid points per orbital
├── radius_grid: [M, N_max]    # 2D grid matrix
└── radius_basis: [M, N_max]   # radial functions
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Flat structure | AI/tensor-friendly, no nested groups |
| Per-orbital cutoff | Supports heterogeneous cutoff (SIESTA) |
| Zero-padding | Fixed-shape arrays for JAX compatibility |
| Bohr units | Native to OpenMX/HPRO, avoids conversion errors |

### Inferred Values

```python
lmax = len(mul_list) - 1
total_orbitals = sum(mul_list)
max_cutoff = cutoff_radii.max()
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

### DFT Software Compatibility

| Software | Grid Type | Cutoff Behavior |
|----------|-----------|-----------------|
| OpenMX | Shared logarithmic | Uniform |
| SIESTA | Independent | Heterogeneous per orbital |
| FHI-aims | Independent | Heterogeneous (planned) |

---

## 4. Field Descriptions

### 4.1 `mul_list` (1D Integer Array)
- **Shape**: `[lmax+1]`
- **Description**: Number of orbitals per angular momentum channel (s, p, d, ...)

### 4.2 `cutoff_radii` (1D Float Array)
- **Shape**: `[M]`
- **Description**: Per-orbital physical cutoff radius in Bohr. Used for graph edge pruning.

### 4.3 `grid_length` (1D Integer Array)
- **Shape**: `[M]`
- **Description**: Number of valid grid points per orbital. The **only** basis for generating accurate masks.

### 4.4 `radius_grid` (2D Float Array)
- **Shape**: `[M, N_max]`
- **Valid Region** (`j < grid_length[i]`): Original grid coordinates
- **Padding Region** (`j >= grid_length[i]`): Linear extrapolation

**Padding Rule**:
```python
r_max = grid[ni - 1]
dr = r_max / ni
for j in range(ni, n_max):
    radius_grid[i, j] = r_max + dr * (j - ni + 1)
```

### 4.5 `radius_basis` (2D Float Array)
- **Shape**: `[M, N_max]`
- **Valid Region**: Original radial function values
- **Padding Region**: `0.0`

---

## 5. JAX Consumption Pattern

```python
import jax.numpy as jnp

def process_basis(radius_grid, radius_basis, grid_length, N_max):
    """Generate mask using integer indexing."""
    mask_matrix = jnp.arange(N_max)[None, :] < grid_length[:, None]
    features_masked = radius_basis * mask_matrix
    r_emb = RBF_expansion(radius_grid)  # Safe due to smooth padding
    return features_masked * r_emb
```

---

## 6. CLI Usage

### Basis Conversion

```bash
# Batch convert PAO files
dock convert openmx convert-basis pao_folder/ basis/ --pattern "*7.0.pao"

# Single file
dock convert openmx convert-single Fe6.0H.pao basis/Fe6.0H.h5
```

### Overlap Calculation

```bash
# With auto-conversion (recommended)
dock compute overlap openmx calc openmx.in basis/ --raw-basis-dir pao/

# Using existing basis.h5
dock compute overlap openmx calc openmx.in basis/

# Custom parameters
dock compute overlap openmx calc openmx.in basis/ --ecut 50 --kdense 91.6
```

### Orbital Selection Syntax

| Syntax | Meaning |
|--------|---------|
| `Mo7.0-s3p2d2` | 3 s + 2 p + 2 d orbitals |
| `Fe6.0H` | Use all orbitals |

---

## 7. OpenMX Integration Details

### 7.1 Grid Type: Linear vs Logarithmic (Critical Fix)

**Problem with Logarithmic Grid (HPRO Original):**

HPRO's spherical Bessel transform used logarithmic grids:
- Log grids are dense near nucleus, sparse at large r
- At high k, `j_l(kr)` oscillates rapidly
- Sparse sampling at large r → numerical noise
- **Result**: Max error 1.48 for ecut=1800

**Solution: Uniform Linear Grid (OpenMX Style):**

OpenMX interpolates PAO data to uniform linear grid before Bessel transform:
- Consistent sampling density across all r values
- **Result**: 100% entries < 1e-4, max error 5.07e-05

| Grid Type | Ecut (Ha) | Max Error | % < 1e-4 |
|-----------|-----------|-----------|----------|
| Logarithmic | 50 | 2.55e-04 | 97.7% |
| Logarithmic | 1800 | 1.48e+00 | 95.6% |
| **Linear** | **50** | **5.07e-05** | **100%** |
| Linear | 1800 | 3.89e-05 | 100% |

**Implementation:**
```python
def _interpolate_to_linear_grid(r_orig, func_orig, rcut, ell, rdense):
    from scipy.interpolate import CubicSpline
    
    npoints = max(int(rcut * rdense), 10)
    r_linear = np.linspace(0, rcut, npoints)
    
    spline = CubicSpline(r_orig, func_orig)
    func_linear = spline(r_linear)
    
    if ell == 0:
        func_linear[r_linear < r_orig[0]] = func_orig[0]
    else:
        func_linear[r_linear < r_orig[0]] = 0.0
    
    func_linear[r_linear > rcut] = 0.0
    
    return GridFunc(LinearRGD(0, rcut, npoints), func_linear, l=ell, rcut=rcut)
```

### 7.2 Parameter Design

| OpenMX Parameter | Default | Our Parameter | Formula |
|------------------|---------|---------------|---------|
| `1DFFT.EnergyCutoff` | 3600 Ry | `Ecut = 1800 Ha` | `kmax = sqrt(2*Ecut)` |
| `1DFFT.NumGridK` | 900 | `kdense = 15` | `grid_nq = kmax * kdense` |
| `1DFFT.NumGridR` | 900 | `rdense = 100` | `grid_nr = rcut * rdense` |

### 7.3 Validation Results

| System | Atoms | Max Error | Entries < 1e-4 | Notes |
|--------|-------|-----------|----------------|-------|
| MoTe2 | 3 | 3.89e-05 | **100%** | All 131 pairs matched |

**Key Discoveries:**
1. OpenMX PAO files store `R(r)` directly with normalization `∫ R(r)² r² dr = 1.0`
2. Linear grid is critical for accuracy
3. Atoms are reordered by species (continuous grouping) for DeepH compatibility

---

## 8. Bug Fix History

| Problem | Cause | Solution |
|---------|-------|----------|
| High-k noise | Logarithmic grid sparse at large r | Interpolate to linear grid |
| Grid type mismatch | LinearRGD for log grid | Use appropriate grid type |
| Extra orbitals loaded | Default value for unselected L | Skip unselected L |
| Unit mismatch | Angstrom passed to HPRO | Convert to Bohr |
| FRAC coords not handled | Missing FRAC case | `positions @ lattice` |
| Atom order mismatch | DeepH requires species grouping | Auto-reorder atoms |

---

## 9. Performance Notes

**Current Implementation:**
- Algorithm: Direct spherical Bessel transform O(Nk × Nr)
- Performance: Seconds for typical systems (not a bottleneck)

**Future Optimization (TODO):**
Fast Bessel transform (Talman 1978) if needed for large datasets.

---

## 10. References

| Resource | Path |
|----------|------|
| Usage Demo | `examples/compute/overlap/demo.ipynb` |
| Convert Demo | `examples/convert/openmx/demo.ipynb` |
| Test Cases | `tests/compute/overlap/openmx/test_basis_convert.py` |

---

**Maintainer**: DeepH Team <deeph-pack@outlook.com>
