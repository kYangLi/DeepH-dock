# DFT Format Converters Design

**Status**: ✅ Implemented  
**Version**: 0.9.11  
**Last Updated**: 2025-03-08

---

## Problem Statement

Different DFT software packages use different output formats and conventions. How to provide a unified interface to convert all these formats into a standardized DeepH format?

---

## Design Goals

1. **Unified Output** - All converters produce the same DeepH standard format
2. **Extensibility** - Easy to add new DFT converters
3. **Robustness** - Handle different basis sets, spin-orbit coupling, etc.
4. **Performance** - Support parallel processing for large datasets

---

## Solution Overview

Each DFT converter follows a common pipeline:

```
DFT Output → Parser → Intermediate Format → DeepH Format Writer
```

---

## Common Conversion Pipeline

### 1. Directory Structure Handling

```python
def _find_data_dirs(root_dir: Path, depth: int) -> list[Path]:
    """Find data directories at specified depth"""
    if depth < 0:
        return [root_dir]
    
    data_dirs = []
    for subdir in root_dir.iterdir():
        if subdir.is_dir():
            data_dirs.extend(_find_data_dirs(subdir, depth - 1))
    
    return data_dirs
```

**Usage**:
- `depth=-1`: Single structure in `input_dir/`
- `depth=0`: Multiple structures in `input_dir/*/`
- `depth=1`: Multiple structures in `input_dir/tier1/*/`

### 2. Structure Parsing

Extract crystal structure information:

```python
def parse_structure(dft_output: Path) -> dict:
    """Parse crystal structure from DFT output
    
    Returns:
        {
            "lattice": np.ndarray,        # (3, 3)
            "elements_unique": List[str],
            "elements_counts": List[int],
            "atomic_numbers": np.ndarray, # (N_atoms,)
            "cart_coords": np.ndarray,    # (N_atoms, 3)
            "frac_coords": np.ndarray,    # (N_atoms, 3)
        }
    """
    # Parser-specific implementation
    pass
```

### 3. Basis Set Information

Extract orbital information:

```python
def parse_basis_info(dft_output: Path) -> dict:
    """Parse basis set information
    
    Returns:
        {
            "elements_orbital_map": {
                "Element": [l1, l2, ...]  # Orbital angular momenta
            },
            "orbital_counts": {
                "Element": [n1, n2, ...]  # Number of orbitals per l
            }
        }
    """
    # Parser-specific implementation
    pass
```

**Example**:
```python
{
    "elements_orbital_map": {
        "C": [0, 0, 1, 1, 2],  # 2s, 2p, 3d
        "H": [0, 0, 1]         # 1s, 2p
    }
}
```

### 4. Matrix Extraction

Extract Hamiltonian and overlap matrices:

```python
def extract_matrices(dft_output: Path, spinful: bool) -> dict:
    """Extract matrices from DFT output
    
    Returns:
        {
            "atom_pairs": np.ndarray,      # (N_edges, 5): [Rx, Ry, Rz, i_atom, j_atom]
            "overlap": np.ndarray,         # Flattened overlap matrix
            "hamiltonian": np.ndarray,     # Flattened Hamiltonian matrix
            "chunk_shapes": np.ndarray,    # (N_edges, 2)
            "chunk_boundaries": np.ndarray,# (N_edges+1,)
        }
    """
    # Parser-specific implementation
    pass
```

### 5. Spin-Orbit Coupling Handling

For spinful systems, matrices expand:

```python
def expand_to_spinful(
    atom_pairs: np.ndarray,
    entries: np.ndarray,
    chunk_shapes: np.ndarray,
    chunk_boundaries: np.ndarray,
) -> tuple:
    """Expand non-spinful matrix to spinful
    
    Non-spinful block: [N_orb_i × N_orb_j]
    Spinful block: [[↑↑, ↑↓], [↓↑, ↓↓]] = [2*N_orb_i × 2*N_orb_j]
    """
    spinful_entries = []
    
    for i, (start, end) in enumerate(zip(chunk_boundaries[:-1], chunk_boundaries[1:])):
        block = entries[start:end].reshape(chunk_shapes[i])
        
        # Create spinful block
        zero_block = np.zeros_like(block)
        spinful_block = np.block([
            [block, zero_block],
            [zero_block, block]
        ])
        
        spinful_entries.extend(spinful_block.flatten())
    
    # Update chunk info (2x shape, 4x size)
    new_chunk_shapes = chunk_shapes * 2
    new_chunk_boundaries = np.cumsum([0] + [s[0]*s[1] for s in new_chunk_shapes])
    
    return atom_pairs, np.array(spinful_entries), new_chunk_shapes, new_chunk_boundaries
```

---

## Converter-Specific Details

### SIESTA

**Input Files**:
- `siesta.HSX` - Hamiltonian and overlap matrices
- `siesta.XV` - Atomic coordinates
- `siesta.EIG` - Eigenvalues (optional)
- `*.ORB_INDX` - Orbital indices

**Key Challenges**:
- Real spherical harmonics ordering differs from standard
- Apply transformation matrix for orbital ordering
- Handle numeric vs analytical basis sets

**Transformation Matrix**:
```python
SIESTA_BASIS_ORDER = {
    0: [0],                          # s
    1: [0, 1, 2],                    # px, py, pz
    2: [0, 1, 2, 3, 4],              # d orbitals
}

SIESTA_BASIS_PARITY = {
    0: [1],
    1: [-1, 1, -1],
    2: [1, -1, 1, -1, 1],
}

def transform_siesta_basis(matrix_block, l_values):
    """Transform SIESTA basis to standard ordering"""
    # Apply parity and reordering
    pass
```

### OpenMX

**Input Files**:
- `scfout` - Binary file with all matrices
- `*.dat` - Input files (for structure)

**Key Challenges**:
- Binary format parsing
- Complex basis set with multiple radial functions
- Spin-orbit coupling support

### FHI-aims

**Input Files**:
- `geometry.in` - Structure
- `KS_eigenvectors` - Eigenvectors (optional)
- Matrix files generated by modified FHI-aims

**Key Challenges**:
- Support both periodic and molecular systems
- Handle different basis set tiers
- Extract from modified output files

### ABACUS

**Input Files**:
- `OUT.suffix/` - Output directory
- `STRU` - Structure file
- `mdata` - Matrix data

**Key Challenges**:
- Chinese DFT code, documentation in Chinese
- Different file organization
- Custom matrix storage format

---

## DeepH Format Utilities

### Standardization

Remove chemical potential gauge freedom:

```python
def standardize_hamiltonian(hamiltonian_path: Path, overlap_path: Path):
    """Standardize Hamiltonian by removing chemical potential
    
    H_std = H - μ * S
    
    where μ is determined by trace condition.
    """
    # Load H and S
    # Compute μ = Tr(HS) / Tr(S)
    # H_new = H - μ * S
    pass
```

### Core Hamiltonian

Subtract atomic core contributions:

```python
def subtract_core_hamiltonian(
    hamiltonian_path: Path,
    core_h_path: Path,
):
    """Subtract atomic core Hamiltonian
    
    H_eff = H - H_core
    
    where H_core contains atomic contribution.
    """
    pass
```

### Format Versioning

Handle format upgrades/downgrades:

```python
def upgrade_dataset(old_dir: Path, new_dir: Path):
    """Upgrade old DeepH format to new version"""
    pass

def downgrade_dataset(new_dir: Path, old_dir: Path):
    """Downgrade new format to old version (for compatibility)"""
    pass
```

---

## Performance Considerations

### 1. Parallel Processing

```python
# Process multiple structures in parallel
from deepx_dock.parallel import parallel_map

results = parallel_map(
    convert_single,
    data_dirs,
    n_jobs=n_jobs,
    desc="Converting"
)
```

### 2. HDF5 Compression

```python
# Use compression for large matrices
with h5py.File(output_path, 'w') as f:
    f.create_dataset('entries', data=entries, compression='gzip', compression_opts=6)
```

### 3. Memory Efficiency

```python
# Process one structure at a time to save memory
for data_dir in data_dirs:
    convert_single(data_dir)
    # Matrices are freed after each iteration
```

---

## Testing Strategy

Each converter should have:

1. **Unit tests** - Test parser functions individually
2. **Integration tests** - Test full conversion pipeline
3. **Comparison tests** - Compare output against reference data

```bash
# Test structure
tests/convert/siesta/
├── run_test.sh           # Integration test script
├── siesta.bak/           # Reference input data (symlink)
└── deeph.bak/            # Reference output data (symlink)
```

---

## Extension Guide

To add a new DFT converter:

1. **Study the DFT output format**
   - File structure
   - Data organization
   - Basis set conventions

2. **Implement parser functions**
   - Structure parser
   - Basis info parser
   - Matrix parser

3. **Handle special cases**
   - Spin-orbit coupling
   - Non-standard basis ordering
   - Unit conversions

4. **Test thoroughly**
   - Multiple test cases
   - Edge cases
   - Performance tests

---

## Common Issues

### 1. Basis Set Mismatch

**Problem**: Different DFT codes use different orbital ordering.

**Solution**: Apply transformation matrices based on DFT-specific conventions.

### 2. Unit Conversion

**Problem**: Different units (Hartree, eV, Bohr, Angstrom).

**Solution**: Use constants from `CONSTANT.py`:

```python
from deepx_dock.CONSTANT import HARTREE_TO_EV, BOHR_TO_ANGSTROM

energy_eV = energy_hartree * HARTREE_TO_EV
distance_ang = distance_bohr * BOHR_TO_ANGSTROM
```

### 3. Periodic vs Molecular

**Problem**: Molecular systems have no k-points, periodic systems do.

**Solution**: Detect system type and handle accordingly:

```python
if is_molecular:
    # Only Gamma point
    k_points = np.array([[0.0, 0.0, 0.0]])
else:
    # Read k-points from DFT output
    k_points = parse_k_points(dft_output)
```

---

**Last Updated**: 2025-03-08
