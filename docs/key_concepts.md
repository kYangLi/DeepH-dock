# Key Concepts

This section describes the core data formats used by the DeepH project for electronic structure calculations and materials modeling. In the latest version of [DeepH-pack](https://github.com/kYangLi/DeepH-pack-docs), we have adopted a new folder layout that is more lightweight, user-friendly, and optimized for high I/O throughput.

## Overview

DeepH utilizes a standardized set of data formats to represent atomic structures, electronic properties, and force field information. These formats enable interoperability between different computational modules and ensure consistent data processing throughout the workflow.

### Folder Structure

```bash
dft
  ├── 0
  │   ├── POSCAR
  │   ├── info.json
  │   ├── overlap.h5
  │   ├── hamiltonian.h5     (optional)
  │   ├── density_matrix.h5  (optional)
  │   ├── potential_r.h5     (optional)
  │   ├── charge_density.h5  (optional)
  │   ├── force.h5           (optional)
  │   └── ...
  ├── 1
  └── ...
```

### File Descriptions

* The root directory for all DFT raw data is named `dft/`.
* Subfolders inside (e.g., `0`, `1`, or `structure_001`) can use free-form labels or numerical indices.

| File Type | Status | Format | Description |
| :--- | :--- | :--- | :--- |
| `POSCAR` | Required | Text | Atomic structure (VASP format) |
| `info.json` | Required | JSON | System metadata and basis set info |
| `overlap.h5` | Required | HDF5 | Overlap matrix (S) in sparse AO basis |
| `hamiltonian.h5` | Optional | HDF5 | Hamiltonian matrix (H) |
| `density_matrix.h5` | Optional | HDF5 | Density matrix |
| `potential_r.h5` | Optional | HDF5 | Real-space potential matrix |
| `charge_density.h5` | Optional | HDF5 | Charge density matrix |
| `force.h5` | Optional | HDF5 | Atomic forces |

## File Types and Their Purposes

### 1. POSCAR - Atomic Structure Information

This file follows the standard `POSCAR` format and contains the crystal structure information:

* Lattice vectors
* Atomic positions
* Element types

**Example:**

```text
H2O POSCAR File
1.0
    10.0   0.0   0.0
     0.0  10.0   0.0
     0.0   0.0  10.0
O  H
2  1
Direct
    0.0     0.0     0.0
    0.757   0.586   0.0
    0.243   0.586   0.0
```

### 2. info.json - Metadata and System Information

The `info.json` file stores metadata and system-specific parameters in JSON format.

**Example for a Hamiltonian task (water molecule):**

```json
{
    "atoms_quantity": 3,
    "orbits_quantity": 23,
    "orthogonal_basis": false,
    "spinful": false,
    "fermi_energy_eV": -2.29107782,
    "elements_orbital_map": {
        "O": [0, 0, 1, 1, 2],
        "H": [0, 0, 1]
    }
}
```

**Example for a force field task:**

```json
{
    "atoms_quantity": 21,
    "elements_force_rcut_map": {
        "O": 5.0,
        "H": 5.0
    },
    "max_num_neighbors": 500
}
```

### 3. HDF5 Files for Electronic Structure Properties

DeepH uses HDF5 files to store atom-pair-resolved electronic structure properties:

#### Common Files

* `overlap.h5` - Overlap matrices
* `hamiltonian.h5` - Hamiltonian matrices
* `density_matrix.h5` - Density matrices

#### Component Descriptions

Each HDF5 file contains the following keys:

| Key | Shape | Description |
| ----- | ------- | ------------- |
| `atom_pairs` | (N, 5) | Integer matrix where N is the number of edges. Each row contains 5 integers: `[R1, R2, R3, i_atom, j_atom]`, representing a coupling between the $i$-th atom in the central unit cell and the $j$-th atom in the periodic image cell specified by the lattice vector indices $(i, j, k)$.  |
| `chunk_boundaries` | (N+1,) | 1D integer array marking boundaries for each edge's data in the entries array |
| `chunk_shapes` | (N, 2) | Integer matrix where each row gives the shape of the submatrix for the corresponding edge |
| `entries` | (M,) | Flattened 1D array of floating-point values containing all matrix elements |

1. **`atom_pairs`**
   * Shape: `N_edge × 5` array
   * Stores edges/"hoppings" in format `[R1, R2, R3, i_atom, j_atom]`
   * `R1, R2, R3`: Relative lattice shift along three lattice vectors
   * `i_atom, j_atom`: Index of start/end atoms (0-indexed, matches `POSCAR` order)

2. **`entries`**
   * 1-D array containing all matrix elements for edges in `atom_pairs`
   * Blocks `A_{i,j,R}` are flattened and concatenated

3. **`chunk_boundaries`**
   * Shape: `(N_edge+1,)` array
   * Records split indexes of blocks in `entries`

4. **`chunk_shapes`**
   * Shape: `N_edge × 2` array
   * Records shapes of each block

#### Spin-Polarized Systems

For systems with `spinful=true`:

* `overlap.h5` remains unchanged
* `hamiltonian.h5` and `density_matrix.h5` expand to include spin
* `chunk_shapes` doubles in size
* `chunk_boundaries` becomes four times larger

Each block becomes a 4-part matrix:

$$
A_{i,j,R} = \begin{bmatrix}
A_{i,j,R,\uparrow,\uparrow} & A_{i,j,R,\uparrow,\downarrow} \\
A_{i,j,R,\downarrow,\uparrow} & A_{i,j,R,\downarrow,\downarrow}
\end{bmatrix}
$$

Each sub-block maintains the same size as in the non-spinful case.

**Important Note:** The `atom_pairs` array must be identical across all `*.h5` files within the same directory.

#### Code Example: Extracting Hamiltonian Matrix Elements

```python
import h5py

def extract_hamiltonian(filepath):
    """Extract Hamiltonian matrix elements from an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        atom_pairs = f['atom_pairs'][:]
        chunk_boundaries = f['chunk_boundaries'][:]
        chunk_shapes = f['chunk_shapes'][:]
        entries = f['entries'][:]
  
    H_tb = {}
    for i, ap in enumerate(atom_pairs):
        start = chunk_boundaries[i]
        end = chunk_boundaries[i+1]
        shape = chunk_shapes[i]
        H_tb[tuple(ap)] = entries[start:end].reshape(shape)
  
    return H_tb

# Usage
H_matrices = extract_hamiltonian('hamiltonian.h5')
```

### 4. Real-Space Grid-Resolved Properties

These HDF5 files store properties on a real-space grid:

* `charge_density.h5` - Electron charge density
* `potential_r.h5` - Local potential

Each HDF5 file contains the following keys:

| Key | Shape | Description |
| --- | ----- | ----------- |
| `shape` | (3,) | Integer array specifying grid divisions in x, y, z directions |
| `entries` | (M,) | Flattened 1D array that can be reshaped to `shape` |

#### Code Example: Reading Grid Data

```python
import numpy as np
import h5py

def read_grid_data(filepath):
    """Read and reshape real-space grid data."""
    with h5py.File(filepath, 'r') as f:
        shape = f['shape'][:]
        entries = f['entries'][:]
  
    return entries.reshape(shape)

# Usage
charge_density = read_grid_data('charge_density.h5')
```

### 5. Force Field Properties (force.h5)

The `force.h5` file contains atom-resolved force field information.

Each HDF5 file contains the following keys:

| Key | Shape | Description |
| --- | ----- | ----------- |
| `cell` | (3, 3) | Lattice vectors |
| `energy` | scalar | Total energy of the system |
| `force` | (N, 3) | Forces on N atoms in x, y, z directions |
| `stress` | (6,) | Stress tensor components in [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation) |

#### Code Example: Reading Force Data

```python
import h5py

def read_force_data(filepath):
    """Read force field data from force.h5."""
    with h5py.File(filepath, 'r') as f:
        cell = f['cell'][:] if 'cell' in f else None
        energy = f['energy'][()] if 'energy' in f else None
        force = f['force'][:]
        stress = f['stress'][:] if 'stress' in f else None
  
    return {
        'cell': cell,
        'energy': energy,
        'force': force,
        'stress': stress
    }

# Usage
force_data = read_force_data('force.h5')
```

## Data Flow in DeepH-dock

Understanding these formats is crucial for working with DeepH-dock:

1. **Input**: DFT software outputs are converted to these standardized formats
2. **Processing**: DeepH modules operate on the data using these consistent representations
3. **Output**: Results are stored in the same formats for interoperability

For more detailed specifications and updates to these formats, please refer to the latest documentation and the [`examples/`](https://github.com/kYangLi/DeepH-dock/tree/main/examples) directory in the repository.
