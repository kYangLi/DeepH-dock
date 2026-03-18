# Electronic Structure Computation Design

**Status**: ✅ Implemented  
**Version**: 0.9.11  
**Last Updated**: 2025-03-08

---

## Problem Statement

How to efficiently compute electronic structure properties (band structure, DOS, Fermi level) from DeepH Hamiltonian matrices?

---

## Design Goals

1. **High Performance** - Efficient matrix operations and parallelization
2. **Robustness** - Handle ill-conditioned matrices
3. **Accuracy** - Multiple numerical methods for different scenarios
4. **Flexibility** - Support various k-point paths and energy ranges

---

## Solution Overview

```
H(R) → Fourier Transform → H(k) → Diagonalization → Eigenvalues → Band Structure/DOS
```

---

## Core Architecture

### 1. Matrix Object Hierarchy

```python
class AOMatrixR:
    """Atomic orbital matrix in real space"""
    
    def __init__(self, Rs: np.ndarray, MRs: np.ndarray):
        self.Rs = Rs    # R-vectors
        self.MRs = MRs  # Matrices in real space
    
    def r2k(self, ks: np.ndarray) -> np.ndarray:
        """Fourier transform to reciprocal space"""
        phase = np.exp(2j * np.pi * np.matmul(ks, self.Rs.T))
        return np.matmul(phase, self.MRs.reshape(len(self.Rs), -1))
```

```python
class AOMatrixObj:
    """Matrix loaded from DeepH format"""
    
    def __init__(self, data_path: Path, matrix_file: str):
        # Load from HDF5
        self._load_matrices(data_path / matrix_file)
        self._load_structure(data_path / "POSCAR")
        self._load_info(data_path / "info.json")
    
    def Sk(self, k: np.ndarray) -> np.ndarray:
        """Get overlap matrix at k-point"""
        return self._overlap_r.r2k(k[np.newaxis, :])[0]
    
    def Hk(self, k: np.ndarray) -> np.ndarray:
        """Get Hamiltonian at k-point"""
        return self._hamiltonian_r.r2k(k[np.newaxis, :])[0]
```

### 2. Eigenvalue Calculation

```python
class HamiltonianObj(AOMatrixObj):
    """Hamiltonian with eigenvalue calculation"""
    
    def diag(self, ks: np.ndarray, **kwargs) -> np.ndarray:
        """Diagonalize Hamiltonian at multiple k-points"""
        eigenvalues = []
        
        for k in ks:
            Hk = self.Hk(k)
            Sk = self.Sk(k)
            
            # Solve generalized eigenvalue problem
            if self._use_sparse:
                eigs = sparse_eigsh(Hk, Sk, k=self._k_sparse)
            else:
                eigs = direct_diag(Hk, Sk)
            
            eigenvalues.append(eigs)
        
        return np.array(eigenvalues)
```

---

## Key Algorithms

### 1. Fourier Transform

Real space → Reciprocal space:

```
H(k) = Σ_R H(R) * exp(2πi * k·R)
```

```python
def r2k(Rs: np.ndarray, MRs: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Fourier transform"""
    phase = np.exp(2j * np.pi * np.dot(k, Rs.T))
    return np.sum(phase[:, None, None] * MRs, axis=0)
```

### 2. Generalized Eigenvalue Problem

Solve: `H(k) * ψ = E * S(k) * ψ`

```python
def direct_diag(Hk: np.ndarray, Sk: np.ndarray) -> np.ndarray:
    """Direct diagonalization"""
    # Cholesky decomposition: S = L * L†
    L = np.linalg.cholesky(Sk)
    L_inv = np.linalg.inv(L)
    
    # Transform: L⁻¹ * H * L⁻† * φ = E * φ
    H_transformed = L_inv @ Hk @ L_inv.T.conj()
    
    # Standard eigenvalue problem
    eigenvalues = np.linalg.eigvalsh(H_transformed)
    
    return np.sort(eigenvalues)
```

### 3. Sparse Diagonalization

For large matrices, use iterative methods:

```python
from scipy.sparse.linalg import eigsh

def sparse_eigsh(Hk: np.ndarray, Sk: np.ndarray, k: int = 10) -> np.ndarray:
    """Sparse diagonalization for large systems"""
    # Solve for k smallest eigenvalues
    eigenvalues, _ = eigsh(Hk, k=k, M=Sk, which='SA')
    return np.sort(eigenvalues)
```

---

## Ill-Conditioned Eigenvalue Handling

### Problem

Overlap matrix `S(k)` may be nearly singular, causing numerical instability.

### Solutions

#### 1. Window Regularization

```python
class WindowRegularization:
    """Remove states outside energy window"""
    
    def __init__(self, window: tuple[float, float]):
        self.emin, self.emax = window
    
    def regularize(self, Hk: np.ndarray, Sk: np.ndarray) -> tuple:
        # Diagonalize S to find null space
        S_eigvals, S_eigvecs = np.linalg.eigh(Sk)
        
        # Keep only states above threshold
        mask = S_eigvals > 1e-10
        P = S_eigvecs[:, mask]
        
        # Project H and S
        H_new = P.T @ Hk @ P
        S_new = P.T @ Sk @ P
        
        return H_new, S_new
```

#### 2. Orbital Removal

```python
class OrbitalRemoval:
    """Remove problematic orbitals"""
    
    def __init__(self, orbital_indices: list[int]):
        self.orbital_indices = orbital_indices
    
    def remove_orbitals(self, Hk: np.ndarray, Sk: np.ndarray) -> tuple:
        # Create mask for orbitals to keep
        mask = np.ones(Hk.shape[0], dtype=bool)
        mask[self.orbital_indices] = False
        
        # Remove orbitals
        H_new = Hk[mask][:, mask]
        S_new = Sk[mask][:, mask]
        
        return H_new, S_new
```

---

## Band Structure Calculation

```python
def calc_band(
    hamiltonian: HamiltonianObj,
    k_path: np.ndarray,
    **kwargs
) -> np.ndarray:
    """Calculate band structure along k-path"""
    
    # Diagonalize at each k-point
    eigenvalues = hamiltonian.diag(k_path, **kwargs)
    
    return eigenvalues
```

**k-path format**:
```python
# High-symmetry k-points
k_points = {
    'Γ': [0.0, 0.0, 0.0],
    'X': [0.5, 0.0, 0.0],
    'M': [0.5, 0.5, 0.0],
}

# Generate path
k_path = generate_k_path(k_points, n_points=100)
```

---

## DOS Calculation

### 1. Gaussian Smearing

```python
def calc_dos_gaussian(
    eigenvalues: np.ndarray,
    energy_grid: np.ndarray,
    sigma: float = 0.1,
) -> np.ndarray:
    """Calculate DOS with Gaussian smearing"""
    
    dos = np.zeros_like(energy_grid)
    
    for eig in eigenvalues.flatten():
        dos += gaussian(energy_grid, mu=eig, sigma=sigma)
    
    return dos / (np.sqrt(2 * np.pi) * sigma)
```

### 2. Tetrahedron Method

```python
def calc_dos_tetrahedron(
    eigenvalues: np.ndarray,
    k_weights: np.ndarray,
    energy_grid: np.ndarray,
) -> np.ndarray:
    """Calculate DOS with tetrahedron method (more accurate)"""
    from libtetrabz import dos
    
    # Use libtetrabz library
    dos_values = dos(eigenvalues, k_weights, energy_grid)
    
    return dos_values
```

---

## Fermi Level Calculation

### 1. Counting Method

```python
def find_fermi_counting(
    eigenvalues: np.ndarray,
    n_electrons: int,
    temperature: float = 0.0,
) -> float:
    """Find Fermi level by electron counting"""
    
    # Sort eigenvalues
    e_sorted = np.sort(eigenvalues.flatten())
    
    if temperature == 0.0:
        # T=0: Fermi level is at n_electrons-th eigenvalue
        return e_sorted[n_electrons - 1]
    else:
        # Finite T: Use Fermi-Dirac distribution
        return find_fermi_fd(e_sorted, n_electrons, temperature)
```

### 2. Tetrahedron Method

```python
def find_fermi_tetrahedron(
    eigenvalues: np.ndarray,
    k_weights: np.ndarray,
    n_electrons: int,
) -> float:
    """Find Fermi level with tetrahedron method"""
    from libtetrabz import fermi_int
    
    # Integration to find Fermi level
    E_fermi = fermi_int(eigenvalues, k_weights, n_electrons)
    
    return E_fermi
```

---

## Performance Optimization

### 1. K-point Parallelization

```python
from deepx_dock.parallel import parallel_map

def diag_parallel(hamiltonian, ks, n_jobs=-1):
    """Parallel diagonalization over k-points"""
    
    def diag_single(k):
        return hamiltonian.diag(k[np.newaxis, :])[0]
    
    eigenvalues = parallel_map(
        diag_single,
        ks,
        n_jobs=n_jobs,
        desc="Diagonalizing"
    )
    
    return np.array(eigenvalues)
```

### 2. Sparse Matrix Storage

```python
# Store only non-zero H(R) terms
from scipy.sparse import csr_matrix

H_sparse = csr_matrix(H_dense)
```

### 3. JIT Compilation

```python
import numba

@numba.jit(nopython=True)
def r2k_fast(Rs, MRs, k):
    """JIT-compiled Fourier transform"""
    phase = np.exp(2j * np.pi * np.dot(k, Rs.T))
    return np.sum(phase[:, None, None] * MRs, axis=0)
```

---

## Usage Examples

### Band Structure

```bash
# Calculate band structure
dock compute eigen calc-band data_dir/ -k K_PATH -o band.h5

# Plot band structure
dock compute eigen plot-band band.h5 -o band.png
```

### DOS

```bash
# Calculate DOS
dock compute eigen calc-dos data_dir/ -e -10,10 -n 1000 -o dos.h5
```

### Fermi Level

```bash
# Find Fermi level
dock compute eigen find-fermi data_dir/ -n 18
```

---

**Last Updated**: 2025-03-08
