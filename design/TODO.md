# TODO - Planned Features

**Status Legend**: ✅ Complete | 🚧 In Progress | 🔜 Planned | ❓ Under Discussion

---

## High Priority

### 🔜 Improve User Documentation

- ✅ OpenMX overlap demo notebook
- ✅ OpenMX convert demo notebook
- 🔜 SIESTA overlap documentation
- 🔜 PySCF documentation improvements

### 🔜 Add More DFT Converters

- 🔜 VASP support (most requested)
- 🔜 CP2K support
- 🔜 Quantum ESPRESSO improvements

---

## Medium Priority

### 🔜 Performance Optimization

- 🔜 GPU acceleration for eigenvalue calculations
- 🔜 Streaming processing for large datasets
- 🔜 Memory-efficient matrix operations

### 🔜 Enhanced Analysis Tools

- 🔜 Automated error diagnosis
- 🔜 Learning curve analysis
- 🔜 Feature importance analysis

### 🔜 Workflow Automation

- 🔜 End-to-end pipeline (DFT → training → prediction)
- 🔜 Automated hyperparameter tuning
- 🔜 Integration with HPC schedulers

---

## Completed Features (2025)

### Core Infrastructure
- ✅ CLI auto-registration system
- ✅ Unified DeepH data format
- ✅ Multi-DFT converter support (SIESTA, OpenMX, FHI-aims, ABACUS, QE)
- ✅ Parallel processing support (ThreadPoolExecutor)

### Electronic Structure
- ✅ Band structure calculation
- ✅ DOS calculation (Gaussian + Tetrahedron)
- ✅ Fermi level finding
- ✅ Ill-conditioned eigenvalue handling

### Analysis Tools
- ✅ Multi-dimensional error analysis
- ✅ Dataset analysis tools
- ✅ Equivariance testing

### Basis & Overlap (2025-03-25)
- ✅ Standardized basis.h5 format with per-orbital cutoff
- ✅ Unified species_pbe.h5 (PAO+VPS in one file)
- ✅ Linear grid interpolation for Bessel transform (100% < 1e-4)
- ✅ CLI: `convert-species`, `calc-species`, `calc`

---

**Last Updated**: 2025-03-25
**Maintainer**: DeepH Team <deeph-pack@outlook.com>
