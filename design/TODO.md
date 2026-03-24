# TODO - Planned Features

**Status Legend**: ✅ Complete | 🚧 In Progress | 🔜 Planned | ❓ Under Discussion

---

## High Priority

### 🔜 Improve User Documentation

**Description**: Complete user-facing documentation for all modules.

**Progress**:
- ✅ OpenMX overlap demo notebook
- ✅ OpenMX convert demo notebook
- 🔜 SIESTA overlap documentation
- 🔜 PySCF documentation improvements

---

### 🔜 Add More DFT Converters

**Description**: Add support for additional DFT codes.

**Planned Converters**:
- 🔜 VASP support (most requested)
- 🔜 CP2K support
- 🔜 Quantum ESPRESSO improvements

---

## Medium Priority

### 🔜 Performance Optimization

**Description**: Improve performance for large-scale calculations.

**Planned Features**:
- 🔜 GPU acceleration for eigenvalue calculations
- 🔜 Streaming processing for large datasets
- 🔜 Memory-efficient matrix operations
- 🔜 Caching layer for intermediate results

---

### 🔜 Enhanced Analysis Tools

**Description**: Add more analysis capabilities.

**Planned Features**:
- 🔜 Automated error diagnosis
- 🔜 Learning curve analysis
- 🔜 Feature importance analysis
- 🔜 Interactive visualization

---

### 🔜 Workflow Automation

**Description**: Automate common workflows.

**Planned Features**:
- 🔜 End-to-end pipeline (DFT → training → prediction)
- 🔜 Automated hyperparameter tuning
- 🔜 Batch job management
- 🔜 Integration with HPC schedulers

---

## Future Considerations

### ❓ Plugin System

**Description**: Allow external packages to extend DeepH-dock.

### ❓ Web Interface

**Description**: Web-based interface for common tasks.

### ❓ Machine Learning Integration

**Description**: Direct integration with DeepH-pack training pipeline.

---

## Completed Features (2025)

- ✅ CLI auto-registration system
- ✅ Unified DeepH data format
- ✅ Multi-DFT converter support (SIESTA, OpenMX, FHI-aims, ABACUS, QE)
- ✅ Band structure calculation
- ✅ DOS calculation (Gaussian + Tetrahedron)
- ✅ Fermi level finding
- ✅ Ill-conditioned eigenvalue handling
- ✅ Multi-dimensional error analysis
- ✅ Dataset analysis tools
- ✅ Equivariance testing
- ✅ Parallel processing support (ThreadPoolExecutor)
- ✅ Design documentation
- ✅ **Basis standardization & OpenMX overlap calculation** (2025-03-24)
  - Standardized basis.h5 format (flat, AI-friendly)
  - PAO → HDF5 batch conversion CLI
  - OpenMX input parsing
  - HPRO integration for overlap calculation
  - Verification: 131 pairs, max error 2.5e-4

---

**Last Updated**: 2025-03-24
**Maintainer**: DeepH Team <deeph-pack@outlook.com>
