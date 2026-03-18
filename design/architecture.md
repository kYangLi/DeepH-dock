# System Architecture

**Status**: ✅ Implemented  
**Version**: 0.9.11  
**Last Updated**: 2025-03-08

---

## Problem Statement

How to design a modular, extensible system that bridges multiple DFT software packages with the DeepH deep learning workflow while maintaining high performance and ease of use?

---

## Design Goals

1. **Modularity** - Clear separation between different functional domains
2. **Extensibility** - Easy to add new DFT converters, computation functions, and analysis tools
3. **Automation** - Minimal boilerplate for CLI command registration
4. **Standardization** - Unified data format across all DFT codes
5. **Performance** - Multi-level parallel processing support

---

## Solution Overview

DeepH-dock uses a **four-layer modular architecture** with an **auto-registration CLI system**:

```
┌─────────────────────────────────────────────────────┐
│                 CLI Layer (dock)                     │
│         Auto-registration via decorators            │
└────────────────┬────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬────────────┐
    │            │            │            │
┌───▼───┐   ┌───▼───┐   ┌───▼───┐   ┌───▼───┐
│convert│   │compute│   │analyze│   │design │
└───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘
    │            │            │            │
┌───▼────────────▼────────────▼────────────▼───┐
│         Unified Data Format Layer              │
│   (POSCAR + info.json + HDF5 matrices)        │
└───────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. CLI Auto-Registration System

**Problem**: Managing 30+ CLI commands manually is error-prone and hard to maintain.

**Solution**: Decorator-based auto-registration with automatic discovery.

#### Implementation

```python
# deepx_dock/_cli/registry.py
class FunctionRegistry:
    def register(self, cli_name, cli_help, cli_args):
        def decorator(func):
            # Extract module path automatically
            auto_module = self._extract_module(func)
            # Store function info
            self._function_info[module_func_name] = {...}
            # Build module tree
            self._build_module_tree(auto_module, func_name)
            return func
        return decorator

# deepx_dock/_cli/__init__.py
def _auto_register_cli():
    """Discover and register all _cli.py files"""
    for cli_file in package_root.rglob("*_cli.py"):
        importlib.import_module(module_full_name)
```

**Usage Pattern**:

```python
# deepx_dock/convert/siesta/_cli.py
from deepx_dock._cli.registry import register

@register(
    cli_name="to-deeph",
    cli_help="Convert SIESTA output to DeepH format",
    cli_args=[
        click.argument('input_dir'),
        click.option('--jobs-num', default=-1),
    ],
)
def translate_siesta_to_deeph(input_dir, jobs_num):
    # Lazy import for performance
    from .translator import SIESTADatasetTranslator
    translator = SIESTADatasetTranslator(input_dir, jobs_num)
    translator.convert()
```

**Auto-registered as**: `dock convert siesta to-deeph`

**Key Benefits**:
- Zero boilerplate for command registration
- Automatic command hierarchy (module/submodule/command)
- Lazy imports for faster CLI startup
- Easy to add new commands

---

### 2. Unified Data Format

**Problem**: Each DFT code has different output formats, making data processing inconsistent.

**Solution**: Define a unified DeepH standard format.

#### Format Specification

```
<data_dir>/
├── POSCAR              # Crystal structure (required)
├── info.json           # Metadata (required)
├── overlap.h5          # Overlap matrix (required)
├── hamiltonian.h5      # Hamiltonian matrix (optional)
├── density_matrix.h5   # Density matrix (optional)
└── position_matrix.h5  # Position matrix (optional)
```

#### HDF5 Matrix Storage

```python
{
    'atom_pairs': np.ndarray,      # (N_edges, 5): [Rx, Ry, Rz, i_atom, j_atom]
    'chunk_shapes': np.ndarray,    # (N_edges, 2): shape of each block
    'chunk_boundaries': np.ndarray,# (N_edges+1,): data boundaries
    'entries': np.ndarray,         # Flattened matrix elements
}
```

**Design Decisions**:
- **Sparse block storage**: Only store non-zero atom pairs
- **Compression**: Use gzip for HDF5 files (10-20x compression)
- **Chunk organization**: Enables efficient partial loading
- **Spin handling**: Expand blocks for spinful systems (4x size)

**Benefits**:
- Consistent interface across all DFT codes
- Efficient storage and loading
- Easy to add new matrix types
- Compatible with DeepH-pack training pipeline

---

### 3. Modular Architecture

**Problem**: How to organize diverse functionalities (convert/compute/analyze/design)?

**Solution**: Four independent modules with clear responsibilities.

#### Module Responsibilities

```
deepx_dock/
├── convert/        # Data format conversion
│   ├── siesta/    # SIESTA → DeepH
│   ├── openmx/    # OpenMX → DeepH
│   ├── fhi_aims/  # FHI-aims → DeepH
│   ├── abacus/    # ABACUS → DeepH
│   └── deeph/     # DeepH format utilities
├── compute/        # Electronic structure calculations
│   ├── eigen/     # Band structure, DOS, Fermi level
│   └── overlap/   # Overlap matrix calculations
├── analyze/        # Data analysis
│   ├── dataset/   # Feature analysis, data splitting
│   ├── error/     # Error analysis
│   └── dft_equiv/ # Equivariance testing
└── design/         # Structure generation
    ├── twist_2d/  # 2D twisted heterostructures
    └── mattergen_chgnet/  # Structure search templates
```

**Design Principles**:
- **Independence**: Each module can be used standalone
- **Single Responsibility**: Each module has one clear purpose
- **Loose Coupling**: Modules communicate via unified data format
- **High Cohesion**: Related functions grouped together

---

### 4. Parallel Processing Strategy

**Problem**: How to handle large datasets efficiently?

**Solution**: Multi-level parallel processing.

#### Parallel Hierarchy

```
1. MPI Parallel (mpi4py)
   └── Cross-node distributed computing
       └── ThreadPoolExecutor (Python standard library)
           └── Single-node multi-thread
               └── Thread Control (threadpoolctl)
                   └── Single-core optimization
```

#### Implementation Pattern

```python
from deepx_dock.parallel import parallel_map

class Processor:
    def __init__(self, data_dir: str | Path, n_jobs: int = -1):
        self.n_jobs = n_jobs
    
    def process_all(self):
        results = parallel_map(
            self._process_single,
            items,
            n_jobs=self.n_jobs,
            desc="Processing"
        )
        return results
```

**Design Decisions**:
- **ThreadPoolExecutor**: Lower memory overhead, faster startup than multiprocessing
- **n_jobs=-1**: Auto-detect available cores (ThreadPoolExecutor default)
- **n_jobs=1**: Sequential execution (avoid threading overhead)
- **Thread safety**: numpy/scipy/h5py release GIL during operations

**Why ThreadPoolExecutor over joblib**:
1. **Memory efficiency**: Shared memory vs copy per process
2. **Startup speed**: No process spawning overhead
3. **Standard library**: No external dependency
4. **GIL friendly**: DeepH-dock uses numpy/scipy which release GIL
5. **Simpler maintenance**: Standard library API is stable

**Usage**:
```bash
# Use all cores (default)
dock convert siesta to-deeph input output

# Sequential (for small tasks)
dock convert siesta to-deeph input output -j 1

# Specific number of threads
dock convert siesta to-deeph input output -j 8
```

---

## Key Design Patterns

### 1. Registry Pattern

Used for CLI auto-registration and easy extensibility.

```python
# Global registry
registry = FunctionRegistry()
register = registry.register

# Usage
@register(cli_name="cmd", cli_help="...", cli_args=[...])
def my_function(...):
    ...
```

### 2. Lazy Import Pattern

Load heavy dependencies only when needed for faster CLI startup.

```python
def cli_function(...):
    # Lazy import
    from .heavy_module import HeavyClass
    processor = HeavyClass(...)
    processor.run()
```

### 3. Template Method Pattern

Base classes define the structure, subclasses implement specifics.

```python
class BaseTranslator:
    def translate(self):
        # Template method
        data = self.read_input()
        parsed = self.parse_data(data)
        self.write_output(parsed)
    
    @abstractmethod
    def read_input(self):
        pass
    
    @abstractmethod
    def parse_data(self, data):
        pass
```

### 4. Strategy Pattern

Different algorithms for the same task.

```python
# Eigenvalue calculation with different strategies
def calc_eigen(method="direct"):
    if method == "direct":
        return direct_diagonalization(H)
    elif method == "sparse":
        return sparse_solver(H, k=10)
    elif method == "kpm":
        return kernel_polynomial_method(H)
```

---

## Extension Points

### Adding a New Module

1. Create directory: `deepx_dock/<module_name>/`
2. Implement core functionality
3. Create `_cli.py` with `@register` decorators
4. Module auto-discovered on CLI startup

### Adding a New DFT Converter

1. Create directory: `deepx_dock/convert/<dft_code>/`
2. Implement parser and translator
3. Register CLI commands in `_cli.py`
4. Add tests in `tests/convert/<dft_code>/`

### Adding a New Computation

1. Implement computation class in `deepx_dock/compute/<type>/`
2. Register CLI command in `_cli.py`
3. Add tests

---

## Trade-offs

### 1. Auto-registration vs Explicit Registration

**Choice**: Auto-registration via decorators

**Pros**:
- Zero boilerplate
- Automatic hierarchy
- Easy to add commands

**Cons**:
- Less explicit
- Requires `_cli.py` naming convention

### 2. Unified Format vs Format-Specific

**Choice**: Unified DeepH format

**Pros**:
- Consistent interface
- Easy downstream processing
- Simplifies code

**Cons**:
- Conversion overhead
- May lose format-specific features

### 3. Multi-level Parallel vs Simple Parallel

**Choice**: Multi-level (MPI/threads)

**Pros**:
- Maximum flexibility
- Scales from laptop to HPC

**Cons**:
- More complex
- Requires understanding of different backends

---

## Future Improvements

1. **Lazy Loading for CLI** - Further reduce startup time by lazy-loading modules
2. **Plugin System** - Allow external packages to register commands
3. **Configuration Management** - Centralized config file support
4. **Caching Layer** - Cache intermediate results for repeated operations
5. **Streaming Processing** - Process large datasets without loading all into memory

---

## Lessons Learned

1. **Naming conventions matter** - `_cli.py` convention enables auto-discovery
2. **Lazy imports are essential** - Heavy dependencies slow down CLI startup
3. **Standard data format is key** - Unified format simplifies everything downstream
4. **Parallel processing needs control** - Oversubscription degrades performance
5. **Documentation is part of the API** - Clear docstrings help users and developers

---

## Reuse Guide

This architecture can be adapted to other projects that need:

1. **Multiple format converters** with unified output
2. **Modular CLI tools** with many commands
3. **Auto-registration** for extensibility
4. **Parallel processing** for large datasets

**Key files to adapt**:
- `deepx_dock/_cli/registry.py` - Registry system
- `deepx_dock/_cli/__init__.py` - Auto-discovery
- `deepx_dock/CONSTANT.py` - Constants and file names
- `deepx_dock/misc.py` - Utility functions

---

**Last Updated**: 2025-03-08
