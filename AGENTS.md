# DeepH-dock Development Guide for AI Agents

## Build, Install & Test Commands

### Installation
```bash
pip install -e .                  # Editable install (development)
uv build --wheel -o dist ./       # Build distribution wheel
```
- Python version: **3.12-3.14** required

### Testing
```bash
bash tests/run_test_all.sh                           # Run all tests
bash tests/<module>/<submodule>/run_test.sh           # Run single test
bash tests/convert/siesta/run_test.sh                 # Example: SIESTA converter
bash tests/compute/eigen/run_test.sh                  # Example: Eigenvalue calculation
```
- Tests are **bash scripts** that execute CLI commands and validate outputs
- Test scripts invoke `dock` commands and compare results against reference data
- Clean test artifacts: `bash tests/_clean.sh` (interactive)

### Linting & Formatting
```bash
black .                           # Format code
ruff check .                      # Lint code
ruff check --fix .                # Auto-fix linting issues
```
- Line length: **120 characters** (configured in pyproject.toml)

## Code Style Guidelines

### Imports Organization
```python
# Standard library (alphabetically)
from pathlib import Path
from typing import List, Optional, Tuple

# Third-party (alphabetically)
import h5py
import numpy as np
from click import argument, option

# Local imports (alphabetically)
from deepx_dock.parallel import parallel_map
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.misc import load_json_file
```
- Separate groups with blank lines
- Alphabetically sorted within each group
- Import specific functions/classes, not entire modules (except numpy)

### Naming Conventions
- **Functions/Variables**: `snake_case` (e.g., `translate_to_deeph`, `n_jobs`)
- **Classes**: `PascalCase` (e.g., `HamiltonianObj`, `DatasetAnalyzer`)
- **Constants**: `UPPER_CASE` (e.g., `DEEPX_HAMILTONIAN_FILENAME`, `HARTREE_TO_EV`)
- **CLI commands**: `kebab-case` (e.g., `to-deeph`, `calc-band`)
- **Private methods**: `_leading_underscore` (e.g., `_read_h5`, `_validation_check`)
- **Parameter naming**:
  - CLI options: `--jobs-num`, `--tier-num` (kebab-case)
  - CLI function parameters: `jobs_num`, `tier_num` (matches click convention)
  - Class `__init__` parameters: `n_jobs`, `n_tier` (unified `n_` prefix)

### Type Hints
```python
# Use for all function signatures
def load_poscar_file(file_path: str | Path) -> dict:
    ...

def analyze_data(
    data_path: str | Path,
    n_jobs: int = -1,
    n_tier: int = 0,
) -> None:
    ...

# Use Optional for optional parameters
from typing import Optional
def process_file(path: Path, encoding: Optional[str] = None) -> str:
    ...
```

### Docstrings
Use **NumPy style** for classes and complex functions:

```python
def r2k(self, ks: np.ndarray) -> np.ndarray:
    """
    Fourier transform from real space to reciprocal space.

    Parameters
    ----------
    ks : np.ndarray, shape (Nk, 3)
        k-points in fractional coordinates.

    Returns
    -------
    MKs : np.ndarray, shape (Nk, N_b, N_b)
        Matrices in reciprocal space.
    """
    phase = np.exp(2j * np.pi * np.matmul(ks, self.Rs.T))
    MRs_flat = self.MRs.reshape(len(self.Rs), -1)
    Mks_flat = np.matmul(phase, MRs_flat)
    return Mks_flat.reshape(len(ks), *self.MRs.shape[1:])
```

### Comments
- **NO inline comments** unless absolutely necessary for non-obvious logic
- Let code speak through clear naming and structure
- Use docstrings for documentation, not comments
- Exception: Brief comments for complex algorithms or physical constants

### Error Handling
```python
# Use assertions for internal invariants
assert self.data_dir.is_dir(), f"{data_dir} is not a directory"

# Raise explicit exceptions for user-facing errors
if symbol not in PERIODIC_TABLE_SYMBOL_TO_INDEX:
    raise KeyError(f"Unknown element symbol: {symbol}")

# Validate inputs early in __init__
def __init__(self, data_dir: str | Path):
    self.data_dir = Path(data_dir)
    assert self.data_dir.is_dir(), f"{data_dir} is not a directory"
    self.data_dir.mkdir(parents=True, exist_ok=True)
```

## Project Architecture

### Module Structure
```
deepx_dock/
├── _cli/              # CLI auto-registration system
│   ├── __init__.py    # Auto-discovery and command building
│   └── registry.py    # Function registration decorator
├── convert/           # DFT software format converters
│   ├── siesta/
│   ├── openmx/
│   ├── fhi_aims/
│   ├── abacus/
│   └── ...
├── compute/           # Electronic structure calculations
│   ├── eigen/         # Band structure, DOS, Fermi level
│   ├── overlap/       # Overlap matrix calculations
│   └── ...
├── analyze/           # Data analysis tools
│   ├── dataset/       # Feature analysis, data splitting
│   ├── error/         # Error analysis
│   └── dft_equiv/     # Equivariance testing
├── design/            # Structure generation
├── CONSTANT.py        # All constants (UPPER_CASE)
└── misc.py            # Utility functions
```

### Adding New CLI Commands

**Pattern**: Each module has a `_cli.py` file for command registration

1. **Create module directory**: `deepx_dock/<module>/<submodule>/`

2. **Implement core functionality**:
```python
# deepx_dock/convert/new_dft/translator.py
class NewDFTTranslator:
    def __init__(self, input_dir: str | Path, output_dir: str | Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
    
    def translate(self) -> None:
        # Implementation
        pass
```

3. **Create `_cli.py`** with auto-registration:
```python
# deepx_dock/convert/new_dft/_cli.py
import click
from pathlib import Path
from deepx_dock._cli.registry import register

@register(
    cli_name="to-deeph",
    cli_help="Convert NewDFT output to DeepH format",
    cli_args=[
        click.argument('input_dir', type=click.Path(exists=True)),
        click.argument('output_dir', type=click.Path()),
        click.option('--jobs-num', '-j', type=int, default=-1,
                     help='Parallel processing number (-1 for all cores)'),
    ],
)
def translate_newdft_to_deeph(
    input_dir: str | Path,
    output_dir: str | Path,
    jobs_num: int,
):
    from .translator import NewDFTTranslator  # Lazy import
    translator = NewDFTTranslator(input_dir, output_dir, n_jobs=jobs_num)
    translator.translate()
```

4. **Command auto-registered as**: `dock convert new-dft to-deeph`

**Key points**:
- Use `@register` decorator from `deepx_dock._cli.registry`
- Lazy import heavy dependencies inside the function
- Use `click` decorators for arguments and options
- Auto-discovery happens on startup via `_auto_register_cli()`

## Common Patterns

### Path Handling
```python
from pathlib import Path

# Always convert to Path early
def __init__(self, data_dir: str | Path):
    self.data_dir = Path(data_dir)
    assert self.data_dir.is_dir(), f"{data_dir} is not a directory"
    self.data_dir.mkdir(parents=True, exist_ok=True)
```

### Parallel Processing
```python
from deepx_dock.parallel import parallel_map

# Use n_jobs parameter (default: -1 for all cores)
results = parallel_map(process_single_item, items, n_jobs=self.n_jobs, desc="Processing")
```

### HDF5 File Operations
```python
import h5py

# Reading
with h5py.File(filepath, 'r') as f:
    atom_pairs = f['atom_pairs'][:]
    entries = f['entries'][:]

# Writing
with h5py.File(filepath, 'w') as f:
    f.create_dataset('entries', data=entries, compression='gzip')
```

### CLI Function Template
```python
@register(
    cli_name="action-name",
    cli_help="Brief description of what this command does",
    cli_args=[
        click.argument('input_path', type=click.Path(exists=True)),
        click.argument('output_path', type=click.Path()),
        click.option('--jobs-num', '-j', type=int, default=-1),
        click.option('--tier-num', '-t', type=int, default=0),
        click.option('--force', is_flag=True, help='Force overwrite'),
    ],
)
def action_command(
    input_path: str | Path,
    output_path: str | Path,
    jobs_num: int,
    tier_num: int,
    force: bool,
):
    """Brief description of CLI function."""
    from .module import ProcessorClass  # Lazy import
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Validation
    assert input_path.exists(), f"{input_path} does not exist"
    if output_path.exists() and not force:
        click.confirm(f"{output_path} exists. Overwrite?", abort=True)
    
    # Process
    processor = ProcessorClass(input_path, output_path, n_jobs=jobs_num, n_tier=tier_num)
    processor.run()
    click.echo("[done] Processing completed")
```

## Testing Guidelines

### Test Structure
```
tests/
├── run_test_all.sh          # Run all tests
├── _clean.sh                # Interactive cleanup script
├── check_file.sh            # File comparison utility
├── <module>/
│   ├── run_test.sh          # Single test script
│   ├── <reference>.bak/     # Reference data (symlink)
│   └── <output>/            # Generated output
```

### Writing Tests
Tests are **bash scripts** that:
1. Execute `dock` CLI commands
2. Compare outputs against reference data
3. Clean up generated files

```bash
#!/bin/bash
_pwd=$(pwd)
script_path=$(realpath $(dirname $0))

cd ${script_path}
rm -rf output_dir

# Run command
dock convert siesta to-deeph input.bak output_dir -t 0 -j 1

# Validate outputs
for d1 in $(ls output_dir); do
    for f in $(ls output_dir/$d1); do
        bash ../../check_file.sh $f output_dir/$d1/$f reference.bak/$d1/$f
    done
done

cd ${_pwd}
```

### Test Data
- Use symlinks for test data: `ln -s ../../../examples/... data.bak`
- Place reference outputs in `<name>.bak` directories
- Generated outputs should match reference structure

## Important Files

- **`pyproject.toml`**: Build config, dependencies, tool settings (ruff, black)
- **`CONSTANT.py`**: All constants (file names, physical constants, periodic table)
- **`misc.py`**: Utility functions (load/save JSON/TOML/POSCAR, data directory listing)
- **`parallel.py`**: Parallel processing utilities using ThreadPoolExecutor
- **`_cli/registry.py`**: CLI registration decorator system
- **`_cli/__init__.py`**: Auto-discovery mechanism and command building

## Physical Constants

Commonly used constants (defined in `CONSTANT.py`):
```python
HARTREE_TO_EV = 27.2113845
BOHR_TO_ANGSTROM = 0.529177249
EXTREMELY_SMALL_FLOAT = 1.0E-16
```

## Data Format Standards

DeepH standard format for each data point:
```
<data_dir>/
├── POSCAR              # Crystal structure (required)
├── info.json           # Metadata (required)
├── overlap.h5          # Overlap matrix (required)
├── hamiltonian.h5      # Hamiltonian matrix (optional)
├── density_matrix.h5   # Density matrix (optional)
└── position_matrix.h5  # Position matrix (optional)
```

HDF5 file structure:
```python
{
    'atom_pairs': np.ndarray,      # (N_edges, 5): [Rx, Ry, Rz, i_atom, j_atom]
    'chunk_shapes': np.ndarray,    # (N_edges, 2): shape of each block
    'chunk_boundaries': np.ndarray,# (N_edges+1,): data boundaries
    'entries': np.ndarray,         # Flattened matrix elements
}
```
