# Frequently Asked Questions

**Status**: ✅ Implemented  
**Version**: 0.9.11  
**Last Updated**: 2025-03-08

---

## Development Questions

### Q: How do I add a new DFT converter?

A: Follow these steps:

1. Create module: `deepx_dock/convert/new_dft/`
2. Implement translator class with `_read_*` methods
3. Register CLI in `_cli.py` using `@register` decorator
4. Add tests in `tests/convert/new_dft/`

See [development.md](development.md) for detailed guide.

---

### Q: How do I register a new CLI command?

A: Use the `@register` decorator in your `_cli.py` file:

```python
from deepx_dock._cli.registry import register
import click

@register(
    cli_name="my-command",
    cli_help="Description",
    cli_args=[
        click.argument('input_path'),
        click.option('--option-name', default='value'),
    ],
)
def my_function(input_path, option_name):
    # Lazy import
    from .module import MyClass
    processor = MyClass(input_path, option_name)
    processor.run()
```

The command is auto-registered as: `dock <module> <submodule> my-command`

---

### Q: Why use lazy imports in CLI functions?

A: Lazy imports improve CLI startup time. DeepH-dock has 30+ commands, and importing all dependencies would slow down every command. Example:

```python
# ❌ Slow startup
from heavy_module import HeavyClass

@register(...)
def my_cmd(...):
    HeavyClass(...)

# ✅ Fast startup
@register(...)
def my_cmd(...):
    from heavy_module import HeavyClass
    HeavyClass(...)
```

---

### Q: How do I handle paths correctly?

A: Always convert to `Path` objects early:

```python
from pathlib import Path

def __init__(self, data_dir: str | Path):
    self.data_dir = Path(data_dir)  # Convert immediately
    assert self.data_dir.is_dir(), f"{data_dir} is not a directory"
```

---

### Q: How do I enable parallel processing?

A: Use `parallel_map` with the `n_jobs` parameter:

```python
from deepx_dock.parallel import parallel_map

class Processor:
    def __init__(self, n_jobs: int = -1):
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

---

## Data Conversion Questions

### Q: What is the DeepH standard format?

A: Each data point has:

```
<data_dir>/
├── POSCAR              # Crystal structure (required)
├── info.json           # Metadata (required)
├── overlap.h5          # Overlap matrix (required)
└── hamiltonian.h5      # Hamiltonian matrix (optional)
```

See [architecture.md](architecture.md) for format details.

---

### Q: How do I handle spin-orbit coupling?

A: For spinful systems, matrices expand by 4x:

```python
def expand_to_spinful(block):
    """Non-spinful: [N×N] → Spinful: [[↑↑, ↑↓], [↓↑, ↓↓]] = [2N×2N]"""
    zero_block = np.zeros_like(block)
    return np.block([[block, zero_block], [zero_block, block]])
```

---

### Q: What if my DFT code uses different units?

A: Use constants from `CONSTANT.py`:

```python
from deepx_dock.CONSTANT import HARTREE_TO_EV, BOHR_TO_ANGSTROM

energy_eV = energy_hartree * HARTREE_TO_EV
distance_ang = distance_bohr * BOHR_TO_ANGSTROM
```

---

### Q: How do I handle different orbital orderings?

A: Apply transformation matrix based on DFT-specific conventions. Example for SIESTA:

```python
SIESTA_BASIS_ORDER = {
    0: [0],           # s
    1: [0, 1, 2],     # px, py, pz
}

SIESTA_BASIS_PARITY = {
    0: [1],
    1: [-1, 1, -1],
}

def transform_siesta_basis(matrix, l_value):
    order = SIESTA_BASIS_ORDER[l_value]
    parity = SIESTA_BASIS_PARITY[l_value]
    return matrix[order] * parity
```

---

## Computation Questions

### Q: What if eigenvalue calculation fails?

A: You may have ill-conditioned overlap matrix. Try:

1. **Window regularization**: Remove states outside energy window
2. **Orbital removal**: Remove problematic orbitals
3. **Sparse solver**: Use iterative methods for large systems

```bash
# Use ill-conditioned handling
dock compute eigen calc-band data_dir/ --ill-method window --ill-window -10,10
```

---

### Q: How do I choose between direct and sparse diagonalization?

A:
- **Direct**: Small systems (< 500 orbitals), more accurate
- **Sparse**: Large systems (> 500 orbitals), faster

```bash
# Direct (default)
dock compute eigen calc-band data_dir/

# Sparse (for large systems)
dock compute eigen calc-band data_dir/ --sparse --k-sparse 20
```

---

### Q: How do I calculate DOS accurately?

A: Two methods available:

1. **Gaussian smearing**: Fast but less accurate
   ```bash
   dock compute eigen calc-dos data_dir/ --method gaussian --sigma 0.1
   ```

2. **Tetrahedron method**: More accurate but slower
   ```bash
   dock compute eigen calc-dos data_dir/ --method tetrahedron
   ```

---

## Analysis Questions

### Q: How do I analyze model prediction errors?

A: Use multi-dimensional error analysis:

```bash
# Entry-level error
dock analyze error entries dft_dir/ pred_dir/ -o error_entries

# Orbital-resolved error
dock analyze error orbital dft_dir/ pred_dir/ -o error_orbital

# Element-pair error
dock analyze error element-pair dft_dir/ pred_dir/ -o error_element
```

---

### Q: How do I split my dataset for training?

A: Use dataset splitting tool:

```bash
dock analyze dataset split data_dir/ \
    --split-ratio 0.6 0.2 0.2 \
    --split-rng-seed 137 \
    -o dataset_split.json
```

---

### Q: How do I validate equivariance?

A: Test translation and rotation equivariance:

```bash
# Generate test structures
dock analyze dft-equiv gen input_dir/ -o test_structures/

# Run DFT on test structures
# ...

# Test equivariance
dock analyze dft-equiv test test_structures/ pred_dir/ -o equiv_report.json
```

---

## Performance Questions

### Q: How do I speed up data conversion?

A: Use parallel processing:

```bash
# Use all cores
dock convert siesta to-deeph input output -j -1

# Use specific number of cores
dock convert siesta to-deeph input output -j 8
```

---

### Q: How do I reduce memory usage?

A: Several strategies:

1. **Process one structure at a time**: Avoid loading all data
2. **Use disk storage**: Store graphs on disk instead of memory
3. **Reduce batch size**: For computation tasks

```python
# Process iteratively
for data_dir in data_dirs:
    convert_single(data_dir)  # Memory freed after each iteration
```

---

### Q: How do I optimize HDF5 file size?

A: Use compression:

```python
with h5py.File(output_path, 'w') as f:
    f.create_dataset('entries', data=entries, 
                     compression='gzip', compression_opts=6)
```

---

## Testing Questions

### Q: How do I run tests?

A: Use bash test scripts:

```bash
# Run all tests
bash tests/run_test_all.sh

# Run single test
bash tests/convert/siesta/run_test.sh
```

---

### Q: How do I add a test for my new feature?

A: Create a test script:

```bash
# tests/convert/new_dft/run_test.sh
#!/bin/bash
_pwd=$(pwd)
script_path=$(realpath $(dirname $0))

cd ${script_path}
rm -rf output_dir

# Run command
dock convert new-dft to-deeph input.bak output_dir -j 1

# Validate outputs
for d1 in $(ls output_dir); do
    bash ../../check_file.sh $f output_dir/$d1/$f reference.bak/$d1/$f
done

cd ${_pwd}
```

---

## Troubleshooting

### Q: Command not found after adding new module

A: Make sure you:
1. Created `_cli.py` file
2. Used `@register` decorator
3. Named function correctly (no conflicts)
4. Reinstalled: `pip install -e .`

---

### Q: CLI startup is slow

A: Check that you're using lazy imports:

```python
# ❌ Bad - imports on startup
from heavy_module import HeavyClass

@register(...)
def my_cmd(...):
    ...

# ✅ Good - imports only when called
@register(...)
def my_cmd(...):
    from heavy_module import HeavyClass
    ...
```

---

### Q: Parallel processing is slower than sequential

A: Check:
1. **Oversubscription**: Don't use more processes than cores
2. **Small tasks**: Parallel overhead may exceed benefit for small tasks
3. **Memory**: Parallel processes may compete for memory

---

**Last Updated**: 2025-03-08
