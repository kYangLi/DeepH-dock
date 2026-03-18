# Development Guide

**Status**: ✅ Implemented  
**Version**: 0.9.11  
**Last Updated**: 2025-03-08

---

## Overview

This guide explains how to extend DeepH-dock by adding new DFT converters, computation functions, and analysis tools.

---

## Adding a New DFT Converter

### Step 1: Create Module Structure

```bash
mkdir -p deepx_dock/convert/new_dft_code
touch deepx_dock/convert/new_dft_code/__init__.py
touch deepx_dock/convert/new_dft_code/translator.py
touch deepx_dock/convert/new_dft_code/_cli.py
```

### Step 2: Implement Translator Class

```python
# deepx_dock/convert/new_dft_code/translator.py
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
from deepx_dock.parallel import parallel_map

from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME, DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.misc import dump_poscar_file, dump_json_file

class NewDFTTranslator:
    """Convert NewDFT output to DeepH format"""
    
    def __init__(self, input_dir: str | Path, output_dir: str | Path, n_jobs: int = -1):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.n_jobs = n_jobs
        
        # Validate inputs
        assert self.input_dir.is_dir(), f"{input_dir} is not a directory"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_all(self) -> None:
        """Convert all structures in input directory"""
        # Find all data directories
        data_dirs = self._find_data_dirs()
        
        # Parallel processing
        results = parallel_map(
            self._convert_single,
            data_dirs,
            n_jobs=self.n_jobs,
            desc="Converting"
        )
    
    def _find_data_dirs(self) -> list[Path]:
        """Find all data directories"""
        # Implementation depends on NewDFT output structure
        return list(self.input_dir.glob("*/"))
    
    def _convert_single(self, data_dir: Path) -> None:
        """Convert single structure"""
        # 1. Read NewDFT output files
        structure_data = self._read_structure(data_dir)
        hamiltonian_data = self._read_hamiltonian(data_dir)
        overlap_data = self._read_overlap(data_dir)
        
        # 2. Convert to DeepH format
        output_dir = self.output_dir / data_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Write POSCAR
        dump_poscar_file(output_dir / DEEPX_POSCAR_FILENAME, structure_data)
        
        # 4. Write info.json
        info = self._extract_info(hamiltonian_data)
        dump_json_file(output_dir / DEEPX_INFO_FILENAME, info)
        
        # 5. Write matrices to HDF5
        self._write_matrix(output_dir / DEEPX_OVERLAP_FILENAME, overlap_data)
        self._write_matrix(output_dir / DEEPX_HAMILTONIAN_FILENAME, hamiltonian_data)
    
    def _read_structure(self, data_dir: Path) -> dict:
        """Read crystal structure from NewDFT output"""
        # Parse NewDFT structure file
        # Return: {"lattice": np.ndarray, "cart_coords": np.ndarray, ...}
        pass
    
    def _read_hamiltonian(self, data_dir: Path) -> dict:
        """Read Hamiltonian matrix from NewDFT output"""
        # Parse NewDFT Hamiltonian file
        # Return: {"atom_pairs": np.ndarray, "entries": np.ndarray, ...}
        pass
    
    def _write_matrix(self, filepath: Path, matrix_data: dict) -> None:
        """Write matrix to HDF5 file"""
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('atom_pairs', data=matrix_data['atom_pairs'])
            f.create_dataset('entries', data=matrix_data['entries'], compression='gzip')
            # ... other datasets
```

### Step 3: Register CLI Command

```python
# deepx_dock/convert/new_dft_code/_cli.py
import click
from pathlib import Path
from deepx_dock._cli.registry import register

@register(
    cli_name="to-deeph",
    cli_help="Convert NewDFT output to DeepH format",
    cli_args=[
        click.argument('input_dir', type=click.Path(exists=True)),
        click.argument('output_dir', type=click.Path()),
        click.option(
            '--jobs-num', '-j', type=int, default=-1,
            help='Parallel processing number (-1 for all cores)'
        ),
        click.option(
            '--tier-num', '-t', type=int, default=0,
            help='Directory depth for data directories'
        ),
    ],
)
def translate_newdft_to_deeph(
    input_dir: str | Path,
    output_dir: str | Path,
    jobs_num: int,
    tier_num: int,
):
    from .translator import NewDFTTranslator  # Lazy import
    
    translator = NewDFTTranslator(input_dir, output_dir, n_jobs=jobs_num, n_tier=tier_num)
    translator.convert_all()
    click.echo("[done] Conversion completed successfully!")
```

### Step 4: Test the Converter

```bash
# Create test directory
mkdir -p tests/convert/new_dft_code

# Create test script
cat > tests/convert/new_dft_code/run_test.sh << 'EOF'
#!/bin/bash
_pwd=$(pwd)
script_path=$(realpath $(dirname $0))

cd ${script_path}
rm -rf output_dir

# Run conversion
dock convert new-dft-code to-deeph input.bak output_dir -j 1

# Validate outputs
for d1 in $(ls output_dir); do
    for f in $(ls output_dir/$d1); do
        bash ../../check_file.sh $f output_dir/$d1/$f reference.bak/$d1/$f
    done
done

cd ${_pwd}
EOF

chmod +x tests/convert/new_dft_code/run_test.sh
```

---

## Adding a New Computation Function

### Step 1: Implement Core Class

```python
# deepx_dock/compute/new_feature/calculator.py
from pathlib import Path
import numpy as np
import h5py
from typing import Optional

from deepx_dock.compute.eigen.matrix_obj import AOMatrixObj
from deepx_dock.misc import load_json_file

class NewFeatureCalculator:
    """Calculate new electronic structure feature"""
    
    def __init__(self, data_path: str | Path, matrix_file: Optional[str] = None):
        self.data_path = Path(data_path)
        self.matrix_file = matrix_file
        
        # Load basic info
        self.info = load_json_file(self.data_path / "info.json")
        self.n_atoms = self.info["atoms_quantity"]
        self.spinful = self.info["spinful"]
        
        # Load matrices if needed
        if matrix_file:
            self.matrix_obj = AOMatrixObj(
                self.data_path,
                matrix_file_path=self.data_path / matrix_file
            )
    
    def calculate(self, **kwargs) -> np.ndarray:
        """Main calculation method"""
        # Implementation
        result = self._compute_feature(**kwargs)
        return result
    
    def _compute_feature(self, **kwargs) -> np.ndarray:
        """Implement the actual calculation"""
        # Your computation logic here
        pass
```

### Step 2: Register CLI Command

```python
# deepx_dock/compute/new_feature/_cli.py
import click
from pathlib import Path
from deepx_dock._cli.registry import register

@register(
    cli_name="calc-feature",
    cli_help="Calculate new electronic structure feature",
    cli_args=[
        click.argument('data_path', type=click.Path(exists=True)),
        click.option(
            '--output-file', '-o', type=str, default="feature.h5",
            help='Output file name'
        ),
        click.option(
            '--option-name', type=str, default="default_value",
            help='Option description'
        ),
    ],
)
def calculate_new_feature(
    data_path: str | Path,
    output_file: str,
    option_name: str,
):
    from .calculator import NewFeatureCalculator
    
    calculator = NewFeatureCalculator(data_path)
    result = calculator.calculate(option=option_name)
    
    # Save result
    output_path = Path(data_path) / output_file
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('feature', data=result)
    
    click.echo(f"[done] Feature saved to {output_file}")
```

---

## Adding a New Analysis Tool

### Step 1: Implement Analyzer Class

```python
# deepx_dock/analyze/new_analysis/analyzer.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from deepx_dock.misc import get_data_dir_lister, load_json_file

class NewAnalyzer:
    """Analyze new aspect of the data"""
    
    def __init__(
        self,
        data_path: str | Path,
        n_jobs: int = -1,
        n_tier: int = 0,
    ):
        self.data_path = Path(data_path)
        self.n_jobs = n_jobs
        self.n_tier = n_tier
        
        # Get list of data directories
        self.data_dirs = list(get_data_dir_lister(
            self.data_path, depth=n_tier
        ))
    
    def analyze(self, **kwargs) -> dict:
        """Run analysis"""
        results = []
        
        for data_dir in self.data_dirs:
            result = self._analyze_single(data_dir, **kwargs)
            results.append(result)
        
        # Aggregate results
        summary = self._summarize(results)
        return summary
    
    def _analyze_single(self, data_dir: Path, **kwargs) -> dict:
        """Analyze single structure"""
        # Load data
        info = load_json_file(data_dir / "info.json")
        
        # Perform analysis
        # ...
        
        return {"result": ...}
    
    def _summarize(self, results: List[dict]) -> dict:
        """Summarize all results"""
        # Aggregate statistics
        return {"summary": ...}
    
    def plot(self, results: dict, output_path: Optional[Path] = None):
        """Visualize results"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plotting logic
        ax.plot(...)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
```

### Step 2: Register CLI Command

```python
# deepx_dock/analyze/new_analysis/_cli.py
import click
from pathlib import Path
from deepx_dock._cli.registry import register

@register(
    cli_name="analyze-feature",
    cli_help="Analyze new aspect of the data",
    cli_args=[
        click.argument('data_path', type=click.Path(exists=True)),
        click.option('--jobs-num', '-j', type=int, default=-1),
        click.option('--plot', is_flag=True, help='Generate plots'),
        click.option('--output-dir', '-o', type=click.Path()),
    ],
)
def analyze_new_feature(
    data_path: str | Path,
    jobs_num: int,
    plot: bool,
    output_dir: str,
):
    from .analyzer import NewAnalyzer
    
    analyzer = NewAnalyzer(data_path, n_jobs=jobs_num)
    results = analyzer.analyze()
    
    if plot:
        output_path = Path(output_dir) if output_dir else Path(data_path)
        analyzer.plot(results, output_path / "analysis.png")
    
    click.echo("[done] Analysis completed")
```

---

## Testing and Validation

### Unit Testing Pattern

```python
# tests/convert/new_dft_code/test_translator.py
import pytest
from pathlib import Path
import numpy as np

from deepx_dock.convert.new_dft_code.translator import NewDFTTranslator

def test_translator_init(tmp_path):
    """Test translator initialization"""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    
    translator = NewDFTTranslator(input_dir, output_dir)
    
    assert translator.input_dir == input_dir
    assert translator.output_dir == output_dir

def test_structure_parsing(tmp_path):
    """Test structure parsing"""
    # Create test input file
    # ...
    
    translator = NewDFTTranslator(input_dir, output_dir)
    structure = translator._read_structure(input_dir)
    
    assert "lattice" in structure
    assert "cart_coords" in structure
```

### Integration Testing

Use bash scripts for integration tests (see `tests/` directory for examples).

---

## Code Quality Guidelines

### Type Hints

```python
# ✅ Good
def process_data(
    data_path: str | Path,
    n_jobs: int = -1,
    spinful: bool = False,
) -> dict:
    ...

# ❌ Bad
def process_data(data_path, n_jobs=-1, spinful=False):
    ...
```

### Error Handling

```python
# ✅ Good - Validate early
def __init__(self, data_dir: str | Path):
    self.data_dir = Path(data_dir)
    assert self.data_dir.is_dir(), f"{data_dir} is not a directory"
    self.data_dir.mkdir(parents=True, exist_ok=True)

# ✅ Good - Raise explicit exceptions
def parse_orbital_string(s: str) -> List[int]:
    if not s:
        return []
    if s[0] not in 'spdfgh':
        raise ValueError(f"Invalid orbital string: {s}")
    # ...
```

### Parallel Processing

```python
# ✅ Good - Use n_jobs parameter
class Processor:
    def __init__(self, data_dir: str | Path, n_jobs: int = -1):
        self.n_jobs = n_jobs
    
    def process_all(self):
        from deepx_dock.parallel import parallel_map
        results = parallel_map(
            self._process_single,
            items,
            n_jobs=self.n_jobs,
            desc="Processing"
        )
        return results
```

---

## Common Pitfalls

### 1. Not Converting Paths Early

```python
# ❌ Bad
def __init__(self, data_dir):
    self.data_dir = data_dir  # Could be str or Path
    files = self.data_dir.glob("*.h5")  # Error if str!

# ✅ Good
def __init__(self, data_dir: str | Path):
    self.data_dir = Path(data_dir)
    files = self.data_dir.glob("*.h5")  # Works!
```

### 2. Forgetting to Validate Inputs

```python
# ❌ Bad
def __init__(self, data_dir):
    self.data_dir = data_dir
    # No validation, may fail later

# ✅ Good
def __init__(self, data_dir: str | Path):
    self.data_dir = Path(data_dir)
    assert self.data_dir.is_dir(), f"{data_dir} is not a directory"
```

### 3. Not Using Lazy Imports

```python
# ❌ Bad - Slow CLI startup
from deepx_dock.convert.siesta.translator import SIESTADatasetTranslator

@register(...)
def convert_siesta(...):
    translator = SIESTADatasetTranslator(...)
    translator.convert()

# ✅ Good - Fast CLI startup
@register(...)
def convert_siesta(...):
    from deepx_dock.convert.siesta.translator import SIESTADatasetTranslator
    translator = SIESTADatasetTranslator(...)
    translator.convert()
```

---

## Registry Reference

DeepH-dock uses decorators for extensibility:

| Decorator | Purpose | File |
|-----------|---------|------|
| `@register` | Register CLI command | `deepx_dock._cli.registry` |

**Usage**:
```python
from deepx_dock._cli.registry import register

@register(
    cli_name="command-name",
    cli_help="Description",
    cli_args=[...],
)
def my_function(...):
    ...
```

---

**Last Updated**: 2025-03-08
