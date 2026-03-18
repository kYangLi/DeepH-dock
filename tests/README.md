# Tests

This directory contains all tests for DeepH-dock, using pytest framework.

## Test Structure

```
tests/
├── conftest.py              # pytest fixtures and configuration
├── pytest.ini               # pytest configuration
├── fixtures/                # Shared test utilities
│   └── comparison.py        # File comparison tools
│
├── misc/                    # Unit tests for utility modules
│   └── test_misc.py         # Tests for misc.py, CONSTANT.py, etc.
│
├── convert/                 # Convert module tests
│   ├── siesta/
│   │   ├── test_siesta.py   # Test script
│   │   ├── siesta.bak/      # Input data
│   │   └── deeph.bak/       # Reference output
│   ├── openmx/
│   │   ├── test_openmx.py
│   │   └── ...
│   ├── abacus/
│   ├── fhi_aims/
│   ├── hopcp/
│   └── deeph/
│
├── compute/                 # Compute module tests
│   ├── eigen/
│   │   ├── test_eigen.py    # Test script
│   │   ├── eigen.clean/     # Input data
│   │   └── eigen.bak/       # Reference output
│   └── ...
│
└── analyze/                 # Analyze module tests
    ├── error/
    │   ├── test_error.py
    │   ├── benchmark.bak/
    │   ├── infer.clean/
    │   └── infer.bak/
    ├── dataset/
    └── dft_equiv/
```

## Test Categories

### 1. Unit Tests (tests/misc/)
Tests for non-computational utility functions:
- File I/O utilities (misc.py)
- Physical constants (CONSTANT.py)
- Helper functions
- Data format conversions

### 2. Module Tests (tests/convert/, tests/compute/, tests/analyze/)
Tests for each functional module:
- **convert/**: DFT format conversion tests
- **compute/**: Electronic structure calculation tests
- **analyze/**: Data analysis tests

Each module test directory contains:
- `test_*.py`: Test scripts
- `*.bak/`: Benchmark reference data
- Input data (may be symlinks to examples)

## Running Tests

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest
```

### Run Specific Module Tests

```bash
# Run convert module tests
pytest tests/convert -v

# Run compute module tests
pytest tests/compute -v

# Run analyze module tests
pytest tests/analyze -v

# Run unit tests
pytest tests/misc -v
```

### Run Specific Test File

```bash
# SIESTA conversion test
pytest tests/convert/siesta/test_siesta.py -v

# Eigenvalue calculation test
pytest tests/compute/eigen/test_eigen.py -v

# Error analysis test
pytest tests/analyze/error/test_error.py -v
```

### Run Specific Test Function

```bash
pytest tests/convert/siesta/test_siesta.py::test_siesta_to_deeph_conversion -v
```

### Run with Coverage Report

```bash
# Terminal coverage report
pytest --cov=deepx_dock --cov-report=term-missing

# HTML coverage report
pytest --cov=deepx_dock --cov-report=html
```

## Test Data

### Input Data
- Located alongside test scripts in each module directory
- Many are symlinks to `examples/` directory
- Contains original DFT output or intermediate data

### Reference Data (*.bak/)
- Benchmark outputs for validation
- Critical for scientific computing accuracy
- **Must not be modified** - these are the ground truth

## Writing New Tests

### Template for Module Tests

```python
"""
Tests for <module> <functionality>
"""
import pytest
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fixtures.comparison import compare_directories


@pytest.fixture
def module_data():
    """Test data for <module>"""
    test_dir = Path(__file__).parent
    return {
        "input": test_dir / "input_data",
        "reference": test_dir / "reference.bak",
    }


def test_module_functionality(module_data, temp_output_dir):
    """Test <functionality>"""
    input_dir = module_data["input"]
    output_dir = temp_output_dir / "output"
    reference_dir = module_data["reference"]
    
    # Run CLI command
    result = subprocess.run(
        ["dock", "<module>", "<command>", str(input_dir), str(output_dir)],
        capture_output=True,
        text=True,
    )
    
    # Check command succeeded
    assert result.returncode == 0, f"Command failed:\n{result.stderr}"
    
    # Compare with reference
    is_equal, errors = compare_directories(output_dir, reference_dir)
    assert is_equal, "\n".join(errors)


def test_module_cli_help():
    """Test CLI help"""
    result = subprocess.run(
        ["dock", "<module>", "<command>", "--help"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
```

### Template for Unit Tests

```python
"""
Unit tests for utility modules
"""
import pytest
from pathlib import Path
from deepx_dock.misc import some_function


def test_some_function(temp_output_dir):
    """Test description"""
    # Setup
    test_file = temp_output_dir / "test.txt"
    test_file.write_text("test content")
    
    # Execute
    result = some_function(test_file)
    
    # Assert
    assert result == expected_value
```

## Test Fixtures

Common fixtures available in `conftest.py`:

- `tests_dir`: Path to tests directory
- `examples_dir`: Path to examples directory
- `temp_output_dir`: Temporary directory for test outputs (auto-cleaned, in `/tmp`)

Module-specific fixtures are defined in each test file.

## Important Notes

1. **All test outputs go to temporary directories** (`/tmp`), never in project directory
2. **Reference data is sacred** - `.bak` directories contain benchmark data for accuracy validation
3. **No shell scripts** - All tests are pure Python pytest
4. **Coverage reports** show which modules need more tests
5. **Test data locality** - Each test module has its own data, making tests self-contained

## Test Naming Conventions

- **Test files**: `test_<module>.py`
- **Test functions**: `test_<functionality>_<scenario>()`
- **Fixtures**: `<module>_data()`
- **Test directories**: Same as module name (e.g., `siesta/`, `eigen/`)

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'fixtures'`, make sure you're running pytest from the project root directory:

```bash
cd /path/to/DeepH-dock
pytest tests/convert/siesta/test_siesta.py
```

### Missing Dependencies

```bash
pip install -e ".[dev]"
```

### Test Data Issues

If test data is missing:
1. Check symlinks are valid (for input data)
2. Check reference data exists in `.bak` directories
3. Check examples directory is present

### Clean Up Test Artifacts

```bash
# Remove all __pycache__ directories
find tests -type d -name "__pycache__" -exec rm -rf {} +

# Remove .pytest_cache
rm -rf .pytest_cache

# Remove coverage files
rm -rf htmlcov/ .coverage
```

## Best Practices

1. **One test file per module**: Keep tests organized with the code they test
2. **Use fixtures**: Reduce code duplication with pytest fixtures
3. **Test all CLI commands**: Each command should have at least one test
4. **Validate outputs**: Use `compare_directories()` for reference comparison
5. **Test help commands**: Simple way to test CLI registration
6. **Keep tests independent**: Each test should be able to run in isolation
