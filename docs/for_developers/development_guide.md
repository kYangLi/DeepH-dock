# Development Guide

This guide details the technical foundations of `DeepH-dock`, explaining its architecture, data formats, and how to extend it with new functionality. It is intended for developers who wish to integrate new tools or workflows into the platform.

## Architecture Overview

DeepH-dock is built around a modular, CLI-first architecture. All functionalities are organized into primary modules accessible via the `dock` command:

* `analyze`: Data analysis and post-processing tools.
* `compute`: Ab-initio calculation drivers and workflows.
* `convert`: Data format converters between DFT codes and DeepH.
* `design`: Structure generation and manipulation (e.g., twisting, stacking).

Each module contains submodules for specific tasks or software interfaces, ensuring a clean separation of concerns.

## Data Formats & Specifications

A unified data specification is central to DeepH-dock's interoperability. We define a standardized data exchange format and build the complete toolchain around it.

### DeepH Input Format (Core Data Format)

This is the core specification for passing electronic structure data between different computational modules, such as DFT calculations, DeepH training, and inference. It primarily contains Hamiltonian and overlap matrices stored in a specific sparse format, along with corresponding atomic structure information.

**For the detailed technical definition, data structure, and field descriptions of this format, please refer to DeepH-dock [Key Concepts](../key_concepts.md#file-types-and-their-purposes)**.

### Unified Format Conversion Toolset

To integrate the outputs of various DFT software (e.g., VASP, ABACUS, FHI-aims, Quantum ESPRESSO) into the DeepH workflow, DeepH-dock provides a series of **format converters**. Each converter is responsible for parsing the output of its specific software and transforming it into the aforementioned unified **DeepH Input Format**.

**Core Goal & Principle:**
Regardless of the source software, all data will adhere to the same specification after conversion. This ensures consistency, reproducibility, and compatibility throughout the entire DeepH ecosystem's data processing pipeline. Therefore, all newly developed converters must target the output of this unified format.

## Extending DeepH-dock

After forking the source code by following the [Fork and Pull Request Process](./collaboration_guide.md#fork-and-pull-request-process) section in the Contributing Guide, you can set up your development environment by referring to [Install from Source (for Developers)](../installation_and_setup.md#install-from-source-for-developers). Then, you can start developing your feature module.

### Overview: Adding a New Module

To integrate a new DFT package:

1. **Create a Submodule**: Inside `deepx_dock/<module>/`, create a new directory (e.g., `my_dft_code`).
2. **Implement Logic**: Write a Python class/function that parses the native output and returns data in the unified DeepH format.
3. **Register a CLI Command**: This is crucial for user access.

### Registering the CLI Command

To make your new functionality accessible via DeepH-dock's main command-line interface, you must register it with the CLI system. This process is straightforward and follows a consistent pattern across all modules.

### The Registration Process

1. **Create `_cli.py` File**: Inside your module directory, create a file named `_cli.py`.

2. **Use the `@register` Decorator**: Import and use the `@register` decorator from `deepx_dock._cli.registry` to define your command.

3. **Define Command Structure**: Specify the command name, help text, and arguments using the decorator parameters.

### Example Implementation

Here's a complete example for registering a new converter command:

```python
# File: deepx_dock/convert/my_dft_code/_cli.py

import click
from pathlib import Path
from deepx_dock._cli.registry import register

@register(
    cli_name="export",  # The subcommand name
    cli_help="Convert MyDFTCode output to DeepH format.",  # Help text
    cli_args=[  # Command arguments and options
        click.argument('input_dir', type=click.Path(exists=True)),
        click.argument('output_dir', type=click.Path()),
        click.option(
            '--jobs-num', '-j',
            type=int,
            default=-1,
            help='Number of parallel workers (-1 for all cores)'
        ),
        click.option(
            '--precision', default='double', 
            help='Numerical precision (single/double).'
        ),
    ],
)
def my_converter(input_dir: Path, output_dir: Path, jobs_num: int, precision: str):
    """
    Core implementation function.
  
    This function is called when the command is executed.
    It should contain or import the actual conversion logic.
    """
    # Import your implementation (lazy importing for faster CLI load)
    from .core_parser import convert_to_deeph_format
  
    # Call the actual implementation (pass as n_jobs to class)
    convert_to_deeph_format(input_dir, output_dir, n_jobs=jobs_num, precision=precision)
```

### Command Availability

Once registered by `@register`, your command becomes automatically available through the DeepH-dock CLI:

```bash
# Command structure: dock [module] [submodule] [command]
dock convert my-dft-code export <input_dir> <output_dir>

# With options
dock convert my-dft-code export <input_dir> <output_dir> --precision single

# With parallel processing
dock convert my-dft-code export <input_dir> <output_dir> -j 4
```

### Key Components Explained

| Component | Purpose | Example |
| ----------- | --------- | --------- |
| `cli_name` | Defines the subcommand name | `"export"` |
| `cli_help` | Brief description shown in help | `"Convert MyDFTCode output..."` |
| `cli_args` | Arguments and options using `click` | `click.argument`, `click.option` |
| Decorated function | Implementation called when command runs | `my_converter()` |

### Using Click for Argument Definition

DeepH-dock uses the [Click](https://click.palletsprojects.com/) library for CLI construction. Within `cli_args`, you can define:

* **Required arguments**: Use `click.argument()`
* **Optional options**: Use `click.option()`
* **Various types**: `click.Path()`, `click.INT`, `click.FLOAT`, `click.BOOL`
* **Validation**: Add constraints and validation rules

**Example with validation:**

```python
cli_args=[
    click.argument('input_dir', type=click.Path(exists=True, dir_okay=True)),
    click.option(
        '--iterations', 
        type=click.IntRange(1, 1000),
        default=100,
        help='Number of iterations (1-1000)'
    ),
    click.option(
        '--verbose', 
        is_flag=True,
        help='Enable verbose output'
    ),
]
```

### Best Practices

1. **Keep Imports Light**: Place heavy imports inside the function body (as shown in the example) to ensure the CLI loads quickly.

2. **Provide Clear Help**: Write descriptive help text for both the command and its options.

3. **Validate Inputs**: Use Click's built-in validation or add custom validation in your function.

4. **Error Handling**: Include appropriate error messages and exit codes for common failure scenarios.

5. **Testing**: Test your CLI command with various inputs to ensure it behaves correctly.

6. **Follow Parameter Naming Conventions**: DeepH-dock uses a consistent naming convention for parallel processing parameters:

   | Position | Name Pattern | Example |
   |----------|--------------|---------|
   | CLI option | `--jobs-num`, `--tier-num` | `-j 4 -t 0` |
   | CLI function parameter | `jobs_num`, `tier_num` | `def cmd(..., jobs_num: int)` |
   | Class `__init__` parameter | `n_jobs`, `n_tier` | `def __init__(self, ..., n_jobs=-1)` |
   | Passing to class | `n_jobs=jobs_num` | `Class(..., n_jobs=jobs_num)` |

   This ensures consistency across the codebase and follows industry standards (`-j` like `make -j`).

By following this registration pattern, your module will seamlessly integrate with the DeepH-dock command-line ecosystem, making it easily discoverable and usable by other researchers.

## Documentation with Jupyter Notebooks

We strongly encourage contributors to include Jupyter notebook in [`examples/`](https://github.com/kYangLi/DeepH-dock/tree/main/examples) for new features. These notebooks serve multiple purposes:

1. **Live Documentation**: All Jupyter notebook (`.ipynb`) files are automatically converted into live documentation.
2. **Functional Tests**: Validate that your implementation works as expected.
3. **Educational Resources**: Help users understand the capabilities and best practices.

These notebooks are executed as part of our CI/CD pipeline, ensuring they remain up-to-date and functional. When contributing new features, we encourage adding corresponding example notebooks that demonstrate usage and serve as functional tests (see the [Documentation with Jupyter Notebooks, Development Guide](./for_developers/development_guide.md#documentation-with-jupyter-notebooks)).

### Notebook Requirements

When submitting a new feature, please include a Jupyter notebook in the [`examples/`](https://github.com/kYangLi/DeepH-dock/tree/main/examples) directory with the following characteristics:

* **Clear Structure**: Start with a brief introduction explaining what the notebook demonstrates.
* **Minimal Working Example**: Include a complete, runnable example that users can adapt.
* **Explanatory Comments**: Use markdown cells to explain each step.
* **Visualizations**: When applicable, include plots or visual outputs.
* **Testing Cells**: Include assertions or checks to verify the outputs are correct.

### CI/CD Integration

Our documentation system automatically renders these notebooks and includes them in the online documentation. The CI/CD pipeline will:

1. Convert them to HTML/PDF formats for documentation.
2. Embed them in the appropriate sections of the documentation website.

For an example of a well-documented notebook, see the existing examples in the [`examples/`](https://github.com/kYangLi/DeepH-dock/tree/main/examples) directory.

## Best Practices & Conventions

* **Testing**: Systematic testing is achieved through executable code cells in Jupyter notebooks within the [`examples/`](https://github.com/kYangLi/DeepH-dock/tree/main/examples) directory. The CI/CD pipeline automatically runs these notebooks, validating the functionality and serving as live documentation.
* **Documentation**: Include docstrings for all public functions and classes. Update the main documentation if needed.
* **Dependencies**: Keep new dependencies minimal. If essential, add them to `pyproject.toml` with appropriate version ranges.

## Next Steps

After familiarizing yourself with these concepts, you are ready to contribute code. Please follow the collaborative process outlined in the [Collaboration Guide](./collaboration_guide.md) to submit your changes.
