<!-- markdownlint-disable MD033 MD036 -->
<h1><p align="center">
  <img src="./docs/_image/logo-large.svg" alt="DeepH-dock Logo" width="500">
</p></h1>

<div align="center">

### Integrating deep learning into first-principles calculations

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![GitHub Issues](https://img.shields.io/github/issues/kYangLi/DeepH-dock.svg)](https://github.com/kYangLi/DeepH-dock/issues)
[![GitHub Stars](https://img.shields.io/github/stars/kYangLi/DeepH-dock.svg?style=social)](https://github.com/kYangLi/DeepH-dock/stargazers)

*Modular, extensible bridge between first-principles calculations and the DeepH method*
</div>

*DeepH-dock* is a modular, extensible interface platform for quantum materials calculations, dedicated to building efficient and reliable bridges between first-principles calculations and the *DeepH (deep learning Hamiltonian) method*. This platform integrates multiple density functional theory (DFT) software interfaces, supports DeepH predictions, and provides standardized data processing. **It also functions independently as a post-processing tool for DFT calculations.**

At the core of *DeepH-dock* is **a unified and flexible interface layer that seamlessly connects mainstream DFT packages with the DeepH workflow**, enabling users to generate and utilize deep learning-based Hamiltonians with minimal effort. DeepH-dock offers first-class support for heterogeneous computational environments, allowing researchers to orchestrate complex multi-software workflows through a consistent Python API.

*DeepH-dock* also establishes [a unified data format tailored for machine learning in materials science](https://docs.deeph-pack.com/deeph-dock/en/latest/key_concepts.html), facilitating efficient implementations of both force fields and electronic structure methods.

For the most comprehensive usage documentation, please visit [https://docs.deeph-pack.com/deeph-dock](https://docs.deeph-pack.com/deeph-dock/en/latest/).

For the latest version-specific documentation of DeepH-pack, please refer to the [corresponding repository](https://github.com/kYangLi/DeepH-pack-docs).

---

- [Core Features](#core-features)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
- [Citation](#citation)
- [Application Scenarios](#application-scenarios)
- [Contributing](#contributing)
  - [Development Workflow for Contributors](#development-workflow-for-contributors)
- [License](#license)
- [Support \& Contact](#support--contact)

## Core Features

- **Multi-Software Compatibility:** Seamlessly bridges major DFT packages (e.g., SIESTA, QE, OpenMX, FHI-aims, ABACUS) with the DeepH deep learning family and downstream tight-binding toolchains.

- **High-Performance Computational Core:** Implements high-performance algorithms for automated structure generation, overlap matrix calculation, Hamiltonian symmetrization, and fast diagonalization via KPM/Lanczos.

- **Standardized Workflows:** Streamlines the full research lifecycle with automated pipelines for DFT data generation, preprocessing, post-analysis, and custom composite workflows.

- **Utility Toolkit:** Empowers developers with a robust set of tools for multi-level parallel computing (MPI/Loky), versatile format conversion, and rigorous data integrity validation.

## Quick Start

### Installation

```bash
pip install git+https://github.com/kYangLi/DeepH-dock
```

### Basic Usage

run `dock -h`,

```bash
Usage: dock [OPTIONS] COMMAND [ARGS]...

  DeepH-dock: Materials Computation and Data Analysis Toolkit.

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  analyze
  compute
  convert
  design
  ls       List all available commands.
```

## Citation

*Any and all use of this software, in whole or in part, should clearly acknowledge and link to this repository.*

If you use this code in your academic work, please cite **the complete package featuring the latest implementation, methodology, and workflow of [DeepH](https://github.com/kYangLi/DeepH-pack-docs)**:

[Yang Li, Yanzhen Wang, Boheng Zhao, *et al*. DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations. arXiv:2601.02938 (2026)](https://arxiv.org/abs/2601.02938)

```bibtex
@article{li2026deeph,
    title={DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations},
    author={Li, Yang and Wang, Yanzhen and Zhao, Boheng and Gong, Xiaoxun and Wang, Yuxiang and Tang, Zechen and Wang, Zixu and Yuan, Zilong and Li, Jialin and Sun, Minghui and Chen, Zezhou and Tao, Honggeng and Wu, Baochun and Yu, Yuhang and Li, He and da Jornada, Felipe H. and Duan, Wenhui and Xu, Yong },
    journal={arXiv preprint arXiv:2601.02938},
    year={2026}
}
```

## Application Scenarios

- **High-Throughput Materials Calculations**: Automated generation and processing of electronic structure data for large material sets
- **Deep Learning Training Data Preparation**: Preparing standardized training datasets for neural network models
- **Tight-Binding Parameter Fitting**: Connecting DFT calculations with tight-binding model parameterization
- **Method Development and Validation**: Rapid implementation and testing of new computational algorithms

## Contributing

We welcome contributions from the community! To ensure a consistent and maintainable codebase, please follow the structured workflow below when adding new functionality.

### Development Workflow for Contributors

A brief summary is provided below. Complete documentation can be found in the [Development Guide](https://docs.deeph-pack.com/deeph-dock/en/latest/for_developers/development_guide.html).

1. **Identify the Target Module**

    First, determine which major module your contribution belongs to, based on the project architecture (e.g., `analyze`, `compute`, `convert`, `design`).

2. **Create Your Submodule**

    Create a new folder for your submodule within the corresponding parent module. For example, if you are implementing an interface between FHI-aims and DeepH, create it under `deepx_dock/convert/fhi_aims/`.

3. **Implement Your Functionality**

    Package the core logic you wish to add as a well-defined Python class or function. This makes the code reusable and easier to test.

4. **Register a Command-Line Interface (CLI)**

    **This is the most important step.** To expose your functionality through the main `dock` command, you must register your function inside the `_cli.py` file within your submodule directory.

    In this file, define your CLI command using the `click` library and the provided `@register` decorator.

**Example:**
If you create `deepx_dock/convert/fhi_aims/_cli.py` with the following content:

```python
import click
from pathlib import Path
from deepx_dock._cli.registry import register

@register(
    cli_name="my-func",
    cli_help="A brief description of what my-func does.",
    cli_args=[
        click.argument('fhiaims_dir', type=click.Path()),
        click.argument('deeph_dir', type=click.Path()),
        click.option(
            '--parallel-num', '-p', type=int, default=-1,
            help="The parallel processing number, -1 for using all of the cores."
        ),
    ],
)
def my_func(fhiaims_dir: Path, deeph_dir: Path, parallel_num: int):
    # Your implementation logic here
    from .my_wanderful_function import my_real_func
    my_real_func(fhiaims_dir, deeph_dir, parallel_num)
```

Then, on the user end, a new command will be available:

```bash
dock convert fhi-aims my-func
```

The command's usage (arguments, options, help text) is entirely defined by the `click` decorators specified in `cli_args`. Please refer to the [official click documentation](https://click.palletsprojects.com/) for details on defining arguments and options.

## License

This project is licensed under the GPL 3.0 License - see the [LICENSE](LICENSE) file for details.

## Support & Contact

- 📖 **Documentation**: [Documentation Link](https://docs.deeph-pack.com/deeph-dock/en/latest/)
- 🐛 **Issue Reporting**: [GitHub Issues](https://github.com/kYangLi/DeepH-dock/issues)
<!-- - 💬 **Discussions**: [GitHub Discussions]() -->

---

*DeepH-dock is community-driven development for quantum materials calculations, aiming to promote openness and reproducibility in computational materials science research.*
