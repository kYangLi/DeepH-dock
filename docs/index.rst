DeepH-dock
==========

.. div:: sd-text-left sd-font-italic

    *Modular, extensible bridge between first-principles calculations and the DeepH method*


----

`DeepH-dock <https://github.com/kYangLi/DeepH-dock/>`_ is a modular, extensible interface platform for quantum materials calculations, dedicated to building efficient and reliable bridges between first-principles calculations and *the DeepH (deep learning Hamiltonian) method*. This platform integrates multiple density functional theory (DFT) software interfaces, supports DeepH predictions, and provides standardized data processing. **It also functions independently as a post-processing tool for DFT calculations.**

At the core of `DeepH-dock` is **a unified and flexible interface layer that seamlessly connects mainstream DFT packages with the DeepH workflow**, enabling users to generate and utilize deep learning-based Hamiltonians with minimal effort. `DeepH-dock` offers first-class support for heterogeneous computational environments, allowing researchers to orchestrate complex multi-software workflows through a consistent Python API. Designed to significantly lower the technical barrier and enhance reproducibility in large-scale quantum materials simulations, `DeepH-dock` is the product of extensive refinement driven by real-world research needs.

*DeepH-dock* also establishes a unified data format tailored for machine learning in materials science, facilitating efficient implementations of both force fields and electronic structure methods.

Features
^^^^^^^^

.. grid::

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Versatile
            :class-card: sd-border-0
            :shadow: none
            :class-title: sd-fs-5

            .. div:: sd-font-normal

                Seamlessly works with major DFT software (`FHI-aims <https://www.fhi-aims.org>`_, `SIESTA <https://siesta-project.org/>`_, `Quantum ESPRESSO <https://www.quantum-espresso.org>`_, `OpenMX <https://openmx-square.org>`_, etc.), deep learning models (DeepH series), and tight-binding toolchains, offering broad compatibility across computational materials science.

                

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Standardized
            :class-card: sd-border-0
            :shadow: none
            :class-title: sd-fs-5

            .. div:: sd-font-normal

                Establish a unified data specification that serves as the foundation for DeepH calculations while bridging the electronic structure output formats of most mainstream DFT software packages.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: High-Performance
            :class-card: sd-border-0
            :shadow: none
            :class-title: sd-fs-5

            .. div:: sd-font-normal

                Leverages optimized algorithms (e.g., KPM and Lanczos) for rapid matrix operations, automated workflows, and robust Hamiltonian processing. Its utility toolkit further enhances efficiency through multi-level parallelism (MPI/Loky), data conversion, and validation tools.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Collaborative & Open-Sourced
            :class-card: sd-border-0
            :shadow: none
            :class-title: sd-fs-5

            .. div:: sd-font-normal

                DeepH-dock is more than an open-source project, it's a collaborative platform we build together. We invite you to code, discuss, and shape the future of materials computation with us, fostering a vibrant ecosystem where every contributor propels the field forward.

Installation
^^^^^^^^^^^^

Install the latest version from the repository via pip:

.. code-block:: bash

    pip install git+https://github.com/kYangLi/DeepH-dock

For detailed guidance including step-by-step installation, troubleshooting, and development environment setup, please refer to `Installation & Setup <./installation_and_setup.html>`_.



Basic usage
^^^^^^^^^^^

DeepH-dock provides both a command-line interface (CLI) and a Python API for flexible usage.

**1. Command-line interface**

Run the ``dock`` command with the ``-h`` flag to view the help menu and available subcommands:

.. code-block:: bash

    dock -h

.. code-block:: bash

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

**2. Python API**

You can also integrate DeepH-dock directly into your Python scripts. The following example demonstrates how to use the ``Twist2D`` module to construct a twisted bilayer structure:

.. code-block:: python

    from deepx_dock.design.twist_2d.twist import Twist2D
    
    # Initialize a Twist2D object
    twist_2d = Twist2D()
    m, n = 7, 8

    # Create the twisted 2D material
    twist_2d.add_layer([m, n], [-n, m+n], prim_poscar="./POSCAR-C")
    twist_2d.add_layer([n, m], [-m, n+m], prim_poscar="./POSCAR-BN")
    twist_2d.twist_layers()

    # Export the resulting structure to a POSCAR file
    twist_2d.write_res_to_poscar()



Citation
^^^^^^^^

If you use this code in your academic work, please cite the complete package featuring the latest implementation, methodology, and workflow of `DeepH <https://github.com/kYangLi/DeepH-pack-docs>`_:

`Yang Li, Yanzhen Wang, Boheng Zhao, et al. DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations. arXiv:2601.02938 (2026) <https://arxiv.org/abs/2601.02938>`_

.. code-block:: bibtex

    @article{li2026deeph,
        title={DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations},
        author={Li, Yang and Wang, Yanzhen and Zhao, Boheng and Gong, Xiaoxun and Wang, Yuxiang and Tang, Zechen and Wang, Zixu and Yuan, Zilong and Li, Jialin and Sun, Minghui and Chen, Zezhou and Tao, Honggeng and Wu, Baochun and Yu, Yuhang and Li, He and da Jornada, Felipe H. and Duan, Wenhui and Xu, Yong },
        journal={arXiv preprint arXiv:2601.02938},
        year={2026}
    }


----

.. toctree::
    :hidden:
    :maxdepth: 1
    
    installation_and_setup
    basic_usage
    key_concepts
    capabilities/index
    for_developers/index
    citation_and_license
