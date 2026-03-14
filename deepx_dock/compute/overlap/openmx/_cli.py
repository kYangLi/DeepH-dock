"""
CLI interface for OpenMX overlap matrix calculation.
"""

import click
from pathlib import Path
import numpy as np

from deepx_dock._cli.registry import register


@register(
    cli_name="calc",
    cli_help="Calculate overlap matrix using OpenMX-style algorithm",
    cli_args=[
        click.argument("data_dir", type=click.Path(exists=True)),
        click.argument("basis_dir", type=click.Path(exists=True)),
        click.option("--cutoff", "-c", type=float, default=15.0, help="Cutoff distance in Angstrom"),
        click.option("--output", "-o", type=click.Path(), default="overlap.h5", help="Output file name"),
        click.option("--spinful", "-s", is_flag=True, help="Consider spin degree of freedom"),
    ],
)
def calc_overlap(data_dir: str | Path, basis_dir: str | Path, cutoff: float, output: str | Path, spinful: bool):
    """
    Calculate overlap matrix from structure and basis.

    Example
    -------
    dock compute overlap openmx calc ./data ./basis -c 10.0 -o overlap.h5
    """
    from deepx_dock.compute.overlap.openmx import CPP_AVAILABLE
    from deepx_dock.misc import load_poscar_file
    import h5py

    data_dir = Path(data_dir)
    basis_dir = Path(basis_dir)
    output = Path(output)

    if not CPP_AVAILABLE:
        click.echo("Warning: C++ extension not available.", err=True)
        click.echo(
            "Please build it first:\n  cd deepx_dock/compute/overlap/openmx/cpp\n  python setup.py build_ext --inplace"
        )
        return

    poscar_file = data_dir / "POSCAR"
    if not poscar_file.exists():
        click.echo(f"Error: POSCAR file not found: {poscar_file}", err=True)
        return

    click.echo(f"Loading structure from {poscar_file}...")

    with open(poscar_file, "r") as f:
        lines = f.readlines()

    comment = lines[0].strip()
    scale = float(lines[1])

    cell = []
    for i in range(2, 5):
        cell.append([float(x) * scale for x in lines[i].split()])
    cell = np.array(cell, dtype=np.float64)

    species_line = lines[5].split()
    num_atoms_line = lines[6].split()

    species_counts = {}
    for i, (symbol, count_str) in enumerate(zip(species_line, num_atoms_line)):
        from deepx_dock.CONSTANT import PERIODIC_TABLE_SYMBOL_TO_INDEX

        atomic_num = PERIODIC_TABLE_SYMBOL_TO_INDEX.get(symbol.capitalize())
        if atomic_num is None:
            click.echo(f"Error: Unknown element symbol: {symbol}", err=True)
            return
        species_counts[atomic_num] = int(count_str)

    positions = []
    species_ids = []

    coord_type = lines[7].strip().lower()[0]

    line_idx = 8
    for atomic_num, count in species_counts.items():
        for _ in range(count):
            coords = [float(x) for x in lines[line_idx].split()[:3]]

            if coord_type == "d":
                pos = np.dot(np.array(coords), cell)
            else:
                pos = np.array(coords) * scale

            positions.append(pos)
            species_ids.append(atomic_num)
            line_idx += 1

    positions = np.array(positions, dtype=np.float64)
    species_ids = np.array(species_ids, dtype=np.int32)

    click.echo(f"  Atoms: {len(positions)}")
    click.echo(f"  Species: {list(species_counts.keys())}")

    from deepx_dock.compute.overlap.openmx.calculator import OverlapCalculator

    click.echo(f"\nInitializing calculator with basis from {basis_dir}...")
    calc = OverlapCalculator(basis_dir)

    calc.set_structure(positions, species_ids, cell)

    basis_names = {}
    for atomic_num in species_counts.keys():
        element_bases = list(basis_dir.glob(f"*.h5"))

        found = False
        for h5_file in element_bases:
            symbol = h5_file.stem
            from deepx_dock.CONSTANT import PERIODIC_TABLE_SYMBOL_TO_INDEX

            if PERIODIC_TABLE_SYMBOL_TO_INDEX.get(symbol) == atomic_num:
                basis_names[atomic_num] = "7.0"
                found = True
                break

        if not found:
            click.echo(f"Warning: No basis found for element {atomic_num}", err=True)

    info_file = data_dir / "info.json"
    if info_file.exists():
        import json

        with open(info_file) as f:
            info = json.load(f)
            if "basis" in info:
                basis_names.update(info["basis"])

    if not basis_names:
        click.echo("Error: No basis sets specified", err=True)
        return

    click.echo(f"Setting basis: {basis_names}")
    calc.set_basis(basis_names)

    click.echo(f"\nComputing overlap matrix (cutoff={cutoff} Å)...")

    try:
        S = calc.compute(cutoff=cutoff)

        click.echo(f"Saving to {output}...")
        output.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output, "w") as f:
            S_dense = S.toarray()
            f.create_dataset("overlap", data=S_dense, compression="gzip")
            f.attrs["total_basis_size"] = calc.total_basis_size
            f.attrs["cutoff"] = cutoff
            f.attrs["num_atoms"] = len(positions)

        click.echo(f"\n[done] Overlap matrix saved to {output}")
        click.echo(f"  Matrix size: {calc.total_basis_size} x {calc.total_basis_size}")
        click.echo(f"  Non-zero elements: {S.nnz}")

    except NotImplementedError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo(
            "\nOverlap computation requires completing the C++ implementation.\n"
            "Current status: Basic structure and mathematical functions implemented.\n"
            "Next steps: Implement k-space integration and angle coupling."
        )
    except Exception as e:
        click.echo(f"\nError during calculation: {e}", err=True)
        import traceback

        traceback.print_exc()
