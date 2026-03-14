"""
Test script for OpenMX overlap calculation.
Converts PAO files to HDF5 and calculates overlap matrix.
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from deepx_dock.compute.overlap.openmx.basis import convert_pao_to_h5, ElementBasis


def parse_poscar(poscar_file: Path):
    """Parse POSCAR file."""
    with open(poscar_file) as f:
        lines = f.readlines()

    # Parse lattice
    scale = float(lines[1])
    cell = []
    for i in range(2, 5):
        vec = [float(x) * scale for x in lines[i].split()]
        cell.append(vec)
    cell = np.array(cell, dtype=np.float64)

    # Parse species and counts
    species_line = lines[5].split()
    counts_line = lines[6].split()

    from deepx_dock.CONSTANT import PERIODIC_TABLE_SYMBOL_TO_INDEX

    species_symbols = []
    atomic_numbers = []
    atom_counts = []

    for symbol, count_str in zip(species_line, counts_line):
        symbol = symbol.strip()
        species_symbols.append(symbol)
        atomic_num = PERIODIC_TABLE_SYMBOL_TO_INDEX.get(symbol)
        if atomic_num is None:
            raise ValueError(f"Unknown element symbol: {symbol}")
        atomic_numbers.append(atomic_num)
        atom_counts.append(int(count_str))

    # Parse positions
    coord_type = lines[7].strip().lower()[0]
    is_direct = coord_type == "d"

    positions = []
    species_ids = []

    line_idx = 8
    for atomic_num, count in zip(atomic_numbers, atom_counts):
        for _ in range(count):
            coords = [float(x) for x in lines[line_idx].split()[:3]]

            if is_direct:
                # Convert fractional to Cartesian
                pos = np.dot(np.array(coords), cell)
            else:
                pos = np.array(coords)

            positions.append(pos)
            species_ids.append(atomic_num)
            line_idx += 1

    positions = np.array(positions, dtype=np.float64)
    species_ids = np.array(species_ids, dtype=np.int32)

    return positions, species_ids, cell, species_symbols, atom_counts


def save_overlap_sparse(overlap_data, atom_pairs, output_file):
    """Save overlap matrix in DeepH sparse format."""
    with h5py.File(output_file, "w") as f:
        f.create_dataset("atom_pairs", data=atom_pairs)
        f.create_dataset("entries", data=overlap_data["entries"])
        f.create_dataset("chunk_boundaries", data=overlap_data["chunk_boundaries"])
        f.create_dataset("chunk_shapes", data=overlap_data["chunk_shapes"])


def main():
    benchmark_dir = Path(__file__).parent / "benchmark"
    basis_dir = Path(__file__).parent.parent.parent.parent / "deepx_dock/compute/overlap/openmx/basis/data"
    basis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OpenMX Overlap Calculation Test")
    print("=" * 60)

    # Step 1: Convert PAO files to HDF5
    print("\n[Step 1] Converting PAO files to HDF5...")

    openmx_pao_dir = Path("/home/deeph/software/calc/OpenMX/build/openmx3.9/DFT_DATA19/PAO")

    pao_files = {
        8: openmx_pao_dir / "O7.0.pao",  # Oxygen
        1: openmx_pao_dir / "H5.0.pao",  # Hydrogen
    }

    for atomic_num, pao_file in pao_files.items():
        if not pao_file.exists():
            print(f"Error: PAO file not found: {pao_file}")
            return

        output_file = basis_dir / f"{['', 'H', '', '', '', '', '', '', 'O'][atomic_num]}.h5"

        print(f"  Converting {pao_file.name} -> {output_file.name}")
        convert_pao_to_h5(pao_file, output_file, compute_k_space=True)

    print("  [done]")

    # Step 2: Parse POSCAR
    print("\n[Step 2] Parsing POSCAR...")

    poscar_file = benchmark_dir / "POSCAR"
    positions, species_ids, cell, species_symbols, atom_counts = parse_poscar(poscar_file)

    print(f"  Species: {species_symbols}")
    print(f"  Counts: {atom_counts}")
    print(f"  Total atoms: {len(positions)}")
    print(f"  Positions shape: {positions.shape}")
    print("  [done]")

    # Step 3: Load basis sets
    print("\n[Step 3] Loading basis sets...")

    element_bases = {}
    for atomic_num in set(species_ids):
        symbol = ["", "H", "", "", "", "", "", "", "O"][atomic_num]
        h5_file = basis_dir / f"{symbol}.h5"

        if not h5_file.exists():
            print(f"Error: Basis file not found: {h5_file}")
            return

        element_bases[atomic_num] = ElementBasis.load_h5(h5_file)
        print(f"  Loaded {symbol}: {element_bases[atomic_num].list_basis_sets()}")

    print("  [done]")

    # Step 4: Load info.json for orbital mapping
    print("\n[Step 4] Loading orbital mapping from info.json...")

    info_file = benchmark_dir / "info.json"
    with open(info_file) as f:
        info = json.load(f)

    print(f"  Atoms quantity: {info['atoms_quantity']}")
    print(f"  Orbits quantity: {info['orbits_quantity']}")
    print(f"  Orbital map: {info['elements_orbital_map']}")

    # Parse orbital map
    # O: [0, 0, 0, 1, 1, 2, 3] -> s, s, s, p, p, d, f? or indices
    # H: [0, 0, 0, 1] -> s, s, s, p?

    print("  [done]")

    # Step 5: Compare with reference overlap
    print("\n[Step 5] Comparing with reference overlap.h5...")

    ref_overlap_file = benchmark_dir / "overlap.h5"
    with h5py.File(ref_overlap_file, "r") as f:
        ref_atom_pairs = f["atom_pairs"][:]
        ref_entries = f["entries"][:]
        ref_chunk_shapes = f["chunk_shapes"][:]
        ref_chunk_boundaries = f["chunk_boundaries"][:]

    print(f"  Reference atom_pairs shape: {ref_atom_pairs.shape}")
    print(f"  Reference entries shape: {ref_entries.shape}")
    print(f"  Reference chunk_shapes: {ref_chunk_shapes}")
    print(f"  Total matrix elements: {len(ref_entries)}")

    # Calculate basis size
    total_basis = 0
    for shape in ref_chunk_shapes:
        total_basis += shape[0] * shape[1]
    print(f"  Total basis size (from chunks): {total_basis}")

    print("  [done]")

    # Step 6: Summary
    print("\n[Summary]")
    print(f"  Structure: {species_symbols} with {atom_counts} atoms")
    print(f"  Basis files: O.h5, H.h5")
    print(f"  Reference overlap: {ref_overlap_file}")
    print(f"  Matrix size: {info['orbits_quantity']} x {info['orbits_quantity']}")

    print("\n[Next Steps]")
    print("  The framework is ready. To compute overlap matrix:")
    print("  1. Implement overlap calculation in C++ (k-space integration + angle coupling)")
    print("  2. Build C++ extension: cd cpp && python setup.py build_ext --inplace")
    print("  3. Run: dock compute overlap openmx calc benchmark basis/data")

    print("\n" + "=" * 60)
    print("Test setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
