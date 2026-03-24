#!/usr/bin/env python3
"""
Test script for OpenMX basis standardization and overlap calculation.

This script demonstrates:
1. Parsing OpenMX PAO files
2. Converting to standardized HDF5 format
3. Parsing OpenMX input files
4. (Requires MPI) Computing overlap matrices

Usage:
    python test_openmx_overlap.py
"""

from pathlib import Path
import sys


def test_basis_conversion():
    """Test PAO → HDF5 conversion"""
    print("=" * 60)
    print("TEST 1: PAO to HDF5 Conversion")
    print("=" * 60)

    from deepx_dock.convert.openmx.basis_convert import parse_openmx_pao, save_basis_to_hdf5, parse_basis_definition

    test_pao = Path("/home/deeph/software/calc/OpenMX/build/openmx3.9/DFT_DATA19/PAO/Fe5.5H.pao")

    if not test_pao.exists():
        print(f"⚠️  Test file not found: {test_pao}")
        return False

    print(f"\nParsing: {test_pao.name}")
    data = parse_openmx_pao(test_pao)

    print(f"✓ Element: {data['element']} (Z={data['atomic_number']})")
    print(f"✓ Lmax: {data['lmax']}, Mul_max: {data['mul_max']}")
    print(f"✓ Radial cutoff: {data['radial_cutoff']} Bohr")
    print(f"✓ Grid points: {data['grid_num']}")
    print(f"✓ Orbitals: {len(data['orbitals'])} L blocks")

    for L in sorted(data["orbitals"].keys()):
        print(f"  - L={L}: {len(data['orbitals'][L])} orbitals")

    output_h5 = Path("/tmp/Fe5.5H.h5")
    save_basis_to_hdf5(data, output_h5)
    print(f"\n✓ Saved to: {output_h5}")

    print("\n" + "=" * 60)
    print("TEST 1: PASSED ✓")
    print("=" * 60)
    return True


def test_basis_definition_parsing():
    """Test basis definition string parsing"""
    print("\n" + "=" * 60)
    print("TEST 2: Basis Definition Parsing")
    print("=" * 60)

    from deepx_dock.convert.openmx.basis_convert import parse_basis_definition

    test_cases = [
        ("Fe6.0H-s2p2d2", "Fe6.0H", {0: 2, 1: 2, 2: 2}),
        ("C7.0-s2p1d1", "C7.0", {0: 2, 1: 1, 2: 1}),
        ("Fe6.0H", "Fe6.0H", {}),
    ]

    all_passed = True
    for input_str, expected_name, expected_sel in test_cases:
        name, sel = parse_basis_definition(input_str)
        passed = name == expected_name and sel == expected_sel

        status = "✓" if passed else "✗"
        print(f"{status} {input_str} -> name={name}, sel={sel}")

        if not passed:
            print(f"  Expected: name={expected_name}, sel={expected_sel}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("TEST 2: PASSED ✓")
    else:
        print("TEST 2: FAILED ✗")
    print("=" * 60)
    return all_passed


def test_openmx_input_parsing():
    """Test OpenMX input file parsing"""
    print("\n" + "=" * 60)
    print("TEST 3: OpenMX Input Parsing")
    print("=" * 60)

    from deepx_dock.compute.overlap.openmx.parse_input import parse_openmx_input, openmx_input_to_structure

    test_input = Path("/home/deeph/software/calc/OpenMX/build/openmx3.9/work/Fe_Bulk_jx.dat")

    if not test_input.exists():
        print(f"⚠️  Test file not found: {test_input}")
        return False

    print(f"\nParsing: {test_input.name}")
    data = parse_openmx_input(test_input)

    print(f"✓ Species definitions: {len(data['species_definition'])}")
    for species, info in data["species_definition"].items():
        print(f"  - {species}: basis={info['basis_name']}, orb_sel={info['orbital_selection']}")

    print(f"✓ Coordinate unit: {data['coordinate_unit']}")
    print(f"✓ Number of atoms: {len(data['atoms'])}")
    print(f"✓ Lattice shape: {data['lattice'].shape}")

    print("\nConverting to structure:")
    structure_data = openmx_input_to_structure(data)
    print(f"✓ Atomic numbers: {structure_data['atomic_numbers']}")
    print(f"✓ Positions shape: {structure_data['positions_cart'].shape}")
    print(f"✓ Lattice shape: {structure_data['lattice'].shape}")

    print("\n" + "=" * 60)
    print("TEST 3: PASSED ✓")
    print("=" * 60)
    return True


def test_hdf5_loading():
    """Test loading basis.h5 (without HPRO/MPI)"""
    print("\n" + "=" * 60)
    print("TEST 4: HDF5 Structure Verification")
    print("=" * 60)

    import h5py
    import numpy as np

    h5_file = Path("/tmp/Fe5.5H.h5")

    if not h5_file.exists():
        print(f"⚠️  HDF5 file not found: {h5_file}")
        print("   Run test_basis_conversion() first")
        return False

    print(f"\nInspecting: {h5_file}")

    with h5py.File(h5_file, "r") as f:
        print("\n✓ Top-level datasets:")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                print(f"  - {key}: shape={f[key].shape}, dtype={f[key].dtype}")
            else:
                print(f"  - {key}/ (group)")

        print("\n✓ File-level attributes:")
        for key, val in f.attrs.items():
            print(f"  - {key}: {val}")

        print("\n✓ Radial grid:")
        print(
            f"  - radial_grid: shape={f['radial_grid'].shape}, range=[{f['radial_grid'][0]:.6f}, {f['radial_grid'][-1]:.6f}]"
        )

        print("\n✓ Orbital structure:")
        mul_list = f["mul_list"][:]
        print(f"  - mul_list: {mul_list}")
        print(f"  - lmax: {len(mul_list) - 1}")
        print(f"  - total_orbitals: {np.sum(mul_list)}")

        print("\n✓ Radial basis matrix:")
        print(f"  - radial_basis: shape={f['radial_basis'].shape}")

    print("\n" + "=" * 60)
    print("TEST 4: PASSED ✓")
    print("=" * 60)
    return True


def test_calculator_structure():
    """Test calculator module structure (no MPI required)"""
    print("\n" + "=" * 60)
    print("TEST 5: Module Structure Verification")
    print("=" * 60)

    print("\n✓ Checking convert/openmx structure:")

    from pathlib import Path

    convert_dir = Path("/home/deeph/software/calc/DeepH-dock/deepx_dock/convert/openmx")

    expected_convert_files = [
        "__init__.py",
        "basis_convert.py",
        "_cli.py",
    ]

    all_found = True
    for filename in expected_convert_files:
        filepath = convert_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename} exists")
        else:
            print(f"  ✗ {filename} NOT FOUND")
            all_found = False

    print("\n✓ Checking compute/overlap/openmx structure:")

    compute_dir = Path("/home/deeph/software/calc/DeepH-dock/deepx_dock/compute/overlap/openmx")

    expected_compute_files = [
        "__init__.py",
        "_cli.py",
        "loader.py",
        "calculator.py",
        "parse_input.py",
    ]

    for filename in expected_compute_files:
        filepath = compute_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename} exists")
        else:
            print(f"  ✗ {filename} NOT FOUND")
            all_found = False

    print("\n✓ Verifying parse_input.py is in compute (not convert):")
    parse_in_convert = convert_dir / "parse_input.py"
    parse_in_compute = compute_dir / "parse_input.py"

    if not parse_in_convert.exists():
        print(f"  ✓ parse_input.py NOT in convert/openmx (correct)")
    else:
        print(f"  ✗ parse_input.py FOUND in convert/openmx (wrong location)")
        all_found = False

    if parse_in_compute.exists():
        print(f"  ✓ parse_input.py FOUND in compute/overlap/openmx (correct)")
    else:
        print(f"  ✗ parse_input.py NOT in compute/overlap/openmx (wrong)")
        all_found = False

    print("\n" + "=" * 60)
    if all_found:
        print("TEST 5: PASSED ✓")
    else:
        print("TEST 5: FAILED ✗")
    print("=" * 60)
    return all_found


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("OpenMX Basis Standardization Test Suite")
    print("=" * 60)

    tests = [
        test_basis_definition_parsing,
        test_basis_conversion,
        test_openmx_input_parsing,
        test_hdf5_loading,
        test_calculator_structure,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(results)

    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    if all(results):
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
