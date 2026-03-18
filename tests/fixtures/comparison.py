import h5py
import numpy as np
from pathlib import Path
from typing import Union, Tuple


def compare_h5_files(file1: Union[str, Path], file2: Union[str, Path], threshold: float = 1e-14) -> Tuple[bool, str]:
    """
    Compare two HDF5 files

    Returns:
        (is_equal, error_message)
    """
    file1, file2 = Path(file1), Path(file2)

    if not file1.exists():
        return False, f"File {file1} does not exist"
    if not file2.exists():
        return False, f"File {file2} does not exist"

    with h5py.File(file1, "r") as f1, h5py.File(file2, "r") as f2:
        keys1, keys2 = set(f1.keys()), set(f2.keys())
        if keys1 != keys2:
            missing_in_1 = keys2 - keys1
            missing_in_2 = keys1 - keys2
            msg_parts = []
            if missing_in_1:
                msg_parts.append(f"Keys missing in {file1}: {missing_in_1}")
            if missing_in_2:
                msg_parts.append(f"Keys missing in {file2}: {missing_in_2}")
            return False, "; ".join(msg_parts)

        for key in f1.keys():
            data1 = np.array(f1[key][()])
            data2 = np.array(f2[key][()])

            if data1.shape != data2.shape:
                return False, f"Key {key} shape mismatch: {data1.shape} vs {data2.shape}"

            if np.issubdtype(data1.dtype, np.number):
                max_diff = np.max(np.abs(data1 - data2))
                if max_diff > threshold:
                    return False, f"Key {key} values differ: max_diff={max_diff:.2e}"
            else:
                if not np.array_equal(data1, data2):
                    return False, f"Key {key} values differ"

    return True, ""


def compare_dat_files(file1: Union[str, Path], file2: Union[str, Path], threshold: float = 1e-14) -> Tuple[bool, str]:
    """Compare two .dat files"""
    file1, file2 = Path(file1), Path(file2)

    if not file1.exists():
        return False, f"File {file1} does not exist"
    if not file2.exists():
        return False, f"File {file2} does not exist"

    try:
        a1 = np.loadtxt(file1)
        a2 = np.loadtxt(file2)
        max_diff = np.max(np.abs(a1 - a2))
        if max_diff > threshold:
            return False, f"Values differ: max_diff={max_diff:.2e}"
        return True, ""
    except ValueError:
        text1 = file1.read_text().strip()
        text2 = file2.read_text().strip()
        if text1 != text2:
            return False, "Text content differs"
        return True, ""


def compare_text_files(file1: Union[str, Path], file2: Union[str, Path]) -> Tuple[bool, str]:
    """Compare two text files"""
    file1, file2 = Path(file1), Path(file2)

    if not file1.exists():
        return False, f"File {file1} does not exist"
    if not file2.exists():
        return False, f"File {file2} does not exist"

    text1 = file1.read_text()
    text2 = file2.read_text()

    if text1 != text2:
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            if line1 != line2:
                return False, f"Line {i + 1} differs:\n  {line1}\n  {line2}"
        if len(lines1) != len(lines2):
            return False, f"Different number of lines: {len(lines1)} vs {len(lines2)}"
        return False, "Files differ"

    return True, ""


def compare_petsc_files(file1: Union[str, Path], file2: Union[str, Path], threshold: float = 1e-14) -> Tuple[bool, str]:
    """
    Compare two PETSc binary files

    Requires petsc4py to be installed
    """
    try:
        from petsc4py import PETSc
        from scipy.sparse import csr_matrix
    except ImportError:
        return False, "PETSc or scipy not available"

    file1, file2 = Path(file1), Path(file2)

    if not file1.exists():
        return False, f"File {file1} does not exist"
    if not file2.exists():
        return False, f"File {file2} does not exist"

    def load_petsc(filename):
        viewer = PETSc.Viewer().createBinary(filename, mode="r")
        R_mat = PETSc.Mat()
        R_mat.load(viewer)
        indptr, indices, data = R_mat.getValuesCSR()
        data = data.real.astype(np.int32)
        R_array = csr_matrix((data, indices, indptr), shape=R_mat.getSize()).toarray()
        R_mat.destroy()
        blocks = {}
        for key in R_array:
            block_mat = PETSc.Mat()
            block_mat.load(viewer)
            indptr, indices, data = block_mat.getValuesCSR()
            blocks[tuple(key.tolist())] = csr_matrix((data, indices, indptr), shape=block_mat.getSize())
            block_mat.destroy()
        viewer.destroy()
        return blocks

    try:
        data1 = load_petsc(file1)
        data2 = load_petsc(file2)
    except Exception as e:
        return False, f"Failed to load PETSc files: {e}"

    # Compare keys
    keys1, keys2 = set(data1.keys()), set(data2.keys())
    if keys1 != keys2:
        missing_in_1 = keys2 - keys1
        missing_in_2 = keys1 - keys2
        msg_parts = []
        if missing_in_1:
            msg_parts.append(f"Keys missing in {file1}: {missing_in_1}")
        if missing_in_2:
            msg_parts.append(f"Keys missing in {file2}: {missing_in_2}")
        return False, "; ".join(msg_parts)

    # Compare matrices
    for key in data1.keys():
        matrix1 = data1[key]
        matrix2 = data2[key]

        if matrix1.shape != matrix2.shape:
            return False, f"Key {key} shape mismatch: {matrix1.shape} vs {matrix2.shape}"

        max_diff = np.max(np.abs(matrix1 - matrix2))
        if max_diff > threshold:
            return False, f"Key {key} values differ: max_diff={max_diff:.2e}"

    return True, ""


def compare_directories(dir1: Union[str, Path], dir2: Union[str, Path], threshold: float = 1e-14) -> Tuple[bool, list]:
    """
    Compare all files in two directories recursively

    Returns:
        (all_equal, error_messages)
    """
    dir1, dir2 = Path(dir1), Path(dir2)
    errors = []

    if not dir1.exists():
        return False, [f"Directory {dir1} does not exist"]
    if not dir2.exists():
        return False, [f"Directory {dir2} does not exist"]

    files1 = sorted([p.relative_to(dir1) for p in dir1.rglob("*") if p.is_file()])
    files2 = sorted([p.relative_to(dir2) for p in dir2.rglob("*") if p.is_file()])

    if files1 != files2:
        missing_in_1 = set(files2) - set(files1)
        missing_in_2 = set(files1) - set(files2)
        if missing_in_1:
            errors.append(f"Files missing in {dir1}: {missing_in_1}")
        if missing_in_2:
            errors.append(f"Files missing in {dir2}: {missing_in_2}")
        return False, errors

    for rel_file in files1:
        file1 = dir1 / rel_file
        file2 = dir2 / rel_file

        suffix = rel_file.suffix
        if suffix == ".h5":
            is_equal, msg = compare_h5_files(file1, file2, threshold)
        elif suffix == ".dat":
            is_equal, msg = compare_dat_files(file1, file2, threshold)
        elif suffix == ".petsc":
            is_equal, msg = compare_petsc_files(file1, file2, threshold)
        else:
            is_equal, msg = compare_text_files(file1, file2)

        if not is_equal:
            errors.append(f"{rel_file}: {msg}")

    return len(errors) == 0, errors
