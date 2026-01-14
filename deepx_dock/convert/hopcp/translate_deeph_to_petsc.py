from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py

from functools import partial
from joblib import Parallel, delayed

from scipy.sparse import csr_matrix

try:
    from petsc4py import PETSc
except Exception as e:
    print("[error] The petsc4py is not well installed.")

from deepx_dock.misc import load_json_file, dump_toml_file
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME
from deepx_dock.CONSTANT import EXTREMELY_SMALL_FLOAT
from deepx_dock.misc import get_data_dir_lister

DEEPX_NECESSARY_FILES = {DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME}

PETSC_HAMILTONIAN_FILENAME = "HR.petsc"
PETSC_OVERLAP_FILENAME = "SR.petsc"
PETSC_INFO_FILENAME = "misc.toml"


def validation_check_deeph(root_dir: Path, prev_dirname: Path):
    all_files = {str(v.name) for v in root_dir.iterdir()}
    if DEEPX_NECESSARY_FILES.issubset(set(all_files)):
        yield prev_dirname


class DeepHtoPETScTranslator:
    def __init__(
        self, deeph_dir, petsc_dir, export_S=True, export_H=True, n_jobs=1, n_tier=1
    ):
        self.deeph_dir = Path(deeph_dir)
        self.petsc_dir = Path(petsc_dir)
        self.export_S = export_S
        self.export_H = export_H
        self.n_jobs = n_jobs
        self.n_tier = n_tier
        assert self.deeph_dir.is_dir(), f"{deeph_dir} is not a directory"
        self.petsc_dir.mkdir(parents=True, exist_ok=True)

    def transfer_all_deeph_to_petsc(self):
        worker = partial(
            self.transfer_one_deeph_to_petsc,
            deeph_path=self.deeph_dir,
            petsc_path=self.petsc_dir,
            export_S=self.export_S,
            export_H=self.export_H,
        )
        data_dir_lister = get_data_dir_lister(
            self.deeph_dir, self.n_tier, validation_check_deeph
        )
        Parallel(n_jobs=self.n_jobs)(
            delayed(worker)(dir_name) for dir_name in tqdm(data_dir_lister, desc="Data")
        )

    @staticmethod
    def transfer_one_deeph_to_petsc(
        dir_name: str, deeph_path: Path, petsc_path: Path, export_S=True, export_H=True
    ):
        try:
            deeh_dir_path = deeph_path / dir_name
            if not deeh_dir_path.is_dir():
                return
            petsc_dir_path = petsc_path / dir_name
            petsc_dir_path.mkdir(parents=True, exist_ok=True)
            #
            writer = PETScWriter(deeh_dir_path, petsc_dir_path)
            writer.dump_data(export_S=export_S, export_H=export_H)
        except Exception as e:
            print(f"Error in {dir_name}: {e}")


class PETScWriter:
    def __init__(self, deeph_path, petsc_path):
        self.deeph_path = Path(deeph_path)
        self.petsc_path = Path(petsc_path)

    def dump_data(self, export_S=True, export_H=True):
        self._read_poscar()
        self._read_info()
        self.R_set = None
        if export_S:
            self._read_matrix("overlaps", DEEPX_OVERLAP_FILENAME)
            self._translate_matrix("overlaps")
            self._dump_matrix("overlaps", PETSC_OVERLAP_FILENAME)
        if export_H:
            self._read_matrix("hamiltonians", DEEPX_HAMILTONIAN_FILENAME)
            self._translate_matrix("hamiltonians", consider_spin=self.spinful)
            self._dump_matrix("hamiltonians", PETSC_HAMILTONIAN_FILENAME)
        # Dump data
        self._dump_info_toml()
        self._dump_poscar()

    def _read_poscar(self):
        poscar_path = self.deeph_path / DEEPX_POSCAR_FILENAME
        with open(poscar_path, "r") as f:
            lines = f.readlines()
        # system name.
        self.system_name = lines[0].strip()
        # Scale for lattice vector and cartesian coordinates
        scale = [float(x) for x in lines[1].split()]
        scale = [scale[0], scale[0], scale[0]] if len(scale) == 1 else scale
        assert scale[0] > 0.0 and scale[1] > 0.0 and scale[2] > 0.0, (
            f"in {poscar_path}, the second line must be positive, but got {scale}."
        )
        # Lattice vector
        self.lattice = np.array(
            [line.split()[:3] for line in lines[2:5]], dtype=np.float64
        )
        self.lattice = np.array([vec * s for vec, s in zip(self.lattice, scale)])
        # Element symbols and number of atoms
        self.elem_symbols_unique = lines[5].split()
        self.elem_counts = [int(num) for num in lines[6].split()]
        assert len(self.elem_symbols_unique) == len(self.elem_counts), (
            f"in {poscar_path}, the 6th line (element symbols) must has the same length as the 7th line (number of atoms), but got {len(self.elem_symbols_unique)} and {len(self.elem_counts)}."
        )
        self.atom_num = sum(self.elem_counts)
        # Cartesian or Direct coordinates
        coords_mode = (
            "Cartesian"
            if (lines[7][0].lower() == "c" or lines[7][0].lower() == "k")
            else "Direct"
        )
        # Coordinates
        assert len(lines) >= 8 + self.atom_num, (
            f"in {poscar_path}, the number of lines must be at least 8 + {self.atom_num}, but got {len(lines)}."
        )
        self.coords = np.array(
            [line.split()[:3] for line in lines[8 : 8 + self.atom_num]],
            dtype=np.float64,
        )
        if coords_mode == "Cartesian":
            self.coords *= scale
        else:
            self.coords = self.coords @ self.lattice

    def _read_info(self):
        info_path = self.deeph_path / DEEPX_INFO_FILENAME
        info = load_json_file(info_path)
        self.spinful = info["spinful"]
        self.orbits_quantity = info["orbits_quantity"]
        self.fermi_energy_eV = info["fermi_energy_eV"]
        self.elements_orbital_dict = info["elements_orbital_map"]
        element_orbital_counts = np.array(
            [
                sum(self.elements_orbital_dict[element]) * 2
                + len(self.elements_orbital_dict[element])
                for element in self.elem_symbols_unique
            ]
        )
        self.atom_orbital_counts = np.repeat(element_orbital_counts, self.elem_counts)
        self.atom_orbital_cumsum = np.concatenate(
            (np.array([0]), np.cumsum(self.atom_orbital_counts))
        )

    def _read_matrix(self, item, filename: str):
        file_path = self.deeph_path / filename
        with h5py.File(file_path, "r") as f:
            atom_pairs = np.array(f["atom_pairs"][:], dtype=np.int64)
            chunk_boundaries = np.array(f["chunk_boundaries"][:], dtype=np.int64)
            chunk_shapes = np.array(f["chunk_shapes"][:], dtype=np.int64)
            entries = np.array(f["entries"][:])
        value = {
            tuple(map(int, atom_pairs[i, :])): entries[
                chunk_boundaries[i] : chunk_boundaries[i + 1]
            ].reshape(chunk_shapes[i])
            for i in range(atom_pairs.shape[0])
        }
        setattr(self, item, value)
        if not self.R_set:
            self.R_set = np.unique(atom_pairs[:, :3], axis=0)
            self.R_set = set(map(tuple, self.R_set.tolist()))

    def _translate_matrix(self, item, consider_spin=False):
        value = getattr(self, item)
        matrix_dim = self.orbits_quantity * (self.spinful + 1)
        rows = {R: [] for R in self.R_set}
        cols = {R: [] for R in self.R_set}
        data = {R: [] for R in self.R_set}
        small_value = 10 * EXTREMELY_SMALL_FLOAT
        for key, matrix in value.items():
            R = tuple(key[:3])
            ia, ja = key[3], key[4]
            i_start_up = self.atom_orbital_cumsum[ia]
            i_end_up = self.atom_orbital_cumsum[ia + 1]
            j_start_up = self.atom_orbital_cumsum[ja]
            j_end_up = self.atom_orbital_cumsum[ja + 1]
            rows_up, cols_up = np.meshgrid(
                np.arange(i_start_up, i_end_up),
                np.arange(j_start_up, j_end_up),
                indexing="ij",
            )
            rows_up = rows_up.ravel()
            cols_up = cols_up.ravel()

            matrix = matrix.astype(np.complex128)
            non_zero_indices = np.where(np.abs(matrix) > small_value)
            non_zero_values = matrix[non_zero_indices]
            if len(non_zero_values) == 0:
                continue

            local_rows, local_cols = non_zero_indices
            block_rows = self.atom_orbital_counts[ia]
            block_cols = self.atom_orbital_counts[ja]

            if self.spinful:
                i_start_down = i_start_up + self.orbits_quantity
                i_end_down = i_end_up + self.orbits_quantity
                j_start_down = j_start_up + self.orbits_quantity
                j_end_down = j_end_up + self.orbits_quantity
                rows_down, cols_down = np.meshgrid(
                    np.arange(i_start_down, i_end_down),
                    np.arange(j_start_down, j_end_down),
                    indexing="ij",
                )
                rows_down = rows_down.ravel()
                cols_down = cols_down.ravel()

                if consider_spin:
                    # uu
                    mask_uu = (local_rows < self.atom_orbital_counts[ia]) & (
                        local_cols < self.atom_orbital_counts[ja]
                    )
                    r_local = local_rows[mask_uu]
                    c_local = local_cols[mask_uu]
                    flat_indices = r_local * block_cols + c_local
                    rows[R].extend(rows_up[flat_indices])
                    cols[R].extend(cols_up[flat_indices])
                    data[R].extend(non_zero_values[mask_uu])

                    # ud
                    mask_ud = (local_rows < self.atom_orbital_counts[ia]) & (
                        local_cols >= self.atom_orbital_counts[ja]
                    )
                    r_local = local_rows[mask_ud]
                    c_local = local_cols[mask_ud] - block_cols
                    flat_indices = r_local * block_cols + c_local
                    rows[R].extend(rows_up[flat_indices])
                    cols[R].extend(cols_down[flat_indices])
                    data[R].extend(non_zero_values[mask_ud])

                    # du
                    mask_du = (local_rows >= self.atom_orbital_counts[ia]) & (
                        local_cols < self.atom_orbital_counts[ja]
                    )
                    r_local = local_rows[mask_du] - block_rows
                    c_local = local_cols[mask_du]
                    flat_indices = r_local * block_cols + c_local
                    rows[R].extend(rows_down[flat_indices])
                    cols[R].extend(cols_up[flat_indices])
                    data[R].extend(non_zero_values[mask_du])

                    # dd
                    mask_dd = (local_rows >= self.atom_orbital_counts[ia]) & (
                        local_cols >= self.atom_orbital_counts[ja]
                    )
                    r_local = local_rows[mask_dd] - block_rows
                    c_local = local_cols[mask_dd] - block_cols
                    flat_indices = r_local * block_cols + c_local
                    rows[R].extend(rows_down[flat_indices])
                    cols[R].extend(cols_down[flat_indices])
                    data[R].extend(non_zero_values[mask_dd])
                else:
                    flat_indices = local_rows * block_cols + local_cols
                    rows[R].extend(rows_up[flat_indices])
                    cols[R].extend(cols_up[flat_indices])
                    data[R].extend(non_zero_values)
                    rows[R].extend(rows_down[flat_indices])
                    cols[R].extend(cols_down[flat_indices])
                    data[R].extend(non_zero_values)
            else:
                flat_indices = local_rows * block_cols + local_cols
                rows[R].extend(rows_up[flat_indices])
                cols[R].extend(cols_up[flat_indices])
                data[R].extend(non_zero_values)

        value_R = {}
        for R in self.R_set:
            if len(data[R]) == 0:
                value_R[R] = csr_matrix((matrix_dim, matrix_dim), dtype=np.complex128)
            else:
                value_R[R] = csr_matrix(
                    (data[R], (rows[R], cols[R])),
                    shape=(matrix_dim, matrix_dim),
                    dtype=np.complex128,
                )

        setattr(self, item, value_R)

    def _dump_matrix(self, item, filename: str):
        value = getattr(self, item)
        file_path = self.petsc_path / filename
        viewer = PETSc.Viewer().createBinary(str(file_path), mode="w")
        R_array = np.array(list(self.R_set), dtype=np.int64)
        R_array = PETScWriter._get_petsc_dense_from_ndarray(R_array)
        R_array.view(viewer)

        for block in value.values():
            block = PETScWriter._get_petsc_csr_from_scipy(block)
            block.view(viewer)

    def _dump_info_toml(self):
        info_path = self.petsc_path / PETSC_INFO_FILENAME

        spin_info = 3 if self.spinful else 0

        orbits_type_out = "{ "
        for element, count in zip(self.elem_symbols_unique, self.elem_counts):
            orbits_type = self.elements_orbital_dict[element]
            elem_orbitals_type = (
                "{ " + " ".join(map(lambda s: str(s) + ",", orbits_type)) + " }, "
            )
            orbits_type_out += elem_orbitals_type * count
        orbits_type_out += "}"

        misc = {
            "is_basis_orthogonal": 0,
            "spin_info": spin_info,
            "R_quantity": len(self.R_set),
            "matrix_size": self.orbits_quantity * (self.spinful + 1),
            "fermi_energy": self.fermi_energy_eV,
            "orbits_type": orbits_type_out,
        }
        dump_toml_file(info_path, {"misc": misc})

    def _dump_poscar(self):
        poscar_path = self.petsc_path / DEEPX_POSCAR_FILENAME
        direct_coords = self.coords @ np.linalg.inv(self.lattice)
        poscar = [
            "POSCAR generated by DeepH-dock \n",
            "1.0\n",
            "  " + " ".join(map(str, self.lattice[0])) + "\n",
            "  " + " ".join(map(str, self.lattice[1])) + "\n",
            "  " + " ".join(map(str, self.lattice[2])) + "\n",
            " ".join(self.elem_symbols_unique) + "\n",
            " " + " ".join(map(str, self.elem_counts)) + "\n",
            "Direct\n",
        ] + [
            "  " + " ".join(map(str, direct_coords[i, :])) + "\n"
            for i in range(self.atom_num)
        ]
        with open(poscar_path, "w") as f:
            f.writelines(poscar)

    @staticmethod
    def _get_petsc_dense_from_ndarray(dense_matrix: np.ndarray):
        rows, cols = dense_matrix.shape
        petsc_matrix = PETSc.Mat().create()
        petsc_matrix.setSizes([rows, cols])
        petsc_matrix.setType(PETSc.Mat.Type.DENSE)
        petsc_matrix.setUp()

        for i in range(rows):
            petsc_matrix.setValues(i, range(cols), dense_matrix[i, :])
        petsc_matrix.assemble()

        return petsc_matrix

    @staticmethod
    def _get_petsc_csr_from_scipy(sparse_matrix: csr_matrix):
        petsc_matrix = PETSc.Mat().create()
        petsc_matrix.setSizes(sparse_matrix.shape)
        petsc_matrix.setType(PETSc.Mat.Type.AIJ)
        petsc_matrix.setUp()

        data = sparse_matrix.data
        indices = sparse_matrix.indices
        indptr = sparse_matrix.indptr

        petsc_matrix.setValuesCSR(indptr, indices, data)
        petsc_matrix.assemble()

        return petsc_matrix
