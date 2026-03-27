import struct
import numpy as np
import h5py
from pathlib import Path
from functools import partial

from deepx_dock.parallel import parallel_map
from deepx_dock.CONSTANT import DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME, DEEPX_PREDICT_HAMILTONIAN_FILENAME
from deepx_dock.misc import get_data_dir_lister, load_json_file, load_poscar_file
from deepx_dock.convert.deeph.translate_old_dataset_to_new import BASIS_TRANS_WIKI2OPENMX
from deepx_dock.convert.openmx.translate_openmx_to_deeph import OPENMX_SCFOUT_FILENAME
from deepx_dock.convert.openmx.translate_openmx_to_deeph import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from deepx_dock.convert.openmx.translate_openmx_to_deeph import BinaryFileReader


def validation_check_scfout(root_dir: Path, prev_dirname: Path):
    all_files = [str(v.name) for v in root_dir.iterdir()]
    if OPENMX_SCFOUT_FILENAME in all_files:
        yield prev_dirname


class DeepHToOpenMXTranslator:
    """
    Translator for converting DeepH Hamiltonians to OpenMX scfout format.

    This class replaces Hamiltonian matrices in scfout files with DeepH predictions.

    Args:
        openmx_data_dir (str or Path): Root directory containing subdirectories with
                                  openmx.scfout files
        deeph_data_dir (str or Path): Root directory containing subdirectories with
                                 hamiltonian.h5 files
        output_dir (str or Path): Output directory for new scfout files
        n_jobs (int, optional): Number of parallel jobs. Default: 1.
        n_tier (int, optional): Number of tiers of the dataset. Default: 0.

    Examples:
        >>> translator = DeepHToOpenMXTranslator(
        ...     openmx_data_dir="./openmx_calc",
        ...     deeph_data_dir="./deeph_output",
        ...     output_dir="./new_scfout",
        ...     n_jobs=4,
        ...     n_tier=0
        ... )
        >>> translator.transfer_all()
    """

    def __init__(self, openmx_data_dir, deeph_data_dir, output_dir, n_jobs=1, n_tier=0):
        self.openmx_data_dir = Path(openmx_data_dir)
        self.deeph_data_dir = Path(deeph_data_dir)
        self.output_dir = Path(output_dir)
        self.n_jobs = n_jobs
        self.n_tier = n_tier
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def transfer_all_deeph_to_openmx(self):
        """Process all subdirectories."""
        worker = partial(
            self._transfer_one,
            openmx_path=self.openmx_data_dir,
            deeph_path=self.deeph_data_dir,
            output_path=self.output_dir,
        )
        data_dir_lister = get_data_dir_lister(
            self.openmx_data_dir, self.n_tier, validation_check_scfout
        )
        parallel_map(worker, data_dir_lister, n_jobs=self.n_jobs, desc="Data")

    @staticmethod
    def _transfer_one(dir_name, openmx_path, deeph_path, output_path):
        """Process a single directory."""
        openmx_dir_path = openmx_path / dir_name
        deeph_dir_path = deeph_path / dir_name
        output_dir = output_path / dir_name
        if not openmx_dir_path.is_dir() or not deeph_dir_path.is_dir():
            return
        try:
            writer = OpenMXWriter(openmx_dir_path, deeph_dir_path, output_dir)
            writer.replace_hamiltonian()
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")


class OpenMXWriter:
    def __init__(self, openmx_path, deeph_path, output_path):
        self.scfout_path = Path(openmx_path) / OPENMX_SCFOUT_FILENAME
        self.H_path = Path(deeph_path) / DEEPX_HAMILTONIAN_FILENAME
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        #
        self.scfout_reader = BinaryFileReader(self.scfout_path)
        #
        self._load_poscar(Path(deeph_path) / DEEPX_POSCAR_FILENAME)
        self._load_info_json(Path(deeph_path) / DEEPX_INFO_FILENAME)
        self._read_scfout_info()
        # Record the starting offset of DFT matrices
        self.matrix_offset = self.scfout_reader.offset

    def replace_hamiltonian(self):
        """Main workflow for replacing Hamiltonian in scfout file."""
        self.hamiltonian = self._read_h5(self.H_path, self.spinful, 1.0/HARTREE_TO_EV)
        self._check_matrix_info(self.hamiltonian, spinful=self.spinful)
        self._basis_transform_to_openmx(self.hamiltonian["entries"], self.spinful)
        if self.spinful:
            self.hamiltonian_openmx = self._extract_openmx_entries_spinful(self.hamiltonian["entries"])
        else:
            self.hamiltonian_openmx = [self.hamiltonian["entries"], ]
        self._write_new_scfout()
    
    def _load_info_json(self, info_path):
        info = load_json_file(info_path)
        #
        self.spinful = info["spinful"]
        self.fermi_energy_eV = info["fermi_energy_eV"]
        self.elem_orb_map = info["elements_orbital_map"]
        #
        self.basis_trans_index = {}
        for elem, orbs in self.elem_orb_map.items():
            orbital_num_list = np.array([2 * orb_l + 1 for orb_l in orbs])
            orbital_cumsum = np.concatenate((np.array([0]), np.cumsum(orbital_num_list, axis=0)), axis=0)[:-1]
            index = []
            for orb_l, orb_cum in zip(orbs, orbital_cumsum):
                index.append(BASIS_TRANS_WIKI2OPENMX(orb_l) + orb_cum)
            self.basis_trans_index[elem] = np.concatenate(index, axis=0)
    
    def _load_poscar(self, poscar_path):
        poscar = load_poscar_file(poscar_path)
        self.atom_elem = [ele for ele, num in zip(poscar["elements_unique"], poscar["elements_counts"]) for _ in range(num)]

    def _read_version_and_spinful_flag(self):
        _version_and_spinful_flag = self.scfout_reader.read("i")[0]
        _openmx_version = _version_and_spinful_flag // 4
        assert _openmx_version == 3, "You are not using the OpenMX 3.9 version!"
        return _version_and_spinful_flag % 4

    def _get_scfout_matrix_info(self):
        """Calculate matrix information for understanding h5 data structure."""
        atom_pairs = []
        chunk_shapes = []
        chunk_boundaries = [0,]
        for i_atom in range(self.atoms_quantity):
            atom_i_orb_quantity = self.orbit_quantity_list[i_atom]
            for j, j_atom in enumerate(self.fnna_indices_list[i_atom]):
                atom_j_orb_quantity = self.orbit_quantity_list[j_atom]
                j_cell = self.fnna_cell_indices_list[i_atom][j]
                atom_pairs.append(list(self.R_ijk[j_cell]) + [i_atom, j_atom])
                chunk_shapes.append((atom_i_orb_quantity, atom_j_orb_quantity))
                _size = atom_i_orb_quantity * atom_j_orb_quantity
                chunk_boundaries.append(chunk_boundaries[-1] + _size)
        return {
            "atom_pairs": np.array(atom_pairs),
            "chunk_shapes": np.array(chunk_shapes),
            "chunk_boundaries": np.array(chunk_boundaries),
        }

    def _check_matrix_info(self, obs, spinful=False):
        if not np.allclose(self.matrix_info["atom_pairs"], obs["atom_pairs"]):
            raise ValueError("The atom_pairs mismatch between *.scfout and *.h5 file!")
        if not np.allclose(self.matrix_info["chunk_shapes"]*(1+spinful), obs["chunk_shapes"]):
            raise ValueError("The chunk_shapes mismatch between *.scfout and *.h5 file!")
        if not np.allclose(self.matrix_info["chunk_boundaries"]*(1+spinful)**2, obs["chunk_boundaries"]):
            raise ValueError("The chunk_boundaries mismatch between *.scfout and *.h5 file!")

    def _read_scfout_info(self):
        """Read scfout header information."""
        # Read basic parameters
        self.atoms_quantity = self.scfout_reader.read("i")[0]
        # Read spin info
        self.spin_info = self._read_version_and_spinful_flag()
        self.spinful = (0!=self.spin_info)
        # Skip CLR atom numbers (3 integers)
        self.scfout_reader.skip("3i")
        # Read R quantity (number of unit cells)
        self.R_quantity = self.scfout_reader.read("i")[0] + 1
        # Read r_order_max
        self.r_order_max = self.scfout_reader.read("i")[0]
        # Skip R_xyz coordinates
        self.scfout_reader.skip(f"{4*self.R_quantity}d")
        # Read R_ijk (cell indices)
        self.R_ijk = np.array(
            self.scfout_reader.read(f"{4*self.R_quantity}i")
        ).reshape((self.R_quantity, 4))[:, 1:]
        # Read orbital quantities per atom
        self.orbit_quantity_list = np.array(
            self.scfout_reader.read(f"{self.atoms_quantity}i")
        )
        self.orbit_cumsum = np.insert(np.cumsum(self.orbit_quantity_list), 0, 0)
        self.orbits_quantity = int(self.orbit_cumsum[-1])
        # Read FNAN quantities (number of first nearest neighbors)
        self.fnna_quantity_list = np.array(
            self.scfout_reader.read(f"{self.atoms_quantity}i")
        )
        # Read FNAN indices (convert to 0-indexed)
        self.fnna_indices_list = [
            np.array(self.scfout_reader.read(f"{fnna_quantity+1}i")) - 1
            for fnna_quantity in self.fnna_quantity_list
        ]
        # Read FNAN cell indices
        self.fnna_cell_indices_list = [
            np.array(self.scfout_reader.read(f"{fnna_quantity+1}i"))
            for fnna_quantity in self.fnna_quantity_list
        ]
        # Skip lattice vectors, reciprocal vectors, and atomic coordinates
        self.scfout_reader.skip("12d")  # lattice vectors
        self.scfout_reader.skip("12d")  # reciprocal vectors
        self.scfout_reader.skip(f"{4*self.atoms_quantity}d")
        # Build matrix info
        self.matrix_info = self._get_scfout_matrix_info()

    def _read_h5(self, h5_path, spinful, unit_convertion):
        """Read deepx h5 file."""
        dtype = np.float64 if not spinful else np.complex128
        with h5py.File(h5_path, "r") as f:
            atom_pairs = np.array(f["atom_pairs"][:], dtype=np.int64)
            chunk_shapes = np.array(f["chunk_shapes"][:], dtype=np.int64)
            chunk_boundaries = np.array(f["chunk_boundaries"][:], dtype=np.int64)
            entries = np.array(f["entries"][:], dtype=dtype) * unit_convertion
        return {
            "atom_pairs": atom_pairs,
            "chunk_shapes": chunk_shapes,
            "chunk_boundaries": chunk_boundaries,
            "entries": entries,
        }

    def _basis_transform_to_openmx(self, entries, spinful):
        for i_pair, atom_pair in enumerate(self.matrix_info["atom_pairs"]):
            chunk_shape = self.matrix_info["chunk_shapes"][i_pair] * (spinful+1)
            chunk_boundary = self.matrix_info["chunk_boundaries"][i_pair] * (spinful+1)**2
            block = entries[chunk_boundary:chunk_boundary+chunk_shape[0]*chunk_shape[1]].reshape(chunk_shape)
            transform_index1 = self.basis_trans_index[self.atom_elem[atom_pair[3]]]
            transform_index2 = self.basis_trans_index[self.atom_elem[atom_pair[4]]]
            entries[chunk_boundary:chunk_boundary+chunk_shape[0]*chunk_shape[1]] = self._transform(block, transform_index1, transform_index2, spinful).reshape(-1)
        return entries

    def _transform(self, matrix, transform_index1, transform_index2, isspinful):
        if isspinful:
            a = matrix.shape[0] // 2
            b = matrix.shape[1] // 2
            matrix = matrix.reshape((2, a, 2, b)).transpose((0, 2, 1, 3)).reshape((4, a, b))
            matrix = matrix[:, transform_index1, :][:, :, transform_index2]
            matrix = matrix.reshape((2, 2, a, b)).transpose((0, 2, 1, 3)).reshape((2 * a, 2 * b))
            return matrix
        else:
            matrix = matrix[transform_index1, :][:, transform_index2]
            return matrix

    def _extract_openmx_entries_spinful(self, entries):
        """
        Extract spin components from full matrix for non-collinear spin case.

        scfout stores:
        - H0_real: H_up_up real
        - H1_real: H_down_down real
        - H2_real: H_up_down real
        - H3_real: H_up_down imag (part of iHNL)
        - H0_imag: H_up_up imag
        - H1_imag: H_down_down imag
        - H2_imag: H_up_down imag (part of iHNL)
        """
        entries_list = [[] for _ in range(7)]
        for i_pair, atom_pair in enumerate(self.matrix_info["atom_pairs"]):
            chunk_shape = self.matrix_info["chunk_shapes"][i_pair] * 2
            chunk_boundary = self.matrix_info["chunk_boundaries"][i_pair] * 4
            block = entries[chunk_boundary:chunk_boundary+chunk_shape[0]*chunk_shape[1]].reshape(chunk_shape)
            # Extract spin components
            H_up_up     = block[:chunk_shape[0]//2, :chunk_shape[1]//2]
            H_down_down = block[chunk_shape[0]//2:, chunk_shape[1]//2:]
            H_up_down   = block[:chunk_shape[0]//2, chunk_shape[1]//2:]
            # Store components
            entries_list[0].extend(H_up_up.real.reshape(-1))
            entries_list[1].extend(H_down_down.real.reshape(-1))
            entries_list[2].extend(H_up_down.real.reshape(-1))
            entries_list[3].extend(H_up_down.imag.reshape(-1))
            entries_list[4].extend(H_up_up.imag.reshape(-1))
            entries_list[5].extend(H_down_down.imag.reshape(-1))
            entries_list[6].extend(np.zeros_like(H_up_down.imag.reshape(-1)))
        # Convert to numpy arrays
        for i, data in enumerate(entries_list):
            entries_list[i] = np.array(data)
        return entries_list

    def _get_overlap_binary_size(self):
        """Calculate the binary_size of overlap matrix."""
        n_elements = self.matrix_info["chunk_boundaries"][-1]
        return n_elements * struct.calcsize("d")

    def _get_hamiltonian_binary_size(self):
        """Calculate the binary_size of Hamiltonian matrix."""
        n_elements = self.matrix_info["chunk_boundaries"][-1]
        if 0 == self.spin_info:
            n_parts = 1
        elif 1 == self.spin_info:
            n_parts = 2
        elif 3 == self.spin_info:
            n_parts = 7
        else:
            raise ValueError(f'Invalid spin info: {self.spin_info}')
        return n_parts * n_elements * struct.calcsize("d")

    def _get_density_matrix_binary_size(self):
        """Calculate the binary_size of density matrix."""
        n_elements = self.matrix_info["chunk_boundaries"][-1]
        if 0 == self.spin_info:
            n_parts = 3
        elif 1 == self.spin_info:
            n_parts = 4
        elif 3 == self.spin_info:
            n_parts = 6
        else:
            raise ValueError(f'Invalid spin info: {self.spin_info}')
        return n_parts * n_elements * struct.calcsize("d")

    def _write_new_scfout(self):
        """Write the new scfout file with replaced Hamiltonian."""
        output_file = self.output_path / OPENMX_SCFOUT_FILENAME
        overlap_binary_size = self._get_overlap_binary_size()
        hamiltonian_binary_size = self._get_hamiltonian_binary_size()
        density_matrix_binary_size = self._get_density_matrix_binary_size()
        matrices_binary_size = (
            hamiltonian_binary_size + # Hamiltonian
            overlap_binary_size*(1+3*(1+self.r_order_max-1)+3) + # Overlap, position, momentum
            density_matrix_binary_size # Density matrix
        )
        with open(self.scfout_path, "rb") as fr, open(output_file, "wb") as fw:
            # Write header (same as original)
            header_data = self.scfout_reader.data[: self.matrix_offset]
            fw.write(header_data)
            # Write new Hamiltonian
            for data in self.hamiltonian_openmx:
                fw.write(struct.pack(f"{len(data)}d", *data))
            # Write remaining data (overlap matrix, etc.)
            fr.seek(self.matrix_offset + hamiltonian_binary_size)
            fw.write(fr.read())
            # Overwrite fermi energy
            fw.seek(self.matrix_offset + matrices_binary_size + struct.calcsize("i"))
            fw.write(struct.pack("d", self.fermi_energy_eV/HARTREE_TO_EV))

