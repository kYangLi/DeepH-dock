from pathlib import Path
from typing import Tuple
import numpy as np
import json
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from deepx_dock.misc import load_poscar_file


ROOT_PATH = Path(__file__).resolve().parent

# Wigner-D-matrix
with h5py.File(ROOT_PATH / "jd.h5", 'r') as f:
    num = len(list(f.keys()))
    jd = [np.array(f[f"l={ll}"][:], dtype=np.float64) for ll in range(num)]

def d_z(theta, l):  # noqa: E741
    if l == 0:
        return np.array([[1.]])
    else:
        sins = np.sin(np.arange(1, l+1)*theta)
        coss = np.cos(np.arange(1, l+1)*theta)
        mat = np.zeros((2*l+1, 2*l+1), dtype=np.float64)
        mat[l, l] = 1.0
        mat[range(l-1, -1,   -1), range(l-1, -1,   -1)] = coss
        mat[range(l+1, 2*l+1, 1), range(l+1, 2*l+1, 1)] = coss
        mat[range(l-1, -1,   -1), range(l+1, 2*l+1, 1)] = sins
        mat[range(l+1, 2*l+1, 1), range(l-1, -1,   -1)] = -sins
        return mat

def d_y(theta, l): # noqa: E741
    if l == 0:
        return np.array([[1.]])
    else:
        return jd[l] @ d_z(theta, l) @ jd[l]

def wigner_D_mat(alpha, beta, gamma, l): # noqa: E741
    if l == 0:
        return np.array([[1.]])
    else:
        return d_z(alpha, l) @ d_y(beta, l) @ d_z(gamma, l)

## Matrix form of observables
def get_obs_array(obs_file, R, orbital_loc):
    obs = h5py.File(obs_file, 'r')
    obs_each_orbital = np.empty((orbital_loc[-1], orbital_loc[-1]), dtype=np.float64)
    ## unpack
    atom_pairs = obs["atom_pairs"]
    chunk_boundaries = obs["chunk_boundaries"]
    chunk_shapes = obs["chunk_shapes"]
    entries = obs["entries"]
    ## construction
    for i_atom_pair, atom_pair in enumerate(atom_pairs):
        if list(atom_pair[0:3]) != list(R):
            continue
        [i, j] = atom_pair[3:5]
        data_block = np.array(entries[chunk_boundaries[i_atom_pair]:chunk_boundaries[i_atom_pair+1]]).reshape(chunk_shapes[i_atom_pair])
        obs_each_orbital[orbital_loc[i]:orbital_loc[i+1], orbital_loc[j]:orbital_loc[j+1]] = data_block
    return obs_each_orbital


def test_equiv(
    dft_dir: str | Path, target: str = "hamiltonian", 
    R: Tuple[int, int, int] = (0, 0, 0), pair: Tuple[int, int] = (-1, -1)
):
    dft_dir = Path(dft_dir)
    structure_0 = load_poscar_file(dft_dir / "0" / "POSCAR")
    species = structure_0["elements_unique"]
    atom_nums = structure_0["elements_counts"]

    with open(dft_dir / "0" / "info.json", "r") as f:
        info = json.load(f)
        orbital_types = [info["elements_orbital_map"][ele] for ele in species]

    orbital_nums = [
        np.sum(np.array(orbital_type)*2+1) for orbital_type in orbital_types
    ]

    loc = [0]
    for atom_num in atom_nums:
        loc.append(loc[-1]+atom_num)
    atoms = []
    orbital_types_tot = []
    for i in range(len(species)):
        atoms += [species[i]]*atom_nums[i]
        orbital_types_tot += orbital_types[i]*atom_nums[i]
    orbital_loc = [0]
    orbital_type_loc = [0]
    for i in range(0,loc[-1]):
        species_i = species.index(atoms[i])
        orbital_num = orbital_nums[species_i]
        orbital_loc.append(orbital_loc[-1]+orbital_num)
        orbital_type = orbital_types[species_i]
        for orbital_type_j in orbital_type:
            orbital_type_loc.append(orbital_type_loc[-1]+2*orbital_type_j+1)
    if pair[0] > 0 and pair[1] > 0:
        selected_orbitals = [
            np.arange(orbital_loc[pair[0]],orbital_loc[pair[0]+1]),
            np.arange(orbital_loc[pair[1]],orbital_loc[pair[1]+1])
        ]
    else:
        selected_orbitals = [
            np.arange(orbital_loc[0],orbital_loc[-1]),
            np.arange(orbital_loc[0],orbital_loc[-1])
        ]

    print("[do] analyzing ...")
    entries_list = []
    obs_data_file_0 = dft_dir / "0" / f"{target}.h5"
    data_each_orbital_0 = get_obs_array(obs_data_file_0, R, orbital_loc)
    data_each_orbital_0 = data_each_orbital_0[selected_orbitals[0], :][:, selected_orbitals[1]]
    entries_list.append(data_each_orbital_0.reshape(-1))
    for dir_now in tqdm(list(dft_dir.iterdir())):
        mat_name = dir_now.name
        if not (dft_dir / mat_name).is_dir():
            continue
        if mat_name == "0":
            continue
        ## Observable
        obs_data_file = dft_dir / mat_name / f"{target}.h5"
        data_each_orbital = get_obs_array(obs_data_file, R, orbital_loc)

        alpha, beta, gamma = np.loadtxt(dft_dir / mat_name / "euler_angle", dtype=np.float64)
        wigners = [wigner_D_mat(alpha,beta,gamma, ll) for ll in range(len(jd))]
        rot_each_orbital = np.zeros((orbital_loc[-1], orbital_loc[-1]))
        for i in range(len(orbital_types_tot)):
            rot_each_orbital[
                orbital_type_loc[i]:orbital_type_loc[i+1], 
                orbital_type_loc[i]:orbital_type_loc[i+1]
            ] = wigners[orbital_types_tot[i]]

        data_each_orbital_rot_back = rot_each_orbital.T @ data_each_orbital @ rot_each_orbital
        data_each_orbital_rot_back = data_each_orbital_rot_back[selected_orbitals[0], :][:, selected_orbitals[1]]

        entries_list.append(data_each_orbital_rot_back.reshape(-1))

    entries_list = np.array(entries_list)
    entries_mean = np.mean(entries_list, axis=0)
    tot_ma = np.mean(np.abs(entries_list))
    struct_rmse = np.sqrt(np.mean((entries_list - entries_mean)**2, axis=1))
    print("[data] Mean absolute value of entries:", np.mean(tot_ma))
    print("[data] Root mean square error (RMSE) of entries:", np.mean(struct_rmse))
    print("[note] For Hamiltonian matrix, a RMSE below 1e-4 eV is preferred.")

    plt.hist(np.array(struct_rmse))
    plt.xlabel("Equivariance RMSE")
    plt.ylabel("Count")
    plt.savefig(f"{dft_dir}/../equiv_mae.png")

