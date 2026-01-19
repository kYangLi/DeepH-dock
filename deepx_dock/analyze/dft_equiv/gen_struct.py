from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from copy import deepcopy
import random
import math
import shutil
from tqdm import tqdm

from deepx_dock.misc import load_poscar_file, dump_poscar_file


def gen_struct(
    poscar_dir: str | Path, generate_num: int, 
    if_rot_coord: bool, if_rot_lat: bool, if_translate: bool,
    if_move_to_origin: bool, dump_decimals: int = 10
):
    poscar_dir = Path(poscar_dir)
    
    dir_0 = poscar_dir / "0"
    structure_0 = load_poscar_file(dir_0 / "POSCAR")
    lat_0 = structure_0["lattice"]
    coords_0 = structure_0["cart_coords"]
    mass_center_0 = np.mean(coords_0, axis=0, keepdims=True)

    structure_i = deepcopy(structure_0)
    random.seed(42)
    for mat_i in tqdm(range(1,generate_num)):
        dir_i = poscar_dir / str(mat_i)
        if dir_i.exists():
            shutil.rmtree(dir_i)
        shutil.copytree(dir_0, dir_i)
        poscar_path = dir_i / "POSCAR"
        if poscar_path.exists():
            poscar_path.unlink()

        alpha = random.random() * 2.0 * math.pi  # [0, 2pi) p(alpha) = uniform
        beta = math.acos(2.0 * random.random() - 1.0)  # [0, pi] p(beta) = sin(beta)
        gamma = random.random() * 2.0 * math.pi  # [0, 2pi) p(gamma) = uniform
        euler_angle_file = dir_i / "euler_angle"
        content = f"{alpha} {beta} {gamma}"  if if_rot_coord else "0.0 0.0 0.0"
        euler_angle_file.write_text(content)

        rot = Rotation.from_euler('zyz', [gamma, beta, alpha], degrees=False)
        shift = np.array([[random.random(), random.random(), random.random()]]) * 0.1

        if if_rot_lat and if_rot_coord:
            lat_i = rot.apply(lat_0)
            coords_i = rot.apply(coords_0)
            if if_move_to_origin:
                coords_i -= rot.apply(mass_center_0)
        elif if_rot_lat:
            lat_i = rot.apply(lat_0)
            coords_i = coords_0 - mass_center_0
            if not if_move_to_origin:
                coords_i += rot.apply(mass_center_0)
        elif if_rot_coord:
            lat_i = lat_0
            coords_i = rot.apply(coords_0 - mass_center_0)
            if not if_move_to_origin:
                coords_i += mass_center_0
        if if_translate:
            coords_i += shift

        structure_i["lattice"] = lat_i
        structure_i["cart_coords"] = coords_i
        dump_poscar_file(dir_i / "POSCAR", structure_i, direct=False, dump_decimals=dump_decimals)


