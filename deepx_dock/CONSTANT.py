from datetime import datetime

# Math
EXTREMELY_LARGE_FLOAT = 1.23456789e10
EXTREMELY_SMALL_FLOAT = 1.0E-16

# Physics
HARTREE_TO_EV = 27.2113845
BOHR_TO_ANGSTROM = 0.529177249

# Time stamp
TIME_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Material related
PERIODIC_TABLE_INDEX_TO_SYMBOL = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S',
    17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V',
    24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb',
    38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru',
    45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb',
    52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce',
    59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb',
    66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf',
    73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au',
    80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn',
    87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np',
    94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
    101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg',
    107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn',
    113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og',
    131: 'My1', 132: 'My2', 133: 'My3', 134: 'My4', 135: 'My5', 136: 'My6'
}
PERIODIC_TABLE_SYMBOL_TO_INDEX = {
    v:k for (k, v) in PERIODIC_TABLE_INDEX_TO_SYMBOL.items()
}

# - Data files
DEEPX_POSCAR_FILENAME = "POSCAR"
DEEPX_INFO_FILENAME = "info.json"
DEEPX_OVERLAP_FILENAME = "overlap.h5"
DEEPX_HAMILTONIAN_FILENAME = "hamiltonian.h5"
DEEPX_PREDICT_HAMILTONIAN_FILENAME = "hamiltonian_pred.h5"
DEEPX_DENSITY_MATRIX_FILENAME = "density_matrix.h5"
DEEPX_POSITION_MATRIX_FILENAME = "position_matrix.h5"
DEEPX_VR_FILENAME = "potential_r.h5"
DEEPX_NECESSARY_FILES = {DEEPX_POSCAR_FILENAME, DEEPX_INFO_FILENAME}
DEEPX_BAND_FILENAME = "band.h5"
DEEPX_EIGVAL_FILENAME = "eigval.h5"
DEEPX_DOS_FILENAME = "dos.h5"
DEEPX_K_PATH_FILENAME = "K_PATH"
DFT_DIRNAME = "dft"
DATASET_SPLIT_FILENAME = "dataset_split.json"
FERMI_ENERGY_INFO_FILENAME = "fermi_energy.json"

SIESTA_HSX_FILENAME = "siesta.HSX"
SIESTA_EIG_FILENAME = "siesta.EIG"
SIESTA_DM_FILENAME = "siesta.DM"

# - Folder names
DEEPX_DFT_FOLDER = "dft"
DEEPX_BASIS_FOLDER = "basis"
