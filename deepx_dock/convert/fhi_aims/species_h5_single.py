"""Single-run FHI-aims output-basis to species_aims_{xc}.h5 converter.

This module focuses on one run directory and provides:
1. Parsing of control.in species blocks.
2. Parsing and pairing of output basis wave/kinetic files.
3. Standardization to overlap-compatible species schema.
4. HDF5 writing and structural validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List

import h5py
import numpy as np


L_CHAR_TO_INT = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
    "h": 5,
    "i": 6,
    "j": 7,
    "k": 8,
}


@dataclass
class RadialRecord:
    kind2: str
    sp_idx: int
    i_fn: int
    n: int
    l: int
    r: np.ndarray | None = None
    u: np.ndarray | None = None
    k: np.ndarray | None = None


@dataclass
class SpeciesModel:
    element: str
    species_name: str
    nljz: np.ndarray
    cutoff_radii: np.ndarray
    grid_length: np.ndarray
    radius_grid_raw: List[np.ndarray]
    radius_data_raw: List[np.ndarray]
    kinetic_data_raw: List[np.ndarray]
    val_density_nljz: np.ndarray
    val_density_cutoff_radii: np.ndarray
    val_density_grid_length: np.ndarray
    val_density_radius_grid_raw: List[np.ndarray]
    val_density_radius_data_raw: List[np.ndarray]
    fn_type_codes: np.ndarray
    orbital_index_local: np.ndarray
    control_meta: Dict[str, object]


@dataclass
class PackedSpeciesModel:
    element: str
    species_name: str
    nljz: np.ndarray
    cutoff_radii: np.ndarray
    grid_length: np.ndarray
    radius_grid_padded: np.ndarray
    radius_data_padded: np.ndarray
    kinetic_data_padded: np.ndarray
    val_density_nljz: np.ndarray
    val_density_cutoff_radii: np.ndarray
    val_density_grid_length: np.ndarray
    val_density_radius_grid_padded: np.ndarray
    val_density_radius_data_padded: np.ndarray
    fn_type_codes: np.ndarray
    orbital_index_local: np.ndarray
    control_meta: Dict[str, object]


@dataclass
class RunExportModel:
    xc_functional: str
    source: str
    units_length: str
    global_nmax: int
    species: Dict[str, PackedSpeciesModel]


PATTERN_INT_L = re.compile(
    r"^(?P<kin>kin_)?(?P<kind2>[a-z]{2})_"
    r"(?P<sp_idx>\d+)_(?P<i_fn>\d+)_(?P<n>\d+)_(?P<l>\d+)\.dat$"
)

PATTERN_CHAR_L = re.compile(
    r"^(?P<kin>kin_)?(?P<kind2>[a-z]{2})_"
    r"(?P<sp_idx>\d+)_(?P<i_fn>\d+)_(?P<n>\d+)_(?P<l>[a-z])\.dat$"
)

PATTERN_FREE_RHO = re.compile(r"^(?P<element>[A-Z][a-z]?)_free_rho\.dat$")


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_control_species(control_path: str | Path) -> Dict[str, object]:
    """Parse species-local parameters from control.in."""
    control_path = Path(control_path)
    if not control_path.exists():
        raise FileNotFoundError(f"control.in not found: {control_path}")

    species_order: List[str] = []
    species_params: Dict[str, Dict[str, object]] = {}
    global_wave_threshold = None
    xc_functional = "unknown"
    current_species: str | None = None

    with open(control_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = _strip_comment(raw)
            if not line:
                continue
            tok = line.split()
            key = tok[0]

            if key == "xc" and len(tok) >= 2:
                xc_functional = tok[1].lower()
                continue

            if key == "species" and len(tok) >= 2:
                current_species = tok[1]
                species_order.append(current_species)
                species_params[current_species] = {
                    "wave_threshold": None,
                    "radial_base": None,
                    "radial_multiplier": None,
                    "angular_grids": [],
                    "basis_dep_cutoff": None,
                    "l_hartree": None,
                    "orbital_lines": [],
                }
                continue

            if key == "wave_threshold" and current_species is None and len(tok) >= 2:
                global_wave_threshold = float(tok[1])
                continue

            if current_species is None:
                continue

            params = species_params[current_species]

            if key == "wave_threshold" and len(tok) >= 2:
                params["wave_threshold"] = float(tok[1])
            elif key == "radial_base" and len(tok) >= 3:
                params["radial_base"] = [float(tok[1]), float(tok[2])]
            elif key == "radial_multiplier" and len(tok) >= 2:
                params["radial_multiplier"] = int(tok[1])
            elif key in {"angular_grids", "angular_grid", "division"} and len(tok) >= 3:
                params["angular_grids"].append([float(tok[-2]), float(tok[-1])])
            elif key == "basis_dep_cutoff" and len(tok) >= 2:
                params["basis_dep_cutoff"] = float(tok[1])
            elif key == "l_hartree" and len(tok) >= 2:
                params["l_hartree"] = int(tok[1])
            elif key in {"ionic", "atomic", "hydro", "confined", "gaussian", "sto"}:
                params["orbital_lines"].append(line)

    if global_wave_threshold is not None:
        for species_name in species_order:
            if species_params[species_name]["wave_threshold"] is None:
                species_params[species_name]["wave_threshold"] = global_wave_threshold

    if len(species_order) == 0:
        raise ValueError("No species blocks found in control.in")

    return {
        "species_order": species_order,
        "species_params": species_params,
        "global_wave_threshold": global_wave_threshold,
        "xc_functional": xc_functional,
    }


def _parse_basis_filename(name: str):
    match = PATTERN_INT_L.match(name)
    if match is not None:
        return {
            "is_kin": bool(match.group("kin")),
            "kind2": match.group("kind2"),
            "sp_idx": int(match.group("sp_idx")),
            "i_fn": int(match.group("i_fn")),
            "n": int(match.group("n")),
            "l": int(match.group("l")),
        }

    match = PATTERN_CHAR_L.match(name)
    if match is None:
        return None

    l_char = match.group("l").lower()
    if l_char not in L_CHAR_TO_INT:
        return None

    return {
        "is_kin": bool(match.group("kin")),
        "kind2": match.group("kind2"),
        "sp_idx": int(match.group("sp_idx")),
        "i_fn": int(match.group("i_fn")),
        "n": int(match.group("n")),
        "l": int(L_CHAR_TO_INT[l_char]),
    }


def _load_two_col_array(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected two-column data in {path}")
    return arr


def _extract_element_from_species_name(species_name: str) -> str:
    match = re.match(r"^([A-Z][a-z]?)", species_name)
    if match is None:
        raise ValueError(f"E_BAD_SPECIES_NAME: cannot extract element from '{species_name}'")
    return match.group(1)


def parse_output_basis_run(run_dir: str | Path) -> List[RadialRecord]:
    """Parse output basis files in a run directory."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run_dir is not a directory: {run_dir}")

    records: Dict[tuple, RadialRecord] = {}
    matched_any = False

    for path in sorted(run_dir.glob("*.dat")):
        parsed = _parse_basis_filename(path.name)
        if parsed is None:
            continue
        matched_any = True

        key = (
            parsed["kind2"],
            parsed["sp_idx"],
            parsed["i_fn"],
            parsed["n"],
            parsed["l"],
        )
        rec = records.setdefault(
            key,
            RadialRecord(
                kind2=parsed["kind2"],
                sp_idx=parsed["sp_idx"],
                i_fn=parsed["i_fn"],
                n=parsed["n"],
                l=parsed["l"],
            ),
        )

        arr = _load_two_col_array(path)
        if parsed["is_kin"]:
            rec.k = arr[:, 1].astype(np.float64)
        else:
            rec.r = arr[:, 0].astype(np.float64)
            rec.u = arr[:, 1].astype(np.float64)

    if not matched_any:
        raise ValueError(f"No output-basis .dat files matched parser in {run_dir}")

    out = sorted(records.values(), key=lambda x: (x.sp_idx, x.i_fn, x.n, x.l, x.kind2))
    for rec in out:
        if rec.r is None or rec.u is None:
            raise ValueError(
                "E_PARSE_FILENAME/E_MISSING_WAVE: "
                f"missing wave data for kind2={rec.kind2}, sp={rec.sp_idx}, i_fn={rec.i_fn}, n={rec.n}, l={rec.l}"
            )
        if rec.k is None:
            raise ValueError(
                "E_MISSING_KINETIC: "
                f"missing kinetic data for kind2={rec.kind2}, sp={rec.sp_idx}, i_fn={rec.i_fn}, n={rec.n}, l={rec.l}"
            )
        if rec.r.ndim != 1 or rec.u.ndim != 1 or rec.k.ndim != 1:
            raise ValueError("E_GRID_MISMATCH: basis arrays must be 1D")
        if len(rec.r) != len(rec.u) or len(rec.u) != len(rec.k):
            raise ValueError(
                "E_GRID_MISMATCH: "
                f"len(r,u,k)=({len(rec.r)},{len(rec.u)},{len(rec.k)}) for sp={rec.sp_idx}, i_fn={rec.i_fn}"
            )

    return out


def parse_free_rho_run(run_dir: str | Path) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse <Element>_free_rho.dat files for val_density export."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run_dir is not a directory: {run_dir}")

    free_rho_by_element: Dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for path in sorted(run_dir.glob("*_free_rho.dat")):
        match = PATTERN_FREE_RHO.match(path.name)
        if match is None:
            continue

        element = match.group("element")
        if element in free_rho_by_element:
            raise ValueError(f"E_DUP_FREE_RHO_FOR_ELEMENT: duplicated free rho file for element {element}")

        arr = _load_two_col_array(path)
        r = arr[:, 0].astype(np.float64)
        rho = arr[:, 1].astype(np.float64)

        if len(r) <= 1:
            raise ValueError(f"E_TOO_SHORT_FREE_RHO: {path}")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(rho)):
            raise ValueError(f"E_NONFINITE_FREE_RHO: {path}")
        if np.any(np.diff(r) <= 0.0):
            raise ValueError(f"E_NONMONOTONIC_FREE_RHO_GRID: {path}")

        free_rho_by_element[element] = (r, rho)

    if len(free_rho_by_element) == 0:
        raise ValueError(f"E_MISSING_FREE_RHO_FILES: no *_free_rho.dat found in {run_dir}")

    return free_rho_by_element


def effective_last_index(u: np.ndarray, k: np.ndarray, tol: float) -> int:
    mask = (np.abs(u) > tol) | (np.abs(k) > tol)
    if np.any(mask):
        return int(np.where(mask)[0][-1])
    return int(len(u) - 1)


def effective_last_index_1d(data: np.ndarray, tol: float, strict_nonzero: bool = False) -> int:
    mask = np.abs(data) > tol
    if np.any(mask):
        return int(np.where(mask)[0][-1])
    if strict_nonzero:
        raise ValueError("E_ZERO_VAL_DENSITY: all points are below trimming tolerance")
    return int(len(data) - 1)


def convert_free_rho_output_to_rho(val_r: np.ndarray, val_density_raw: np.ndarray) -> np.ndarray:
    """Convert AIMS free-rho output to rho(r): rho = d / (16*pi^2*r^2)."""
    r_safe = np.where(np.abs(val_r) < 1e-15, 1e-15, val_r)
    return val_density_raw / (16.0 * (np.pi**2) * (r_safe**2))


def _kind2_to_code(kind2_values: Iterable[str]) -> np.ndarray:
    uniq = sorted(set(kind2_values))
    table = {v: i for i, v in enumerate(uniq)}
    return np.asarray([table[v] for v in kind2_values], dtype=np.int32)


def build_species_models(
    control: Dict[str, object],
    records: List[RadialRecord],
    run_dir: str | Path,
    tol: float = 5e-7,
):
    """Build species models from control mapping and parsed records."""
    order = control["species_order"]
    params = control["species_params"]
    free_rho_by_element = parse_free_rho_run(run_dir)

    grouped: Dict[str, List[RadialRecord]] = {s: [] for s in order}
    for rec in records:
        if rec.sp_idx < 1 or rec.sp_idx > len(order):
            raise ValueError(
                f"E_SPIDX_OUT_OF_RANGE: sp_idx={rec.sp_idx}, n_species={len(order)}"
            )
        species_name = order[rec.sp_idx - 1]
        grouped[species_name].append(rec)

    models: Dict[str, SpeciesModel] = {}

    for species_name, recs in grouped.items():
        recs.sort(key=lambda x: (x.i_fn, x.n, x.l, x.kind2))

        element = _extract_element_from_species_name(species_name)
        if element not in free_rho_by_element:
            raise ValueError(
                f"E_MISSING_FREE_RHO_FOR_SPECIES: species={species_name}, expected {element}_free_rho.dat"
            )

        val_r_full, val_density_full_raw = free_rho_by_element[element]
        val_density_full = convert_free_rho_output_to_rho(val_r_full, val_density_full_raw)
        val_i_last = effective_last_index_1d(val_density_full, tol=tol, strict_nonzero=True)
        val_n_eff = int(val_i_last + 1)

        nljz_rows = []
        cutoffs = []
        grid_len = []
        rgrid_rows = []
        rdata_rows = []
        kdata_rows = []
        kind2_rows = []

        zeta_counter: Dict[tuple, int] = {}
        for rec in recs:
            zeta_key = (rec.n, rec.l)
            zeta_counter[zeta_key] = zeta_counter.get(zeta_key, 0) + 1
            zeta = zeta_counter[zeta_key]

            i_last = effective_last_index(rec.u, rec.k, tol=tol)
            n_eff = int(i_last + 1)
            r_eff = rec.r[:n_eff]
            # HPRO overlap path expects phi(r) ~= u(r)/r rather than raw u(r).
            rr_safe = np.where(np.abs(r_eff) < 1e-15, 1e-15, r_eff)
            u_eff = rec.u[:n_eff] / rr_safe

            nljz_rows.append([rec.n, rec.l, 0, zeta])
            grid_len.append(n_eff)
            cutoffs.append(float(rec.r[i_last]))
            rgrid_rows.append(r_eff)
            rdata_rows.append(u_eff)
            kdata_rows.append(rec.k[:n_eff])
            kind2_rows.append(rec.kind2)

        models[species_name] = SpeciesModel(
            element=species_name,
            species_name=species_name,
            nljz=np.asarray(nljz_rows, dtype=np.int32),
            cutoff_radii=np.asarray(cutoffs, dtype=np.float64),
            grid_length=np.asarray(grid_len, dtype=np.int32),
            radius_grid_raw=rgrid_rows,
            radius_data_raw=rdata_rows,
            kinetic_data_raw=kdata_rows,
            val_density_nljz=np.asarray([[0, 0, 0, 1]], dtype=np.int32),
            val_density_cutoff_radii=np.asarray([float(val_r_full[val_i_last])], dtype=np.float64),
            val_density_grid_length=np.asarray([val_n_eff], dtype=np.int32),
            val_density_radius_grid_raw=[val_r_full[:val_n_eff]],
            val_density_radius_data_raw=[val_density_full[:val_n_eff]],
            fn_type_codes=_kind2_to_code(kind2_rows),
            orbital_index_local=np.arange(len(recs), dtype=np.int32),
            control_meta=params[species_name],
        )

    return models


def _pad_rows(rows: List[np.ndarray], nmax: int) -> np.ndarray:
    out = np.zeros((len(rows), nmax), dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def pack_species_models(models: Dict[str, SpeciesModel], xc_functional: str) -> RunExportModel:
    """Pack variable-length rows to padded arrays with global_nmax."""
    if len(models) == 0:
        raise ValueError("No species model to pack")

    global_nmax = 0
    for model in models.values():
        if len(model.grid_length) > 0:
            global_nmax = max(global_nmax, int(np.max(model.grid_length)))
        if len(model.val_density_grid_length) > 0:
            global_nmax = max(global_nmax, int(np.max(model.val_density_grid_length)))

    if global_nmax <= 0:
        raise ValueError("Invalid global_nmax while packing species models")

    packed: Dict[str, PackedSpeciesModel] = {}
    for species_name, model in models.items():
        packed[species_name] = PackedSpeciesModel(
            element=model.element,
            species_name=model.species_name,
            nljz=model.nljz,
            cutoff_radii=model.cutoff_radii,
            grid_length=model.grid_length,
            radius_grid_padded=_pad_rows(model.radius_grid_raw, global_nmax),
            radius_data_padded=_pad_rows(model.radius_data_raw, global_nmax),
            kinetic_data_padded=_pad_rows(model.kinetic_data_raw, global_nmax),
            val_density_nljz=model.val_density_nljz,
            val_density_cutoff_radii=model.val_density_cutoff_radii,
            val_density_grid_length=model.val_density_grid_length,
            val_density_radius_grid_padded=_pad_rows(model.val_density_radius_grid_raw, global_nmax),
            val_density_radius_data_padded=_pad_rows(model.val_density_radius_data_raw, global_nmax),
            fn_type_codes=model.fn_type_codes,
            orbital_index_local=model.orbital_index_local,
            control_meta=model.control_meta,
        )

    return RunExportModel(
        xc_functional=xc_functional,
        source="aims",
        units_length="bohr",
        global_nmax=global_nmax,
        species=packed,
    )


def _to_float2_or_nan(v):
    if v is None:
        return np.asarray([np.nan, np.nan], dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def _to_int_or_minus1(v):
    if v is None:
        return np.int32(-1)
    return np.int32(v)


def _to_float_or_nan(v):
    if v is None:
        return np.float64(np.nan)
    return np.float64(v)


def _to_string_array(v):
    if v is None:
        return np.asarray([], dtype=h5py.string_dtype(encoding="utf-8"))
    return np.asarray(v, dtype=h5py.string_dtype(encoding="utf-8"))


def write_species_h5(output_file: str | Path, model: RunExportModel) -> None:
    """Write packed model into overlap-compatible species HDF5."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as f:
        f.attrs["xc_functional"] = model.xc_functional
        f.attrs["source"] = model.source
        f.attrs["global_nmax"] = np.int32(model.global_nmax)
        f.attrs["units_length"] = model.units_length

        for species_name in sorted(model.species.keys()):
            sp = model.species[species_name]

            grp = f.create_group(species_name)
            grp.attrs["element"] = sp.element
            grp.attrs["species_name"] = species_name
            grp.attrs["basis_source"] = "output basis"
            grp.attrs["xc_functional"] = model.xc_functional
            grp.attrs["radial_value_convention"] = "aims_output_basis_div_r"
            grp.attrs["val_density_value_convention"] = "aims_output_free_rho_div_16pi2r2"

            basis = grp.create_group("basis")
            basis.create_dataset("nljz_list", data=sp.nljz)
            basis.create_dataset("cutoff_radii", data=sp.cutoff_radii)
            basis.create_dataset("grid_length", data=sp.grid_length)
            basis.create_dataset("radius_grid", data=sp.radius_grid_padded, compression="gzip")
            basis.create_dataset("radius_data", data=sp.radius_data_padded, compression="gzip")
            basis.create_dataset("kinetic_data", data=sp.kinetic_data_padded, compression="gzip")

            val_density = grp.create_group("val_density")
            val_density.create_dataset("nljz_list", data=sp.val_density_nljz)
            val_density.create_dataset("cutoff_radii", data=sp.val_density_cutoff_radii)
            val_density.create_dataset("grid_length", data=sp.val_density_grid_length)
            val_density.create_dataset("radius_grid", data=sp.val_density_radius_grid_padded, compression="gzip")
            val_density.create_dataset("radius_data", data=sp.val_density_radius_data_padded, compression="gzip")

            ext = grp.create_group("extensions")
            ext.create_dataset("orbital_index_local", data=sp.orbital_index_local)
            ext.create_dataset("fn_type_codes", data=sp.fn_type_codes)
            ext.create_dataset("control_radial_base", data=_to_float2_or_nan(sp.control_meta.get("radial_base")))
            ext.create_dataset(
                "control_radial_multiplier",
                data=_to_int_or_minus1(sp.control_meta.get("radial_multiplier")),
            )
            ext.create_dataset(
                "control_angular_grids",
                data=np.asarray(sp.control_meta.get("angular_grids", []), dtype=np.float64),
            )
            ext.create_dataset(
                "control_basis_dep_cutoff",
                data=_to_float_or_nan(sp.control_meta.get("basis_dep_cutoff")),
            )
            ext.create_dataset("control_l_hartree", data=_to_int_or_minus1(sp.control_meta.get("l_hartree")))
            ext.create_dataset(
                "control_wave_threshold",
                data=_to_float_or_nan(sp.control_meta.get("wave_threshold")),
            )
            ext.create_dataset(
                "control_orbital_lines",
                data=_to_string_array(sp.control_meta.get("orbital_lines", [])),
            )
            ext.create_dataset(
                "parse_notes",
                data=np.asarray("control.in + output basis only", dtype=h5py.string_dtype(encoding="utf-8")),
            )


def validate_species_h5(output_file: str | Path) -> None:
    """Validate minimal compatibility for species loader + overlap input chain."""
    output_file = Path(output_file)
    with h5py.File(output_file, "r") as f:
        if "global_nmax" not in f.attrs:
            raise ValueError("Missing root attribute: global_nmax")
        global_nmax = int(f.attrs["global_nmax"])

        for species_name in f.keys():
            b = f[f"{species_name}/basis"]
            required = [
                "nljz_list",
                "cutoff_radii",
                "grid_length",
                "radius_grid",
                "radius_data",
                "kinetic_data",
            ]
            for key in required:
                if key not in b:
                    raise ValueError(f"Missing dataset: {species_name}/basis/{key}")

            nrows = int(b["nljz_list"].shape[0])
            if int(b["cutoff_radii"].shape[0]) != nrows:
                raise ValueError(f"Row mismatch in cutoff_radii for {species_name}")
            if int(b["grid_length"].shape[0]) != nrows:
                raise ValueError(f"Row mismatch in grid_length for {species_name}")
            if int(b["radius_grid"].shape[0]) != nrows:
                raise ValueError(f"Row mismatch in radius_grid for {species_name}")
            if int(b["radius_data"].shape[0]) != nrows:
                raise ValueError(f"Row mismatch in radius_data for {species_name}")
            if int(b["kinetic_data"].shape[0]) != nrows:
                raise ValueError(f"Row mismatch in kinetic_data for {species_name}")

            if int(b["radius_grid"].shape[1]) != global_nmax:
                raise ValueError(f"global_nmax mismatch in radius_grid for {species_name}")
            if int(b["radius_data"].shape[1]) != global_nmax:
                raise ValueError(f"global_nmax mismatch in radius_data for {species_name}")
            if int(b["kinetic_data"].shape[1]) != global_nmax:
                raise ValueError(f"global_nmax mismatch in kinetic_data for {species_name}")

            grid_length = b["grid_length"][:]
            cutoff = b["cutoff_radii"][:]
            rgrid = b["radius_grid"][:]
            for i in range(nrows):
                n_eff = int(grid_length[i])
                if n_eff <= 0:
                    raise ValueError(f"Non-positive grid_length for {species_name}, row {i}")
                rg = rgrid[i, :n_eff]
                if np.any(np.diff(rg) <= 0.0):
                    raise ValueError(f"radius_grid not strictly increasing for {species_name}, row {i}")
                if not np.isclose(cutoff[i], rg[-1], atol=1e-10, rtol=1e-8):
                    raise ValueError(f"cutoff mismatch for {species_name}, row {i}")

            v_path = f"{species_name}/val_density"
            if v_path not in f:
                raise ValueError(f"Missing group: {v_path}")

            v = f[v_path]
            val_required = [
                "nljz_list",
                "cutoff_radii",
                "grid_length",
                "radius_grid",
                "radius_data",
            ]
            for key in val_required:
                if key not in v:
                    raise ValueError(f"Missing dataset: {v_path}/{key}")

            vrows = int(v["nljz_list"].shape[0])
            if vrows != 1:
                raise ValueError(f"val_density nljz_list row count must be 1 for {species_name}")
            if int(v["cutoff_radii"].shape[0]) != vrows:
                raise ValueError(f"val_density row mismatch in cutoff_radii for {species_name}")
            if int(v["grid_length"].shape[0]) != vrows:
                raise ValueError(f"val_density row mismatch in grid_length for {species_name}")
            if int(v["radius_grid"].shape[0]) != vrows:
                raise ValueError(f"val_density row mismatch in radius_grid for {species_name}")
            if int(v["radius_data"].shape[0]) != vrows:
                raise ValueError(f"val_density row mismatch in radius_data for {species_name}")

            if int(v["radius_grid"].shape[1]) != global_nmax:
                raise ValueError(f"val_density global_nmax mismatch in radius_grid for {species_name}")
            if int(v["radius_data"].shape[1]) != global_nmax:
                raise ValueError(f"val_density global_nmax mismatch in radius_data for {species_name}")

            expected_val_nljz = np.asarray([[0, 0, 0, 1]], dtype=np.int32)
            if not np.array_equal(v["nljz_list"][:], expected_val_nljz):
                raise ValueError(f"val_density nljz_list mismatch for {species_name}")

            v_n_eff = int(v["grid_length"][0])
            if v_n_eff <= 0:
                raise ValueError(f"Non-positive val_density grid_length for {species_name}")

            v_rg = v["radius_grid"][0, :v_n_eff]
            if np.any(np.diff(v_rg) <= 0.0):
                raise ValueError(f"val_density radius_grid not strictly increasing for {species_name}")

            v_cutoff = float(v["cutoff_radii"][0])
            if not np.isclose(v_cutoff, v_rg[-1], atol=1e-10, rtol=1e-8):
                raise ValueError(f"val_density cutoff mismatch for {species_name}")

            if not np.allclose(v["radius_grid"][0, v_n_eff:], 0.0, atol=1e-12, rtol=0.0):
                raise ValueError(f"val_density padded radius_grid must be zero for {species_name}")
            if not np.allclose(v["radius_data"][0, v_n_eff:], 0.0, atol=1e-12, rtol=0.0):
                raise ValueError(f"val_density padded radius_data must be zero for {species_name}")


def convert_single_run_to_species_h5(
    run_dir: str | Path,
    output_h5: str | Path,
    xc_functional: str | None = None,
    tol: float = 5e-7,
) -> Path:
    """Parse one run directory and write species_aims_{xc}.h5."""
    run_dir = Path(run_dir)
    control = parse_control_species(run_dir / "control.in")
    records = parse_output_basis_run(run_dir)
    models = build_species_models(control, records, run_dir=run_dir, tol=tol)

    xc = xc_functional.lower() if xc_functional else str(control["xc_functional"]).lower()
    if not xc or xc == "unknown":
        xc = "unknown"

    export_model = pack_species_models(models, xc_functional=xc)
    write_species_h5(output_h5, export_model)
    validate_species_h5(output_h5)
    return Path(output_h5)


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert one FHI-aims run directory to species_aims_{xc}.h5"
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing control.in and basis dat files")
    parser.add_argument("output_h5", type=Path, help="Output species HDF5 path")
    parser.add_argument("--xc-functional", type=str, default=None, help="Override xc functional in output attrs")
    parser.add_argument("--tol", type=float, default=5e-7, help="Tail truncation tolerance for basis/val_density")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    out = convert_single_run_to_species_h5(
        run_dir=args.run_dir,
        output_h5=args.output_h5,
        xc_functional=args.xc_functional,
        tol=args.tol,
    )
    print(f"[done] Exported: {out}")


if __name__ == "__main__":
    main()
