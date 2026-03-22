import re
import subprocess
from pathlib import Path


class AimsPatcher:
    """
    Apply DeepH patch to FHI-aims source code to enable DeepX warmstart support.

    The patch enables FHI-aims to load Hamiltonian matrices predicted by DeepX
    models and use them as initial guess for SCF calculations.
    """

    _PATCH_DIR = Path(__file__).parent / "patch"

    def __init__(self, aims_src_dir: str | Path):
        self.aims_src_dir = Path(aims_src_dir)
        self._patch_file: Path | None = None
        self._target_version: str | None = None

    @classmethod
    def get_supported_versions(cls) -> list[str]:
        """
        Return list of supported FHI-aims versions.

        Scans patch directory for patch_deepx_fhiaims_*.diff files and extracts
        version numbers from filenames.

        Returns
        -------
        list[str]
            Sorted list of supported version strings (e.g., ["250822_1", "251231"])
        """
        versions = []
        if not cls._PATCH_DIR.is_dir():
            return versions
        for patch_file in cls._PATCH_DIR.glob("patch_deepx_fhiaims_*.diff"):
            try:
                version = cls._extract_version_from_patch_name(patch_file.name)
                versions.append(version)
            except ValueError:
                continue
        return sorted(versions)

    @classmethod
    def _extract_version_from_patch_name(cls, patch_name: str) -> str:
        """
        Extract FHI-aims version from patch filename.

        Parameters
        ----------
        patch_name : str
            Patch filename, e.g., "patch_deepx_fhiaims_250822_1.diff"

        Returns
        -------
        str
            Version string, e.g., "250822_1" or "250822"

        Raises
        ------
        ValueError
            If version cannot be extracted from filename
        """
        match = re.search(r"patch_deepx_fhiaims_(\d{6}(?:_\d+)?)\.diff", patch_name)
        if not match:
            raise ValueError(
                f"Cannot extract version from patch filename: {patch_name}. "
                "Expected pattern: patch_deepx_fhiaims_YYMMDD_N.diff or patch_deepx_fhiaims_YYMMDD.diff"
            )
        return match.group(1)

    def _detect_aims_version(self) -> str | None:
        """
        Detect FHI-aims version from source directory.

        Reads the first line of README.md which contains version info like:
        "# FHI-aims code distribution, 250822_1"

        Returns
        -------
        str | None
            Version string (e.g., "250822_1") or None if not found
        """
        readme_path = self.aims_src_dir / "README.md"
        if not readme_path.is_file():
            return None
        with open(readme_path, "r") as f:
            first_line = f.readline().strip()
        match = re.search(r"(\d{6}(?:_\d+)?)", first_line)
        if match:
            return match.group(1)
        return None

    def _find_patch_for_version(self, version: str) -> Path | None:
        """
        Find patch file matching the given version.

        Parameters
        ----------
        version : str
            FHI-aims version string (e.g., "250822_1" or "250822")

        Returns
        -------
        Path | None
            Path to the patch file, or None if not found
        """
        if not self._PATCH_DIR.is_dir():
            return None

        for patch_file in self._PATCH_DIR.glob("patch_deepx_fhiaims_*.diff"):
            try:
                patch_version = self._extract_version_from_patch_name(patch_file.name)
                if patch_version == version:
                    return patch_file
            except ValueError:
                continue
        return None

    def _check_already_patched(self) -> bool:
        """Check if aims source has already been patched."""
        deeph_interface_dir = self.aims_src_dir / "src" / "deeph_interface"
        return deeph_interface_dir.is_dir()

    def apply_patch(self, force: bool = False) -> None:
        """
        Apply the patch to FHI-aims source code.

        Parameters
        ----------
        force : bool
            If True, apply patch even if already patched (may fail)

        Raises
        ------
        RuntimeError
            If validation fails or patch cannot be applied
        """
        if not self.aims_src_dir.is_dir():
            raise RuntimeError(f"Directory does not exist: {self.aims_src_dir}")

        src_dir = self.aims_src_dir / "src"
        if not src_dir.is_dir():
            raise RuntimeError(f"Not a valid FHI-aims source directory: {self.aims_src_dir}/src not found")

        detected_version = self._detect_aims_version()
        if detected_version is None:
            raise RuntimeError(f"Cannot detect FHI-aims version from {self.aims_src_dir}/README.md")

        self._patch_file = self._find_patch_for_version(detected_version)
        if self._patch_file is None:
            supported = self.get_supported_versions()
            if supported:
                raise RuntimeError(
                    f"FHI-aims version {detected_version} is not supported. Supported versions: {', '.join(supported)}"
                )
            else:
                raise RuntimeError(
                    f"FHI-aims version {detected_version} is not supported. No patch files found in {self._PATCH_DIR}"
                )

        self._target_version = detected_version

        if self._check_already_patched() and not force:
            raise RuntimeError(
                "FHI-aims source already patched (src/deeph_interface exists). Use --force to re-apply patch."
            )

        patch_cmd = [
            "patch",
            "-p1",
            "-i",
            str(self._patch_file),
            "-d",
            str(self.aims_src_dir),
        ]

        result = subprocess.run(patch_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Patch failed with return code {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        self._write_usage_file()

    def _generate_usage_markdown(self) -> str:
        """Generate detailed usage guide in markdown format."""
        return f"""# DeepH Warmstart for FHI-aims

This document explains how to build and use FHI-aims with DeepX warmstart support.

## Overview

The patch enables FHI-aims to load Hamiltonian matrices predicted by DeepX (DeepH-JAX) models and use them as the initial guess for SCF calculations. This can significantly accelerate SCF convergence for complex systems.

- **Target FHI-aims version**: {self._target_version}
- **Patch file**: `{self._patch_file.name if self._patch_file else "N/A"}`

---

## 1. Build FHI-aims with HDF5 Support

The patched FHI-aims requires HDF5 for reading DeepH Hamiltonian files.

### Step 1: Configure HDF5 Path

Edit `initial_cache.cmake` in the FHI-aims source directory and set your HDF5 installation path:

```cmake
set(DEEPH_HDF5_ROOT "/path/to/your/hdf5" CACHE PATH "Path to HDF5 installation root" FORCE)
```

### Step 2: Build FHI-aims

```bash
cd {self.aims_src_dir}
mkdir build_deeph && cd build_deeph
cmake -S .. -B . -C ../initial_cache.cmake
make -j 8
```

The resulting binary `aims.x` will have DeepH warmstart support enabled.

---

## 2. Use DeepH Warmstart for SCF Acceleration

### Directory Structure

```
your_calculation/
├── control.in              # FHI-aims input
├── geometry.in             # Structure file
└── deepx_warm/             # DeepH output (symlink or directory)
    ├── POSCAR              # Crystal structure
    ├── hamiltonian.h5      # Predicted Hamiltonian matrix
    ├── overlap.h5          # Overlap matrix
    └── info.json           # Metadata
```

### Quick Setup

You can create a symlink to your DeepH output directory:

```bash
cd your_calculation
ln -s /path/to/deeph_output ./deepx_warm
```

Alternatively, you can name the directory `deeph_warm` (both names are supported).

### Enable Warmstart in control.in

Add the following line to your `control.in`:

```
use_deepx_warmstart .true.
```

**Behavior**:

| Setting | Description |
|---------|-------------|
| `.true.` | FHI-aims loads Hamiltonian from `./deepx_warm/` (or `./deeph_warm/`) and uses it as the initial guess for SCF |
| `.false.` or omitted | Normal FHI-aims behavior (no DeepH warmstart) |

### Run FHI-aims

```bash
mpirun -np 4 aims.x > aims.out
```

---

## 3. Output Messages

When DeepX warmstart is active, you'll see messages like:

```
DeepH-JAX (DeepX) warmstart requested. Will search for ./deepx_warm or ./deeph_warm.
DeepX restart: loaded real-space Hamiltonian from ./deepx_warm or ./deeph_warm.
DeepX restart: using injected real-space Hamiltonian in first SCF iteration.
DeepX restart: applying one-shot no-mix density update in this iteration.
DeepX restart: first-iteration density hotstart (no Pulay/Broyden mixing history used).
```

---

## 4. Troubleshooting

### HDF5 Not Found During Build

**Problem**: CMake cannot find HDF5

**Solution**: Check `DEEPH_HDF5_ROOT` in `initial_cache.cmake` points to a valid HDF5 installation

### Hamiltonian Loading Fails

**Problem**: Error "use_deepx_warmstart is .true. but warm directory not found"

**Solution**: 
- Ensure `./deepx_warm/` or `./deeph_warm/` directory exists
- The directory must be in the same location as `control.in`
- Verify `hamiltonian.h5` exists in that directory

### Basis Set Mismatch

**Problem**: Error about chunk shape validation failing

**Solution**: 
- Ensure your DeepH model was trained with the same basis set
- Verify `info.json` in DeepH output matches your FHI-aims calculation
- Use the same species defaults

---

## 5. Technical Details

### Modified Files

The patch modifies the following FHI-aims source files:

| File | Modification |
|------|--------------|
| `CMakeLists.txt` | Add `USE_DEEPH_INTERFACE` option |
| `src/CMakeLists.txt` | Add DeepX interface source files |
| `src/dimensions.f90` | Register `use_deepx_warmstart` keyword |
| `src/read_control.f90` | Parse `use_deepx_warmstart` keyword |
| `src/runtime_choices.f90` | Add `use_deepx_warmstart` variable |
| `src/scf_solver.f90` | Integrate DeepX warmstart into SCF loop |
| `src/Makefile.backend` | Add DeepX interface compilation rules |
| `src/Makefile.hdf5` | Add HDF5 C library linking |

### New Files

| File | Purpose |
|------|---------|
| `initial_cache.cmake` | CMake configuration template |
| `src/deeph_interface/include/rs_mx_trans.f90` | Fortran interface module |
| `src/deeph_interface/src/rs_mx_trans.c` | C implementation for HDF5 I/O |

---

## 6. References

- [DeepH-dock Documentation](https://docs.deeph-pack.com/deeph-dock)
- [DeepH-pack Documentation](https://github.com/kYangLi/DeepH-pack-docs)
- [FHI-aims Manual](https://fhi-aims.org)

---

## 7. Citation

If you use this patch in your research, please cite:

1. DeepH-pack paper (for the method)
2. FHI-aims paper (for the DFT code)

---

*This file was auto-generated by `dock convert fhi-aims patch-aims` command.*
"""

    def _write_usage_file(self) -> None:
        """Write usage guide markdown file to aims source directory."""
        usage_file = self.aims_src_dir / "DEEPH_WARMSTART_USAGE.md"
        content = self._generate_usage_markdown()
        with open(usage_file, "w") as f:
            f.write(content)

    @property
    def usage_file_path(self) -> Path:
        """Path to the usage guide file that will be written."""
        return self.aims_src_dir / "DEEPH_WARMSTART_USAGE.md"

    @property
    def target_version(self) -> str:
        """Target FHI-aims version for this patch."""
        if self._target_version is None:
            raise RuntimeError("Version not yet detected")
        return self._target_version

    @property
    def patch_file(self) -> Path:
        """Path to the patch file."""
        if self._patch_file is None:
            raise RuntimeError("Patch file not yet selected")
        return self._patch_file
