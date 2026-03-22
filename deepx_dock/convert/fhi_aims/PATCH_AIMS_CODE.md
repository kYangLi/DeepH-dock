# FHI-aims DeepX Warmstart Patch

This patch enables FHI-aims to restart SCF calculations using Hamiltonian matrices predicted by DeepX (DeepH-JAX) models.

## Patch Information

- **Patch File**: `patch_deepx_fhiaims_20250822_1.diff`
- **Target Version**: FHI-aims 250822_1 (Aug 25, 2025 release, update 1)
- **Patch Version**: 250822_1

## Prerequisites

1. **FHI-aims Source Code**: Version 250822_1
   - Download from the [official FHI-aims website](https://fhi-aims.org/news/releases)

2. **HDF5 Library**: Required for reading DeepX Hamiltonian files
   - Version 1.10+ recommended
   - Must have C library installed

3. **CMake**: Version 3.15+
   - For building with CMake support

## How to Apply the Patch

### Step 1: Navigate to FHI-aims Source Directory

```bash
cd fhi-aims.250822 # Or whatever it calls
```

### Step 2: Copy Patch File

```bash
cp /path/to/patch_deepx_fhiaims_20250822_1.diff .
```

### Step 3: Apply the Patch

```bash
patch -p1 < patch_deepx_fhiaims_20250822_1.diff
```

If successful, you should see output like:

```txt
patching file CMakeLists.txt
patching file .gitignore
patching file initial_cache.cmake
patching file src/CMakeLists.txt
patching file src/deeph_interface/include/rs_mx_trans.f90
patching file src/deeph_interface/src/rs_mx_trans.c
patching file src/dimensions.f90
patching file src/fock_routines/calculate_fock_matrix_4.f90
patching file src/Makefile.backend
patching file src/Makefile.hdf5
patching file src/read_control.f90
patching file src/runtime_choices.f90
patching file src/scf_solver.f90
```

## How to Build FHI-aims with DeepX Support

### Option 1: Using CMake (Recommended)

1. Customize the provided `initial_cache.cmake`:

   ```bash
   # Edit initial_cache.cmake to set your compilers and library paths
   vim initial_cache.cmake
   ```

2. Configure and build:

   ```bash
   mkdir build_deeph
   cd build_deeph
   cmake -S .. -B . -C ../initial_cache.cmake
   make -j 8
   ```

### Option 2: Using Makefile

1. Set environment variables:

   ```bash
   export HDF5_HOME=/path/to/your/hdf5
   export USE_DEEPH_INTERFACE=yes
   ```

2. Build:

   ```bash
   cd src
   make -f Makefile.hdf5
   ```

## How to Use DeepX Warmstart

### Step 1: Prepare DeepX Hamiltonian Data

Before running FHI-aims, you need to prepare the Hamiltonian data from your DeepX model:

1. Run DeepX model inference to generate Hamiltonian prediction
2. Save the Hamiltonian matrix in HDF5 format (see DeepX documentation)
3. Place the files in a directory named either:
   - `./deepx_warm/`
   - `./deeph_warm/` (alias dirname)

The directory should contain:

```bash
deepx_warm/
├── hamiltonian.h5     # Hamiltonian matrix in DeepH/DeepX HDF5 format
├── overlap.h5         # (Optional) Overlap matrix
└── ...
```

### Step 2: Add Control.in Keyword

Add the following keyword to your `control.in` file:

```bash
use_deepx_warmstart .true.
```

### Step 3: Run FHI-aims

Run FHI-aims as usual. The code will:

1. Search for `./deepx_warm/` or `./deeph_warm/` directory
2. Load the Hamiltonian matrix from `hamiltonian.h5`
3. Use it as the initial guess for the first SCF iteration
4. Perform one-shot density update (no mixing) in the first iteration
5. Continue normal SCF iterations from iteration 2

## Output Messages

When DeepX warmstart is active, you'll see messages like:

```txt
DeepH-JAX (DeepX) warmstart requested. Will search for ./deepx_warm or ./deeph_warm.
DeepX restart: loaded real-space Hamiltonian from ./deepx_warm or ./deeph_warm.
DeepX restart: using injected real-space Hamiltonian in first SCF iteration.
DeepX restart: applying one-shot no-mix density update in this iteration.
DeepX restart: first-iteration density hotstart (no Pulay/Broyden mixing history used).
```

## Troubleshooting

### Patch Fails to Apply

**Problem**: Patch fails with "patching file XXX: Hunk #1 FAILED"

**Solution**:

- Ensure you're using FHI-aims version 250822
- Make sure you're applying from the correct directory level (should be in `fhi-aims.250822/`)
- Check if the file has been modified from the original

### HDF5 Not Found During Build

**Problem**: CMake or Makefile cannot find HDF5

**Solution**:

- Set `HDF5_ROOT` or `HDF5_HOME` environment variable
- For CMake: set in `initial_cache.cmake`
- For Makefile: export `HDF5_HOME=/path/to/hdf5`

### Hamiltonian Loading Fails

**Problem**: Error message "use_deepx_warmstart is .true. but warm directory not found"

**Solution**:

- Create `./deepx_warm/` or `./deeph_warm/` directory
- Ensure `hamiltonian.h5` exists in that directory
- Verify HDF5 file is readable (use `h5dump -H hamiltonian.h5`)

### Basis Set Mismatch

**Problem**: Error about chunk shape validation failing

**Solution**:

- Ensure DeepX model was trained with the same basis set
- Verify `info.json` in DeepX output matches your FHI-aims calculation
- Check that the same species defaults are used

## Technical Details

### Modified Files

The patch modifies the following FHI-aims source files:

1. **CMakeLists.txt**: Add `USE_DEEPH_INTERFACE` option
2. **src/CMakeLists.txt**: Add DeepX interface source files
3. **src/dimensions.f90**: Register `use_deepx_warmstart` keyword
4. **src/read_control.f90**: Parse `use_deepx_warmstart` keyword
5. **src/runtime_choices.f90**: Add `use_deepx_warmstart` logical variable
6. **src/scf_solver.f90**: Integrate DeepX warmstart into SCF loop
7. **src/Makefile.backend**: Add DeepX interface compilation rules
8. **src/Makefile.hdf5**: Add HDF5 C library linking

### New Files

The patch adds the following new files:

1. **initial_cache.cmake**: CMake configuration template
2. **src/deeph_interface/include/rs_mx_trans.f90**: Fortran interface module
3. **src/deeph_interface/src/rs_mx_trans.c**: C implementation for HDF5 I/O

### Algorithm

The DeepX warmstart algorithm:

1. **First SCF iteration**:
   - Load Hamiltonian from `deepx_warm/hamiltonian.h5`
   - Map DeepX basis order to FHI-aims basis order
   - Convert units (eV → Hartree)
   - Inject into FHI-aims Hamiltonian array
   - Skip real-space integration
   - Perform one-shot density update (no mixing)

2. **Second and subsequent iterations**:
   - Normal SCF procedure with Pulay/Broyden mixing

### Basis Set Mapping

The patch handles the basis set ordering difference between DeepX and FHI-aims:

- DeepX: Atoms sorted by chemical symbol, then by Z, then by index
- DeepX: Orbitals ordered as per FHI-aims internal order (no sorting)
- FHI-aims: Native basis ordering

The mapping is performed automatically in `rs_mx_trans.c`.

## Citation

If you use this patch in your research, please cite:

1. DeepH-pack paper (for the method)
2. FHI-aims paper (for the DFT code)

## License

This patch is provided under the same license as FHI-aims. The DeepX-related code portions are licensed under GPL-3.0-or-later.

## Contact

For questions or issues related to this patch:

- DeepX/DeepH team: <deeph-pack@outlook.com>
- FHI-aims: See official FHI-aims documentation
