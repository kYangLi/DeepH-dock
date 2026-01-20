from pathlib import Path

# Defined from ase/calculators/aims.py
float_keys = [
    'charge',
    'charge_mix_param',
    'default_initial_moment',
    'fixed_spin_moment',
    'hartree_convergence_parameter',
    'harmonic_length_scale',
    'ini_linear_mix_param',
    'ini_spin_mix_param',
    'initial_moment',
    'MD_MB_init',
    'MD_time_step',
    'prec_mix_param',
    'set_vacuum_level',
    'spin_mix_param',
]

exp_keys = [
    'basis_threshold',
    'occupation_thr',
    'sc_accuracy_eev',
    'sc_accuracy_etot',
    'sc_accuracy_forces',
    'sc_accuracy_rho',
    'sc_accuracy_stress',
]

string_keys = [
    'communication_type',
    'density_update_method',
    'KS_method',
    'mixer',
    'output_level',
    'packed_matrix_format',
    'relax_unit_cell',
    'restart',
    'restart_read_only',
    'restart_write_only',
    'spin',
    'total_energy_method',
    'qpe_calc',
    'xc',
    'species_dir',
    'run_command',
    'plus_u',
]

int_keys = [
    'empty_states',
    'ini_linear_mixing',
    'max_relaxation_steps',
    'max_zeroin',
    'multiplicity',
    'n_max_pulay',
    'sc_iter_limit',
    'walltime',
]

bool_keys = [
    'collect_eigenvectors',
    'compute_forces',
    'compute_kinetic',
    'compute_numerical_stress',
    'compute_analytical_stress',
    'compute_heat_flux',
    'distributed_spline_storage',
    'evaluate_work_function',
    'final_forces_cleaned',
    'hessian_to_restart_geometry',
    'load_balancing',
    'MD_clean_rotations',
    'MD_restart',
    'override_illconditioning',
    'override_relativity',
    'restart_relaxations',
    'squeeze_memory',
    'symmetry_reduced_k_grid',
    'use_density_matrix',
    'use_dipole_correction',
    'use_local_index',
    'use_logsbt',
    'vdw_correction_hirshfeld',
]

list_keys = [
    'init_hess',
    'k_grid',
    'k_offset',
    'MD_run',
    'MD_schedule',
    'MD_segment',
    'mixer_threshold',
    'occupation_type',
    'output',
    'cube',
    'preconditioner',
    'relativistic',
    'relax_geometry',
]

def read_from_control_in(control_in_path: str | Path):
    control_in_path = Path(control_in_path)
    parameters = {}
    
    with open(control_in_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # Strip comments
        if '#' in line:
            line = line.split('#')[0]
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        key = parts[0]
        args = parts[1:]
        
        if key == 'output':
            if 'output' not in parameters:
                parameters['output'] = []
            parameters['output'].append(" ".join(args))
            continue
            
        # Values handling
        if key in float_keys or key in exp_keys:
            if args:
                parameters[key] = float(args[0])
        elif key in int_keys:
            if args:
                parameters[key] = int(args[0])
        elif key in bool_keys:
            if not args:
                # e.g. vdw_correction_hirshfeld
                parameters[key] = True
            else:
                val = args[0].lower()
                if val == '.true.':
                    parameters[key] = True
                elif val == '.false.':
                    parameters[key] = False
                else:
                    # Fallback or error? Aims uses .true./.false.
                    # But vdw_correction_hirshfeld is bool key but switch-like.
                    pass
        elif key in list_keys:
            # If it's a list key, we might want to store as list or string.
            # k_grid is clearly a list of ints.
            # relativistic is "atomic_zora scalar" (string or list of strings)
            
            # Check known types in dataclass to decide?
            # or always return native types.
            
            if key == 'k_grid':
                parameters[key] = [int(x) for x in args]
            elif key == 'k_offset':
                parameters[key] = [float(x) for x in args]
            else:
                    # Default to string for complex stuff like relativistic or occupation_type
                    # ASE write_control handles list or string.
                    parameters[key] = " ".join(args)
        elif key in string_keys:
            parameters[key] = " ".join(args)
        else:
            # not set species info
            if ("species" in key) or ("species" in args):
                break
            # Unknown key
            # try to guess
            if not args:
                parameters[key] = True
            elif len(args) == 1:
                val = args[0]
                try:
                    parameters[key] = int(val)
                except ValueError:
                    try:
                        parameters[key] = float(val)
                    except ValueError:
                        if val.lower() == '.true.': 
                            parameters[key] = True
                        elif val.lower() == '.false.': 
                            parameters[key] = False
                        else: 
                            parameters[key] = val
            else:
                parameters[key] = " ".join(args)

    # Construct object
    # We need to map parameters to fields and extra

    return parameters

if __name__ == '__main__':
    # Create a dummy control.in
    ctrl_in_path = Path('./control.in')
    print(read_from_control_in(ctrl_in_path))
