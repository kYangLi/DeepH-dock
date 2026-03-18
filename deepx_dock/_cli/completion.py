"""
Shell completion for DeepX-dock CLI.

Usage:
    dock completion init bash
    dock completion update
"""

import click
from pathlib import Path
from typing import Dict, List, Any
import shutil


# ==============================================================================
# Part 1: Command Extraction
# ==============================================================================


def _extract_params_from_cli_args(cli_args: List) -> List[str]:
    """
    Extract parameter names from cli_args decorator list.

    Strategy: Inspect decorator closure variables to extract parameter names.

    Args:
        cli_args: Click decorator list (click.option, click.argument, etc.)

    Returns:
        Parameter name list, e.g. ['-b', '--benchmark-dft-dir', '--target', '-h', '--help']
    """
    params = []

    for decorator in cli_args:
        try:
            # Check if decorator has closure variables
            if hasattr(decorator, "__closure__") and decorator.__closure__:
                # Get parameter type (Option or Argument)
                param_type = None
                if len(decorator.__closure__) > 1:
                    param_type = decorator.__closure__[1].cell_contents

                # Get parameter names (opts tuple)
                if len(decorator.__closure__) > 2:
                    opts_tuple = decorator.__closure__[2].cell_contents

                    # Only include options (start with -)
                    if isinstance(opts_tuple, tuple):
                        for opt in opts_tuple:
                            if isinstance(opt, str) and opt.startswith("-"):
                                params.append(opt)
        except Exception:
            pass

    params = sorted(set(params))

    return params


def extract_commands_from_registry() -> Dict[str, Dict[str, Any]]:
    """
    Extract all commands and parameters from registry.

    Returns:
        {
            'analyze.error.entries': {
                'params': ['-b', '--benchmark-dft-dir', '--target', ...],
                'help': 'Error distribution for each entries...'
            },
            ...
        }
    """
    from deepx_dock._cli.registry import registry

    commands = {}

    all_functions = registry.list_functions()

    for module_func_name in all_functions:
        info = registry.get_function_info(module_func_name)

        if info:
            cli_args = info.get("cli_args", [])
            params = _extract_params_from_cli_args(cli_args)

            params.extend(["-h", "--help"])
            params = sorted(set(params))

            commands[module_func_name] = {"params": params, "help": info.get("cli_help", "")}

    return commands


# ==============================================================================
# Part 2: Completion Generation
# ==============================================================================


def generate_bash_completion(commands: Dict[str, Dict[str, Any]]) -> str:
    """Generate Bash static completion script."""

    top_commands = set()
    for cmd_key in commands.keys():
        parts = cmd_key.split(".")
        if len(parts) >= 1:
            top_commands.add(parts[0])

    script = (
        '''# DeepX-dock Bash Completion
# Generated automatically - DO NOT EDIT

_dock_static_completion() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"
    local cmd="${COMP_WORDS[1]}"
    local subcmd="${COMP_WORDS[2]}"
    local subsubcmd="${COMP_WORDS[3]}"
    
    # Level 1: Main commands
    if [[ $COMP_CWORD -eq 1 ]]; then
        COMPREPLY=($(compgen -W "'''
        + " ".join(sorted(top_commands))
        + """ ls completion" -- "$cur"))
        return 0
    fi
    
    # Level 2: Subcommands
    if [[ $COMP_CWORD -eq 2 ]]; then
        case "$cmd" in
"""
    )

    for top_cmd in sorted(top_commands):
        subcommands = set()
        for cmd_key in commands.keys():
            if cmd_key.startswith(f"{top_cmd}."):
                parts = cmd_key.split(".")
                if len(parts) >= 2:
                    subcommands.add(parts[1])

        if subcommands:
            script += f'''            {top_cmd})
                COMPREPLY=($(compgen -W "{" ".join(sorted(subcommands))}" -- "$cur"))
                return 0
                ;;
'''

    # Add completion command subcommands manually
    script += """            completion)
                COMPREPLY=($(compgen -W "init update" -- "$cur"))
                return 0
                ;;
"""

    script += """            *)
                ;;
        esac
    fi
    
    # Level 3: Sub-subcommands
    if [[ $COMP_CWORD -eq 3 ]]; then
        case "$cmd $subcmd" in
"""

    for top_cmd in sorted(top_commands):
        second_level = set()
        for cmd_key in commands.keys():
            if cmd_key.startswith(f"{top_cmd}."):
                parts = cmd_key.split(".")
                if len(parts) >= 2:
                    second_level.add(".".join(parts[:2]))

        for second_cmd in sorted(second_level):
            third_level = set()
            for cmd_key in commands.keys():
                if cmd_key.startswith(f"{second_cmd}."):
                    parts = cmd_key.split(".")
                    if len(parts) >= 3:
                        third_level.add(parts[2])

            if third_level:
                script += f'''            '{second_cmd.replace(".", " ")}')
                COMPREPLY=($(compgen -W "{" ".join(sorted(third_level))}" -- "$cur"))
                return 0
                ;;
'''

    script += """            *)
                ;;
        esac
    fi
    
    # Parameters completion (only when user types - or --)
    if [[ "$cur" == -* ]]; then
        case "$cmd $subcmd $subsubcmd" in
"""

    for cmd_key, cmd_info in commands.items():
        params = cmd_info.get("params", [])
        if params:
            parts = cmd_key.split(".")
            if len(parts) == 1:
                case_pattern = f"            '{parts[0]}  '"
            elif len(parts) == 2:
                case_pattern = f"            '{parts[0]} {parts[1]} '"
            else:
                case_pattern = f"            '{parts[0]} {parts[1]} {parts[2]}'"

            script += f'''{case_pattern})
                COMPREPLY=($(compgen -W "{" ".join(params)}" -- "$cur"))
                return 0
                ;;
'''

    script += """        esac
    fi
    
    # File path completion (default)
    compopt -o default
    COMPREPLY=()
    return 0
}

_dock_check_version() {
    local dock_exe=$(which dock 2>/dev/null)
    if [[ -z "$dock_exe" ]]; then
        return 1
    fi
    
    local cache_dir="$HOME/.cache/deepx"
    local path_cache="$cache_dir/dock_path.txt"
    local mtime_cache="$cache_dir/dock_mtime.txt"
    
    mkdir -p "$cache_dir" 2>/dev/null
    
    local cached_path=$(cat "$path_cache" 2>/dev/null)
    local cached_mtime=$(cat "$mtime_cache" 2>/dev/null)
    local current_mtime=$(stat -c %Y "$dock_exe" 2>/dev/null)
    
    if [[ "$cached_path" != "$dock_exe" ]] || [[ "$cached_mtime" != "$current_mtime" ]]; then
        echo "$dock_exe" > "$path_cache"
        echo "$current_mtime" > "$mtime_cache"
        return 0
    fi
    
    return 1
}

_dock_lazy_completion() {
    if ! command -v dock &>/dev/null; then
        return
    fi
    
    if _dock_check_version; then
        dock completion update >/dev/null 2>&1
        source ~/.cache/deepx/dock_completion.bash
        return
    fi
    
    _dock_static_completion "$@"
}

complete -F _dock_lazy_completion dock
"""

    return script


def generate_zsh_completion(commands: Dict[str, Dict[str, Any]]) -> str:
    """Generate Zsh static completion script."""

    top_commands = set()
    for cmd_key in commands.keys():
        parts = cmd_key.split(".")
        if len(parts) >= 1:
            top_commands.add(parts[0])

    script = """#compdef dock
# DeepX-dock Zsh Completion
# Generated automatically - DO NOT EDIT

_dock() {
    local curcontext="$curcontext" state line
    typeset -A opt_args
    
    _arguments -C \\
        '1: :->cmds' \\
        '2: :->subcmds' \\
        '3: :->subsubcmds' \\
        '*::args:->args'
    
    case $state in
        cmds)
            _arguments \\
                '(-h --help)'{-h,--help}'[Show help]' \\
                '--version[Show version]'
            _describe 'command' cmds
            ;;
        subcmds)
            case $line[1] in
"""

    for top_cmd in sorted(top_commands):
        subcommands = set()
        for cmd_key in commands.keys():
            if cmd_key.startswith(f"{top_cmd}."):
                parts = cmd_key.split(".")
                if len(parts) >= 2:
                    subcommands.add(parts[1])

        if subcommands:
            subcmd_list = " ".join(sorted(subcommands))
            script += f"""                {top_cmd})
                    _describe '{top_cmd} subcommand' '({subcmd_list})'
                    ;;
"""

    # Add completion command subcommands manually
    script += """                completion)
                    _describe 'completion subcommand' '(init update)'
                    ;;
"""

    script += """                completion)
                    _describe 'completion subcommand' '(init update)'
                    ;;
            esac
            ;;
        subsubcmds)
            case "$line[1] $line[2]" in
"""

    for top_cmd in sorted(top_commands):
        second_level = set()
        for cmd_key in commands.keys():
            if cmd_key.startswith(f"{top_cmd}."):
                parts = cmd_key.split(".")
                if len(parts) >= 2:
                    second_level.add(".".join(parts[:2]))

        for second_cmd in sorted(second_level):
            third_level = set()
            for cmd_key in commands.keys():
                if cmd_key.startswith(f"{second_cmd}."):
                    parts = cmd_key.split(".")
                    if len(parts) >= 3:
                        third_level.add(parts[2])

            if third_level:
                third_list = " ".join(sorted(third_level))
                script += f"""                {second_cmd.replace(".", " ")})
                    _describe 'subcommand' '({third_list})'
                    ;;
"""

    script += """            esac
            ;;
        args)
            case "$line[1] $line[2] $line[3]" in
"""

    for cmd_key, cmd_info in commands.items():
        params = cmd_info.get("params", [])
        if params:
            parts = cmd_key.split(".")
            if len(parts) == 1:
                cmd_pattern = f"{parts[0]}  )"
            elif len(parts) == 2:
                cmd_pattern = f"{parts[0]} {parts[1]} )"
            else:
                cmd_pattern = f"{parts[0]} {parts[1]} {parts[2]})"

            args_str = " \\\n".join([f"                '{p}[{p}]'" for p in params[:5]])
            if len(params) > 5:
                args_str += f" \\\n                # ... and {len(params) - 5} more options"

            script += f"""                {cmd_pattern}
                    _arguments \\
{args_str}
                    ;;
"""

    script += """            esac
            ;;
    esac
}

cmds=(
"""

    for top_cmd in sorted(top_commands):
        script += f"    '{top_cmd}:{top_cmd} commands'\n"

    script += """    'ls:List all available commands'
    'completion:Shell completion management'
)

_dock
"""

    return script


def generate_fish_completion(commands: Dict[str, Dict[str, Any]]) -> str:
    """Generate Fish static completion script."""

    script = """# DeepX-dock Fish Completion
# Generated automatically - DO NOT EDIT

complete -c dock -f

"""

    top_commands = set()
    for cmd_key in commands.keys():
        parts = cmd_key.split(".")
        if len(parts) >= 1:
            top_commands.add(parts[0])

    for top_cmd in sorted(top_commands):
        script += f"complete -c dock -n '__fish_use_subcommand' -a '{top_cmd}'\n"

    script += "complete -c dock -n '__fish_use_subcommand' -a 'ls'\n"
    script += "complete -c dock -n '__fish_use_subcommand' -a 'completion'\n\n"

    script += "# Subcommands\n"
    for top_cmd in sorted(top_commands):
        subcommands = set()
        for cmd_key in commands.keys():
            if cmd_key.startswith(f"{top_cmd}."):
                parts = cmd_key.split(".")
                if len(parts) >= 2:
                    subcommands.add(parts[1])

        for subcmd in sorted(subcommands):
            script += f"complete -c dock -n '__fish_seen_subcommand_from {top_cmd}; and not __fish_seen_subcommand_from {subcmd}' -a '{subcmd}'\n"

    script += "\n# Completion subcommands\n"
    script += "complete -c dock -n '__fish_seen_subcommand_from completion; and not __fish_seen_subcommand_from init update' -a 'init'\n"
    script += "complete -c dock -n '__fish_seen_subcommand_from completion; and not __fish_seen_subcommand_from init update' -a 'update'\n"

    script += "\n# Parameters\n"

    for cmd_key, cmd_info in commands.items():
        params = cmd_info.get("params", [])
        if params:
            parts = cmd_key.split(".")
            if len(parts) >= 2:
                conditions = [f"__fish_seen_subcommand_from {part}" for part in parts]
                condition = "; and ".join(conditions)

                for param in params[:10]:
                    param_name = param.lstrip("-")
                    script += f"complete -c dock -n '{condition}' -l '{param_name}'\n"

    return script


# ==============================================================================
# Part 3: CLI Commands
# ==============================================================================


@click.group(name="completion")
def completion_group():
    """Shell completion management commands."""
    pass


@completion_group.command(name="init")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
@click.option("--auto-setup", is_flag=True, help="Automatically add to shell config without asking.")
def init_completion(shell: str, auto_setup: bool):
    """Initialize shell completion for dock.

    This will generate the completion script and optionally add it to your shell config.

    Example:
        dock completion init bash
        dock completion init bash --auto-setup
    """
    commands = extract_commands_from_registry()

    if shell == "bash":
        script = generate_bash_completion(commands)
        cache_path = Path.home() / ".cache" / "deepx" / "dock_completion.bash"
        config_file = Path.home() / ".bashrc"
    elif shell == "zsh":
        script = generate_zsh_completion(commands)
        cache_path = Path.home() / ".cache" / "deepx" / "dock_completion.zsh"
        config_file = Path.home() / ".zshrc"
    elif shell == "fish":
        script = generate_fish_completion(commands)
        cache_path = Path.home() / ".cache" / "deepx" / "dock_completion.fish"
        config_file = None

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path.write_text(script)

    dock_exe_path = shutil.which("dock")
    if dock_exe_path:
        dock_exe = Path(dock_exe_path)
        dock_mtime = int(dock_exe.stat().st_mtime)
        (cache_path.parent / "dock_path.txt").write_text(str(dock_exe))
        (cache_path.parent / "dock_mtime.txt").write_text(str(dock_mtime))

    click.echo(click.style("✅ Completion script generated successfully!", fg="green", bold=True))
    click.echo(f"📁 Location: {cache_path}")
    click.echo()

    if shell == "fish":
        click.echo(click.style("🐟 Fish Shell Detected", fg="cyan", bold=True))
        click.echo()
        click.echo("Fish will automatically load the completion script.")
        click.echo("You may need to restart your shell or run:")
        click.echo(click.style("   exec fish", fg="cyan"))
        click.echo()
        click.echo("Try it out:")
        click.echo(click.style("   dock <TAB>", fg="green"))
        return

    # For bash and zsh
    setup_line = f"[ -f {cache_path} ] && source {cache_path}"

    # Check if already in config file
    already_configured = False
    if config_file and config_file.exists():
        config_content = config_file.read_text()
        if setup_line in config_content:
            already_configured = True

    if already_configured:
        click.echo(click.style("ℹ️  Completion is already configured in your shell.", fg="yellow"))
        click.echo(f"📄 Config file: {config_file}")
        click.echo()
        click.echo("To apply changes, reload your shell:")
        click.echo(click.style(f"   source {config_file}", fg="cyan"))
        click.echo()
        click.echo("Or try it now:")
        click.echo(click.style(f"   source {cache_path}", fg="cyan"))
        click.echo(click.style("   dock <TAB>", fg="green"))
        return

    # Ask user if they want to auto-setup
    should_setup = auto_setup
    if not auto_setup:
        click.echo(click.style("📝 Setup Options:", fg="cyan", bold=True))
        click.echo()
        click.echo("Would you like to automatically add the completion to your shell config?")
        click.echo(f"This will add the following line to {config_file}:")
        click.echo(click.style(f"   {setup_line}", fg="yellow"))
        click.echo()
        should_setup = click.confirm("Add to shell config?", default=True)

    if should_setup:
        # Add to config file
        with open(config_file, "a") as f:
            f.write(f"\n# DeepX-dock completion\n{setup_line}\n")

        click.echo()
        click.echo(click.style("✅ Successfully added to shell config!", fg="green", bold=True))
        click.echo(f"📄 Updated: {config_file}")
        click.echo()
        click.echo("To apply changes now, run:")
        click.echo(click.style(f"   source {config_file}", fg="cyan"))
        click.echo()
        click.echo("Or try it in the current session:")
        click.echo(click.style(f"   source {cache_path}", fg="cyan"))
        click.echo(click.style("   dock <TAB>", fg="green"))
    else:
        click.echo()
        click.echo(click.style("📝 Manual Setup Instructions:", fg="cyan", bold=True))
        click.echo()
        click.echo("Add this line to your shell config file:")
        click.echo(click.style(f"   {setup_line}", fg="yellow"))
        click.echo()
        click.echo(f"Then reload your shell:")
        click.echo(click.style(f"   source {config_file}", fg="cyan"))
        click.echo()
        click.echo("Or try it now:")
        click.echo(click.style(f"   source {cache_path}", fg="cyan"))
        click.echo(click.style("   dock <TAB>", fg="green"))


@completion_group.command(name="update")
def update_completion():
    """Update completion cache (usually called automatically)."""
    commands = extract_commands_from_registry()

    cache_dir = Path.home() / ".cache" / "deepx"
    cache_dir.mkdir(parents=True, exist_ok=True)

    bash_script = generate_bash_completion(commands)
    (cache_dir / "dock_completion.bash").write_text(bash_script)

    zsh_script = generate_zsh_completion(commands)
    (cache_dir / "dock_completion.zsh").write_text(zsh_script)

    fish_script = generate_fish_completion(commands)
    (cache_dir / "dock_completion.fish").write_text(fish_script)

    click.echo("✅ Completion cache updated.")
