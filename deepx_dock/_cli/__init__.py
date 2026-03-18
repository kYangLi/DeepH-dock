import importlib
import click
from pathlib import Path
from typing import List

from deepx_dock._version import __version__
from deepx_dock._cli.registry import registry


# ------------------------------------------------------------------------------
# Custom Click Group to preserve command order
# ------------------------------------------------------------------------------
class OrderedClickGroup(click.Group):
    def list_commands(self, ctx):
        return list(self.commands.keys())


# ------------------------------------------------------------------------------
# Auto register click function
# ------------------------------------------------------------------------------
def _auto_register_cli():
    # Define necessary parameters
    top_modules = ["analyze", "compute", "convert", "design"]
    package_root = Path(__file__).parent.parent
    # Discover the functions under the given top module
    for module in top_modules:
        module_path = package_root / module
        # Check
        if not module_path.is_dir():
            click.echo(f"[warning] Module '{module}' not found, skipping")
            continue
        # Find all cli file and register
        for cli_file in module_path.rglob("*_cli.py"):
            # Skip pyc dir
            if "__pycache__" in str(cli_file):
                continue
            # import (register) the target module
            relative_path = cli_file.relative_to(package_root)
            module_parts = list(relative_path.with_suffix("").parts)
            module_full_name = f"deepx_dock.{'.'.join(module_parts)}"
            importlib.import_module(module_full_name)


_auto_register_cli()


# ------------------------------------------------------------------------------
# Transfer the registered click function into command line
# ------------------------------------------------------------------------------
def _create_command(info):
    func = info["func"]

    def make_command():
        # Define the command
        @click.command(
            name=info["cli_name"],
            help=info["cli_help"],
            context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
        )
        @click.pass_context
        def command(ctx, **kwargs):
            ctx.ensure_object(dict)
            func(**kwargs)

        # Add click options
        for arg_decorator in info["cli_args"]:
            command = arg_decorator(command)
        # Return!
        return command

    return make_command()


def _add_commands_to_group(curr_group: click.Group, module_parts):
    # Deal with the functions inside current module
    functions = registry.get_functions_in_module(module_parts)
    for func_name in functions:
        module_func_name = f"{'.'.join(module_parts)}.{func_name}"

        info = registry.get_function_info(module_func_name)
        if info:
            command = _create_command(info)
            curr_group.add_command(command)
    # Deal with the sub-module inside current module
    submodules = registry.get_submodules(module_parts)
    for submodule_name in submodules:
        new_module_parts = module_parts + [submodule_name]

        @curr_group.group(name=submodule_name)
        def sub_group(): ...

        # Recursion!
        _add_commands_to_group(sub_group, new_module_parts)


def create_cli():
    # Create the root group
    @click.group(
        cls=OrderedClickGroup,
        name="dock",
        help="DeepH-dock: Materials computation and data analysis toolkit.",
        context_settings={"help_option_names": ["-h", "--help"], "show_default": True, "max_content_width": 120},
        invoke_without_command=True,
    )
    @click.option("-v", "--version", is_flag=True, help="Show the version and exit.")
    @click.pass_context
    def cli(ctx, version):
        if version:
            from deepx_dock._version import __version__

            click.echo(f"deepx-dock v{__version__}")
            ctx.exit()
        ctx.ensure_object(dict)
        if ctx.invoked_subcommand is None:
            click.echo(ctx.get_help())

    # Get module tree
    module_tree = registry.get_module_tree()
    # Create group for each top module
    for top_module_name in module_tree.keys():
        # Make top module group function
        def make_module_group(module_name):
            @cli.group(name=module_name, invoke_without_command=True)
            @click.pass_context
            def module_group(ctx, **kwargs):
                if ctx.invoked_subcommand is None:
                    click.echo(ctx.get_help())
                    ctx.exit()

            return module_group

        # Get top module group
        module_group = make_module_group(top_module_name)
        # Recursively add all parts below current top group
        _add_commands_to_group(module_group, [top_module_name])

    # Add a command to list all of the commands available
    @cli.command(name="ls", help="List all available commands.")
    def list_commands():
        # Prepare the necessary parameters
        import textwrap

        COLORS = {
            "command": "green",
            "module": "blue",
            "highlight": "cyan",
            "title": "bright_white",
            "separator": "yellow",
            "description": "white",
        }
        try:
            terminal_width = click.get_terminal_size()[0]
        except:
            terminal_width = 80

        def format_command_string(module: str, cli_name: str) -> str:
            full_command = f"dock {' '.join(module.split('.'))} {cli_name}"
            return click.style(full_command, fg=COLORS["command"], bold=True)

        def format_description(cli_help: str, indent: int = 2) -> List[str]:
            if not cli_help:
                return []
            # Figure out the wrapped lines format
            available_width = terminal_width - indent
            wrapped_lines = textwrap.wrap(cli_help, width=available_width, subsequent_indent=" " * indent)
            if wrapped_lines:
                wrapped_lines[0] = " " * indent + wrapped_lines[0]
            # Get!
            return wrapped_lines

        def display_command_info(module_func_name: str):
            info = registry.get_function_info(module_func_name)
            if not info:
                return
            # Prepare the command
            module = info["module"]
            cli_name = info["cli_name"]
            cli_help = info["cli_help"]
            # Print the command
            click.echo(format_command_string(module, cli_name))
            # Print the help
            if cli_help:
                desc_lines = format_description(cli_help)
                for line in desc_lines:
                    click.echo(click.style(line, fg=COLORS["description"]))
                click.echo()

        # Start Print
        functions = registry.list_functions()
        click.echo(click.style("✨ Available Commands ✨", fg=COLORS["title"], bold=True))
        # Count the command quantity
        click.echo(click.style(f"Total: {len(functions)} commands", fg=COLORS["highlight"]))
        # Split line
        separator = click.style("─" * min(terminal_width, 80), fg=COLORS["separator"])
        click.echo(separator)
        click.echo()
        # Show the command with group
        # - Get grouped commands
        grouped_commands = {}
        for module_func_name in functions:
            info = registry.get_function_info(module_func_name)
            if info:
                module = info["module"]
                grouped_commands.setdefault(module, []).append(module_func_name)
        # - Show grouped commands
        for module in sorted(grouped_commands.keys()):
            # Show title
            module_title = click.style(f"📦 {module}", fg=COLORS["module"], bold=True)
            click.echo(module_title)
            click.echo(click.style("─" * (len(module) + 4), fg=COLORS["module"]))
            # Show all sub commands
            for module_func_name in sorted(grouped_commands[module]):
                display_command_info(module_func_name)
            click.echo()

        # 使用说明
        click.echo(click.style("💡 Usage Examples:", fg=COLORS["highlight"], bold=True))
        click.echo(
            "  "
            + click.style("dock", fg=COLORS["command"])
            + click.style(" module command [options]", fg="bright_white")
        )
        click.echo("  " + click.style("dock --help", fg=COLORS["command"]) + " - Show general help")
        click.echo("  " + click.style("dock module --help", fg=COLORS["command"]) + " - Show help for a module")
        click.echo(
            "  " + click.style("dock module command --help", fg=COLORS["command"]) + " - Show help for a command"
        )
        click.echo(separator)

    # Add completion command group
    from deepx_dock._cli.completion import completion_group

    cli.add_command(completion_group)

    # Reorder commands: ls second to last, completion last
    from collections import OrderedDict

    ordered_commands = OrderedDict()
    desired_order = ["analyze", "compute", "convert", "design", "ls", "completion"]
    for name in desired_order:
        if name in cli.commands:
            ordered_commands[name] = cli.commands[name]
    cli.commands = ordered_commands

    # Return CLI!
    return cli


cli = create_cli()
__all__ = ["cli", "registry"]
