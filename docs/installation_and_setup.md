<!-- markdownlint-disable MD033 -->
# Installation & Setup

## Install UV

To begin, configure your environment with uv, a fast and versatile Python package manager written in Rust. Please follow the installation instructions on the [official uv website](https://docs.astral.sh/uv/#installation).

On Linux or macOS, you can install `uv` with a single command (requires an internet connection):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

It is highly recommended that configuring high-performance mirrors based on your IP location. For example, for users in China, you cloud using the mirror provided by [TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)

```bash
# Add the following lines into ~/.config/uv/uv.toml
[[index]]
url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/"
default = true
```

## Create `deeph` python virtual environment

Create `python 3.13` environments with `uv`:

``` bash
mkdir ~/.uvenv
cd ~/.uvenv
uv venv deeph --python=3.13 # Create `deeph` venv in current dir
```

Then, the uv virtual environment can be activate with command,

```bash
source ~/.uvenv/deeph/bin/activate
```

Conveniently, all files installed into the `deeph` venv will be located in the ~/.uvenv/deeph directory.

## Quick Install (for Common Users)

Ensure you've activated the uv environment as described in the previous section, and that you're currently in the `deeph` environment you created.

Execute the following commands to automatically install `DeepH-dock` and all its dependencies:

```bash
uv pip install git+https://github.com/kYangLi/DeepH-dock
```

> **Note**: During the installation process, an internet connection is required. DeepH-dock and DeepH-pack can be installed under the same python venv.

## Install from Source (for Developers)

Ensure you've activated the uv environment as described in the previous section, and that you're currently in the `deeph` environment you created.

Execute the following commands to establish the development environment.

```bash
git clone https://github.com/kYangLi/DeepH-dock.git
# or, git clone https://github.com/<YourAccount>/DeepH-dock.git
# after fork the repository

cd DeepH-dock
uv pip install -e .[docs]
```

## Commandline Auto-Completion

Set up intelligent tab completion to save time and reduce errors when using the `dock` command. After configuration, press <kbd>Tab</kbd> to auto-complete subcommands, options, and file paths while in the `deeph` environment.

### Setup Instructions

#### Add to Shell Configuration

Paste the appropriate alias into your shell's config file:

- **Bash** (`~/.bashrc`):

    ```bash
    alias act-deeph='source ${HOME}/.uvenv/deeph/bin/activate ; eval "$(_DOCK_COMPLETE=bash_source dock)"'
    ```

- **Zsh** (`~/.zshrc`):

    ```bash
    alias act-deeph='source ${HOME}/.uvenv/deeph/bin/activate ; eval "$(_DOCK_COMPLETE=zsh_source dock)"'
    ```

- **Fish** (`~/.config/fish/completions/dock.fish`):

    ```bash
    alias act-deeph='source ${HOME}/.uvenv/deeph/bin/activate ; _DOCK_COMPLETE=fish_source dock | source'
    ```

Restart your terminal session to reload the shell settings.

#### Activate the Completion

Execute this command in your terminal:

```bash
act-deeph
```

This will:

- Activate the `deeph` Python virtual environment
- Enable tab completion for `dock` in your current session

### Using Auto-Completion

1. Type `dock` followed by a space
2. Press <kbd>Tab</kbd> to:
   - View available subcommands/options
   - Auto-complete partially typed arguments
   - Suggest file paths when relevant

> **Note**: Auto-completion persists only for the terminal session where `act-deeph` was executed. Re-run `act-deeph` in new sessions.
