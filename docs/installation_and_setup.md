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

For publish version:
```bash
uv pip install deepx-dock
```

For development version:
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

## Shell Auto-Completion

DeepH-dock provides intelligent tab completion for `dock` commands, enabling fast and error-free command entry. The completion system supports **Bash**, **Zsh**, and **Fish** shells.

### Quick Setup

Run the following command to initialize shell completion:

```bash
dock completion init bash    # For Bash users
dock completion init zsh     # For Zsh users
dock completion init fish    # For Fish users
```

The command will:
1. Generate the completion script
2. Check if completion is already configured
3. Ask whether to automatically add it to your shell config file (default: Y)
4. Provide instructions for immediate activation

### Interactive Setup (Recommended)

When you run `dock completion init bash`, you'll see:

```
✅ Completion script generated successfully!
📁 Location: /home/user/.cache/deepx/dock_completion.bash

📝 Setup Options:

Would you like to automatically add the completion to your shell config?
This will add the following line to /home/user/.bashrc:
   [ -f /home/user/.cache/deepx/dock_completion.bash ] && source ...

Add to shell config? [Y/n]: 
```

- Press **Enter** (or type **Y**) to automatically add to your shell config
- Type **n** for manual setup instructions

### Automatic Mode (Skip Confirmation)

To skip the confirmation prompt and automatically configure completion:

```bash
dock completion init bash --auto-setup
```

### Activate Immediately

After setup, reload your shell configuration to enable completion immediately:

**Bash:**
```bash
source ~/.bashrc
```

**Zsh:**
```bash
source ~/.zshrc
```

**Fish:**
```bash
exec fish
```

Or load completion in the current session without restarting:

```bash
source ~/.cache/deepx/dock_completion.bash  # Bash
source ~/.cache/deepx/dock_completion.zsh   # Zsh
```

### Using Auto-Completion

Once configured, press <kbd>Tab</kbd> to auto-complete:

```bash
dock <TAB>                        # View all commands
dock analyze <TAB>                # View analyze subcommands
dock analyze error <TAB>          # View error subcommands
dock analyze error entries --<TAB> # View available options
```

### Features

- **Fast Performance**: < 20ms response time (vs 500ms with traditional Click completion)
- **Smart Detection**: Automatically detects if completion is already configured
- **Version Aware**: Automatically updates when switching virtual environments or reinstalling
- **Multi-Shell Support**: Native support for Bash, Zsh, and Fish

### Manual Setup

If you prefer manual configuration or the automatic setup fails, add this line to your shell config file:

**Bash** (`~/.bashrc`):
```bash
[ -f ~/.cache/deepx/dock_completion.bash ] && source ~/.cache/deepx/dock_completion.bash
```

**Zsh** (`~/.zshrc`):
```bash
[ -f ~/.cache/deepx/dock_completion.zsh ] && source ~/.cache/deepx/dock_completion.zsh
```

**Fish** (`~/.config/fish/completions/dock.fish`):
```bash
# Fish automatically loads completions from ~/.cache/deepx/dock_completion.fish
```

### Troubleshooting

**Completion not working?**

1. Verify completion is loaded:
   ```bash
   complete -p dock  # Bash
   ```

2. Check if script exists:
   ```bash
   ls ~/.cache/deepx/dock_completion.bash
   ```

3. Reload manually:
   ```bash
   source ~/.cache/deepx/dock_completion.bash
   ```

**Completion outdated after update?**

The completion cache updates automatically when it detects version changes. You can also manually update:

```bash
dock completion update
```
