# Essential Software (Python, Anaconda, Jupyter)

## Introduction

Setting up the right software stack is crucial for AI development. This guide covers the essential tools you'll need: Python, package managers, Jupyter notebooks, and development environments.

## Python

### Why Python for AI?

Python is the de facto language for AI and machine learning:

**Advantages:**
- Simple, readable syntax
- Extensive AI/ML libraries
- Large community and resources
- Industry standard
- Rapid prototyping

**Key Libraries:**
- NumPy - Numerical computing
- Pandas - Data manipulation
- Matplotlib/Seaborn - Visualization
- Scikit-learn - Traditional ML
- TensorFlow/PyTorch - Deep learning

### Python Version

**Recommended: Python 3.10 or 3.11**
- Good compatibility with AI frameworks
- Stable and well-supported
- Balance of features and stability

**Avoid:**
- Python 3.12+ - Too new, compatibility issues
- Python 3.8 and below - Outdated, missing features
- Python 2.x - Deprecated, don't use

### Installing Python

#### Linux (Ubuntu/Debian)

```bash
# Check current version
python3 --version

# Install Python 3.11
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install pip
sudo apt install python3-pip

# Verify installation
python3.11 --version
pip3 --version
```

#### Windows

**Option 1: Official Installer (Recommended for WSL2)**
```powershell
# Download from python.org
# During installation:
# ✓ Add Python to PATH
# ✓ Install pip
```

**Option 2: Microsoft Store**
```powershell
# Search "Python 3.11" in Microsoft Store
# Click Install
# Automatically adds to PATH
```

**Option 3: WSL2 (Best for AI Development)**
```bash
# In WSL2 Ubuntu
sudo apt update
sudo apt install python3.11 python3-pip
```

#### macOS

**Option 1: Homebrew (Recommended)**
```bash
# Install Homebrew first (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Verify
python3.11 --version
```

**Option 2: Official Installer**
- Download from python.org
- Run installer
- Adds to PATH automatically

### Python Virtual Environments

**Why Use Virtual Environments?**
- Isolate project dependencies
- Avoid version conflicts
- Reproducible environments
- Clean system Python

#### venv (Built-in)

```bash
# Create virtual environment
python3 -m venv myenv

# Activate
# Linux/Mac:
source myenv/bin/activate
# Windows:
myenv\Scripts\activate

# Install packages
pip install numpy pandas

# Deactivate
deactivate
```

#### virtualenvwrapper (Advanced)

```bash
# Install
pip install virtualenvwrapper

# Add to ~/.bashrc or ~/.zshrc
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh

# Create environment
mkvirtualenv ml-env

# Switch environments
workon ml-env

# List environments
lsvirtualenv

# Delete environment
rmvirtualenv ml-env
```

## Anaconda / Miniconda

### What is Anaconda?

Anaconda is a distribution of Python and R for scientific computing and data science. It includes:
- Python interpreter
- Conda package manager
- 250+ pre-installed packages
- Jupyter Notebook
- Spyder IDE

### Anaconda vs. Miniconda vs. Miniforge

| Feature | Anaconda | Miniconda | Miniforge |
|---------|----------|-----------|-----------|
| **Size** | 3+ GB | 400 MB | 400 MB |
| **Pre-installed packages** | 250+ | None | None |
| **GUI** | Anaconda Navigator | No | No |
| **Default channel** | Anaconda (slower) | Anaconda | conda-forge (faster) |
| **Best for** | Beginners | Advanced users | Apple Silicon, advanced users |

**Recommendation:**
- **Beginners:** Anaconda (everything included)
- **Advanced:** Miniconda (minimal, customize)
- **Apple Silicon:** Miniforge (optimized for M1/M2/M3)

### Installing Anaconda

#### Linux

```bash
# Download installer
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# Run installer
bash Anaconda3-2024.02-1-Linux-x86_64.sh

# Follow prompts
# Accept license
# Choose installation location
# Initialize Anaconda3 (yes)

# Restart terminal or:
source ~/.bashrc

# Verify
conda --version
```

#### Windows

```powershell
# Download from anaconda.com
# Run installer
# Choose "Just Me" or "All Users"
# Add to PATH (optional, not recommended)
# Register as default Python (optional)

# Open Anaconda Prompt
conda --version
```

#### macOS

```bash
# Intel Macs: Download from anaconda.com
# Apple Silicon: Use Miniforge instead

# Miniforge (Recommended for M1/M2/M3)
brew install miniforge

# Or manual install:
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh

# Verify
conda --version
```

### Conda Basics

#### Creating Environments

```bash
# Create environment with specific Python version
conda create -n ml-env python=3.11

# Create with packages
conda create -n dl-env python=3.11 numpy pandas matplotlib

# Activate environment
conda activate ml-env

# Deactivate
conda deactivate

# List environments
conda env list

# Remove environment
conda env remove -n ml-env
```

#### Installing Packages

```bash
# Install from conda
conda install numpy pandas scikit-learn

# Install from conda-forge (often newer versions)
conda install -c conda-forge transformers

# Install from pip (if not available in conda)
pip install some-package

# Update package
conda update numpy

# Update all packages
conda update --all

# Update conda itself
conda update conda
```

#### Managing Environments

```bash
# Export environment
conda env export > environment.yml

# Create from file
conda env create -f environment.yml

# Clone environment
conda create --name new-env --clone old-env

# List installed packages
conda list

# Search for package
conda search pytorch
```

### Conda vs. Pip

**Use Conda When:**
- Installing packages with C/C++ dependencies
- Need specific versions of Python
- Want environment management
- Installing scientific packages

**Use Pip When:**
- Package not available in conda
- Need latest version
- Installing pure Python packages
- Lighter weight solution

**Best Practice:**
- Create conda environment
- Install major packages with conda
- Use pip for remaining packages
- Avoid mixing conda and pip when possible

## Jupyter Notebook / JupyterLab

### What is Jupyter?

Interactive computing environment for:
- Code execution in cells
- Inline visualization
- Markdown documentation
- Sharing and collaboration

**Use Cases:**
- Data exploration
- Prototyping
- Teaching and learning
- Reporting results
- Sharing research

### Jupyter Notebook vs. JupyterLab

**Jupyter Notebook:**
- Classic interface
- Simple and lightweight
- Single document interface
- Good for beginners

**JupyterLab:**
- Modern interface
- Multiple documents in tabs
- Integrated file browser
- Extensions and customization
- Recommended for serious work

### Installing Jupyter

#### With Anaconda (Already Included)

```bash
# Launch Jupyter Notebook
jupyter notebook

# Launch JupyterLab
jupyter lab
```

#### With pip

```bash
# Activate your virtual environment
source myenv/bin/activate

# Install Jupyter Notebook
pip install notebook

# Install JupyterLab (recommended)
pip install jupyterlab

# Launch
jupyter lab
```

#### With conda

```bash
# Create environment with Jupyter
conda create -n jupyter-env python=3.11 jupyterlab

# Activate
conda activate jupyter-env

# Launch
jupyter lab
```

### Jupyter Configuration

#### Change Default Directory

```bash
# Generate config file
jupyter notebook --generate-config

# Edit config file
# Linux/Mac: ~/.jupyter/jupyter_notebook_config.py
# Windows: C:\Users\YourName\.jupyter\jupyter_notebook_config.py

# Add line:
c.NotebookApp.notebook_dir = '/path/to/your/notebooks'
```

#### Enable Extensions

```bash
# Install extensions manager
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Popular extensions:
# - Table of Contents
# - Variable Inspector
# - ExecuteTime
# - Code folding
```

#### Remote Access

```bash
# Generate password
jupyter notebook password

# Start with remote access
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Access from another computer:
# http://your-ip-address:8888
```

### Jupyter Kernels

**What are Kernels?**
- Backend that executes code
- Different environments/languages
- Switch between Python versions

#### Managing Kernels

```bash
# Install ipykernel in your environment
conda activate ml-env
pip install ipykernel

# Add environment as kernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ML)"

# List kernels
jupyter kernelspec list

# Remove kernel
jupyter kernelspec uninstall ml-env
```

### Jupyter Tips and Tricks

#### Keyboard Shortcuts

**Command Mode (press Esc):**
- `A` - Insert cell above
- `B` - Insert cell below
- `DD` - Delete cell
- `M` - Convert to Markdown
- `Y` - Convert to Code
- `Shift+Enter` - Run cell and select below

**Edit Mode (press Enter):**
- `Tab` - Code completion
- `Shift+Tab` - Show documentation
- `Ctrl+]` - Indent
- `Ctrl+[` - Dedent

#### Magic Commands

```python
# Timing
%time result = slow_function()  # Time single execution
%timeit fast_function()  # Time multiple executions

# System commands
!pip install numpy
!ls -la

# Load external code
%load script.py

# Run external script
%run script.py

# Matplotlib inline
%matplotlib inline

# Auto-reload modules
%load_ext autoreload
%autoreload 2

# Show variables
%whos

# Debug
%debug  # Enter debugger after exception
```

#### Best Practices

1. **Use descriptive names** for notebooks
2. **Restart kernel and run all** before sharing
3. **Clear output** before committing to git
4. **Use Markdown cells** for documentation
5. **Keep cells focused** (one task per cell)
6. **Version control** with git (use .gitignore for outputs)

## Additional Essential Tools

### Git (Version Control)

```bash
# Linux
sudo apt install git

# Mac
brew install git

# Windows
# Download from git-scm.com

# Configure
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Basic usage
git init
git add .
git commit -m "Initial commit"
git push origin main
```

### Text Editors / IDEs

**VS Code (Recommended)**
- Free and open source
- Excellent Python support
- Jupyter integration
- Remote development
- Extensions ecosystem

**PyCharm**
- Professional Python IDE
- Excellent debugging
- Code completion
- Paid (Community edition free)

**Vim / Neovim**
- Terminal-based
- Highly customizable
- Steep learning curve
- Very efficient

### Terminal Multiplexers

**tmux (Linux/Mac)**
```bash
# Install
sudo apt install tmux  # Linux
brew install tmux      # Mac

# Basic usage
tmux new -s session-name
# Ctrl+B, D to detach
tmux attach -t session-name
```

**screen (Alternative)**
```bash
# Start session
screen -S session-name

# Detach: Ctrl+A, D
# Reattach
screen -r session-name
```

## Setting Up Your First AI Environment

### Complete Setup Example

```bash
# 1. Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 2. Create AI environment
conda create -n ai-lab python=3.11 -y
conda activate ai-lab

# 3. Install essential packages
conda install -c conda-forge \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyterlab \
    ipykernel \
    -y

# 4. Install PyTorch (check pytorch.org for latest)
# For NVIDIA GPU:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 5. Install additional tools
pip install transformers datasets accelerate

# 6. Add kernel to Jupyter
python -m ipykernel install --user --name=ai-lab --display-name="Python (AI Lab)"

# 7. Launch JupyterLab
jupyter lab
```

### Test Installation

```python
# In Jupyter notebook, test imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import transformers

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting Common Issues

### Conda is slow

```bash
# Use mamba (faster conda alternative)
conda install mamba -c conda-forge

# Use mamba instead of conda
mamba install numpy pandas
```

### Import errors

```bash
# Verify correct environment is activated
conda env list  # Check * next to current env

# Reinstall package
pip uninstall package-name
pip install package-name

# Check Python path
python -c "import sys; print(sys.executable)"
```

### Jupyter kernel not found

```bash
# List kernels
jupyter kernelspec list

# Reinstall kernel
conda activate your-env
pip install ipykernel
python -m ipykernel install --user --name=your-env
```

### CUDA not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
# Visit pytorch.org for installation command

# Verify in Python
import torch
print(torch.cuda.is_available())
```

## Next Steps

- **[Installing Ollama](./08-installing-ollama.md)** - Run LLMs locally
- **[AI Frameworks](./09-ai-frameworks.md)** - TensorFlow, PyTorch, Hugging Face
- **[Security and Network Setup](./10-security-network.md)** - Secure your lab

## Resources

- Python Documentation - https://docs.python.org/3/
- Anaconda Documentation - https://docs.anaconda.com/
- Jupyter Documentation - https://jupyter.org/documentation
- Real Python - https://realpython.com/
- Python Package Index (PyPI) - https://pypi.org/

