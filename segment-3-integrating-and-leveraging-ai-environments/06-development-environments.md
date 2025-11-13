# Development Environments for AI

## Introduction

A well-configured development environment is crucial for productive AI work. This guide covers the most popular development environments for AI/ML, including Jupyter Notebooks, IDEs, coding agents, and remote development setups. We'll explore how to configure each for optimal AI development workflows.

## Jupyter Notebooks and JupyterLab

### Why Jupyter for AI?

Jupyter notebooks are ideal for AI development because they:
- Enable interactive experimentation
- Combine code, visualizations, and documentation
- Support rapid prototyping and iteration
- Facilitate sharing and collaboration
- Provide inline visualization of results

### Installing JupyterLab

```bash
# Install JupyterLab
pip install jupyterlab

# Install extensions
pip install jupyterlab-git jupyterlab-lsp
pip install jupyter-ai  # AI assistant for Jupyter

# Install additional kernels
pip install ipykernel

# Launch JupyterLab
jupyter lab
```

### JupyterLab Configuration

```bash
# Generate configuration file
jupyter lab --generate-config

# Edit configuration
nano ~/.jupyter/jupyter_lab_config.py
```

```python
# ~/.jupyter/jupyter_lab_config.py

# Allow remote access
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False

# Set password (generate hash first)
# jupyter lab password
c.ServerApp.password = 'your-hashed-password'

# Increase data rate limit for large outputs
c.ServerApp.iopub_data_rate_limit = 10000000

# Set working directory
c.ServerApp.root_dir = '/home/user/notebooks'

# Enable extensions
c.LabApp.collaborative = True
```

### Essential Jupyter Extensions

```bash
# Install nbextensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Install useful extensions
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyterlab-plotly
jupyter labextension install @jupyterlab/toc

# Install code formatting
pip install jupyterlab-code-formatter black isort
jupyter labextension install @ryantam626/jupyterlab_code_formatter
```

### Jupyter Notebook Best Practices

#### 1. Notebook Structure

```python
# Cell 1: Imports and Setup
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure matplotlib
%matplotlib inline
plt.style.use('seaborn-v0_8')

# Cell 2: Configuration
CONFIG = {
    'model_name': 'meta-llama/Llama-3.2-3B-Instruct',
    'batch_size': 32,
    'learning_rate': 2e-5,
    'epochs': 3,
    'max_length': 512
}

# Cell 3: Load Data
# ...

# Cell 4: Model Definition
# ...

# Cell 5: Training Loop
# ...

# Cell 6: Evaluation
# ...

# Cell 7: Visualization
# ...
```

#### 2. Use Magic Commands

```python
# Timing
%time result = expensive_function()
%timeit fast_function()

# Profiling
%prun slow_function()

# Memory usage
%load_ext memory_profiler
%memit memory_intensive_function()

# Auto-reload modules
%load_ext autoreload
%autoreload 2

# Display environment variables
%env

# Run shell commands
!pip install package
!nvidia-smi

# Display multiple outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

#### 3. Interactive Widgets

```python
import ipywidgets as widgets
from IPython.display import display

# Create interactive controls
learning_rate = widgets.FloatSlider(
    value=1e-4,
    min=1e-6,
    max=1e-2,
    step=1e-6,
    description='LR:',
    continuous_update=False
)

batch_size = widgets.IntSlider(
    value=32,
    min=8,
    max=128,
    step=8,
    description='Batch:'
)

def train_model(lr, bs):
    print(f"Training with lr={lr}, batch_size={bs}")
    # Training code here

# Interactive training
widgets.interact(train_model, lr=learning_rate, bs=batch_size)
```

### Remote Jupyter Access

#### Method 1: SSH Tunneling

```bash
# On remote server
jupyter lab --no-browser --port=8888

# On local machine
ssh -N -L 8888:localhost:8888 user@remote-server

# Access at http://localhost:8888
```

#### Method 2: HTTPS with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot

# Generate certificate
sudo certbot certonly --standalone -d your-domain.com

# Configure Jupyter
jupyter lab --generate-config
```

```python
# ~/.jupyter/jupyter_lab_config.py
c.ServerApp.certfile = '/etc/letsencrypt/live/your-domain.com/fullchain.pem'
c.ServerApp.keyfile = '/etc/letsencrypt/live/your-domain.com/privkey.pem'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 443
```

### JupyterHub for Multi-User

```bash
# Install JupyterHub
pip install jupyterhub
pip install notebook

# Generate config
jupyterhub --generate-config

# Edit config
nano jupyterhub_config.py
```

```python
# jupyterhub_config.py
c.JupyterHub.ip = '0.0.0.0'
c.JupyterHub.port = 8000
c.Spawner.default_url = '/lab'
c.Spawner.notebook_dir = '~/notebooks'

# Use PAM authenticator (Unix users)
c.JupyterHub.authenticator_class = 'jupyterhub.auth.PAMAuthenticator'

# Resource limits per user
c.Spawner.mem_limit = '8G'
c.Spawner.cpu_limit = 2
```

## Visual Studio Code (VS Code)

### Why VS Code for AI?

- Excellent Python support
- Integrated Jupyter notebooks
- Remote development capabilities
- AI-powered coding assistants
- Git integration
- Extensive extension ecosystem

### Essential Extensions for AI Development

```bash
# Install VS Code extensions via CLI
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.vscode-jupyter-cell-tags
code --install-extension ms-toolsai.vscode-jupyter-slideshow
code --install-extension GitHub.copilot
code --install-extension GitHub.copilot-chat
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-vscode-remote.remote-containers
code --install-extension ms-azuretools.vscode-docker
code --install-extension eamodio.gitlens
code --install-extension donjayamanne.githistory
code --install-extension mhutchie.git-graph
code --install-extension yzhang.markdown-all-in-one
code --install-extension ms-python.black-formatter
code --install-extension ms-python.isort
```

### VS Code Configuration for AI

```json
// settings.json
{
    // Python
    "python.defaultInterpreterPath": "/home/user/ai-env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "python.sortImports.args": ["--profile", "black"],
    
    // Jupyter
    "jupyter.askForKernelRestart": false,
    "jupyter.interactiveWindow.textEditor.executeSelection": true,
    
    // Editor
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "editor.rulers": [100],
    "editor.suggestSelection": "first",
    
    // Files
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.ipynb_checkpoints": true
    },
    "files.watcherExclude": {
        "**/data/**": true,
        "**/models/**": true,
        "**/checkpoints/**": true
    },
    
    // Terminal
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.fontSize": 14,
    
    // Git
    "git.autofetch": true,
    "git.confirmSync": false,
    
    // Remote
    "remote.SSH.remotePlatform": {
        "gpu-server": "linux"
    }
}
```

### Remote Development with VS Code

#### SSH Remote Development

```bash
# Install Remote-SSH extension
code --install-extension ms-vscode-remote.remote-ssh

# Configure SSH
# .ssh/config
Host gpu-server
    HostName 192.168.1.100
    User username
    IdentityFile ~/.ssh/id_rsa
    ForwardAgent yes
```

In VS Code:
1. Press `F1` → "Remote-SSH: Connect to Host"
2. Select your configured host
3. VS Code will connect and install server components
4. Open your project folder on the remote machine

#### Dev Containers

```dockerfile
# .devcontainer/Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and AI libraries
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers accelerate datasets jupyter

# Set working directory
WORKDIR /workspace
```

```json
// .devcontainer/devcontainer.json
{
    "name": "AI Development",
    "build": {
        "dockerfile": "Dockerfile"
    },
    "runArgs": ["--gpus", "all"],
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-toolsai.jupyter",
                "GitHub.copilot"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/usr/bin/python3"
            }
        }
    },
    "forwardPorts": [8888],
    "postCreateCommand": "pip install -r requirements.txt"
}
```

### Debugging AI Code in VS Code

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Train Model",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/train.py",
            "console": "integratedTerminal",
            "args": [
                "--config", "configs/train.yaml",
                "--debug"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }
        },
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        },
        {
            "name": "Python: Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            }
        }
    ]
}
```

## PyCharm

### PyCharm for AI Development

PyCharm Professional offers excellent AI development features:
- Scientific mode with interactive plots
- Jupyter notebook support
- Remote interpreter support
- Database tools
- Professional debugging

### Configuration

```python
# Settings → Project → Python Interpreter
# Add remote interpreter via SSH

# Settings → Tools → Python Scientific
# Enable scientific mode

# Settings → Editor → Code Style → Python
# Set line length to 100
# Enable "Optimize imports on the fly"

# Settings → Build, Execution, Deployment → Python Debugger
# Enable "Gevent compatible debugging"
# Enable "PyQt compatible debugging"
```

### Remote Development in PyCharm

1. **Configure Deployment**:
   - Tools → Deployment → Configuration
   - Add SFTP server
   - Configure mappings

2. **Set Remote Interpreter**:
   - Settings → Project → Python Interpreter
   - Add → SSH Interpreter
   - Configure SSH connection
   - Select interpreter path

3. **Sync Files**:
   - Tools → Deployment → Automatic Upload
   - Enable automatic sync

## Coding Agents and AI Assistants

### GitHub Copilot

```bash
# Install in VS Code
code --install-extension GitHub.copilot
code --install-extension GitHub.copilot-chat
```

**Usage Tips**:
```python
# Write descriptive comments for better suggestions
# Generate a PyTorch dataset class for image classification
class ImageDataset(torch.utils.data.Dataset):
    # Copilot will suggest the implementation

# Use Copilot Chat for explanations
# Ask: "Explain this code" or "How can I optimize this?"

# Generate tests
# Write: "def test_model_forward():" and let Copilot complete
```

### Cursor IDE

Cursor is an AI-first IDE built on VS Code:

```bash
# Download from cursor.sh
# Import VS Code settings and extensions
```

**Features**:
- AI-powered code generation
- Natural language code editing
- Codebase-aware suggestions
- Multi-file editing

### Codeium

Free alternative to Copilot:

```bash
# Install in VS Code
code --install-extension Codeium.codeium

# Or in Jupyter
pip install jupyter-codeium
```

### Amazon CodeWhisperer

```bash
# Install AWS Toolkit
code --install-extension amazonwebservices.aws-toolkit-vscode

# Sign in with AWS account
# Enable CodeWhisperer
```

## Integrated Development Workflows

### Example 1: Jupyter + VS Code Workflow

```python
# 1. Explore in Jupyter notebook
# notebooks/exploration.ipynb

# Quick data exploration
import pandas as pd
df = pd.read_csv('data.csv')
df.head()

# Try different models
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
results = classifier(df['text'].tolist())

# 2. Convert to Python script in VS Code
# src/train.py

import argparse
from pathlib import Path
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer

def main(args):
    # Load data
    df = pd.read_csv(args.data_path)
    
    # Train model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    trainer = Trainer(model=model, ...)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    args = parser.parse_args()
    main(args)
```

### Example 2: Remote Development Workflow

```bash
# Local machine: Edit code in VS Code with Remote-SSH
# Files automatically sync to remote server

# Remote server: Run training
python train.py --config configs/experiment1.yaml

# Local machine: Monitor with TensorBoard
ssh -L 6006:localhost:6006 user@remote-server
# Access TensorBoard at http://localhost:6006

# Local machine: Debug remotely
# Set breakpoints in VS Code
# Attach to remote Python process
```

### Example 3: Container-Based Workflow

```bash
# Develop in VS Code with Dev Container
# All dependencies in container
# Consistent environment across team

# Build and run
docker-compose up -d

# VS Code automatically connects to container
# Code, debug, and test in containerized environment

# Deploy same container to production
docker push myregistry/ai-model:latest
```

## Productivity Tools

### Terminal Multiplexers

#### tmux

```bash
# Install tmux
sudo apt install tmux

# Create session
tmux new -s ai-training

# Detach: Ctrl+b, d
# Reattach: tmux attach -t ai-training

# Split panes
# Horizontal: Ctrl+b, "
# Vertical: Ctrl+b, %

# Example .tmux.conf
cat > ~/.tmux.conf << 'EOF'
# Enable mouse
set -g mouse on

# Start windows at 1
set -g base-index 1

# Status bar
set -g status-style 'bg=colour235 fg=colour136'
EOF
```

#### screen

```bash
# Alternative to tmux
screen -S training

# Detach: Ctrl+a, d
# Reattach: screen -r training
```

### Code Formatting and Linting

```bash
# Install tools
pip install black isort flake8 mypy pylint

# Format code
black src/
isort src/

# Lint
flake8 src/
mypy src/
pylint src/
```

```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.10.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100']

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=10000']
EOF

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Collaboration Tools

### Git Workflows

```bash
# Feature branch workflow
git checkout -b feature/new-model
# Make changes
git add .
git commit -m "Add new model architecture"
git push origin feature/new-model
# Create pull request

# Keep branch updated
git checkout main
git pull origin main
git checkout feature/new-model
git rebase main
```

### Code Review Tools

- **GitHub**: Pull requests, code review, CI/CD
- **GitLab**: Similar to GitHub, self-hosted option
- **Gerrit**: Advanced code review
- **Review Board**: Open-source code review

### Documentation

```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Initialize documentation
sphinx-quickstart docs

# Build documentation
cd docs
make html

# View at docs/_build/html/index.html
```

## Conclusion

The right development environment significantly impacts productivity in AI development. Whether you prefer the interactive nature of Jupyter notebooks, the comprehensive features of VS Code or PyCharm, or the AI-powered assistance of modern coding agents, choosing and configuring your tools properly will accelerate your AI development workflow.

Key recommendations:
- **Exploration**: Use Jupyter notebooks
- **Development**: Use VS Code or PyCharm
- **Remote Work**: Configure SSH remote development
- **Collaboration**: Use Git and code review tools
- **Productivity**: Leverage AI coding assistants
- **Consistency**: Use containers and pre-commit hooks

## Additional Resources

- [JupyterLab Documentation](https://jupyterlab.readthedocs.io/)
- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [PyCharm Documentation](https://www.jetbrains.com/pycharm/learn/)
- [GitHub Copilot](https://github.com/features/copilot)
- [tmux Cheat Sheet](https://tmuxcheatsheet.com/)

