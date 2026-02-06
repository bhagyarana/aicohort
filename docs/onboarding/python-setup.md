---
sidebar_position: 4
title: Python Setup
description: Set up your Python development environment
---

# Python Environment Setup

This guide will help you set up a proper Python development environment for the Agentic AI Training.

## Installing Python

### Windows

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check "Add Python to PATH"
4. Click "Install Now"

```powershell
# Verify installation
python --version
pip --version
```

### macOS

```bash
# Using Homebrew (recommended)
brew install python@3.11

# Verify installation
python3 --version
pip3 --version
```

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python
sudo apt install python3.11 python3.11-venv python3-pip

# Verify installation
python3 --version
pip3 --version
```

## Virtual Environment Setup

:::tip Why Virtual Environments?
Virtual environments isolate project dependencies, preventing conflicts between different projects.
:::

### Creating a Virtual Environment

```bash
# Navigate to your project directory
cd ~/projects/agentic-ai-training

# Create virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Verifying Activation

When activated, your prompt should show `(venv)`:

```bash
(venv) $ python --version
Python 3.11.x
```

## Installing Dependencies

### Core Dependencies

Create a `requirements.txt` file with the essential packages:

```txt title="requirements.txt"
# LangChain ecosystem
langchain>=0.3.0
langchain-openai>=0.3.0
langchain-google-genai>=2.0.0
langchain-groq>=0.3.0
langchain-community>=0.3.0
langgraph>=0.2.0

# Vector stores
chromadb>=1.0.0
faiss-cpu>=1.8.0

# Embeddings
sentence-transformers>=3.0.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0

# Jupyter for notebooks
jupyter>=1.0.0
ipykernel>=6.0.0
```

### Installing Packages

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### Verify Installation

```python
# Run Python and test imports
python -c "import langchain; print(f'LangChain: {langchain.__version__}')"
python -c "import langgraph; print('LangGraph: OK')"
```

## IDE Setup

### VS Code (Recommended)

1. Install [Visual Studio Code](https://code.visualstudio.com/)
2. Install the Python extension
3. Install Jupyter extension for notebooks

**Recommended Extensions:**
- Python (Microsoft)
- Pylance
- Jupyter
- Python Indent
- autoDocstring

**Configure Python Interpreter:**
```
Ctrl/Cmd + Shift + P → "Python: Select Interpreter" → Choose your venv
```

### PyCharm

1. Open your project folder
2. Go to Settings → Project → Python Interpreter
3. Add your virtual environment as the interpreter

## Jupyter Notebook Setup

```bash
# Install Jupyter kernel for your venv
python -m ipykernel install --user --name=agentic-ai --display-name="Agentic AI"

# Start Jupyter
jupyter notebook
```

### VS Code Notebooks

1. Open any `.ipynb` file
2. Select your kernel (top right)
3. Choose "Agentic AI" or your venv

## Project Structure

Recommended project structure:

```
agentic-ai-training/
├── venv/                   # Virtual environment
├── notebooks/              # Jupyter notebooks
│   ├── module-1/
│   ├── module-2/
│   └── ...
├── src/                    # Source code
│   ├── agents/
│   ├── tools/
│   └── utils/
├── data/                   # Data files
├── .env                    # Environment variables (API keys)
├── .gitignore             # Git ignore file
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

### Creating the Structure

```bash
# Create directories
mkdir -p notebooks/{module-1,module-2,module-3,module-4,module-5}
mkdir -p src/{agents,tools,utils}
mkdir data

# Create essential files
touch .env .gitignore requirements.txt README.md
```

### .gitignore Template

```gitignore title=".gitignore"
# Virtual environment
venv/
.venv/

# Environment variables
.env
.env.local

# Python cache
__pycache__/
*.pyc
*.pyo

# Jupyter checkpoints
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# Data files (optional)
data/*.pdf
data/*.csv

# Vector store data
chroma_db/
```

## Troubleshooting

### Common Issues

**Issue: `pip` command not found**
```bash
# Use pip3 instead
pip3 install package-name

# Or use Python module
python -m pip install package-name
```

**Issue: Permission denied**
```bash
# Use --user flag
pip install --user package-name
```

**Issue: SSL Certificate Error**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package-name
```

**Issue: Package conflicts**
```bash
# Create fresh environment
rm -rf venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Verification Script

Create and run this script to verify your setup:

```python title="verify_setup.py"
#!/usr/bin/env python3
"""Verify the Agentic AI training environment setup."""

import sys

def check_python_version():
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    print(f"Python 3.10+ required, found {version.major}.{version.minor}")
    return False

def check_packages():
    packages = [
        "langchain",
        "langchain_openai",
        "langchain_google_genai",
        "langgraph",
        "chromadb",
        "dotenv",
    ]

    all_ok = True
    for package in packages:
        try:
            __import__(package)
            print(f"  {package}: OK")
        except ImportError:
            print(f"  {package}: MISSING")
            all_ok = False
    return all_ok

if __name__ == "__main__":
    print("Checking Python version...")
    py_ok = check_python_version()

    print("\nChecking packages...")
    pkg_ok = check_packages()

    if py_ok and pkg_ok:
        print("\nSetup complete! You're ready to start.")
    else:
        print("\nSome issues found. Please fix before continuing.")
```

Run it:
```bash
python verify_setup.py
```

## Next Steps

Your Python environment is ready! Next:

1. Configure your [API Keys](./api-keys-setup)
2. Start [Module 1: LangChain Fundamentals](/learn/modules/module-1)
