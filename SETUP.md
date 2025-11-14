# Python Environment Setup and CmdStan Installation

This guide walks you through setting up a Python virtual environment, installing dependencies, and configuring CmdStan for Bayesian modeling.

**Recommended IDE:** We strongly recommend using [Visual Studio Code (VS Code)](https://code.visualstudio.com/) for this workshop, as it provides excellent support for Jupyter notebooks, Python development, and integrated terminal access.

We will use _Stan_ to do some hands-on exercises during the workshop. We recommend installing _CmdStan_ beforehand as it takes several minutes.

## Prerequisites

- Python 3.8 or higher
- [Visual Studio Code](https://code.visualstudio.com/) (recommended)
- Note: CmdStan installation instructions are specific for macOS. <br> Check <https://mc-stan.org/docs/cmdstan-guide/installation.html> for your setup-specific instructions.

## Step 1: Create a Python Virtual Environment

Navigate to the project directory and create a virtual environment:

```bash
cd /Path/to/QI2025  # update this based on your working directory
python3 -m venv .myenv
```

## Step 2: Activate the Virtual Environment

Activate the newly created virtual environment:

```bash
source .myenv/bin/activate
```

You should see `(.myenv)` prepended to your terminal prompt, indicating the virtual environment is active.

## Step 3: Install Required Packages

Install all dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install the following packages:
- numpy (<2.0)
- pandas
- matplotlib
- cmdstanpy
- scipy
- seaborn
- torch
- plotly
- scikit-learn

## Step 4: Select the Virtual Environment in VS Code

To use the virtual environment with Jupyter notebooks in VS Code:

1. Open the notebook file (`run_stan.ipynb` or any other `.ipynb` file)
2. Click on the kernel selector in the top-right corner of the notebook
3. Select "Select Another Kernel..."
4. Choose "Python Environments..."
5. Select the `.myenv` environment from the list (should be located at `./.myenv/bin/python`)

Alternatively, you can use the Command Palette:
1. Press `Cmd+Shift+P` (macOS) to open the Command Palette
2. Type "Python: Select Interpreter"
3. Select the interpreter from `./.myenv/bin/python`

## Step 5: Install CmdStan

Run the `run_stan.ipynb` notebook to install CmdStan:

1. Open `run_stan.ipynb` in VS Code
2. Ensure the correct kernel (.myenv) is selected
3. Run the first cell, which will:
   - Configure macOS SDK paths to resolve C++ compilation issues
   - Check if CmdStan is installed
   - Automatically install CmdStan if not present

The installation process may take several minutes as it downloads and compiles CmdStan.

## Troubleshooting

### Virtual Environment Not Showing in Kernel List

If the virtual environment doesn't appear in the kernel selector:
1. Make sure you've activated the environment and installed `ipykernel`:
   ```bash
   source .myenv/bin/activate
   pip install ipykernel
   python -m ipykernel install --user --name=.myenv
   ```
2. Reload VS Code window (`Cmd+Shift+P` → "Developer: Reload Window")

### CmdStan Installation Issues

The `run_stan.ipynb` notebook includes macOS-specific fixes for C++ compilation issues. If you encounter problems:
- Ensure Xcode Command Line Tools are installed: `xcode-select --install`
- Verify the SDK path is correctly set by running: `xcrun --show-sdk-path`

## Deactivating the Virtual Environment

When you're done working, deactivate the virtual environment:

```bash
deactivate
```

## Additional Notes

- Always activate the virtual environment before working on this project
- To update packages, run: `pip install --upgrade -r requirements.txt`
- The virtual environment is local to this project and doesn't affect system-wide Python packages
