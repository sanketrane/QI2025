# QI2025 Workshop

This repository contains materials for the QI2025 workshop on Bayesian modeling and computational biology.

## Workshop Contents

### 📂 Projects

1. **Population Growth Perturbations** (`population_growth_perturbations/`)
   - Bayesian models for population dynamics using Stan
   - Logistic growth models (with and without treatment effects)
   - Simple homogeneous models (with and without treatment effects)
   - Observed data with treatment analysis 

2. **Pulse Chase Exercise** (`Pulse_chase_exercise/`)
   - BrdU pulse-chase data analysis and visualization

3. **Spatial 3D** (`spatial_3d/`)
   - Stochastic SIRS spatial modeling

4. **VAE Deep Learning** (`Vae_deeplearn/`)
   - Cell-wise VAE implementation
   - Deep learning models for single-cell data analysis

5. **Trajectory Inference** (`Trajectory_inference/`)
   - Potential trajectory analysis with GFP data
   - Interactive HTML visualization outputs

6. **Conditional VAE** (`conditional_VAE/`)
   - Dynamic model inference with single-cell flow data
   - Combines dynamic modeling with single-cell type data using Pyro
   - Includes Pyro 101 tutorial and dynamical model fitting
   - Simulated time series datasets with ground truth weights

### 📓 Setup Notebooks

- `run_stan.ipynb` - CmdStan installation and setup guide

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Visual Studio Code](https://code.visualstudio.com/) (recommended)


### Quick Start

```bash
# Clone the repository
git clone https://github.com/sanketrane/QI2025.git
cd QI2025

# Create virtual environment
python3 -m venv .myenv
source .myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Open in VS Code
code .
```

### Setup Instructions

For detailed setup instructions, including:
- Creating a Python virtual environment
- Installing dependencies
- Configuring CmdStan for Bayesian modeling
- Selecting the correct kernel in VS Code

Please refer to **[SETUP.md](SETUP.md)** for step-by-step guidance.


## Dependencies

The project uses the following Python packages (see `requirements.txt`):
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `matplotlib` - Plotting and visualization
- `cmdstanpy` - Python interface for Stan
- `scipy` - Scientific computing
- `seaborn` - Statistical data visualization
- `torch` - PyTorch deep learning framework
- `pyro-ppl` - Pyro probabilistic programming
- `plotly` - Interactive visualizations
- `scikit-learn` - Machine learning tools
- `tqdm` - Progress bars
- `umap-learn` - Dimensionality reduction

## Key Frameworks

### Stan Models
Stan models are located in `population_growth_perturbations/stan_models/`:
- Logistic growth models (with and without treatment effects)
- Simple homogeneous models (with and without treatment effects)

### Pyro Framework
The `conditional_VAE/` project uses Pyro for probabilistic programming:
- Variational inference with conditional VAE
- Dynamic systems modeling with single-cell data
- Built on PyTorch for deep learning integration

## Contributing

This repository is for workshop materials. For questions or issues, please contact the workshop organizers.

## Resources

### Documentation
- [Stan Documentation](https://mc-stan.org/docs/)
- [CmdStanPy Documentation](https://mc-stan.org/cmdstanpy/)
- [Pyro Documentation](https://pyro.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### VS Code Extensions
- [Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## License

Workshop materials for educational purposes.
