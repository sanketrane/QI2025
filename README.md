# QI2025 Workshop

This repository contains materials for the QI2025 workshop on Bayesian modeling and computational biology.

## Workshop Contents

### 📂 Projects

- **Population Growth Perturbations**: Bayesian models for population dynamics using Stan
  - Logistic growth models
  - Homogeneous models
  - Treatment effect analysis
  
- **Pulse Chase Exercise**: BrdU pulse-chase data analysis and visualization

- **Spatial 3D**: Stochastic SIRS spatial modeling

- **Trajectory Inference**: Potential trajectory analysis with GFP data

### 📓 Notebooks

- `run_stan.ipynb` - CmdStan installation and setup

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Visual Studio Code](https://code.visualstudio.com/) (recommended)

### Setup Instructions

For detailed setup instructions, including:
- Creating a Python virtual environment
- Installing dependencies
- Configuring CmdStan for Bayesian modeling
- Selecting the correct kernel in VS Code

Please refer to **[SETUP.md](SETUP.md)** for step-by-step guidance.

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

## Dependencies

The project uses the following Python packages (see `requirements.txt`):
- `numpy` (<2.0)
- `pandas`
- `matplotlib`
- `cmdstanpy`
- `scipy`
- `seaborn`
- `torch`
- `plotly`
- `scikit-learn`

## Stan Models

Stan models are located in `population_growth_perturbations/stan_models/`:
- Logistic growth models (with and without treatment effects)
- Simple homogeneous models (with and without treatment effects)

## Contributing

This repository is for workshop materials. For questions or issues, please contact the workshop organizers.

## Resources

- [Stan Documentation](https://mc-stan.org/docs/)
- [CmdStanPy Documentation](https://mc-stan.org/cmdstanpy/)
- [VS Code Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## License

Workshop materials for educational purposes.
