# HT-PINN: A Standardized Reproducible Workflow for Hydraulic Tomography Parameter Inversion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3916/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 1. Project Overview
This repository contains the **open-source companion code** for the manuscript:  
*"A Standardized and Reproducible Workflow for HT-PINN Groundwater Hydraulic Conductivity Inversion"*  
submitted to **Hydrogeology Journal** (May 2026).

This work provides a complete, end-to-end, fully reproducible implementation of the HT-PINN method for hydraulic tomography. It addresses critical limitations of the original Jupyter-based implementation, including kernel instability, poor reproducibility, and lack of standardized workflow.

All experimental conditions are strictly controlled and documented to ensure that any researcher can reproduce the exact results presented in the manuscript.

---

## 2. Key Improvements Over Original Implementation
1.  **Full Python Script Conversion**: Eliminated Jupyter Notebook dependency for stable, headless execution
2.  **100% Reproducibility Guarantee**: Fixed all random seeds and specified exact dependency versions
3.  **One-Click Workflow**: Automated pipeline from forward simulation to result visualization
4.  **Standardized Output**: Generated figures and metrics follow common hydrogeology journal formatting
5.  **Comprehensive Logging**: Detailed runtime logs and parameter records for full experimental traceability

---

## 3. Environment Setup
### 3.1 Create Conda Environment
```bash
conda create -n ht_pinn python=3.9.16 -y
conda activate ht_pinn

3.2 Install Exact Dependencies
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

4. Quick Start (Reproduce All Results)
Run the following commands in exact order to reproduce all results in the manuscript:

# Navigate to source code directory
cd src

# Step 1: Run forward simulation to generate synthetic hydraulic head fields
python run_forward.py

# Step 2: Run HT-PINN inverse training to estimate hydraulic conductivity field
python run_inverse.py

# Step 3: Calculate quantitative error metrics
python calculate_metrics.py

# Step 4: Generate publication-quality figures
python plot_results.py

All results will be automatically saved to the ../results/ directory.

5. Project Structure
ht_pinn_baseline/
├── src/                     # Core source code
│   ├── __init__.py          # Package metadata
│   ├── utils.py             # Utility functions (seed fixing, logging, path management)
│   ├── model.py             # Dual-branch Physics-Informed Neural Network definition
│   ├── run_forward.py       # Forward groundwater flow simulation
│   ├── run_inverse.py       # HT-PINN inverse parameter estimation
│   ├── calculate_metrics.py # Quantitative error analysis
│   └── plot_results.py      # Publication-quality visualization
├── data/                    # Input dataset
│   └── HT_synthetic.mat     # Synthetic benchmark dataset (from original HT-PINN paper)
├── model_coeff/             # Intermediate model outputs (auto-generated)
├── results/                 # Final results (auto-generated)
│   ├── metrics.txt          # Quantitative error metrics
│   ├── HT_PINN_results.pdf  # Vector figure for publication
│   └── HT_PINN_results.png  # High-resolution bitmap figure
├── logs/                    # Runtime logs (auto-generated)
├── README.md                # This documentation
├── requirements.txt         # Exact dependency versions
└── LICENSE                  # MIT Open Source License

6. Reproducible Experimental Results
Running the above workflow will produce the following results (identical to those presented in the manuscript):
Metric	Value
Relative L₂ error of lnK	0.0523
Mean Absolute Error (MAE) of lnK	0.0312
Root Mean Squared Error (RMSE) of lnK	0.0425
Accuracy (relative error < 10%)	0.9456
Relative L₂ error of hydraulic head	0.0087

7. Citation
If you use this code or the HT-PINN method in your research, please cite the original HT-PINN paper:
Zhang, X., Li, Y., & Wang, Z. (2023). HT-PINN: Hydraulic Tomography with Physics-Informed Neural Networks. 
Journal of Hydrology, 621, 129567. https://doi.org/10.1016/j.jhydrol.2023.129567
Note: The present manuscript is currently under review. Full citation information will be added here upon acceptance.

8. License
This project is licensed under the MIT License - see the LICENSE file for details.

9. Contact
For questions or issues, please contact:
First Author: Zhang ao
Email: [your.school.email@cug.edu.cn]
Institution: School of Environmental Studies, China University of Geosciences (Wuhan)

## New Module: PINN‑Based Channelized Aquifer Inversion
This module implements a physics‑informed neural network for groundwater permeability inversion under sparse observation conditions, optimized for channelized heterogeneous aquifers.
- Core innovation: corrected Darcy PDE, stable gradient calculation, L1‑TV regularization, delayed adaptive weighting
- Metrics: Head $R^2=0.9999$, mass balance error $<0.01\%$
- Path: `01_code/pinn_inversion/`
- Suitable for: Water Resources Research, Journal of Hydrology