# ESCA
Code and network data accompanying:

**E. Hormann and R. Lambiotte, “Coarse-graining Directed Networks with Ergodic Sets Preserving Diffusive Dynamics.”**

This repository contains the Python code used to implement the **Ergodic Set Coarse-graining Algorithm (ESCA)** and to generate the numerical results presented in the paper.

## Repository Structure
ESCA/
├── src/
│   ├── 01_ESCA_step_1.py
│   ├── 02_ESCA_step_2.py
│   ├── 03_plot_compressions.py
│   ├── 04_ESCA_core_rewiring.py
│   ├── 05_rewired_core_spectral.py
│   └── mygraph.py
├── real_world_networks/
│   └── Validated/
├── results/
└── figures/

## Code

The scripts are numbered according to the analysis workflow:

* 01_ESCA_step_1.py — identifies the forward and backward ergodic sets and the transient core, and performs the first ESCA coarse-graining step.
* 02_ESCA_step_2.py — constructs the source-to-well transport matrix and performs the SVD-based second ESCA coarse-graining step.
* 03_plot_compressions.py — calculates and visualises the compression statistics.
* 04_ESCA_core_rewiring.py — performs the core-rewiring analysis.
* 05_rewired_core_spectral.py — performs the spectral analysis of the original and rewired core networks.
* mygraph.py — contains functions shared by the analysis scripts.

## Network data

The real_world_networks/Validated/ directory contains the directed networks used in the real-world network analysis. The data has been taken from the ICON Colorado Index of Complex Networks. Note some networks have been zipped due to GitHub size limits. Such files should be unzipped and the zipped version deleted from the folder before running ESCA.


## Reproducibility

The code and network data are provided to facilitate reproduction of the numerical results reported in the paper.

The random-walk simulations and network-rewiring analyses involve stochastic procedures, so small numerical differences may occur between independent runs.

Please cite the associated paper when using this code or the accompanying data.
