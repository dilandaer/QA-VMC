# Quantum-assisted variational Monte Carlo

This open source code provides the actual practice of calculating the spectral gap for the different proposals and the VMC algorithm based on these proposals in https://arxiv.org/abs/2502.20799. In the example folder, we provide the calculation process for a water as an example.

## Requirement

- jax == 0.5.3
- jaxlib == 0.5.3
- netket >= 3.16.0
- numpy >= 1.24.0, < 2.0.0
- scipy >= 1.12.0
- pyscf >= 2.5.0
- mindspore == 2.5.0
- mindquantum == 0.10.0