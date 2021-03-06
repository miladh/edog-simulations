[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/miladh/edog-simulations/master)
# edog-simulations

This repository contains all simulations presented in the paper:

"Firing-Rate Based Network Modeling of the dLGN Circuit: Effects of Cortical Feedback on Spatiotemporal Response Properties of Relay Cells". https://doi.org/10.1371/journal.pcbi.1006156


These simulations use the [pyLGN simulator](http://pylgn.readthedocs.io/en/latest/) and the calculations are done in Jupyter notebooks.

## Installation

- Install [pyLGN](http://pylgn.readthedocs.io/en/latest/installation.html#installation) (v0.9): `conda install -c defaults -c conda-forge -c cinpla pylgn=0.9`
- Clone or download this repo.
- In terminal run `python setup.py`

## Dependencies

- python >=3.5
- matplotlib
- numpy
- scipy
- setuptools
- pillow
- quantities 0.12.1
- pyyaml
- pylgn 0.9
