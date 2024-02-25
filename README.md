# Paper-CUQIpy-2-PDE


This repository contains the code for the paper "CUQIpy – Part II: computational uncertainty quantification for PDE-based inverse problems in Python".

## Installation
Install the package using pip (assuming python is installed):
```
pip install cuqipy
```
Some examples require additional packages like the plugins for FEniCS. These can be installed following the instructions on:
* [CUQIpy-FEniCS](https://github.com/CUQI-DTU/CUQIpy-FEniCS)

## Running the examples
The examples (scripts and notebooks) are organized by folders, one folder for each of the 4 case studies (Poisson, Heat 1D, EIT, and PAT). To run the examples that are written as Jupyter notebooks, you need to install [Jupyter](https://jupyter.org/install). One can also view the notebooks on GitHub by clicking on the notebook files in the folders.

## Case studies
The following case studies are included in this repository:

* [Section 1: Introduction](Poisson) 2D Poisson example in the folder `Poisson`
* [Section 2: Framework for PDE-based Bayesian inverse problems in CUQIpy](heat_1D) 1D Heat example in the folder `heat_1D`, see `heat_1D/README.md` for instructions on how to run the scripts.
* [Section 4: CUQIpy-FEniCS example: Electrical Impedance Tomography (EIT)](EIT) EIT example in the folder `EIT`, see `EIT/README.md` for instructions on how to run the scripts
* [Section 5: Photo-acoustic tomography through user-defined PDE models](PAT) PAT example in the folder `PAT`, see `PAT/README.md` for instructions on how to run the scripts


