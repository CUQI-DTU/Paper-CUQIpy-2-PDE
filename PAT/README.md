# Instructions for running the PAT 1D Bayesian Inverse Problem

To generate figures 10, 11, and 12 in the paper, run the script `PAT.py`
two times, one with the parameters `full_data = True` for the full data case
and the other with `full_data = False` for the partial data case.
The result samples will be saved in the folder `stat`. Then run the script
`plot_paper_figures.py` to generate the figures 10, 11, and 12. The figures will
be saved in the folder `plots`.

The `PAT.py` script performs the sampling using the pCN method for the 
photo-acoustic tomography problem and generates samples and point-estimates for
the posterior.