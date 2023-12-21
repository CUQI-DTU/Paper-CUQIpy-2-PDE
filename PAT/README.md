# Instructions for running the PAT 1D Bayesian Inverse Problem

To generate figures 10, 11, and 12 in the paper, first run the script `PAT.py`
for the full data case (set `full_data = True`) and for the partial data case 
(set `full_data = False`). The result samples will be saved in the folder
`stat`. Then run the script `plot_paper_figures.py` to generate the figures 10,
11, and 12. The figures will be saved in the folder `plots`.

The `PAT.py` script performs the sampling using the pCN method for the 
photo-acoustic tomography problem and generate samples and point-estimates for
the posterior.