# Instructions for running the 1D PAT Bayesian Inverse Problem

To generate the PAT results shown in the paper, run the notebook `PAT.ipynb`. The notebook will generate the PAT results and figures shown in the paper. The results will be saved in folder `stat` and the plots will be saved in folder `plots`. The notebook total running time is approximately 8 hours. 

You can set `Ns`, the number of samples, to a smaller value when calling the method `run_EIT` to get results quicker for a shorter MCMC chain. Note that you will also need to set the burn-in `Nb` and the thinning `Nt` accordingly. 