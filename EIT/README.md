# Instructions for running the EIT Bayesian Inverse Problem

To generate the EIT results shown in the paper, run the notebook `EIT.ipynb`. The notebook will generate the EIT results and figures shown in the paper. The results will be saved in folder `stat` and the plots will be saved in folder `plots`. The notebook total running time is approximately 15 hours. 

You can set `Ns`, the number of samples, to a smaller value when calling the method `run_EIT` to get results quicker for a shorter MCMC chain. Note that you will also need to set the burn-in `Nb` and the thinning `Nt` accordingly. 

Observed data for each noise-levels are available in `/data/`:
  - `/data/obs_circular_inclusion_2_5per_noise.npz` with 5 percent noise-level.
  - `/data/obs_circular_inclusion_2_10per_noise.npz` with 10 percent noise-level.
  - `/data/obs_circular_inclusion_2_20per_noise.npz` with 20 percent noise-level.
