# Instructions for running the EIT Bayesian Inverse Problem

To generate the EIT results shown in the paper, run the script `EIT.py` for each of the different noise levels: $5%$, $10%$, $20%$. Then run the script `plot_paper_figures.py` to generate the figures 6, 7, 8, and 9 in the paper. To set the noise level, change the variable `noise_percent` in the script `EIT.py` to be either `5`, `10`, or `20`. The script `EIT.py` will then generate the sampling results for the corresponding noise level and save it in the folder `/stat/`. The script `plot_paper_figures.py` will then use these results to generate the paper figures.

Observed data for each noise-levels are available in `/obs/`:
    - `/obs/obs_circular_inclusion_2_5per_noise.npz` with 5 percent noise-level.
    - `/obs/obs_circular_inclusion_2_10per_noise.npz` with 10 percent noise-level.
    - `/obs/obs_circular_inclusion_2_20per_noise.npz` with 20 percent noise-level.