# Instructions for running the EIT Bayesian Inverse Problem

To generate the results in figures 7, 8, and 9 in the paper, run the script `EIT.py`.
 
This script uses sampling to draw samples from the posterior of the EIT problem. The scripts uses the samples to then visulize samples from the prior distribution, posterior distribution, visualize the conducticity estimate and the uncertainty for the estimation. Finally the samples are used to visualize the estimate of the Bayesian parameters.

Data regarding different noise-levels are available in `/obs/`:
    - `/obs/obs_circular_inclusion_2_1per_noise.npz` with 1 percent noise-level.
    - `/obs/obs_circular_inclusion_2_5per_noise.npz` with 5 percent noise-level.
    - `/obs/obs_circular_inclusion_2_10per_noise.npz` with 10 percent noise-level.
    - `/obs/obs_circular_inclusion_2_20per_noise.npz` with 20 percent noise-level.
    - `/obs/obs_circular_inclusion_2_50per_noise.npz` with 50 percent noise-level.

change the line 149 to choose the data corresponding the the desired noise-level.
