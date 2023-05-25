# Instructions for running the 1D heat Bayesian Inverse Problem

The code for sections 2.1-2.6 of the paper can be found in the python
notebook `heat_1D_part1.ipynb`.
It generates figures 2, 3 and 4.
And the code for sections 2.7 of the paper can be found in the python
notebook `heat_1D_part2.ipynb`.
It generates figure 5. 

The code can take very long time to run. For experimentation, you can adjust the
parameter `Ns_factor` in the python notebooks to a value less than one to set up
the code to generate less samples. For example `Ns_factor=0.01` will result in 
number of samples that is 1% of the original number of samples. Note, however,
that generating fewer samples could potentially result in poor approximation
of the posterior.
