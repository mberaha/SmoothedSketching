# Code for ``A smoothed-Bayesian approach to frequency recovery from sketched data'' by Beraha, Favaro, and Sesia [(arXiv)](https://arxiv.org/abs/2309.15408)

See below for usage instructions

## Minimax and Worst-Case

To reproduce the numerical minimax and worst-case analysis, see the Jupyter (Python) notebook `Worst Case and Minimax.ipynb`. This requires the Python following packages

```
numpy
pyomo
```

## Main Simulated Examples

The code implementing our methods is written in `Julia` and can be found in the `julia_src` folder. In the `simulation` folders, different `.jl` files can be used to reproduce the numerical experiments. 

We make use of the following packages

```
AdaptiveRejectionSampling v0.2.1
AddPackage v0.1.1
Distributions v0.25.112
Integrals v4.1.0
NNlib v0.8.21
Optimization v3.20.2
OptimizationOptimJL v0.1.14
SpecialFunctions v2.4.0
StatsBase v0.34.3
```

For instance, to run the example on the bigram frequency, from the root folder run

```
julia simulations/word_frequency.jl
```

most experiments benefit from using parallel computing. To this end either prepend the flag `JULIA_NUM_THREADS=XXX` before the terminal command or export an environmental variable.


Plots are produced via the Python notebook `simulations/Report.ipynb` which requires the Python packages

```
numpy>=1.24.4
matplotlib>=3.7.1
pandas>=1.5.3
```

## Conformal Simulation

To run the experiment on conformal inference, you should first install the following Python packages

```
joblib>=1.3.1
matplotlib>=3.7.1
methodtools>=0.4.7
mmh3>=3.0.0
numpy>=1.24.4
pandas>=1.5.3
pyjulia>=0.6.1
scikit-learn>=1.3.0
scipy>=1.10.1
tqdm>=4.65.0
```

Then in a Julia terminal, write
```
using Pkg
Pkg.add("https://github.com/mberaha/Sketch.jl")
```

Then install the `conformalized-sketching` Python package
```
python3 -m pip install git+https://github.com/mberaha/conformalized-sketching.git@cdfb10a3236e54bbb6c80af5b869232c083408a0
``````

and run 
```
python3 simulations/conformal.py
```
