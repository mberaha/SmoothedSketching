# Code for ``Frequency and cardinality recovery from sketched data: a novel approach bridging Bayesian and frequentist views''

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
AdaptiveRejectionSampling
Distributions
Integrals
NNlib
Optimization
OptimizationOptimJL
Random
SpecialFunctions
StatsBase
```

For instance, to run the example on the bigram frequency, from the root folder run

```
julia simulations/word_frequency.jl
```

most experiments benefit from using parallel computing. To this end either prepend the flag `JULIA_NUM_THREADS=XXX` before the terminal command or export an environmental variable.


Plots are produced via the Python notebook `simulations/Report.ipynb` which requires the Python packages

```
numpy
matplotlib
pandas
```

## Conformal Simulation

To run the experiment on conformal inference, you should first install the following Python packages

```
scipy
scikit-learn
tqdm
mmh3
methodtools
pyjulia
```

Then in a Julia terminal, write

```
using Pkg
Pkg.add("https://github.com/mberaha/Sketch.jl")
```

Then install the `conformalized-sketching` Python package

```
python3 -m pip install git+https://github.com/mberaha/conformalized-sketching.git
``````

and run 

```
python3 simulations/conformal.py
```
