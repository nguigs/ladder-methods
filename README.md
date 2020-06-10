# Ladder Transport

This repository gathers the code to illustrate the paper on the convergence analysis of ladder methods for parallel transport.

## Install

Clone the repo:

`git clone https://gitlab.inria.fr/nguigui/ladder-transport.git`

The implementation relies on the package [`geomstats`](http://geomstats.ai) with its default numpy backend. The dependencies
of geomstats and the package from the author's fork can be installed via

```
cd ladder-transport
pip install -r requirements.txt
```

It might be useful to install these packages in a virtualenv or conda env. If the latter is chosen,
note that `autograd` and `geomstats` must be installed via pip:

```
pip install autograd
pip install git+https://github.com/nguigs/geomstats.git@nguigs-pt
```

After a successful install, run the [notebook](Convergence%20Analysis.ipynb) to reproduce all the figures of the paper.