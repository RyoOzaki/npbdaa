# Nonparametric Bayesian Double Articulation Analyzer

This is a Python implementation for Nonparametric Bayesian Double Articulation Analyzer (NPB-DAA). The NPB-DAA can directly acquire language and acoustic models from observed continuous speech signals.

This generative model is called hiererichel Dirichlet process hidden language model (HDP-HLM), which is obtained by extending the hierarchical Dirichlet process hidden semi-Markov model (HDP-HSMM) proposed by Johnson et al. An inference procedure for the HDP-HLM is derived using the blocked Gibbs sampler originally proposed for the HDP-HSMM.

# Description
・NPB_DAA/README - There is a NPB-DAA tutorial in PDF.(In Japanese. English version is coming soon.)

・NPB_DAA/pyhsmm - Python Library for HDP-HSMM. You can get it at [ https://github.com/mattjj/pyhsmm ]. (Please check this VERSION at README)

・NPB_DAA/dahsmm - Python code for NPB-DAA

# Requirement

・Ubuntu 16.04 LTS

・Python 3.6.5

・Numpy
・Scipy
・Matplotlib
・Cython

・pybasicbayes 0.2.2
・pyhsmm 0.1.6

# Install
1. Install Python environments and gcc if you use.
1. Install pybasicbayes.
```
$ git clone https://github.com/mattjj/pybasicbayes
$ cd pybasicbayes
$ python setup.py install
```
1. Install pyhsmm.
```
$ git clone https://github.com/mattjj/pyhsmm
$ cd pyhsmm
$ python setup.py install
```
1. Install pyhlm (this).
```
$ git clone https://github.com/RyoOzaki/npbdaa
$ cd pyhlm
$ python setup.py install
```
1. You can execute sample code in "sample" directory.
```
$ python pyhlm_sample.py
```

# References
・Taniguchi, Tadahiro, Shogo Nagasaka, and Ryo Nakashima. [Nonparametric Bayesian double articulation analyzer for direct language acquisition from continuous speech signals](http://ieeexplore.ieee.org/document/7456220/?arnumber=7456220), 2015.

・Matthew J. Johnson and Alan S. Willsky. [Bayesian Nonparametric Hidden Semi-Markov Models](http://www.jmlr.org/papers/volume14/johnson13a/johnson13a.pdf). Journal of Machine Learning Research (JMLR), 14:673–701, 2013.

# Authors
Tadahiro Taniguch, Ryo Nakashima, Nagasaka Shogo, Tada Yuki, Kaede Hayashi, Ryo Ozaki.

## License
* MIT
    * see LICENSE
