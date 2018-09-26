# Nonparametric Bayesian Double Articulation Analyzer

This is a Python implementation for Nonparametric Bayesian Double Articulation Analyzer (NPB-DAA).
The NPB-DAA can directly acquire language and acoustic models from observed continuous speech signals.

This generative model is called hiererichel Dirichlet process hidden language model (HDP-HLM), which is obtained by extending the hierarchical Dirichlet process hidden semi-Markov model (HDP-HSMM) proposed by Johnson et al.
An inference procedure for the HDP-HLM is derived using the blocked Gibbs sampler originally proposed for the HDP-HSMM.

# Requirement

+ Ubuntu 16.04 LTS
+ Python 3.6.5
+ Numpy 1.14.2
+ Scipy 1.0.1
+ Scikit-learn 0.19.1
+ Matplotlib 2.2.2
+ Joblib 0.11
+ Cython 0.28.2
+ tqdm 4.23.4
+ pybasicbayes 0.2.2
+ pyhsmm 0.1.6

# Installation instructions
1. Install GNU compiler collection to use Cython.
```
$ sudo apt install gcc
```
1. Install the necessary libraries for installation.
```
$ pip install numpy future six
```
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
$ git clone https://github.com/RyoOzaki/npbdaa npbdaa
$ cd npbdaa
$ python setup.py install
```

# Sample source
There is a sample source of NPB-DAA in "sample" directory.
```
$ cd sample
$ python pyhlm_sample.py
```

# References
+ Taniguchi, Tadahiro, Shogo Nagasaka, and Ryo Nakashima. [Nonparametric Bayesian double articulation analyzer for direct language acquisition from continuous speech signals](http://ieeexplore.ieee.org/document/7456220/?arnumber=7456220), 2015.

+ Matthew J. Johnson and Alan S. Willsky. [Bayesian Nonparametric Hidden Semi-Markov Models](http://www.jmlr.org/papers/volume14/johnson13a/johnson13a.pdf). Journal of Machine Learning Research (JMLR), 14:673–701, 2013.

# Authors
Tadahiro Taniguch, Ryo Nakashima, Nagasaka Shogo, Tada Yuki, Kaede Hayashi, and Ryo Ozaki.

# License
+ MIT
  + see LICENSE
