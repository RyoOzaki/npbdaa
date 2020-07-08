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
+ pytest

# Installation instructions
1. Install GNU compiler collection to use Cython.
```
$ sudo apt install gcc
```
1. Install the necessary libraries for installation.
```
$ pip install numpy future six
$ pip install cython
```
1. Install pybasicbayes.
```
$ git clone https://github.com/mattjj/pybasicbayes
$ cd pybasicbayes
$ python setup.py install
```
1. Install pyhsmm.
```
$ git clone https://github.com/RyoOzaki/pyhsmm
$ cd pyhsmm
$ python setup.py install
```
The repository of pyhsmm was forked and updated by Ryo Ozaki.
If you want to install pyhsmm of master repository, please go to https://github.com/mattjj/pyhsmm
But, the master repository's codes include some bugs in cython codes.
1. Install pyhlm (this).
```
$ git clone https://github.com/RyoOzaki/npbdaa npbdaa
$ cd npbdaa
$ python setup.py install
$ python setup.py build_ext --inplace # if you want to use PyCharm, then you shod compile them
$ # See https://geoexamples.com/python/2017/04/20/pycharm-cython.html
```

# Sample source
There is a sample source of NPB-DAA in "sample" directory.
Please run the "unroll_default_config" before run "pyhlm_sample", and you can change the hyperparameters using the config file "hypparams/defaults.config".
```
$ cd sample
$ python unroll_default_config.py
$ python pyhlm_sample.py
$ python summary_and_plot.py
```

# For new data

### 要は何をすれば良いのか

1. セットアップ
    1. プロジェクトのクローン
    1. 各種ライブラリの導入
    1. optional: PyCharmなどを使う場合は Cython のコンパイル
2. データの配置 (sample/DATA/.)
    1. データは一つの観測 (e.g. aioi_aioi) で得た (m, n_feature) の行列
       ただし、その行列は txt として export される。
        * FYI: file name にはセグメントとワードを書いてある (e.g. aioi_aioi.txt)
    1. 学習する txt のリストを `files.txt` として配置 (sample/.)
3. ハイパーパラメータを設定
    1. 必要に応じて `default.config` の以下を更新:
       model, pyhlm, letter_observation, letter_duration, letter_hsmm, superstate, word_length
    1. `unroll_default_config.py` を使って展開 (各ファイル名はよしなにつけてくれる)
4. `pyhlm_sample.py` (あるいは simple_pyhlm_sample.py) を実行
    1. よしなに学習をすすめてくれる模様
5. `summary_and_plot.py` を実行
    1. load model config -> plot results -> ARI の計算などなど
    1. DAAのletterとsegmentのアノテーションの図がそれぞれ `<label>_l.png` と `<label>_s.png` に書き出される
        * FYI: Path モジュールの `import` が無いように見える (cf. https://github.com/RyoOzaki/npbdaa/pull/2/files)
        * FYI: `Log_likelihood.png` の生成は ValueError を起こす

### 質問

1. 分析/報告の手順
    a. Aと同様、A
    b. Bと同様、B
    c. その他
1. 分析にベースラインとの比較が含まれる場合(b/c)、妥当なベースライン
1. 上の3のハイパーパラメータを設定するステップに関するドキュメントは存在するか

# References
+ Taniguchi, Tadahiro, Shogo Nagasaka, and Ryo Nakashima. [Nonparametric Bayesian double articulation analyzer for direct language acquisition from continuous speech signals](http://ieeexplore.ieee.org/document/7456220/?arnumber=7456220), 2015.

+ Matthew J. Johnson and Alan S. Willsky. [Bayesian Nonparametric Hidden Semi-Markov Models](http://www.jmlr.org/papers/volume14/johnson13a/johnson13a.pdf). Journal of Machine Learning Research (JMLR), 14:673–701, 2013.

# Authors
Tadahiro Taniguch, Ryo Nakashima, Nagasaka Shogo, Tada Yuki, Kaede Hayashi, and Ryo Ozaki.

# License
+ MIT
  + see LICENSE
