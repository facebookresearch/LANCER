<p align="center">
<img src='https://arman-z.github.io/files/images/lancer_diagram.png?raw=true' width="600">
</p>

## *Lan*d*s*cap*e* Su*r*rogate: Learning Decision Losses for Mathematical Optimization Under Partial Information

We are introducing *LANCER*, a versatile framework designed to tackle challenging optimization problems, such as those found in nonlinear combinatorial problems, smart predict+optimize framework, etc.

This source code accompanies our paper 📜: <a href="https://arxiv.org/abs/2307.08964"> https://arxiv.org/abs/2307.08964 </a>

Please 🌟star🌟 this repo and cite our paper 📜 if you like (and/or use) our work, thank you!

## Contributors

<a href="https://arman-z.github.io/">Arman Zharmagambetov</a> (First Author),
<a href="http://bamos.github.io/">Brandon Amos</a>,
<a href="https://aaron-ferber.github.io/">Aaron Ferber</a>,
<a href="https://taoanhuang.github.io/">Taoan Huang</a>,
<a href="https://viterbi.usc.edu/directory/faculty/Dilkina/Bistra">Bistra Dilkina</a>,
<a href="https://yuandong-tian.com/">Yuandong Tian</a> (Principal Investigator)

## What is in this release?

This release contains our implementation of LANCER and its application to 1) learning surrogates for mixed-integer nonlinear programming (**MINLP**) and; 2) smart Predict+Optimize (a.k.a. decision-focused learning or **DFL**), but it can also be applied to a range of other large-scale optimization problems, such as hyper-parameter optimization, model-based reinforcement learning, etc.

### 0. Setup

Please follow the steps in [installation.md](./installation.md) to setup the environment.

### 1. Applying LANCER for MINLP

[MINLP/README.md](./MINLP/README.md) contains instructions to validate LANCER for MINLP tasks. We evaluate on two benchmarks: Stochastic Shortest Path and Combinatorial Portfolio Optimization/Selection with 3rd order objective. 

### 2. Applying LANCER for DFL

[DFL/README.md](./DFL/README.md) contains instructions to validate LANCER for DFL tasks. We evaluate on three benchmarks: Shortest Path, Multidimensional Knapsack and Portfolio Optimization/Selection.      

## License
Our source code is under [CC-BY-NC 4.0 license](./LICENSE).