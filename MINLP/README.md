## LANCER: Learning Decision Losses for MINLP 

Before proceeding, please make sure that you have successfully completed all required steps in [installation.md](../installation.md) to setup the environment.

Below we provide detailed examples on how to apply LANCER for mixed-integer nonlinear programs (MINLP). Specifically, we automatically learn surrogate problems that are easy to solve and whose solution provide "good" solution for the original nonlinear problem. 

We first provide scripts to reproduce the results from our paper and then list steps on how to apply LANCER for your own MINLP.


### 1. Stochastic Shortest Path

<p align="center">
<img src='https://arman-z.github.io/files/images/minlp_ssp.png?raw=true' width="600">
</p>

Run the following command to try on the smaller grid size (5x5). **IMPORTANT:** You may want to change ```-thr``` argument to validate different settings: 0.9 for tight deadline, 1.0 for normal deadline and 1.1 for loose deadline.

    $ cd LANCER/MINLP/
    $ python scripts/run_lancer_zero.py -pb ssp -ngpu -buf -init -ni 10 -thr 0.9

Run the following command for 15x15 grid size (change ```-thr``` to try different settings):

    $ python scripts/run_lancer_zero.py -pb ssp -ngpu -buf -init -lls 300 -ni 100 -gamma 0.2 -gn 15 -thr 0.9

### 2. Combinatorial Portfolio Optimization with 3rd order loss

<p align="center">
<img src='https://arman-z.github.io/files/images/minlp_portfolio.png?raw=true' width="400">
</p>

Note: this requires preparing Portfolio Optimization dataset outlined in [installation.md](../installation.md). Moreover, we tested our code on SCIP only, so it **requires SCIP installed on your system**.

On this benchmark, we provide two versions of LANCER: LANCER-zero and LANCER-prior. Please refer to the paper for details. 

Use the following command to run **LANCER-zero**: 

    $ cd LANCER/MINLP/
    $ python scripts/run_lancer_zero.py -pb ps -buf -ngpu -init -sstd 0.5 --init_gamma 0.0 -llr 0.001 -lls 300 -ni 15

Use the following command to run **LANCER-prior**:

    $ python scripts/run_lancer_prior.py -init -ngpu -ni 7 -lmi 2 -cmi 2

Here is some animation for LANCER-zero which illustrated solutions at each alternating optimization iteration. It looks like the initial solution (and first several iterations) mostly care about high returns (rewards). Then the algorithm tries to reduce the "risk minus skewness" while preserving a high return.

<p align="center">
<img src='https://arman-z.github.io/files/images/comb_portf_sol.gif?raw=true' width="400">
</p>

### 3. How to use LANCER for your custom optimization problem?

Several steps must be performed to validate LANCER on other problem and/or to try it with other models:

1. Add your optimization problem to ```MINLP/bb_problems/``` which should implement all abstract methods in ```BaseProblem```. See examples in the folder. 

2. [Optional] If you need to use models other than MLP and DirectC for target mapping (e.g. Transformers), you will have to add them to ```MINLP/models/models_c.py``` and implement all abstract methods in ```BaseCModel```.

3. [Optional] If you need to use models other than MLP for LANCER (e.g. Transformers), you will have to add them to ```MINLP/models/models_lancer.py``` and implement all abstract methods in ```BaseLancer```.

4. Finally, add some bookkeeping code to ```MINLP/scripts/run_lancer_zero.py``` or ```run_lancer_prior.py``` (e.g. read and process your data, parse arguments, etc.) and run it. 

Happy coding 😊

### 4. Running baselines

Additionally, we implemented some of the baselines we compare with. Please refer to ```sciprts/run_baselines.py``` for details.
