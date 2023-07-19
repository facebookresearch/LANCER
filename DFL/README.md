## LANCER: Learning Decision Losses for DFL 

Before proceeding, please make sure that you have successfully completed all required steps in [installation.md](../installation.md) to setup the environment.

Below we provide detailed examples on how to apply LANCER for DFL problems. This includes replicating results from our paper (see figure below) and suggestions on how to apply LANCER for your own task.



<p align="center">
<img src='https://arman-z.github.io/files/images/dfl_sp_ks.png?raw=true' width="600">
</p>

### 1. Shortest Path
    $ cd LANCER/DFL/
    $ python scripts/run_lancer_dfl.py -pb sp -ngpu -s glpk

### 2. Multidimnesional Knapsack with SCIP
    $ cd LANCER/DFL/
    $ python scripts/run_lancer_dfl.py -pb ks -ngpu -cnl 1 -cls 300 -cei 100 --seed 47 -clri 0.005 -cwd 0.001 -ni 7 -lls 200 -clr 0.001 -lmi 10 -nc 16

### 3. Portfolio Optimization

Note: this requires preparing Portfolio Optimization dataset outlined in [installation.md](../installation.md). 

    $ cd LANCER/DFL/
    $ python scripts/run_lancer_dfl.py -pb pf -ngpu -cnl 1 -cls 500 -clri 0.0005 -clr 0.0005 -cei 50 -ni 8 -llr 0.0001 -cwd 0.1 -lwd 0.0 -r 1.0


### 4. How to use LANCER for your custom optimization problem?

Several steps must be performed to validate LANCER on other problem and/or to try it with other models:

1. Add your optimization problem to ```DFL/bb_problems/``` which should implement all abstract methods in ```BaseProblem```. See examples in the folder. 

2. [Optional] If you need to use models other than MLP for target mapping (e.g. Transformers), you will have to add them to ```DFL/models/models_c.py``` and implement all abstract methods in ```BaseCModel```.

3. [Optional] If you need to use models other than MLP for LANCER (e.g. Transformers), you will have to add them to ```DFL/models/models_lancer.py``` and implement all abstract methods in ```BaseLancer```.

4. Finally, add some bookkeeping code to ```DFL/scripts/run_lancer_dfl.py``` (e.g. read and process your data, parse arguments, etc.) and run it. 

Happy coding 😊

### 4. Running baselines

Additionally, we implemented some of the baselines we compare with. Please refer to [baselines/README.md](./baselines/README.md) for details.
