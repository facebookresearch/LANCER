## I Install dependencies

A. (Recommended) Install with conda:

1. Install conda, if you don't already have it, by following the instructions at:
   https://conda.io/projects/conda/en/latest/user-guide/install/index.html

   It is better to reboot your terminal after installing conda so that PATH is updated.
	
2. Create a conda environment that will contain python 3:
	```
   $ conda create -n lancer
	
3. Activate the environment (do this every time you open a new terminal and want to run code):
	```
   $ source activate lancer

4. Install the requirements into this conda environment
	```
   $ cd <path_to_LANCER>
   $ pip install --user -r requirements.txt

5. Allow your code to be able to see the 'LANCER' package
	```
   $ pip install -e .

B. Install on system Python using pip:

1. Install requirements 
   ```
   $ cd <path_to_LANCER>
   $ pip install -r requirements.txt

2. Allow your code to be able to see the 'LANCER' package
   ```
   $ pip install -e .

## II Install Mathematical Optimization software(s)

Our code relies on various optimization toolboxes to solve the underlying problem: linear program, quadratic program, mixed-integer linear program and so on. To enable these capabilities, we require installing at least one of the following:

1. SCIP available at https://www.scipopt.org/doc/html/md_INSTALL.php

    You may want to consider Conda to install SCIP. Conda will install SCIP automatically, hence everything can be installed in a single command (conda environment must be activated):
    ```
   conda install --channel conda-forge pyscipopt

2. GLPK (GNU Linear Programming Kit): https://www.gnu.org/software/glpk/ 
   
    ``` 
   or by simply running (in Ubuntu):
   $ sudo apt-get install glpk-utils
    ```
   
3. GUROBI (not tested, requires license): https://www.gurobi.com/ 

## III [Optional] Prepare Portfolio Optimization dataset

This step is **optional** if you're intended to work on Portfolio Optimization problem. Note: this requires downloading "price_data_2004-01-01_2017-01-01_daily.pt" file located at https://github.com/sanketkshah/LODLs/tree/main/data (alternative link to download: https://dl.fbaipublicfiles.com/lancer/price_data_2004-01-01_2017-01-01_daily.pt). Create a folder named `data/` under `utils/`:
```
$ mkdir utils/data/
```
1. Run the following command to generate dataset for smart P+O (a.k.a. DFL) task:
    ```
    $ python utils/gen_portfolio_data.py --problem dfl

2. Run the following command to generate dataset for MINLP task:
    ```
    $ python utils/gen_portfolio_data.py --problem minlp


## IV [Optional] Install PyEPO to run DFL baselines

If you would like to run DFL baselines (see DFL/baselines/run_baselines.py), you need to install PyEPO. Please follow the steps at: https://github.com/khalil-research/PyEPO
