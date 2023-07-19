## Baseline methods for smart Predict+Optimize problems.

IMPORTANT: running the baselines requires pyepo package installed: https://github.com/khalil-research/PyEPO

The main running script is ```run_baselines.py```. Run the following commands to replicate our experimental results. Note: below scripts will execute a single run of the experiment. To obtain error bars, set ```-ni``` argument to 5.

#### Shortest Path with SPO+
```
python baselines/run_baselines.py -pb sp -b spo
```

#### Shortest Path with DBB
```
python baselines/run_baselines.py -pb sp -b dbb
```

#### Shortest Path with 2-stage
```
python baselines/run_baselines.py -pb sp -b two_stage
```

#### Multidimnesional Knapsack with SPO+
```
python baselines/run_baselines.py -pb ks --seed 47 --n_epochs 20 -lr 0.001 -wd 0.001 -nl 1 -ls 300 -b spo -ws
```

#### Multidimnesional Knapsack with DBB
```
python baselines/run_baselines.py -pb ks --seed 47 --n_epochs 20 -lr 0.0001 -wd 0.001 -nl 1 -ls 300 -b dbb -ws
```

#### Multidimnesional Knapsack with 2-stage
```
python baselines/run_baselines.py -pb ks --seed 47 --n_epochs 100 -lr 0.005 -wd 0.001 -nl 1 -ls 300 -b two_stage
```

<p align="center">
<img src='https://arman-z.github.io/files/images/dfl_sp_ks.png?raw=true' width="600">
</p>