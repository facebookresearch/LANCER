# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import numpy as np
import torch
import random
from utils import nn_utils
from pathlib import Path
from DFL.learners.lancer_learner import LancerLearner
from sklearn.model_selection import train_test_split


def run_on_problem(args):

    perf_list_train, perf_list_test = [], []
    for expt_i in range(args["expt_iters"]):
        seed = args["seed"]+expt_i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args["problem"] == "sp":
            # LP for shortest path model
            from DFL.bb_problems import sp_problem
            bb_problem = sp_problem.ShortestPathProblem(num_feats=5, grid=(5, 5), solver=args["solver"])
            Y, Z = bb_problem.generate_dataset(N=args["ndata"]+1000, deg=6, noise_width=0.5)
            Y_train, Y_test, Z_train, Z_test = train_test_split(Y, Z, test_size=1000, random_state=seed)
            dataset = (Y_train, Y_test, Z_train, Z_test, None, None)
        elif args["problem"] == "ks":
            from DFL.bb_problems import knapsack_problem
            cap, num_items, kdim, p = 45, 100, 5, 256
            weights = np.random.uniform(0, 1, (kdim, num_items))
            # ILP for multidimnesional knapsack
            bb_problem = knapsack_problem.KnapsackProblem(num_feats=p, weights=weights, cap=cap, num_items=num_items,
                                                         kdim=kdim, n_cpus=args["n_cpus"], solver=args["solver"])
            if expt_i == 0:
                # construct random neural net only once
                rnd_nn = nn_utils.build_mlp(input_size=num_items, output_size=p, n_layers=1, size=500,
                                            activation="relu", output_activation="tanh")
            Y, Z = bb_problem.generate_dataset_nn(N=args["ndata"]+1000, rnd_nn=rnd_nn, noise_width=0.1)
            Y_train, Y_test, Z_train, Z_test = train_test_split(Y, Z, test_size=1000, random_state=seed)
            dataset = (Y_train, Y_test, Z_train, Z_test, None, None)
        elif args["problem"] == "pf":
            from DFL.bb_problems import portfolio_problem
            num_feats, num_stocks, alpha = 28, 50, 0.1
            bb_problem = portfolio_problem.PortfolioOptProblem(num_feats, num_stocks, alpha)
            data_path = Path(__file__).parent.parent.parent / "utils/data/portfolio_dfl.npz"
            data_npz = np.load(data_path)
            Y_train, Y_test, Z_train, Z_test = \
                data_npz["Y_train"], data_npz["Y_test"], data_npz["C_train"], data_npz["C_test"]
            Z_train_aux, Z_test_aux = data_npz["C_train_aux"], data_npz["C_test_aux"]
            dataset = (Y_train, Y_test, Z_train, Z_test, Z_train_aux, Z_test_aux)
        else:
            raise NotImplementedError("unknown problem (dataset) type")

        learner = LancerLearner(args, "mlp", "mlp", bb_problem)
        log_dict = learner.run_training_loop(dataset,
                                             n_iter=args["n_iter"],
                                             print_freq=args["print_freq"],
                                             c_max_iter=args["c_max_iter"],
                                             c_nbatch=args["c_nbatch"],
                                             lancer_max_iter=args["lancer_max_iter"],
                                             lancer_nbatch=args["lancer_nbatch"],
                                             c_epochs_init=args["c_epochs_init"],
                                             c_lr_init=args["c_lr_init"],
                                             use_replay_buffer=args["use_buffer"])
        if args["problem"] in ["ks", "sp"]:
            # report normalized regret for ks and sp
            perf_list_train.append(log_dict["regret_tr"][-1])
            perf_list_test.append(log_dict["regret_te"][-1])
        else: # report performances of several baselines:
            lancer_dl = bb_problem.calc_results_baselines(log_dict, Z_test, Z_test_aux)
            perf_list_test.append(lancer_dl)

    if args["problem"] in ["ks", "sp"]:
        print("\n\n======== Train regret mean =", np.mean(perf_list_train), "% std =", np.std(perf_list_train))
        print("======== Test regret mean =", np.mean(perf_list_test), "% std =", np.std(perf_list_test))
    else:
        print("\n\n======== Test DL mean =", np.mean(perf_list_test), ", std =", np.std(perf_list_test))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-pb', type=str, choices=["sp", "ks", "pf"], default="sp",
                        help="name of the problem/dataset: sp (shortest path), ks (multidim knapsack), pf (portfolio)")
    parser.add_argument('--solver', '-s', type=str, choices=["scip", "gurobi", "glpk"], default="scip",
                        help="optimization solver to use, scip is default")
    parser.add_argument('--ndata', '-n', type=int, default=1000,
                        help="dataset size")
    parser.add_argument('--n_iter', '-ni', type=int, default=10,
                        help="number of alternating optimization iterations")
    parser.add_argument('--n_cpus', '-nc', type=int, default=1,
                        help="number of cpus used to parallelize bb_problem (for ks only), default=1")
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', type=int, default=0)
    parser.add_argument('--print_freq', '-pf', type=int, default=1)
    parser.add_argument('--use_buffer', '-buf', action='store_true')
    parser.add_argument('--expt_iters', '-ei', type=int, default=1,
                        help="number of times to repeat expt with different rnd seeds (for mean and std)")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed, for reproducibility")

    # LANCER-related hyperparameters
    parser.add_argument('--lancer_n_layers', '-lnl', type=int, default=2)
    parser.add_argument('--lancer_layer_size', '-lls', type=int, default=100)
    parser.add_argument('--lancer_lr', '-llr', type=float, default=0.001)
    parser.add_argument('--lancer_weight_decay', '-lwd', type=float, default=0.01)
    parser.add_argument('--lancer_opt_type', '-lo', type=str, default="adam")
    parser.add_argument('--lancer_max_iter', '-lmi', type=int, default=5)
    parser.add_argument('--lancer_nbatch', '-lnb', type=int, default=1024)

    # Target model-related hyperparameters
    parser.add_argument('--c_n_layers', '-cnl', type=int, default=0) # default is linear model
    parser.add_argument('--c_layer_size', '-cls', type=int, default=64)
    parser.add_argument('--c_lr', '-clr', type=float, default=0.005)
    parser.add_argument('--c_weight_decay', '-cwd', type=float, default=0.01)
    parser.add_argument('--z_regul', '-r', type=float, default=0.0,
                        help="apply 2-stage regularization, default=0.0")
    parser.add_argument('--c_opt_type', '-co', type=str, default="adam")
    parser.add_argument('--c_max_iter', '-cmi', type=int, default=5)
    parser.add_argument('--c_nbatch', '-cnb', type=int, default=128)
    # Training initial target model using 2-stage (a.k.a. warm-start)
    parser.add_argument('--c_lr_init', '-clri', type=float, default=0.005)
    parser.add_argument('--c_epochs_init', '-cei', type=int, default=30)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)
    print(params)
    run_on_problem(params)


if __name__ == '__main__':
    main()
