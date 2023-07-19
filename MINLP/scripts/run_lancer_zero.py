# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import numpy as np
import torch
import random
from pathlib import Path
from MINLP.learners import lancer_zero_learner


def start_lancer_zero(dataset, bb_solver, args):
    loss_list = []
    for i in range(0, len(dataset)):
        learner = lancer_zero_learner.LancerZeroLearner(args, "direct", "mlp", bb_solver)
        log_dict = learner.run_training_loop(dataset[i],
                                             n_iter=args["n_iter"],
                                             print_freq=args["print_freq"],
                                             c_max_iter=args["c_max_iter"],
                                             lancer_max_iter=args["lancer_max_iter"],
                                             lancer_nbatch=args["lancer_nbatch"],
                                             num_samples=args["n_samples"],
                                             rnd_sigma=args["sampling_std"],
                                             init_heuristic=args["init_from_heuristic"],
                                             use_replay_buffer=args["use_buffer"])
        loss_i = np.array(log_dict["dl_tr"]).min()
        loss_list.append(loss_i)
        # print("Instance: ", i, "|", abs(log_dict["dl_tr"][0]), np.abs(log_dict["dl_tr"][1:]).max())
        print("INSTANCE: ", i, " completed, objective: ", loss_i)
    return loss_list


def run_on_problem(args):
    seed = args["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args["problem"] == "ssp":
        from MINLP.bb_problems import stochastic_sp_problem
        bb_problem = stochastic_sp_problem.StochasticShortestPath(grid_n=args["grid_n"], init_gamma=args["init_gamma"])
        dataset = bb_problem.generate_data(num_instances=args["ndata"], thres_mult=args["threshold"], seed=seed)
        loss_list = start_lancer_zero(dataset, bb_problem, args)
        loss_list = np.abs(loss_list) # probabilities are between 0 and 1
        print("\nStochastic shortest path, threshold =", args["threshold"])
        print("Mean objective", np.mean(loss_list), "+-", np.std(loss_list))
    elif args["problem"] == "ps":
        from MINLP.bb_problems import portfolio_problem
        num_stocks, n_data = args["num_stocks"], args["ndata"]
        bb_problem = portfolio_problem.PortfolioSelection(num_feats=0, num_stocks=num_stocks, alpha=args["ps_alpha"],
                                                         beta=args["ps_beta"], lmbd=args["ps_lmbd"],
                                                         min_assets=args["min_assets"], max_assets=args["max_assets"],
                                                         init_gamma=args["init_gamma"])
        data_path = Path(__file__).parent.parent.parent / "utils/data/portfolio_minlp.npz"
        data_npz = np.load(data_path)
        C_test, C_test_aux, C_test_aux1 = data_npz["C_test"], data_npz["C_test_aux"], data_npz["C_test_aux1"]
        C_test, C_test_aux, C_test_aux1 = C_test[:n_data], C_test_aux[:n_data], C_test_aux1[:n_data]
        dataset, _ = bb_problem.aug_dataset_with_x_init(C_test, C_test_aux, C_test_aux1)
        loss_list = start_lancer_zero(dataset, bb_problem, args)
        print("\n*** Combinatorial Portfolio Selection with Nonlinear Obj ***")
        print("\nMean objective", np.mean(loss_list), "+-", np.std(loss_list) / np.sqrt(n_data))
    else:
        raise NotImplementedError("unknown problem (dataset) type")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-pb', type=str, choices=["ssp", "ps"], default="ssp",
                        help="ssp (stochastic shortest path, default), ps (MINLP portfolio selection)")
    parser.add_argument('--n_iter', '-ni', type=int, default=50,
                        help="number of alternating optimization iterations")
    parser.add_argument('--ndata', '-nd', type=int, default=25,
                        help="number of problem instances")
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', type=int, default=0)
    parser.add_argument('--print_freq', '-pf', type=int, default=2)
    parser.add_argument('--use_buffer', '-buf', action='store_true')
    parser.add_argument('--seed', type=int, default=10,
                        help="random seed, for reproducibility")

    # related to initialization
    parser.add_argument('--init_from_heuristic', '-init', action='store_true',
                        help="initialize cost vector from heuristic solution")
    parser.add_argument('--init_gamma', '-gamma', type=float, default=0.5)

    # LANCER-related hyperparameters
    parser.add_argument('--lancer_n_layers', '-lnl', type=int, default=2)
    parser.add_argument('--lancer_layer_size', '-lls', type=int, default=200)
    parser.add_argument('--lancer_lr', '-llr', type=float, default=0.0005)
    parser.add_argument('--lancer_weight_decay', '-lwd', type=float, default=0.0)
    parser.add_argument('--lancer_opt_type', '-lo', type=str, default="adam")
    parser.add_argument('--lancer_max_iter', '-lmi', type=int, default=10)
    parser.add_argument('--lancer_nbatch', '-lnb', type=int, default=1024)

    # Target model-related hyperparameters
    parser.add_argument('--c_lr', '-clr', type=float, default=0.001)
    parser.add_argument('--c_opt_type', '-co', type=str, default="adam")
    parser.add_argument('--c_max_iter', '-cmi', type=int, default=10)

    # sampling related
    parser.add_argument('--n_samples', '-ns', type=int, default=100)
    parser.add_argument('--sampling_std', '-sstd', type=float, default=0.6)

    # Problem specific hyperparams: ssp
    parser.add_argument('--threshold', '-thr', choices=[0.9, 1.0, 1.1], type=float, default=1.1,
                        help="tight (0.9), normal (1.0) and loose (1.1) schedules")
    parser.add_argument('--grid_n', '-gn', type=int, default=5)

    # Problem specific hyperparams: ps
    parser.add_argument('--num_stocks', type=int, default=100)
    parser.add_argument('--ps_alpha', type=float, default=0.1)
    parser.add_argument('--ps_beta', type=float, default=0.5)
    parser.add_argument('--ps_lmbd', type=float, default=0.01)
    parser.add_argument('--min_assets', type=int, default=3)
    parser.add_argument('--max_assets', type=int, default=10)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)
    print(params)
    run_on_problem(params)


if __name__ == '__main__':
    main()
