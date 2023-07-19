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
from MINLP.learners import lancer_prior_learner


def run_trainer_mlp(args):
    seed = args["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args["problem"] == "ps":
        from MINLP.bb_problems import portfolio_problem
        num_stocks = args["num_stocks"]
        bb_problem = portfolio_problem.PortfolioSelection(num_feats=2*num_stocks, num_stocks=num_stocks,
                                                          alpha=args["ps_alpha"], beta=args["ps_beta"], lmbd=args["ps_lmbd"],
                                                          min_assets=args["min_assets"], max_assets=args["max_assets"],
                                                          init_gamma=args["init_gamma"])
        data_path = Path(__file__).parent.parent.parent / "utils/data/portfolio_minlp.npz"
        data_npz = np.load(data_path)
        C_test, C_test_covar, C_test_coskew = data_npz["C_test"], data_npz["C_test_aux"], data_npz["C_test_aux1"]
        C_train, C_train_covar, C_train_coskew = data_npz["C_train"], data_npz["C_train_aux"], data_npz["C_train_aux1"]
        n_test, n_train = args["ndata_test"], args["ndata_train"]
        C_test, C_test_covar, C_test_coskew = C_test[:n_test], C_test_covar[:n_test], C_test_coskew[:n_test]
        C_train, C_train_covar, C_train_coskew = C_train[:n_train], C_train_covar[:n_train], C_train_coskew[:n_train]
        _, init_x_test = bb_problem.aug_dataset_with_x_init(C_test, C_test_covar, C_test_coskew)
        _, init_x_train = bb_problem.aug_dataset_with_x_init(C_train, C_train_covar, C_train_coskew)
        aux_data_test = (C_test, C_test_covar, C_test_coskew, init_x_test)
        aux_data_train = (C_train, C_train_covar, C_train_coskew, init_x_train)
        Y_test = bb_problem.get_features(C_test, C_test_covar, C_test_coskew, init_x_test)
        Y_train = bb_problem.get_features(C_train, C_train_covar, C_train_coskew, init_x_train)
        bb_problem.num_feats = Y_test.shape[1]
    else:
        raise NotImplementedError("unknown problem (dataset) type")

    learner = lancer_prior_learner.LancerPriorLearner(args, "mlp", "mlp", bb_problem)
    log_dict = learner.run_training_loop(Y_train, Y_test, aux_data_train, aux_data_test,
                                         n_iter=args["n_iter"],
                                         print_freq=args["print_freq"],
                                         c_max_iter=args["c_max_iter"],
                                         c_nbatch=args["c_nbatch"],
                                         loss_max_iter=args["lancer_max_iter"],
                                         loss_nbatch=args["lancer_nbatch"],
                                         c_epochs_init=args["c_epochs_init"],
                                         c_lr_init=args["c_lr_init"],
                                         init_heuristic=args["init_from_heuristic"],
                                         use_replay_buffer=args["use_buffer"])
    for k, v in log_dict.items():
        print(k, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-pb', type=str, choices=["ps"], default="ps",
                        help="name of the problem/dataset: ps (portfolio selection)")
    parser.add_argument('--n_iter', '-ni', type=int, default=10,
                        help="number of alternating opt iterations")
    parser.add_argument('--ndata_test', '-nte', type=int, default=25,
                        help="test dataset size")
    parser.add_argument('--ndata_train', '-ntr', type=int, default=200,
                        help="train dataset size")
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', type=int, default=0)
    parser.add_argument('--print_freq', '-pf', type=int, default=1)
    parser.add_argument('--use_buffer', '-buf', action='store_true')
    parser.add_argument('--seed', type=int, default=10)

    # related to initialization
    parser.add_argument('--init_from_heuristic', '-init', action='store_true')
    parser.add_argument('--init_gamma', '-gamma', type=float, default=0.0)

    # LANCER-related hyperparameters
    parser.add_argument('--lancer_n_layers', '-lnl', type=int, default=2)
    parser.add_argument('--lancer_layer_size', '-lls', type=int, default=100)
    parser.add_argument('--lancer_lr', '-llr', type=float, default=0.0001)
    parser.add_argument('--lancer_weight_decay', '-lwd', type=float, default=0.0)
    parser.add_argument('--lancer_opt_type', '-lo', type=str, default="adam")
    parser.add_argument('--lancer_max_iter', '-lmi', type=int, default=5)
    parser.add_argument('--lancer_nbatch', '-lnb', type=int, default=1024)

    # Target model-related hyperparameters
    parser.add_argument('--c_n_layers', '-cnl', type=int, default=1)
    parser.add_argument('--c_layer_size', '-cls', type=int, default=500)
    parser.add_argument('--c_lr', '-clr', type=float, default=0.0001)
    parser.add_argument('--c_weight_decay', '-cwd', type=float, default=0.0)
    parser.add_argument('--c_max_iter', '-cmi', type=int, default=5)
    parser.add_argument('--c_nbatch', '-cnb', type=int, default=128)
    # Training initial target model
    parser.add_argument('--c_lr_init', '-clri', type=float, default=0.001)
    parser.add_argument('--c_epochs_init', '-cei', type=int, default=100)

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
    run_trainer_mlp(params)


if __name__ == '__main__':
    main()
