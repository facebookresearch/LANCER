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


def run_ssp_baselines(args, seed):
    from MINLP.bb_problems import stochastic_sp_problem
    bb_problem = stochastic_sp_problem.StochasticShortestPath(grid_n=args["grid_n"], init_gamma=args["init_gamma"])
    dataset = bb_problem.generate_data(num_instances=args["ndata"], thres_mult=args["threshold"], seed=seed)
    if args["baseline"] == "heur":
        sol_dim, _ = bb_problem.get_c_shapes()
        heur_C = np.zeros((args["ndata"], sol_dim))
        means, variances, thresholds = [], [], []
        for i in range(args["ndata"]):
            ci = bb_problem.get_initial_solution(dataset[i])
            heur_C[i, :] = ci
            means.append(dataset[i][0])
            variances.append(dataset[i][1])
            thresholds.append(dataset[i][2])
        loss_list = bb_problem.eval_surrogate(heur_C, aux_data=(np.array(means), np.array(variances), np.array(thresholds)))
    else: # exact MINLP solution
        print("See Ferber et al. 2022 (SurCo) for the implementation of other baselines")
        raise NotImplementedError
    loss_list = np.abs(loss_list)  # probabilities are between 0 and 1
    print("\nStochastic shortest path, threshold =", args["threshold"], " baseline =", args["baseline"])
    print("Mean objective", np.mean(loss_list), "+-", np.std(loss_list))


def run_portfolio_baselines(args, seed):
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
    dataset, init_x_mat = bb_problem.aug_dataset_with_x_init(C_test, C_test_aux, C_test_aux1)
    print("\n*** Combinatorial Portfolio Selection with Nonlinear Obj ***")
    if args["baseline"] == "heur":
        heur_C = np.zeros((n_data, num_stocks))
        for i in range(n_data):
            ci = bb_problem.get_initial_solution((C_test[i], C_test_aux[i]))
            heur_C[i, :] = ci
        obj_list = bb_problem.eval_surrogate(heur_C, aux_data=(C_test, C_test_aux, C_test_aux1, init_x_mat))
        print("heuristic+MILP result: ", np.mean(obj_list), "+-", np.std(obj_list)/np.sqrt(n_data))
    elif args["baseline"] == "miqp":
        obj_list = bb_problem.run_scip_optimal(C_test, C_test_aux, C_test_aux1, init_x_mat, exact=False)
        print("MIQP approximation result: ", np.mean(obj_list), "+-", np.std(obj_list))
    else:
        obj_list = bb_problem.run_scip_optimal(C_test, C_test_aux, C_test_aux1, init_x_mat, exact=True)
        print("MINLP exact (with timeout) result: ", np.mean(obj_list), "+-", np.std(obj_list))


def run_on_problem(args):
    seed = args["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args["problem"] == "ssp":
        run_ssp_baselines(args, seed)
    elif args["problem"] == "ps":
        run_portfolio_baselines(args, seed)
    else:
        raise NotImplementedError("unknown problem (dataset) type")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-pb', type=str, choices=["ssp", "ps"], default="ssp",
                        help="ssp (stochastic shortest path, default), ps (MINLP portfolio selection)")
    parser.add_argument('--baseline', '-b', type=str, choices=["heur", "miqp", "minlp"], default="heur",
                        help="baseline name: heur - domain heuristic, "
                             "miqp - mixed-integer quadratic program approximation for portfolio, "
                             "minlp - exact MINLP solution (takes long time)")
    parser.add_argument('--ndata', '-nd', type=int, default=25,
                        help="number of problem instances")
    parser.add_argument('--seed', type=int, default=10,
                        help="random seed, for reproducibility")
    parser.add_argument('--init_gamma', '-gamma', type=float, default=0.5)

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
