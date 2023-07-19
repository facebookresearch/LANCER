# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import cvxpy as cp
from DFL.bb_problems import base_problem


class PortfolioOptProblem(base_problem.BaseProblem):
    def __init__(self, num_feats, num_stocks, alpha):
        super(PortfolioOptProblem, self).__init__()
        self.num_feats = num_feats
        self.num_stocks = num_stocks
        self.alpha = alpha
        self.lancer_out_activation = "tanh"
        self.model, self.var, self.ret, self.L = self.build_model()

    def build_model(self, **kwargs):
        x_var = cp.Variable(self.num_stocks)
        L_sqrt_para = cp.Parameter((self.num_stocks, self.num_stocks))
        p_para = cp.Parameter(self.num_stocks)
        constraints = [x_var >= 0, x_var <= 1, cp.sum(x_var) == 1]
        objective = cp.Minimize(self.alpha * cp.sum_squares(L_sqrt_para.T @ x_var) - p_para.T @ x_var)
        problem = cp.Problem(objective, constraints)
        return problem, x_var, p_para, L_sqrt_para

    def _get_objective(self, z, x, sqrt_covar):
        # sqrt_covar must be Cholesky decomposition of covar matrix
        quad_term = np.square(sqrt_covar.T @ x).sum()
        obj = self.alpha * quad_term - z.T @ x
        return obj

    def eval(self, z_pred: np.ndarray, z_true: np.ndarray, **kwargs) -> np.ndarray:
        assert z_pred.shape == z_true.shape
        assert "aux_data" in kwargs
        Q_mat = kwargs["aux_data"]
        N = z_true.shape[0]
        assert Q_mat.shape[0] == N
        sqrt_covar = np.linalg.cholesky(Q_mat)
        f_hat_list = []
        for i in range(N):
            if i%100 == 0:
                print("Solving QP:", i, " out of ", N)
            self.ret.value = z_pred[i]
            self.L.value = sqrt_covar[i]
            self.model.solve(solver=cp.SCS)
            sol = self.var.value
            ############################
            f_hat_i_cp = self._get_objective(z_true[i], sol, sqrt_covar[i])
            f_hat_list.append([f_hat_i_cp])
        return np.array(f_hat_list)

    def get_c_shapes(self):
        return 1, self.num_stocks

    def get_activations(self):
        return "relu", "tanh"  # hidden layer activation, output_activation

    def get_rnd_performance(self, z_true, z_aux):
        z_pred_rnd = np.random.rand(z_true.shape[0], z_true.shape[1])
        f_hat = self.eval(z_pred_rnd, z_true, aux_data=z_aux)
        return f_hat.mean()

    def calc_results_baselines(self, log_dict, Z_test, Z_test_aux):
        # document the value of a random guess
        rnd_dl, num_runs = 0.0, 10
        for rnd_run_i in range(num_runs):
            print("Running random predictor: ", rnd_run_i)
            obj_i = self.get_rnd_performance(Z_test, Z_test_aux)
            rnd_dl += obj_i / num_runs
        # report performances
        opt_dl = log_dict["dl_te_opt"].mean()
        print("\n*************\nRandom Decision Loss =", rnd_dl)
        print("Optimal Decision Loss =", opt_dl)
        opt_minus_rnd = rnd_dl - opt_dl
        print("2-stage Decision Loss (normalized) =", (log_dict["dl_te"][0] - opt_dl) / opt_minus_rnd)
        lancer_dl = (log_dict["dl_te"][-1] - opt_dl) / opt_minus_rnd
        print("Lancer Decision Loss (normalized) =", lancer_dl)
        print("-----------------\n")
        return lancer_dl
