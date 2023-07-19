# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from MINLP.bb_problems import base_problem
import numpy as np
from pyomo import environ as pe
from pyomo import opt as po


class PortfolioSelection(base_problem.BaseProblem):
    def __init__(self, num_feats, num_stocks, alpha, beta, lmbd, min_assets, max_assets,
                 init_gamma, solve_true=False):
        super(PortfolioSelection, self).__init__()
        self.num_feats = num_feats
        self.num_stocks = num_stocks
        self.alpha = alpha  # penalty on quadratic term
        self.beta = beta  # penalty on cubic term
        self.lmbd = lmbd  # penalty on deviation from x_init
        self.min_assets = min_assets
        self.max_assets = max_assets
        self.solve_true = solve_true
        self.init_gamma = init_gamma  # coef for sampling cost vectors
        self.lancer_out_activation = "tanh"

        _model, _vars, _yvars = self.build_model()
        self._model = _model
        self._vars = _vars
        self._yvars = _yvars
        self.sense = "MINIMIZE"
        self._solverfac = po.SolverFactory("scip")

    def build_model(self, **kwargs):
        # min and max proportion to pick from each selected portfolios
        fmin = 0.01
        fmax = 0.2
        m = pe.ConcreteModel("combinatorial portfolio")
        m.x_set = pe.Set(initialize=list(range(self.num_stocks)))
        m.y_set = pe.Set(initialize=list(range(self.num_stocks)))
        m.v_set = pe.Set(initialize=list(range(self.num_stocks)))
        x = pe.Var(m.x_set, domain=pe.NonNegativeReals)
        y = pe.Var(m.y_set, domain=pe.NonNegativeReals)
        v = pe.Var(m.v_set, domain=pe.Binary)
        m.x = x
        m.y = y
        m.v = v
        # constraints
        m.cons = pe.ConstraintList()
        m.cons.add(sum(x[j] for j in range(self.num_stocks)) == 1)
        m.cons.add(sum(v[j] for j in range(self.num_stocks)) <= self.max_assets)
        m.cons.add(sum(v[j] for j in range(self.num_stocks)) >= self.min_assets)
        for i in range(self.num_stocks):
            m.cons.add(fmin * v[i] <= x[i])
            m.cons.add(fmax * v[i] >= x[i])
        m.cons_dev = pe.ConstraintList()
        for i in range(self.num_stocks):
            m.cons_dev.add(x[i] <= y[i])
            m.cons_dev.add(-x[i] <= y[i])
        m.obj = pe.Objective(sense=pe.minimize, expr=0)
        return m, x, y

    def _solve_single_core(self, z_pred, x0):
        self._model.del_component(self._model.cons_dev)
        self._model.del_component(self._model.cons_dev_index)
        self._model.cons_dev = pe.ConstraintList()
        for i in range(self.num_stocks):
            self._model.cons_dev.add(self._vars[i] - x0[i] <= self._yvars[i])
            self._model.cons_dev.add(-self._vars[i] + x0[i] <= self._yvars[i])
        self._model.del_component(self._model.obj)
        obj = sum(-z_pred[j] * self._vars[k] for j, k in enumerate(self._vars)) + \
              self.lmbd * sum(self._yvars[k] for k in self._yvars)
        self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)
        self._solverfac.solve(self._model)
        sol = [pe.value(self._vars[k]) for k in self._vars]
        return sol

    def _eval_true_objective(self, sol, ret, covar, coskew, init_x):
        x = np.array(sol)
        quad_term = x.T @ covar @ x
        cubic_term = -x.T @ coskew @ np.kron(x, x)
        # print("quad_term/cubic_term: ", int(abs(quad_term / cubic_term)), quad_term, cubic_term)
        dev_term = np.abs(x - init_x).sum()
        ret_term = -ret.T @ x
        obj = ret_term + self.alpha * quad_term + self.beta * cubic_term + self.lmbd * dev_term
        return obj

    def eval_surrogate(self, z_pred: np.ndarray, **kwargs) -> np.ndarray:
        assert len(z_pred.shape) > 1
        N = z_pred.shape[0]
        assert "aux_data" in kwargs
        aux_data = kwargs["aux_data"]
        true_returns, true_covars, true_coskew, init_x = aux_data[0], aux_data[1], aux_data[2], aux_data[3]
        if len(true_returns.shape) < 2:
            true_returns = true_returns[None]
            true_covars = true_covars[None]
            true_coskew = true_coskew[None]
            init_x = init_x[None]
        assert true_returns.shape[0] == true_covars.shape[0] == true_coskew.shape[0] == init_x.shape[0] == N
        f_hat_list = np.zeros((N, 1))
        for i in range(N):
            if i%50 == 0:
                print("Solving MIP:", i)
            sol = self._solve_single_core(z_pred[i], init_x[i])
            f_hat_i = self._eval_true_objective(sol=sol, ret=true_returns[i],
                                                covar=true_covars[i], coskew=true_coskew[i], init_x=init_x[i])
            f_hat_list[i, 0] = f_hat_i
        return f_hat_list

    def _solve_single_core_true(self, ret, covar, coskew, x0, exact=False):
        self._model.del_component(self._model.cons_dev)
        self._model.del_component(self._model.cons_dev_index)
        self._model.cons_dev = pe.ConstraintList()
        for i in range(self.num_stocks):
            self._model.cons_dev.add(self._vars[i] - x0[i] <= self._yvars[i])
            self._model.cons_dev.add(-self._vars[i] + x0[i] <= self._yvars[i])

        # quadratic term-------------------
        if self._model.find_component('quad_obj') is not None:
            self._model.del_component(self._model.quad_obj)
        self._model.quad_obj = pe.Var(initialize=0.0, domain=pe.Reals)
        S = 0
        for i in range(self.num_stocks):
            S += covar[i, i] * self._vars[i] * self._vars[i]
            for j in range(i+1, self.num_stocks):
                S += 2*covar[i, j] * self._vars[i] * self._vars[j]
        self._model.cons_dev.add(S == self._model.quad_obj)
        # -----------------------------------

        # cubic term-------------------
        if exact:
            if self._model.find_component('cubic_obj') is not None:
                self._model.del_component(self._model.cubic_obj)
            self._model.cubic_obj = pe.Var(initialize=0.0, domain=pe.Reals)
            CC = 0
            for i in range(self.num_stocks):
                for j in range(self.num_stocks):
                    CC += coskew[i, j * self.num_stocks + j] * self._vars[i] * self._vars[j] * self._vars[j]
                    for k in range(j+1, self.num_stocks):
                        CC += 2*coskew[i, j*self.num_stocks + k] * self._vars[i] * self._vars[j] * self._vars[k]
            self._model.cons_dev.add(CC == self._model.cubic_obj)
        # -----------------------------------
        self._model.del_component(self._model.obj)
        obj = sum(-ret[j] * self._vars[k] for j, k in enumerate(self._vars)) + \
              self.lmbd * sum(self._yvars[k] for k in self._yvars) + \
              self.alpha * self._model.quad_obj
        if exact:
              obj -= self.beta * self._model.cubic_obj
        self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)
        self._solverfac.solve(self._model, options={"limits/time": 60*60}, tee=True)
        try:
            sol, objVal = [pe.value(self._vars[k]) for k in self._vars], pe.value(self._model.obj)
        except:
            sol, objVal = ret, 0
        return sol, objVal

    def run_scip_optimal(self, true_returns, true_covars, true_coskew, init_x, exact=False):
        if len(true_returns.shape) < 2:
            true_returns = true_returns[None]
            true_covars = true_covars[None]
            true_coskew = true_coskew[None]
            init_x = init_x[None]
        N = true_returns.shape[0]
        assert true_returns.shape[0] == true_covars.shape[0] == true_coskew.shape[0] == init_x.shape[0] == N
        f_hat_list = np.zeros((N, 1))
        for i in range(N):
            if exact:
                print("\n\nSolving MINLP:", i)
            else:
                print("\n\nSolving MIQP:", i)
            try:
                sol_true, objVal = self._solve_single_core_true(true_returns[i], true_covars[i],
                                                                true_coskew[i], init_x[i], exact=exact)
                f_hat_list[i, 0] = objVal
                print("Solved so far:", f_hat_list.squeeze(), flush=True)
            except:
                print("Solving MINLP failed")
        return f_hat_list

    def get_c_shapes(self):
        return self.num_stocks, self.num_stocks

    def sample_z(self, N, mean, sigma):
        d = len(mean)
        c_sample = (sigma * mean.mean()) * np.random.randn(N-1, d) + mean
        c_sample = np.concatenate((mean[None], c_sample), axis=0)
        return np.clip(c_sample, a_min=0, a_max=1)

    def get_activations(self):
        return "tanh", "tanh"  # activation, output_activation for c_model

    def get_initial_solution(self, aux_data):
        mean, covar = aux_data[0], aux_data[1]
        variance = np.diag(covar)
        return mean - self.init_gamma * variance

    def aug_dataset_with_x_init(self, return_mat, covar_mat, coskew_mat):
        """
        generate random initial solutions and append them to dataset
        """
        assert len(return_mat.shape) > 1
        assert len(covar_mat.shape) > 1
        assert len(coskew_mat.shape) > 1
        N = return_mat.shape[0]
        assert N == covar_mat.shape[0] == coskew_mat.shape[0]
        init_x = np.random.uniform(0, 1, (N, self.num_stocks))
        init_x = init_x / init_x.sum(axis=1)[:, None]
        datasets_list = []
        for i in range(N):
            datasets_list.append((return_mat[i], covar_mat[i], coskew_mat[i], init_x[i]))
        return datasets_list, init_x

    def get_features(self, return_mat, covar_mat, coskew_mat, init_x):
        """
        Returns features for LANCER prior
        """
        assert len(return_mat.shape) > 1
        assert len(covar_mat.shape) > 1
        assert len(coskew_mat.shape) > 1
        assert len(init_x.shape) > 1
        N = return_mat.shape[0]
        assert N == covar_mat.shape[0] == coskew_mat.shape[0] == init_x.shape[0]
        covar_eigv = np.zeros((N, self.num_stocks))
        for i in range(N):
            # compute the main eigenvector of covar_mat[i]
            _, v = np.linalg.eigh(covar_mat[i])
            covar_eigv[i, :] = v[:, -1]  # eig_vec corresponding to the largest eig_val
        # return np.concatenate((return_mat, covar_eigv), axis=1)
        return np.array(return_mat)
