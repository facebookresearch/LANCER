# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
import ray
from DFL.bb_problems import base_problem
from pyomo import environ as pe
from pyomo import opt as po
from utils import nn_utils


class KnapsackProblem(base_problem.BaseProblem):
    def __init__(self, num_feats, weights, cap, num_items, kdim, n_cpus=1, solver="scip"):
        super(KnapsackProblem, self).__init__()
        self.num_feats = num_feats
        self.weights = np.array(weights)
        self.capacity = np.array([cap] * kdim)
        # changing capacity for minimization problem
        self.capacity = np.sum(self.weights, axis=1) - self.capacity
        self.items = list(range(self.weights.shape[1]))

        self.num_items = num_items  # dim of the cost vector
        self.kdim = kdim
        self.n_cpus = n_cpus

        _model, _vars = self.build_model()
        self._model = _model
        self._vars = _vars
        self._solverfac = po.SolverFactory(solver)

    def build_model(self, **kwargs):
        """
        A method to a SCIP model
        Returns:
            tuple: optimization model and variables
        """
        m = pe.ConcreteModel("knapsack")
        m.its = pe.Set(initialize=self.items)
        x = pe.Var(m.its, domain=pe.Binary)
        m.x = x
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(len(self.capacity)):
            m.cons.add(sum(self.weights[i, j] * x[j] for j in self.items) >= self.capacity[i])
        m.obj = pe.Objective(sense=pe.minimize, expr=0)
        return m, x

    def _solve_single_core(self, z):
        self._model.del_component(self._model.obj)
        obj = sum(z[j] * self._vars[k] for j, k in enumerate(self._vars))
        self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)
        self._solverfac.solve(self._model)
        sol = [pe.value(self._vars[k]) for k in self._vars]
        return sol

    def eval(self, z_pred: np.ndarray, z_true: np.ndarray, **kwargs) -> np.ndarray:
        assert z_pred.shape == z_true.shape
        assert self.n_cpus > 0
        n_cpus = os.cpu_count() if self.n_cpus == -1 else self.n_cpus
        N = z_true.shape[0]
        f_hat_list = np.zeros((N, 1))
        if n_cpus > 1:
            ray.init(num_cpus=n_cpus)
        processes = []
        for i in range(N):
            if i % 100 == 0:
                print("Solving MIP:", i)
            if n_cpus == 1:
                sol = self._solve_single_core(z_pred[i])
                f_hat_i_cp = np.dot(sol, z_true[i])
                f_hat_list[i, 0] = f_hat_i_cp
                continue
            elif len(processes) >= n_cpus or i >= N - 1:
                ray_f_hats = ray.get(processes)
                for f_hat_i_cp, idx in ray_f_hats:
                    f_hat_list[idx, 0] = f_hat_i_cp
                del processes[:]
                del ray_f_hats
            processes.append(_solve_omo_ray.remote(self.weights, self.items, self.capacity,
                                                   z_pred[i], z_true[i], i, self._solverfac))
        if n_cpus > 1:
            ray.shutdown()
        return f_hat_list

    def generate_dataset_nn(self, N, rnd_nn, noise_width=0.0):
        # cost vectors
        z = np.random.uniform(0, 5, (N, self.num_items))
        nn_utils.device = None
        c_tensor = nn_utils.from_numpy(z)
        y_tensor = rnd_nn(c_tensor)
        y = nn_utils.to_numpy(y_tensor)
        # noise
        epsilon = np.random.uniform(1 - noise_width, 1 + noise_width, self.num_feats)
        y_noisy = y * epsilon
        return y_noisy, z

    def get_c_shapes(self):
        return self.num_items, self.num_items


@ray.remote
def _solve_omo_ray(weights, items, capacity, c, c_true, idx, solverfac):
    """
    static ray method to solve the problem using pyomo
    """
    omo_model, omo_var = build_model(weights, items, capacity)
    obj = sum(c[j] * omo_var[k] for j, k in enumerate(omo_var))
    omo_model.obj = pe.Objective(sense=pe.minimize, expr=obj)
    solverfac.solve(omo_model)
    sol = [pe.value(omo_var[k]) for k in omo_var]
    f_hat = np.dot(sol, c_true)
    return f_hat, idx


def build_model(weights, items, capacity):
    """
    to use with ray
    """
    m = pe.ConcreteModel("knapsack")
    m.its = pe.Set(initialize=items)
    x = pe.Var(m.its, domain=pe.Binary)
    m.x = x
    m.cons = pe.ConstraintList()
    # constraints
    for i in range(len(capacity)):
        m.cons.add(sum(weights[i,j] * x[j] for j in items) >= capacity[i])
    return m, x