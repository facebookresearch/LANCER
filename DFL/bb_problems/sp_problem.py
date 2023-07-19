# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from DFL.bb_problems import base_problem
from pyomo import environ as pe
from pyomo import opt as po


class ShortestPathProblem(base_problem.BaseProblem):
    def __init__(self, num_feats, grid, solver="glpk"):
        super(ShortestPathProblem, self).__init__()
        self.num_feats = num_feats
        self.m = grid[0]
        self.n = grid[1]
        # dimension of the cost vector
        self.d = (self.m - 1) * self.n + (self.n - 1) * self.m
        self.arcs = self._get_arcs()
        _model, _vars = self.build_model()
        self._model = _model
        self._vars = _vars
        self._solverfac = po.SolverFactory(solver)

    def _get_arcs(self):
        """
        A helper method to get list of arcs for grid network
        """
        arcs = []
        for i in range(self.m):
            # edges on rows
            for j in range(self.n - 1):
                v = i * self.n + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == self.m - 1:
                continue
            for j in range(self.n):
                v = i * self.n + j
                arcs.append((v, v + self.n))
        return arcs

    def build_model(self, **kwargs):
        """
        A method to build pyomo model: Linear Program
        Returns:
            tuple: optimization model and variables
        """
        m = pe.ConcreteModel(name="shortest path")
        x = pe.Var(self.arcs, name="x", within=pe.NonNegativeReals)
        m.x = x
        m.cons = pe.ConstraintList()

        for i in range(self.m):
            for j in range(self.n):
                v = i * self.n + j
                expr = 0
                for e in self.arcs:
                    # flow in
                    if v == e[1]:
                        expr += x[e]
                    # flow out
                    elif v == e[0]:
                        expr -= x[e]
                # source
                if i == 0 and j == 0:
                    m.cons.add(expr == -1)
                # sink
                elif i == self.m - 1 and j == self.m - 1:
                    m.cons.add(expr == 1)
                # transition
                else:
                    m.cons.add(expr == 0)
        m.obj = pe.Objective(sense=pe.minimize, expr=0)
        return m, x

    def eval(self, z_pred: np.ndarray, z_true: np.ndarray, **kwargs) -> np.ndarray:
        assert z_pred.shape == z_true.shape
        N = z_true.shape[0]
        f_hat_list = []
        # TODO: run this loop in parallel
        for i in range(N):
            if i%100 == 0:
                print("Solving LP:", i, " out of ", N)
            self._model.del_component(self._model.obj)
            obj = sum(z_pred[i, j] * self._vars[k] for j, k in enumerate(self._vars))
            self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)
            self._solverfac.solve(self._model)
            sol = [pe.value(self._vars[k]) for k in self._vars]
            ############################
            f_hat_i_cp = np.dot(sol, z_true[i])
            f_hat_list.append([f_hat_i_cp])
        return np.array(f_hat_list)

    def generate_dataset(self, N, deg=1, noise_width=0):
        """
        Generate synthetic dataset for the DFL shortest path problem
        :param N: number of points
        :param deg: degree of polynomial to enforce nonlinearity
        :param noise_width: add eps noise to the cost vector
        :return: dataset of features x and the ground truth cost vector of edges c
        """
        # random matrix parameter B
        B = np.random.binomial(1, 0.5, (self.d, self.num_feats))
        # feature vectors
        x = np.random.normal(0, 1, (N, self.num_feats))
        # cost vectors
        z = np.zeros((N, self.d))
        for i in range(N):
            # cost without noise
            zi = (np.dot(B, x[i].reshape(self.num_feats, 1)).T / np.sqrt(self.num_feats) + 3) ** deg + 1
            # rescale
            zi /= 3.5 ** deg
            # noise
            epislon = np.random.uniform(1 - noise_width, 1 + noise_width, self.d)
            zi *= epislon
            z[i, :] = zi
        return x, z

    def get_c_shapes(self):
        return self.d, self.d
