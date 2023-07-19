# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from torch import nn
from pyepo.model.omo import optOmoModel
from pyomo import environ as pe
from pyepo import EPO


class ShortestPathModelOMO(optOmoModel):
    def __init__(self, grid):
        self.grid = grid
        self.arcs = self._getArcs()
        super().__init__(solver="scip")

    def _getArcs(self):
        """
        A helper method to get list of arcs for grid network

        Returns:
            list: arcs
        """
        arcs = []
        for i in range(self.grid[0]):
            # edges on rows
            for j in range(self.grid[1] - 1):
                v = i * self.grid[1] + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == self.grid[0] - 1:
                continue
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                arcs.append((v, v + self.grid[1]))
        return arcs

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # sense
        self.modelSense = EPO.MINIMIZE
        # create a model
        m = pe.ConcreteModel(name="shortest path")
        # variables
        x = pe.Var(self.arcs, name="x", within=pe.NonNegativeReals)
        m.x = x
        # flow conservation constraints
        m.cons = pe.ConstraintList()
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
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
                elif i == self.grid[0] - 1 and j == self.grid[0] - 1:
                    m.cons.add(expr == 1)
                # transition
                else:
                    m.cons.add(expr == 0)
        return m, x


class KnapsackModelOMO(optOmoModel):
    """
    This class is optimization model for knapsack problem
    """

    def __init__(self, weights, capacity):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
        """
        self.weights = np.array(weights)
        self.capacity = np.array(capacity)
        # changing capacity for minimization problem
        self.capacity = np.sum(self.weights, axis=1) - self.capacity
        self.items = list(range(self.weights.shape[1]))
        super().__init__(solver="scip")

    def _getModel(self):
        """
        A method to build PyOMO model
        Returns:
            tuple: optimization model and variables
        """
        self.modelSense = EPO.MINIMIZE
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


# build linear model
class BaselineLinearModel(nn.Module):

    def __init__(self, num_feat, n_out):
        super(BaselineLinearModel, self).__init__()
        self.linear = nn.Linear(num_feat, n_out)

    def forward(self, x):
        out = self.linear(x)
        return out


class BaselineNNModel(nn.Module):

    def __init__(self, num_feat, n_out, n_layers, size):
        super(BaselineNNModel, self).__init__()
        self.linear = nn.Linear(num_feat, n_out)
        layers = []
        in_size = num_feat
        for _ in range(n_layers):
            layers.append(nn.Linear(in_size, size))
            layers.append(nn.Tanh())
            in_size = size
        layers.append(nn.Linear(in_size, n_out))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out
