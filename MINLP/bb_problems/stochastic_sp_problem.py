# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from MINLP.bb_problems import base_problem
import networkx as nx
import numpy as np
from scipy import stats


class StochasticShortestPath(base_problem.BaseProblem):
    def __init__(self, grid_n, init_gamma=0.5):
        super(StochasticShortestPath, self).__init__()
        self.grid_n = grid_n
        self.graph = self.build_model()
        self.init_gamma = init_gamma
        self.source = list(self.graph.nodes)[0]
        self.target = list(self.graph.nodes)[-1]
        self.lancer_out_activation = "tanh"

    def build_model(self, **kwargs):
        graph = nx.grid_2d_graph(self.grid_n, self.grid_n)
        for edge in graph.edges:
            graph.edges[edge]["mean"] = 0
            graph.edges[edge]["variance"] = 0
        return graph

    def solve_shortest_path(self, edge_weights):
        edges = list(self.graph.edges)
        G = nx.to_networkx_graph(self.graph)
        for (i, (u, v)) in enumerate(edges):
            G[u][v]['weight'] = edge_weights[i]
        spath = nx.shortest_path(
            G, self.source, self.target, weight='weight', method="bellman-ford"
        )
        spath_edges = list(zip(spath[:-1], spath[1:]))
        return spath_edges

    def update_edge_data(self, mean, variance):
        assert len(mean) == len(variance) == len(self.graph.edges)
        for (i,edge) in enumerate(self.graph.edges):
            self.graph.edges[edge]["mean"] = mean[i]
            self.graph.edges[edge]["variance"] = variance[i]

    def _eval_true_objective(self, path, threshold):
        mean_sum = sum([self.graph.edges[edge]["mean"] for edge in path])
        variance_sum = sum([self.graph.edges[edge]["variance"] for edge in path])
        return -stats.norm.cdf((threshold - mean_sum) / np.sqrt(variance_sum))

    def eval_surrogate(self, z_pred: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluate performance of the surrogate problem w.r.t. original objective
        """
        assert len(z_pred.shape) > 1
        N = z_pred.shape[0]
        assert "aux_data" in kwargs
        aux_data = kwargs["aux_data"]
        means, variances, thresholds = aux_data[0], aux_data[1], aux_data[2]
        if len(means.shape) < 2:
            means = means[None]
            variances = variances[None]
        assert means.shape[0] == variances.shape[0] == N
        f_hat_list = np.zeros((N, 1))
        for i in range(N):
            if i%25 == 0:
                print("Solving Shortest Path:", i)
            self.update_edge_data(mean=means[i], variance=variances[i])
            spath_edges = self.solve_shortest_path(z_pred[i])
            # compute the original nonlinear objective:
            f_hat_i = self._eval_true_objective(path=spath_edges, threshold=thresholds[i])
            f_hat_list[i, 0] = f_hat_i
        return f_hat_list

    def get_c_shapes(self):
        d = 2 * (self.grid_n - 1) * self.grid_n
        return d, d

    def sample_z(self, N, mean, sigma):
        d = len(mean)
        z_sample = (sigma * mean.mean()) * np.random.randn(N-1, d) + mean
        z_sample = np.concatenate((mean[None], z_sample), axis=0)
        return np.clip(z_sample, a_min=0, a_max=1)

    def get_activations(self):
        return "tanh", "sigmoid"  # activation, output_activation for c_model

    def get_initial_solution(self, aux_data):
        mean, variance = aux_data[0], aux_data[1]
        return mean + self.init_gamma * np.sqrt(variance)

    def _generate_grid_instance(self, seed, mean_ub, variance_ub):
        rng = np.random.default_rng(seed)
        d, _ = self.get_c_shapes()
        mean, variance = [], []
        for i in range(d):
            mean.append(rng.uniform(0.1, mean_ub))
            variance.append(rng.uniform(0.1, variance_ub) * (1 - mean[-1]))
        return np.array(mean), np.array(variance)

    def generate_data(self, num_instances: int, thres_mult=1.0, mean_ub=0.2, variance_ub=0.3, seed=0):
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 1000, size=num_instances)
        datasets_list = []
        for s in seeds:
            mean, variance = self._generate_grid_instance(s, mean_ub, variance_ub)
            ltm_path_edges = self.solve_shortest_path(mean)
            self.update_edge_data(mean, variance)
            ltm_mean = sum([self.graph.edges[edge]["mean"] for edge in ltm_path_edges])
            thres = thres_mult * ltm_mean
            datasets_list.append((mean, variance, thres))
        return datasets_list
