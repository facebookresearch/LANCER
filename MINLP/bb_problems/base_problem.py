# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import abc
import numpy as np


class BaseProblem(object, metaclass=abc.ABCMeta):
    def __init__(self):
        super(BaseProblem, self).__init__()
        self.sense = "MINIMIZE"
        self.num_feats = 0
        self.lancer_out_activation = "tanh"

    def build_model(self, **kwargs):
        raise NotImplementedError

    def eval_surrogate(self, z_pred: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates surrogate problem loss (MILP): g(x*),
        where x* is the solution obtained using z_pred
        :param z_pred: predicted problem descriptor for MILP
        :param kwargs: additional problem specific arguments
        :return: f_hat_list = list of losses, one per datapoint
        """
        raise NotImplementedError

    def _eval_true_objective(self, **kwargs):
        """
        Compute the original nonlinear objective
        """
        raise NotImplementedError

    def get_c_shapes(self):
        raise NotImplementedError

    def get_activations(self):
        return "tanh", "relu"  # hidden layer activation, output_activation for c_model

    def get_initial_solution(self, aux_data):
        raise NotImplementedError

    def sample_z(self, N, mean, sigma):
        """
        sample z vector N number of times using Gaussian distribution centered at mean and
        with standard deviation sigma
        :return:
        """
        raise NotImplementedError
