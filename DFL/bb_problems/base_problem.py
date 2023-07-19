# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import abc
import numpy as np


class BaseProblem(object, metaclass=abc.ABCMeta):
    """
    BaseProblem defines and solves underlying mathematical optimization problem
    Use this base class to define your own problem
    """
    def __init__(self):
        super(BaseProblem, self).__init__()
        self.num_feats = 0
        self.lancer_out_activation = "relu"

    def build_model(self, **kwargs):
        raise NotImplementedError

    def eval(self, z_pred: np.ndarray, z_true: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates problem specific decision loss: f(x*; c_true),
        where x* is the solution obtained using z_pred
        :param z_pred: predicted problem descriptors
        :param z_true: ground truth problem descriptors
        :param kwargs: additional problem specific arguments
        :return: f_hat_list = list of decision losses, one per datapoint
        """
        raise NotImplementedError

    def get_c_shapes(self):
        raise NotImplementedError

    def get_activations(self):
        return "tanh", "relu"  # hidden layer(s) and output_activation for c_model
