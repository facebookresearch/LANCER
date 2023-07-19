# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np


def norm_regret(obj, true_obj, sense="MINIMIZE"):
    """
    Computes normalized regret for decision-focused learning
    :param obj: objective values obtained from predicted problem descriptors z_pred
    :param true_obj: objective value obtained from true problem descriptors z_true
    :param sense: MINIMIZE or MAXIMIZE
    :return: normalized regret
    """
    if sense == "MINIMIZE":
        regr = np.sum(obj-true_obj)
    else:
        regr = np.sum(true_obj-obj)
    optsum = np.abs(true_obj).sum()
    return regr / (optsum + 1e-7) # avoid division by zero
