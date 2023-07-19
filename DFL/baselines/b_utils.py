# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from pyepo import EPO


def decision_loss(predmodel, optmodel, dataloader):
    """
    A function to evaluate model performance with normalized true regret

    Args:
        predmodel (nn): a regression neural network for cost prediction
        optmodel (optModel): an PyEPO optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: true regret loss
    """
    # evaluate
    predmodel.eval()
    dl_obj = []
    dl_obj_true = []
    # load data
    for data in dataloader:
        x, c, w, z = data
        # cuda
        if next(predmodel.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # predict
        cp = predmodel(x).to("cpu").detach().numpy()
        # solve
        for j in range(cp.shape[0]):
            # accumulate loss
            true_cost = c[j].to("cpu").detach().numpy()
            true_obj = z[j].item()
            optmodel.setObj(cp[j])
            sol, _ = optmodel.solve()
            # obj with true cost
            obj = np.dot(sol, true_cost)
            dl_obj.append(obj)
            dl_obj_true.append(true_obj)
    # normalized regret
    if optmodel.modelSense == EPO.MINIMIZE:
        regret = np.subtract(dl_obj, dl_obj_true).sum()
    else:
        regret = np.subtract(dl_obj_true, dl_obj).sum()
    regret = regret / (np.abs(dl_obj_true).sum()+1e-7)
    return np.mean(dl_obj), np.mean(dl_obj_true), regret


def decision_loss_sklearn(optmodel, c_pred, c_true):
    loss = 0
    true_loss = 0
    dl_obj = 0
    dl_obj_true = 0
    N = c_pred.shape[0]
    for i in range(N):
        optmodel.setObj(c_pred[i])
        sol, _ = optmodel.solve()
        optmodel.setObj(c_true[i])
        _, true_obj = optmodel.solve()
        # obj with true cost
        obj = np.dot(sol, c_true[i]) # works only for linear model
        if optmodel.modelSense == EPO.MINIMIZE:
            loss += obj - true_obj
        else:
            loss += true_obj - obj
        true_loss += abs(true_obj)
        dl_obj += obj
        dl_obj_true += true_obj
    return dl_obj / N, dl_obj_true / N, loss / (true_loss + 1e-7)
