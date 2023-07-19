# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import sys
import argparse
import pyepo
import torch
import numpy as np
import b_utils
from utils import nn_utils
from torch import nn
from torch.utils.data import DataLoader
from pyepo_classes import ShortestPathModelOMO, KnapsackModelOMO, BaselineLinearModel, BaselineNNModel
from sklearn.model_selection import train_test_split
from sklearn import linear_model


def run_pyepo_trainer(model, loader_train, loader_test, optmodel, num_epochs, optimizer,
                      baseline_model, baseline_type, criterion=None):
    # train model
    model.train()
    # init log
    loss_log = []
    regret_list_test = [pyepo.metric.regret(model, optmodel, loader_test)]
    regret_list_train = [pyepo.metric.regret(model, optmodel, loader_train)]
    for epoch in range(num_epochs):
        # load data
        for i, data in enumerate(loader_train):
            x, c, w, z = data
            # cuda
            if torch.cuda.is_available():
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # forward pass
            cp = model(x)
            if baseline_type == "spo":
                loss = baseline_model(cp, c, w, z).mean()
            else:
                wp = baseline_model(cp)
                zp = (wp * c).sum(1).view(-1, 1)
                loss = criterion(zp, z)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            loss_log.append(loss.item())

        dl_obj, dl_obj_true, regret = b_utils.decision_loss(model, optmodel, loader_test)
        regret_list_test.append(regret)
        print("\nEpoch {:3}, Loss: {:8.4f}, Regret: {:7.4f}%, DL: {:7.3f}, DL true: {:7.3f}".format(epoch + 1,
                                                                                                    loss.item(),
                                                                                                    regret * 100,
                                                                                                    dl_obj,
                                                                                                    dl_obj_true))
        dl_obj, dl_obj_true, regret = b_utils.decision_loss(model, optmodel, loader_train)
        regret_list_train.append(regret)
        print("\t\tRegret train: {:7.4f}%, DL train: {:7.3f}, DL train true: {:7.3f}".format(regret * 100, dl_obj,
                                                                                             dl_obj_true))
        sys.stdout.flush()

    return regret_list_train, regret_list_test


def initial_fit(model, loader_train, num_epochs, lr, weight_decay):
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        # load data
        for i, data in enumerate(loader_train):
            x, c, _, _ = data
            # cuda
            if torch.cuda.is_available():
                x, c = x.cuda(), c.cuda()
            # forward pass
            cp = model(x)
            loss_step = loss(cp, c)
            # backward pass
            optimizer.zero_grad()
            loss_step.backward()
            optimizer.step()
        print("*** Initial fit epoch: ", epoch, ", loss: ", loss_step.item())


def run_baselines(problem, 
                  baseline_type="spo",
                  expt_iters=1, deg=6, n=1000,
                  batch_size=32, num_epochs=25, lr=0.01, weight_decay=0.01,
                  dbb_lmbd=15, warm_start=False, init_seed=42):
    """
    Args:
        optmodel: underlining optimization model
        baseline_type: DFL method (spo, dbb, two_stage)
        expt_iters: number of repetitions (for confidence intervals)
        deg: polynomial degree for feature generation
        n: dataset size
        batch_size: for dbb and spo+
        num_epochs: for dbb and spo+
        lr: for dbb and spo+
    """

    e = 0.5  # noise half-width
    regret_list_train, regret_list_test = [], []
    for expt_i in range(expt_iters):
        seed = init_seed+expt_i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # generate data for grid network (features and costs)
        if problem == "sp":
            grid, p = (5, 5), 5  # grid size and feat dim for sp problem
            optmodel = ShortestPathModelOMO(grid=grid)  # init model
            feats, costs = pyepo.data.shortestpath.genData(num_data=n+1000, num_features=p, grid=grid,
                                                           deg=deg, noise_width=e, seed=seed)
            x_train, x_test, c_train, c_test = train_test_split(feats, costs, test_size=1000, random_state=seed)
        elif problem == "ks":

            from DFL.bb_problems import knapsack_problem
            cap, num_items, kdim, p = 45, 100, 5, 256
            weights = np.random.uniform(0, 1, (kdim, num_items))
            optmodel = KnapsackModelOMO(weights=weights, capacity=[cap] * kdim)
            # used only for generating dataset -----------
            bb_problem = knapsack_problem.KnapsackProblem(num_feats=p, weights=weights, cap=cap, num_items=num_items,
                                                          kdim=kdim, n_cpus=1)
            if expt_i == 0:
                # construct random neural net only once
                rnd_nn = nn_utils.build_mlp(input_size=num_items, output_size=p, n_layers=1, size=500,
                                            activation="relu", output_activation="tanh")
            Y, Z = bb_problem.generate_dataset_nn(N=n + 1000, rnd_nn=rnd_nn, noise_width=0.1)
            x_train, x_test, c_train, c_test = train_test_split(Y, Z, test_size=1000, random_state=seed)
            # ------------------------
        else:
            raise NotImplementedError("unknown problem (dataset) type")
        print("Features:\n", x_train[0])
        print("Costs:\n", c_train[0])
        if baseline_type != "two_stage":
            # get optDataset
            dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
            dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)
            # set data loader
            loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
            # init model
            if args.n_layers > 0:
                reg = BaselineNNModel(x_train.shape[1], c_train.shape[1], args.n_layers, args.layer_size)
            else:
                reg = BaselineLinearModel(x_train.shape[1], c_train.shape[1])
            # cuda
            if torch.cuda.is_available():
                reg = reg.cuda()

            if warm_start:
                # hardcoded for KS problem only
                initial_fit(model=reg, loader_train=loader_train, num_epochs=100, lr=0.005, weight_decay=0.001)
            # set adam optimizer
            optimizer = torch.optim.Adam(reg.parameters(), lr=lr, weight_decay=weight_decay)
            # init SPO+ loss
            if baseline_type == "spo":
                baseline_model = pyepo.func.SPOPlus(optmodel, processes=1)
                criterion = None
            else:
                baseline_model = pyepo.func.blackboxOpt(optmodel, lambd=dbb_lmbd, processes=1)
                criterion = nn.L1Loss()
            regrets_tr, regrets_te = run_pyepo_trainer(reg, loader_train, loader_test, optmodel,
                                                       num_epochs, optimizer, baseline_model,
                                                       baseline_type, criterion=criterion)
            regret_list_train.append(regrets_tr[-1]), regret_list_test.append(regrets_te[-1])
        else:
            if args.n_layers > 0:
                # get optDataset
                dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
                dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)
                # set data loader
                loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
                loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
                
                reg = BaselineNNModel(x_train.shape[1], c_train.shape[1], args.n_layers, args.layer_size)
                # cuda
                if torch.cuda.is_available():
                    reg = reg.cuda()
                # hardcoded for KS problem only
                initial_fit(model=reg, loader_train=loader_train, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay)
                dl_obj, dl_obj_true, regret = b_utils.decision_loss(reg, optmodel, loader_train)
                regret_list_train.append(regret)
                print("\t\tRegret train: {:7.4f}%, DL train: {:7.3f}, DL train true: {:7.3f}".format(regret * 100, dl_obj,
                                                                                                    dl_obj_true))
                dl_obj, dl_obj_true, regret = b_utils.decision_loss(reg, optmodel, loader_test)
                regret_list_test.append(regret)
                print("\t\tRegret test: {:7.4f}%, DL train: {:7.3f}, DL train true: {:7.3f}".format(regret * 100, dl_obj,
                                                                                                    dl_obj_true))
                sys.stdout.flush()
            else:
                predictor = linear_model.Ridge(alpha=weight_decay)
                twostage = pyepo.twostage.sklearnPred(predictor)
                twostage.fit(x_train, c_train)
                c_pred_train = twostage.predict(x_train)
                c_pred_test = twostage.predict(x_test)
                print("\n\nPerformance of 2-stage")
                dl_obj, dl_obj_true, regret = b_utils.decision_loss_sklearn(optmodel, c_pred_train, c_train)
                regret_list_train.append(regret)
                print("\t\tRegret train: {:7.4f}%, DL train: {:7.3f}, DL train true: {:7.3f}".format(regret * 100, dl_obj,
                                                                                                    dl_obj_true))
                dl_obj, dl_obj_true, regret = b_utils.decision_loss_sklearn(optmodel, c_pred_test, c_test)
                regret_list_test.append(regret)
                print("\t\tRegret test: {:7.4f}%, DL test: {:7.3f}, DL test true: {:7.3f}".format(regret * 100, dl_obj,
                                                                                                dl_obj_true))
    print("\n\n\n======== Train regret mean =", np.mean(regret_list_train), " std =", np.std(regret_list_train))
    print("======== Test regret mean =", np.mean(regret_list_test), " std =", np.std(regret_list_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-pb', type=str, choices=["sp", "ks"], default="ks",
                        help="name of the problem/dataset: sp (shortest path), ks (multidim knapsack),")
    parser.add_argument('--baseline', '-b', type=str, choices=["spo", "dbb", "two_stage"], default="two_stage",
                        help="choice of the baseline")
    parser.add_argument('--ndata', '-n', type=int, default=1000,
                        help="dataset size")
    parser.add_argument('--n_iter', '-ni', type=int, default=1,
                        help="number of times to repeat experiment")
    parser.add_argument('--n_epochs', '-ne', type=int, default=25,
                        help="number of epochs")
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', type=int, default=0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.01)
    parser.add_argument('--nbatch', '-nb', type=int, default=32)
    parser.add_argument('--n_layers', '-nl', type=int, default=0)
    parser.add_argument('--layer_size', '-ls', type=int, default=100)
    parser.add_argument('--dbb_lmbd', type=float, default=15.0)
    parser.add_argument('--warm_start', '-ws', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(args)
    run_baselines(problem=args.problem,
                  baseline_type=args.baseline,
                  deg=6,
                  n=args.ndata,
                  expt_iters=args.n_iter,
                  batch_size=args.nbatch,
                  num_epochs=args.n_epochs,
                  lr=args.learning_rate,
                  weight_decay=args.weight_decay,
                  dbb_lmbd=args.dbb_lmbd,
                  warm_start=args.warm_start,
                  init_seed=args.seed)
