# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import sys
import numpy as np
import os
import torch
import random
from pathlib import Path


def compute_moments(prices):
    N = prices.shape[0]
    num_samples = prices.shape[-1]
    num_stocks = prices.shape[-2]

    mean = prices.mean(dim=-1, keepdim=True)
    std = np.sqrt(prices.square().mean(dim=-1, keepdim=True) - mean.square())
    prices_scaled = (prices - mean) / std

    # Compute covariance matrices
    covar = torch.zeros(N, num_stocks, num_stocks)
    for i in range(N):
        covar[i,:,:] = prices_scaled[i,:,:] @ prices_scaled[i,:,:].T
    # normalize and add 0.1*I to make sure covar is p.d.
    covar = covar / (num_samples - 1) + 0.1 * torch.eye(num_stocks)

    # Compute coskewness matrices
    coskew = torch.zeros(N, num_stocks, num_stocks ** 2)
    for i in range(N):
        if i%200 == 0:
            print("Generating co-skewness matrix: ", i, "out of:", N)
        for t in range(num_samples):
            row_i = torch.unsqueeze(prices_scaled[i,:,t], 0)
            coskew[i,:,:] += torch.kron(row_i.T @ row_i, row_i)
        coskew[i,:,:] = coskew[i,:,:] / (num_samples - 1)

    return covar, coskew


def load_portfolio_data(path, num_stocks):
    eps = 1e-12
    print("Loading data...")
    prices_past, prices_now, _, prices_fut, _, _, symbols = torch.load(path)
    total_stocks = len(symbols)
    stocks_subset = random.sample(range(total_stocks), num_stocks)
    prices_past = prices_past[:, stocks_subset]
    prices_now = prices_now[:, stocks_subset].squeeze()
    prices_fut = prices_fut[:, stocks_subset]

    covar_mat, coskew_mat = compute_moments(prices_fut)

    num_features = prices_past.shape[-1]
    prices_past_flat = prices_past.reshape(-1, num_features)
    # feature scaling
    prices_past = torch.div((prices_past - torch.mean(prices_past_flat, dim=0)),
                            (torch.std(prices_past_flat, dim=0) + eps))
    return prices_past.float(), prices_now.float(), covar_mat.float(), coskew_mat.float()


def gen_data_4minlp():
    rand_seed = 42
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    num_stocks = 100
    num_train = 400
    num_all = 800
    data_path = Path(__file__).parent / "data"
    if not os.path.exists(data_path / "price_data_2004-01-01_2017-01-01_daily.pt"):
        print("Please download portfolio data. See installation.md. Aborting...")
        sys.exit()
    X, Y, Y_aux, Y_aux1 = load_portfolio_data(path=data_path / "price_data_2004-01-01_2017-01-01_daily.pt",
                                              num_stocks=num_stocks)
    X_train, Y_train, Y_train_aux, Y_train_aux1 = X[:num_train], Y[:num_train], Y_aux[:num_train], Y_aux1[:num_train]
    X_test, Y_test, Y_test_aux, Y_test_aux1 = X[num_train:num_all], Y[num_train:num_all], Y_aux[num_train:num_all], Y_aux1[num_train:num_all]
    # saving the data
    print(X_train.shape, Y_train.shape, Y_train_aux.shape, Y_train_aux1.shape)
    np.savez_compressed(data_path / "portfolio_minlp.npz",
                        Y_train=X_train, Y_test=X_test.numpy()[:50],
                        C_train=Y_train, C_test=Y_test.numpy()[:50],
                        C_train_aux=Y_train_aux, C_test_aux=Y_test_aux.numpy()[:50],
                        C_train_aux1=Y_train_aux1, C_test_aux1=Y_test_aux1.numpy()[:50])


def gen_data_4dfl():
    rand_seed = 1
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    num_stocks = 50
    num_train = 400
    num_all = 800
    data_path = Path(__file__).parent / "data"
    if not os.path.exists(data_path / "price_data_2004-01-01_2017-01-01_daily.pt"):
        print("Please download portfolio data. See installation.md. Aborting...")
        sys.exit()
    X, Y, Y_aux, Y_aux1 = load_portfolio_data(path=data_path / "price_data_2004-01-01_2017-01-01_daily.pt",
                                              num_stocks=num_stocks)
    X_train, Y_train, Y_train_aux, _ = X[:num_train], Y[:num_train], Y_aux[:num_train], Y_aux1[:num_train]
    X_test, Y_test, Y_test_aux, _ = X[num_train:num_all], Y[num_train:num_all], Y_aux[num_train:num_all], Y_aux1[num_train:num_all]
    # saving the data
    print(X_train.shape, Y_train.shape, Y_train_aux.shape)
    np.savez_compressed(data_path / "portfolio_dfl.npz",
                        Y_train=X_train, Y_test=X_test.numpy(),
                        C_train=Y_train, C_test=Y_test.numpy(),
                        C_train_aux=Y_train_aux, C_test_aux=Y_test_aux.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-pb', type=str, choices=["dfl", "minlp"], default="minlp",
                        help="generate portfolio data for dfl and minlp")
    args = parser.parse_args()
    if args.problem == "dfl":
        print("generating data for dfl")
        gen_data_4dfl()
    else:
        print("generating data for minlp")
        gen_data_4minlp()
