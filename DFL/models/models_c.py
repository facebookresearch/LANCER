# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import abc
import torch
import numpy as np
from DFL.models import models_lancer
from utils import nn_utils
from torch import nn
from torch import optim


class BaseCModel(object, metaclass=abc.ABCMeta):
    def predict(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def initial_fit(self, y: np.ndarray, z_true: np.ndarray, **kwargs):
        """Initialize the model by fitting it to the ground truth z_true"""
        raise NotImplementedError

    def update(self, y: np.ndarray, z_true: np.ndarray, model_loss: models_lancer.BaseLancer):
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError


class MLPCModel(BaseCModel, nn.Module):
    # multilayer perceptron CModel
    def __init__(self,
                 y_dim,
                 z_dim,
                 n_layers,
                 layer_size,
                 learning_rate,
                 opt_type="adam", # "adam" or "sgd"
                 momentum=0.9,
                 weight_decay=0.001,
                 z_regul=0.0,
                 activation="tanh",
                 output_activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.z_regul = z_regul
        #####################################
        self.model_output = nn_utils.build_mlp(input_size=self.y_dim,
                                               output_size=self.z_dim,
                                               n_layers=self.n_layers,
                                               size=self.layer_size,
                                               activation=activation,
                                               output_activation=output_activation)
        self.model_output.to(nn_utils.device)
        self.loss = nn.MSELoss()
        if opt_type == "adam":
            self.optimizer = optim.Adam(params=self.model_output.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.SGD(params=self.model_output.parameters(),
                                       lr=self.learning_rate,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def initial_fit(self, y: np.ndarray, z_true: np.ndarray,
                    learning_rate=0.005, num_epochs=100, batch_size=64, print_freq=1):
        assert y.shape[0] == z_true.shape[0]
        N = y.shape[0]
        n_batches = int(N / batch_size)

        optimizer = optim.Adam(params=self.model_output.parameters(),
                               lr=learning_rate,
                               weight_decay=self.weight_decay)
        self.mode(train=True)
        for itr in range(num_epochs):
            rand_indices = np.random.permutation(N)
            for bi in range(n_batches + 1):
                idxs = rand_indices[bi * batch_size: (bi + 1) * batch_size]
                y_batch = nn_utils.from_numpy(y[idxs])
                z_true_batch = nn_utils.from_numpy(z_true[idxs])
                z_pred_batch = self.forward(y_batch)
                optimizer.zero_grad()
                loss = self.loss(z_pred_batch, z_true_batch)
                loss.backward()
                optimizer.step()
            if itr % print_freq == 0:
                print("*** Initial fit epoch: ", itr, ", loss: ", loss.item())

    def predict(self, y: np.ndarray) -> np.ndarray:
        self.mode(train=False)
        with torch.no_grad():
            if len(y.shape) > 1:
                y_tensor = nn_utils.from_numpy(y)
            else:
                y_tensor = nn_utils.from_numpy(y[None])
            z_pred_tensor = self.forward(y_tensor)
            return nn_utils.to_numpy(z_pred_tensor)

    def mode(self, train=True):
        if train:
            self.model_output.train()
        else:
            self.model_output.eval()

    def forward(self, y_tensor: torch.FloatTensor):
        return torch.squeeze(self.model_output(y_tensor))

    def update(self, y: np.ndarray, z_true: np.ndarray, model_loss: models_lancer.BaseLancer):
        y_tensor = nn_utils.from_numpy(y)
        z_true_tensor = nn_utils.from_numpy(z_true)
        z_pred_tensor = self.forward(y_tensor)
        self.optimizer.zero_grad()
        lancer_loss = model_loss.forward_theta_step(z_pred_tensor, z_true_tensor)
        total_loss = lancer_loss + self.z_regul * self.loss(z_pred_tensor, z_true_tensor)
        total_loss.backward()
        self.optimizer.step()
        return lancer_loss.item()
