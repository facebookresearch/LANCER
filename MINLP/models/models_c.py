# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import abc
import torch
import numpy as np
from utils import nn_utils
from MINLP.models import models_lancer
from torch import nn
from torch import optim


class BaseCModel(object, metaclass=abc.ABCMeta):
    def predict(self, y=None) -> np.ndarray:
        raise NotImplementedError

    def initial_fit(self, **kwargs):
        raise NotImplementedError

    def update(self, model_loss: models_lancer.BaseLancer, y=None):
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError


class DirectCModel(BaseCModel):
    def __init__(self,
                 z_dim,
                 learning_rate,
                 opt_type="adam",
                 momentum=0.9,
                 output_activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model_output = torch.randn(z_dim, requires_grad=True)
        self.model_output.to(nn_utils.device)
        self.loss = nn.MSELoss()
        self.output_activation = nn_utils._str_to_activation[output_activation]
        self.opt_type = opt_type
        if self.opt_type == "adam":
            self.optimizer = optim.Adam(params=[self.model_output], lr=self.learning_rate)
        else:
            self.optimizer = optim.SGD(params=[self.model_output], lr=self.learning_rate,
                                       momentum=self.momentum)

    # save model parameters
    def save(self, filepath):
        torch.save(self.model_output, filepath)

    def mode(self, train=True):
        pass

    def initial_fit(self, z_init: np.ndarray, **kwargs):
        assert len(z_init) == len(self.model_output)
        z_init_tf = nn_utils.from_numpy(z_init)
        self.model_output = z_init_tf.clone().detach().requires_grad_(True)
        self.model_output.to(nn_utils.device)
        if self.opt_type == "adam":
            self.optimizer = optim.Adam(params=[self.model_output], lr=self.learning_rate)
        else:
            self.optimizer = optim.SGD(params=[self.model_output], lr=self.learning_rate,
                                       momentum=self.momentum)

    def predict(self, y=None) -> np.ndarray:
        with torch.no_grad():
            pred = self.output_activation(self.model_output)
            return nn_utils.to_numpy(pred)

    def update(self, model_loss: models_lancer.BaseLancer, y=None):
        if y is not None:
            assert isinstance(y, np.ndarray)
            assert len(y) == self.z_dim
            y_tensor = nn_utils.from_numpy(y)
        else:
            y_tensor = None
        self.optimizer.zero_grad()
        pred = self.output_activation(self.model_output)
        total_loss = model_loss.forward_theta_step(pred, y_tensor)
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()


class MLPCModel(BaseCModel, nn.Module):
    def __init__(self,
                 y_dim,
                 z_dim,
                 n_layers,
                 layer_size,
                 learning_rate,
                 weight_decay=0.001,
                 activation="tanh",
                 output_activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        #####################################
        self.model_output = nn_utils.build_mlp(input_size=self.y_dim,
                                               output_size=self.z_dim,
                                               n_layers=self.n_layers,
                                               size=self.layer_size,
                                               activation=activation,
                                               output_activation=output_activation)
        self.model_output.to(nn_utils.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.model_output.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)

    # save model parameters
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def initial_fit(self, y: np.ndarray, z_init: np.ndarray,
                    learning_rate=0.005, num_epochs=100, batch_size=64, print_freq=1):
        assert y.shape[0] == z_init.shape[0]
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
                z_init_batch = nn_utils.from_numpy(z_init[idxs])
                z_pred_batch = self.forward(y_batch)
                optimizer.zero_grad()
                loss = self.loss(z_pred_batch, z_init_batch)
                loss.backward()
                optimizer.step()
            if itr % print_freq == 0:
                print("*** Initial fit epoch: ", itr, ", loss: ", loss.item())

    def predict(self, y=None) -> np.ndarray:
        assert y is not None
        assert isinstance(y, np.ndarray)
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

    def update(self, model_loss: models_lancer.BaseLancer, y=None):
        assert y is not None
        assert isinstance(y, np.ndarray)
        y_tensor = nn_utils.from_numpy(y)
        z_pred_tensor = self.forward(y_tensor)
        self.optimizer.zero_grad()
        total_loss = model_loss.forward_theta_step(z_pred_tensor)
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
