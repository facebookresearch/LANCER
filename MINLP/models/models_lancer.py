# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import abc
import torch
import numpy as np
from utils import nn_utils
from torch import nn
from torch import optim


class BaseLancer(object, metaclass=abc.ABCMeta):
    def predict(self, z_pred: np.ndarray, y=None) -> np.ndarray:
        raise NotImplementedError

    def forward_theta_step(self, z_pred_tensor: torch.FloatTensor, y=None):
        raise NotImplementedError

    def update(self, z_pred: np.ndarray, f_hat: np.ndarray, **kwargs):
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError


class MLPLancer(BaseLancer, nn.Module):
    def __init__(self,
                 z_dim,
                 f_dim,
                 y_dim,
                 n_layers,
                 layer_size,
                 learning_rate,
                 opt_type="adam",  # "adam" or "sgd"
                 momentum=0.9,
                 weight_decay=0.001,
                 out_activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.f_dim = f_dim
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        #####################################
        self.model_output = nn_utils.build_mlp(input_size=self.z_dim + self.y_dim,
                                               output_size=self.f_dim,
                                               n_layers=self.n_layers,
                                               size=self.layer_size,
                                               output_activation=out_activation)
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

    def predict(self, z_pred: np.ndarray, y=None) -> np.ndarray:
        self.mode(train=False)
        if y is not None:
            assert isinstance(y, np.ndarray)
            assert len(y) == len(z_pred)
        with torch.no_grad():
            if len(z_pred.shape) > 1:
                z_pred_tensor = nn_utils.from_numpy(z_pred)
                if y is not None:
                    y_tensor = nn_utils.from_numpy(y)
            else:
                z_pred_tensor = nn_utils.from_numpy(z_pred[None])
                if y is not None:
                    y_tensor = nn_utils.from_numpy(y[None])
            # return the output of the parametric loss function in numpy format
            if y is not None:
                return nn_utils.to_numpy(self.forward(z_pred_tensor, y_tensor))
            return nn_utils.to_numpy(self.forward(z_pred_tensor))

    def mode(self, train=True):
        self.train(train)

    def forward(self, z_pred_tensor: torch.FloatTensor, y=None):
        if y is not None:
            assert isinstance(y, torch.Tensor)
            if len(y.shape) > 1:
                input_cat = torch.cat((z_pred_tensor, y), dim=1)
            else:
                input_cat = torch.cat((z_pred_tensor, y))
            return self.model_output(input_cat)
        return self.model_output(z_pred_tensor)

    def forward_theta_step(self, z_pred_tensor: torch.FloatTensor, y=None):
        predicted_loss = self.forward(z_pred_tensor, y)
        return torch.mean(predicted_loss)

    def update(self, z_pred: np.ndarray, f_hat: np.ndarray, **kwargs):
        y = kwargs["y"] if "y" in kwargs else None
        if y is not None:
            assert isinstance(y, np.ndarray)
            assert len(y) == len(z_pred)
            y_tensor = nn_utils.from_numpy(y)
        else:
            y_tensor = None
        z_pred_tensor = nn_utils.from_numpy(z_pred)
        f_hat_tensor = nn_utils.from_numpy(f_hat)  # targets
        predictions = self.forward(z_pred_tensor, y_tensor)
        self.optimizer.zero_grad()
        loss = self.loss(predictions, f_hat_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()
