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
    def predict(self, z_pred: np.ndarray, z_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forward_theta_step(self, z_pred_tensor: torch.FloatTensor, z_true_tensor: torch.FloatTensor):
        """
        Used to train a target model in models_c. That is, we minimize the LANCER loss
        :param z_pred_tensor: predicted problem descriptors (in pytorch format)
        :param z_true_tensor: ground truth problem descriptors (in pytorch format)
        :return: LANCER loss
        """
        raise NotImplementedError

    def update(self, z_pred: np.ndarray, z_true: np.ndarray, f_hat: np.ndarray, **kwargs):
        """
        Update parameters of LANCER using (z_pred, z_true) and
        decision loss f_hat (true objective value)
        :param z_pred: predicted problem descriptors
        :param z_true: ground truth problem descriptors
        :param f_hat: decision loss (i.e., true objective value)
        :param kwargs: additional problem specific parameters
        :return: None
        """
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError


class MLPLancer(BaseLancer, nn.Module):
    # multilayer perceptron LANCER Model
    def __init__(self,
                 z_dim,
                 f_dim,
                 n_layers,
                 layer_size,
                 learning_rate,
                 opt_type="adam",  # "adam" or "sgd"
                 momentum=0.9,
                 weight_decay=0.001,
                 out_activation="relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.z_dim = z_dim
        self.f_dim = f_dim
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        #####################################
        self.model_output = nn_utils.build_mlp(input_size=self.z_dim,
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

    def predict(self, z_pred: np.ndarray, z_true: np.ndarray) -> np.ndarray:
        assert z_pred.shape == z_true.shape
        self.mode(train=False)
        with torch.no_grad():
            if len(z_true.shape) > 1:
                z_pred_tensor, z_true_tensor = nn_utils.from_numpy(z_pred), nn_utils.from_numpy(z_true)
            else:
                z_pred_tensor, z_true_tensor = nn_utils.from_numpy(z_pred[None]), nn_utils.from_numpy(z_true[None])
            # return the output of the parametric loss function in numpy format
            return nn_utils.to_numpy(self.forward(z_pred_tensor, z_true_tensor))

    def mode(self, train=True):
        if train:
            self.model_output.train()
        else:
            self.model_output.eval()

    def forward(self, z_pred_tensor: torch.FloatTensor, z_true_tensor: torch.FloatTensor):
        # input = torch.cat((z_pred_tensor, z_true_tensor), dim=1)
        # input = torch.abs(z_true_tensor - z_pred_tensor)
        input = torch.square(z_true_tensor - z_pred_tensor)
        return self.model_output(input)

    def forward_theta_step(self, z_pred_tensor: torch.FloatTensor, z_true_tensor: torch.FloatTensor):
        predicted_loss = self.forward(z_pred_tensor, z_true_tensor)
        return torch.mean(predicted_loss)

    def update(self, z_pred: np.ndarray, z_true: np.ndarray, f_hat: np.ndarray, **kwargs):
        z_pred_tensor = nn_utils.from_numpy(z_pred)
        z_true_tensor = nn_utils.from_numpy(z_true)  # fixed input
        f_hat_tensor = nn_utils.from_numpy(f_hat)  # targets
        predictions = self.forward(z_pred_tensor, z_true_tensor)
        self.optimizer.zero_grad()
        loss = self.loss(predictions, f_hat_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()
