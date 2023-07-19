# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import numpy as np
from MINLP.models import models_lancer, models_c
from MINLP.bb_problems import base_problem
from utils import nn_utils


class LancerZeroLearner:
    def __init__(self,
                 params: dict,
                 c_model_type: str,
                 lancer_model_type: str,
                 bb_problem: base_problem.BaseProblem):
        nn_utils.init_gpu(
            use_gpu=not params['no_gpu'],
            gpu_id=params['which_gpu']
        )
        self.bb_problem = bb_problem
        self.y_dim = self.bb_problem.num_feats
        self.c_out_dim, self.lancer_in_dim = self.bb_problem.get_c_shapes()
        self.f_dim = 1
        c_hidden_activation, c_output_activation = self.bb_problem.get_activations()
        # initialize parametric loss model (LANCER)
        if lancer_model_type == "mlp":
            self.lancer_model = models_lancer.MLPLancer(self.lancer_in_dim, self.f_dim, y_dim=0,
                                                        n_layers=params["lancer_n_layers"],
                                                        layer_size=params["lancer_layer_size"],
                                                        learning_rate=params["lancer_lr"],
                                                        opt_type=params["lancer_opt_type"],
                                                        weight_decay=params["lancer_weight_decay"],
                                                        out_activation=bb_problem.lancer_out_activation)
        else:
            raise NotImplementedError
        # initialize target model
        if c_model_type == "direct":
            self.cmodel = models_c.DirectCModel(self.c_out_dim,
                                                learning_rate=params["c_lr"],
                                                opt_type=params["c_opt_type"],
                                                output_activation=c_output_activation)
        else:
            raise NotImplementedError

    def learn_theta(self, max_iter=1, print_freq=1):
        """
        Fitting model C_{\theta}
        """
        self.lancer_model.mode(train=False)
        self.cmodel.mode(train=True)
        total_iter = 0
        while total_iter < max_iter:
            loss_i = self.cmodel.update(self.lancer_model)
            total_iter += 1
            if total_iter % print_freq == 0:
                print("****** Fitting C model, itr: ", total_iter, ", loss: ", loss_i, flush=True)
            if total_iter >= max_iter:
                break

    def learn_w(self, z_pred, f_hat, max_iter=1, batch_size=64, print_freq=1):
        """
        Fitting LANCER model
        """
        assert z_pred.shape[0] == f_hat.shape[0]
        N = z_pred.shape[0]
        n_batches = int(N / batch_size)
        self.lancer_model.mode(train=True)
        total_iter = 0
        while total_iter < max_iter:
            rand_indices = np.random.permutation(N)
            for bi in range(n_batches + 1):
                idxs = rand_indices[bi * batch_size: (bi + 1) * batch_size]
                f_hat_batch = f_hat[idxs]
                z_pred_batch = z_pred[idxs]
                loss_i = self.lancer_model.update(z_pred_batch, f_hat_batch)
                total_iter += 1
                if total_iter % print_freq == 0:
                    print("****** Fitting lancer, itr: ", total_iter, ", loss: ", loss_i, flush=True)
                if total_iter >= max_iter:
                    break

    def augment_aux(self, N, aux_data):
        """
        Augment sampled points with the original aux_data
        :param N: number of points
        :param aux_data: e.g. covar & coskew matrix in portfolio opt
        :return: repeated aux_data for N points
        """
        new_aux_data = []
        for aux in aux_data:
            if len(aux.shape) == 1:
                new_aux = np.tile(aux, (N, 1)).squeeze()
            else:
                new_aux = np.tile(aux, (N, 1, 1)).squeeze()
            new_aux_data.append(new_aux)
        return new_aux_data

    def run_training_loop(self, dataset, n_iter, **kwargs):
        print_freq = kwargs["print_freq"] if "print_freq" in kwargs else 1
        c_max_iter = kwargs["c_max_iter"] if "c_max_iter" in kwargs else -1
        lancer_max_iter = kwargs["lancer_max_iter"] if "lancer_max_iter" in kwargs else -1
        lancer_nbatch = kwargs["lancer_nbatch"] if "lancer_nbatch" in kwargs else 1000
        num_samples = kwargs["num_samples"] if "num_samples" in kwargs else 100
        rnd_sigma = kwargs["rnd_sigma"] if "rnd_sigma" in kwargs else 0.1
        init_heuristic = kwargs["init_heuristic"] if "init_heuristic" in kwargs else True
        use_replay_buffer = kwargs["use_replay_buffer"] if "use_replay_buffer" in kwargs else False

        aux_data = self.augment_aux(num_samples, dataset)
        if init_heuristic:  # warm start from heuristic solution
            self.cmodel.initial_fit(self.bb_problem.get_initial_solution(dataset))
        z_pred_main = self.cmodel.predict()
        z_pred = self.bb_problem.sample_z(num_samples, z_pred_main, rnd_sigma)

        log_dict = {"dl_tr": [], "loss_tr": []}
        for itr in range(n_iter):
            print("\n ---- Running solver -----")
            f_hat = self.bb_problem.eval_surrogate(z_pred, aux_data=aux_data)
            bidx = np.argmin(f_hat.flatten()) if itr > 0 else 0  # in case a sampled point gets better obj
            if bidx > 0:
                self.cmodel.initial_fit(z_pred[bidx])
            print("\n******* Iter:", itr, "| True objective =", f_hat[bidx][0])

            # if true, adding model evaluations to the "replay buffer"
            if itr == 0 or not use_replay_buffer:
                z_pred_4lancer = np.array(z_pred)
                f_hat_4lancer = np.array(f_hat)
            else:
                z_pred_4lancer = np.vstack((z_pred_4lancer, z_pred))
                f_hat_4lancer = np.vstack((f_hat_4lancer, f_hat))

            # Step over w: learning LANCER
            self.learn_w(z_pred_4lancer, f_hat_4lancer,
                         max_iter=lancer_max_iter, batch_size=lancer_nbatch, print_freq=print_freq)
            # Step over theta: learning target model c
            self.learn_theta(max_iter=c_max_iter, print_freq=print_freq)

            z_pred_main = self.cmodel.predict()
            z_pred = self.bb_problem.sample_z(num_samples, z_pred_main, rnd_sigma)

            f_pred = self.lancer_model.predict(z_pred)
            print("==== F errors (sample mean): Train =", f_pred.mean())
            log_dict["dl_tr"].append(f_hat[bidx][0])
            log_dict["loss_tr"].append(f_pred.mean())

        f_hat = self.bb_problem.eval_surrogate(z_pred, aux_data=aux_data)
        log_dict["dl_tr"].append(np.min(f_hat))
        print("\n******* Iter:", n_iter, "| True objective =", np.min(f_hat))
        return log_dict
