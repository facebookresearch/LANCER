# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from MINLP.models import models_lancer, models_c
from MINLP.bb_problems import base_problem
from utils import nn_utils


class LancerPriorLearner:
    def __init__(self,
                 params: dict,
                 c_model_type: str,
                 loss_model_type: str,
                 bb_problem: base_problem.BaseProblem):
        # TODO: implement proper Logger
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
        if loss_model_type == "mlp":
            self.lancer_model = models_lancer.MLPLancer(self.lancer_in_dim, self.f_dim, y_dim=0,
                                                        n_layers=params["lancer_n_layers"],
                                                        layer_size=params["lancer_layer_size"],
                                                        learning_rate=params["lancer_lr"],
                                                        opt_type=params["lancer_opt_type"],
                                                        weight_decay=params["lancer_weight_decay"],
                                                        out_activation=bb_problem.lancer_out_activation)
        else:
            raise NotImplementedError
        # initialize surrogate model
        if c_model_type == "mlp":
            self.cmodel = models_c.MLPCModel(self.y_dim, self.c_out_dim,
                                             n_layers=params["c_n_layers"],
                                             layer_size=params["c_layer_size"],
                                             learning_rate=params["c_lr"],
                                             weight_decay=params["c_weight_decay"],
                                             activation=c_hidden_activation,
                                             output_activation=c_output_activation)
        else:
            raise NotImplementedError

    def learn_theta(self, y, max_iter=1, batch_size=64, print_freq=1):
        N = y.shape[0]
        n_batches = int(N / batch_size)
        self.lancer_model.mode(train=False)
        self.cmodel.mode(train=True)
        total_iter = 0
        while total_iter < max_iter:
            rand_indices = np.random.permutation(N)
            for bi in range(n_batches + 1):
                idxs = rand_indices[bi * batch_size: (bi + 1) * batch_size]
                y_batch = y[idxs]
                loss_i = self.cmodel.update(self.lancer_model, y_batch)
                total_iter += 1
                if total_iter % print_freq == 0:
                    print("****** Fitting C model, itr: ", total_iter, ", loss: ", loss_i, flush=True)
                if total_iter >= max_iter:
                    break

    def learn_w(self, z_pred, f_hat, max_iter=1, batch_size=64, print_freq=1, y=None):
        assert z_pred.shape[0] == f_hat.shape[0]
        if y is not None:
            assert y.shape[0] == z_pred.shape[0]
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
                y_batch = None if y is None else y[idxs]
                loss_i = self.lancer_model.update(z_pred_batch, f_hat_batch, y=y_batch)
                total_iter += 1
                if total_iter % print_freq == 0:
                    print("****** Fitting lancer, itr: ", total_iter, ", loss: ", loss_i, flush=True)
                if total_iter >= max_iter:
                    break

    def run_training_loop(self, Y_train, Y_test, aux_tr, aux_te, n_iter, **kwargs):
        print_freq = kwargs["print_freq"] if "print_freq" in kwargs else 1
        c_max_iter = kwargs["c_max_iter"] if "c_max_iter" in kwargs else -1
        c_nbatch = kwargs["c_nbatch"] if "c_nbatch" in kwargs else 128
        c_epochs_init = kwargs["c_epochs_init"] if "c_epochs_init" in kwargs else 30
        c_lr_init = kwargs["c_lr_init"] if "c_lr_init" in kwargs else 0.005
        lancer_max_iter = kwargs["lancer_max_iter"] if "lancer_max_iter" in kwargs else -1
        lancer_nbatch = kwargs["lancer_nbatch"] if "lancer_nbatch" in kwargs else 1000
        init_heuristic = kwargs["init_heuristic"] if "init_heuristic" in kwargs else True
        use_replay_buffer = kwargs["use_replay_buffer"] if "use_replay_buffer" in kwargs else False

        if init_heuristic:  # warm start from heuristic solution
            z_initial = np.zeros((Y_train.shape[0], self.c_out_dim))
            for i in range(Y_train.shape[0]):
                aux_tr_i = [aux_tr[j][i] for j in range(len(aux_tr))]
                z_initial[i,:] = self.bb_problem.get_initial_solution(aux_tr_i)
            self.cmodel.initial_fit(y=Y_train, z_init=z_initial,
                                    learning_rate=c_lr_init, num_epochs=c_epochs_init,
                                    batch_size=c_nbatch, print_freq=print_freq)
        z_pred = self.cmodel.predict(Y_train)
        z_pred_test = self.cmodel.predict(Y_test)

        log_dict = {"dl_tr": [], "loss_tr": [], "dl_te": []}
        for itr in range(n_iter):
            print("\n ---- Running solver for train -----")
            f_hat = self.bb_problem.eval_surrogate(z_pred, aux_data=aux_tr)
            print("\n ---- Running solver for test -----")
            f_hat_test = self.bb_problem.eval_surrogate(z_pred_test, aux_data=aux_te)
            print("\n******* Iter:", itr, "| Train DL =", np.mean(f_hat), " | Test DL =", np.mean(f_hat_test), "\n")

            ### Adding model evaluations to the replay buffer
            if itr == 0 or not use_replay_buffer:
                c_pred_4lancer = np.array(z_pred)
                y_4lancer = np.array(Y_train)
                f_hat_4lancer = np.array(f_hat)
            else:
                c_pred_4lancer = np.vstack((c_pred_4lancer, z_pred))
                y_4lancer = np.vstack((y_4lancer, Y_train))
                f_hat_4lancer = np.vstack((f_hat_4lancer, f_hat))

            # Step over parametric loss function
            self.learn_w(c_pred_4lancer, f_hat_4lancer,
                         max_iter=lancer_max_iter, batch_size=lancer_nbatch, print_freq=print_freq)
            # Step over parametric function
            self.learn_theta(y=Y_train, max_iter=c_max_iter, batch_size=c_nbatch, print_freq=print_freq)

            z_pred = self.cmodel.predict(Y_train)
            z_pred_test = self.cmodel.predict(Y_test)
            f_pred = self.lancer_model.predict(z_pred)
            print("==== F errors: Train =", f_pred.mean())
            log_dict["dl_tr"].append(f_hat.mean())
            log_dict["dl_te"].append(f_hat_test.mean())
            log_dict["loss_tr"].append(f_pred.mean())

        f_hat = self.bb_problem.eval_surrogate(z_pred, aux_data=aux_tr)
        f_hat_test = self.bb_problem.eval_surrogate(z_pred_test, aux_data=aux_te)
        print("\n******* Iter:", n_iter, "| Train DL =", np.mean(f_hat), " | Test DL =", np.mean(f_hat_test))
        log_dict["dl_tr"].append(f_hat.mean())
        log_dict["dl_te"].append(f_hat_test.mean())

        print("\nTraining completed...")
        return log_dict
