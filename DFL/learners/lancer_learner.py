# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from utils import nn_utils, dfl_utils
from DFL.models import models_lancer, models_c
from DFL.bb_problems import base_problem


class LancerLearner:
    def __init__(self, params: dict, c_model_type: str, lancer_model_type: str, bb_problem: base_problem.BaseProblem):
        nn_utils.init_gpu(
            use_gpu=not params['no_gpu'],
            gpu_id=params['which_gpu']
        )
        self.bb_problem = bb_problem
        self.y_dim = self.bb_problem.num_feats
        self.c_out_dim, self.lancer_in_dim = self.bb_problem.get_c_shapes()
        self.f_dim = 1
        c_hidden_activation, c_output_activation = self.bb_problem.get_activations()
        if lancer_model_type == "mlp":
            self.lancer_model = models_lancer.MLPLancer(self.lancer_in_dim, self.f_dim,
                                                        n_layers=params["lancer_n_layers"],
                                                        layer_size=params["lancer_layer_size"],
                                                        learning_rate=params["lancer_lr"],
                                                        opt_type=params["lancer_opt_type"],
                                                        weight_decay=params["lancer_weight_decay"],
                                                        out_activation=bb_problem.lancer_out_activation)
        else:
            raise NotImplementedError
        if c_model_type == "mlp":
            self.cmodel = models_c.MLPCModel(self.y_dim, self.c_out_dim,
                                             n_layers=params["c_n_layers"],
                                             layer_size=params["c_layer_size"],
                                             learning_rate=params["c_lr"],
                                             opt_type=params["c_opt_type"],
                                             weight_decay=params["c_weight_decay"],
                                             z_regul=params["z_regul"],
                                             activation=c_hidden_activation,
                                             output_activation=c_output_activation)
        else:
            raise NotImplementedError

    def learn_theta(self, y, z_true, max_iter=1, batch_size=64, print_freq=1):
        """
        Fitting target model C_{\theta}
        """
        assert y.shape[0] == z_true.shape[0]
        N = y.shape[0]
        n_batches = int(N / batch_size)
        self.lancer_model.mode(train=False)
        self.cmodel.mode(train=True)
        total_iter = 0
        while total_iter < max_iter:
            rand_indices = np.random.permutation(N)
            for bi in range(n_batches + 1):
                idxs = rand_indices[bi * batch_size: (bi + 1) * batch_size]
                z_true_batch = z_true[idxs]
                y_batch = y[idxs]
                loss_i = self.cmodel.update(y_batch, z_true_batch, self.lancer_model)
                total_iter += 1
                if total_iter % print_freq == 0:
                    print("****** Fitting target model C, itr: ", total_iter, ", lancer loss: ", loss_i, flush=True)
                if total_iter >= max_iter:
                    break

    def learn_w(self, z_pred, z_true, f_hat, max_iter=1, batch_size=64, print_freq=1):
        """
        Fitting LANCER model
        """
        assert z_pred.shape == z_true.shape
        assert z_true.shape[0] == f_hat.shape[0]
        N = z_true.shape[0]
        n_batches = int(N / batch_size)
        self.lancer_model.mode(train=True)
        total_iter = 0
        while total_iter < max_iter:
            rand_indices = np.random.permutation(N)
            for bi in range(n_batches + 1):
                idxs = rand_indices[bi * batch_size: (bi + 1) * batch_size]
                f_hat_batch = f_hat[idxs]
                z_true_batch = z_true[idxs]
                z_pred_batch = z_pred[idxs]
                loss_i = self.lancer_model.update(z_pred_batch, z_true_batch, f_hat_batch)
                total_iter += 1
                if total_iter % print_freq == 0:
                    print("****** Fitting lancer, itr: ", total_iter, ", loss: ", loss_i, flush=True)
                if total_iter >= max_iter:
                    break

    def run_training_loop(self, dataset, n_iter, **kwargs):
        c_max_iter = kwargs["c_max_iter"] if "c_max_iter" in kwargs else 100
        c_nbatch = kwargs["c_nbatch"] if "c_nbatch" in kwargs else 128
        c_epochs_init = kwargs["c_epochs_init"] if "c_epochs_init" in kwargs else 30
        c_lr_init = kwargs["c_lr_init"] if "c_lr_init" in kwargs else 0.005
        lancer_max_iter = kwargs["lancer_max_iter"] if "lancer_max_iter" in kwargs else 100
        lancer_nbatch = kwargs["lancer_nbatch"] if "lancer_nbatch" in kwargs else 1000
        print_freq = kwargs["print_freq"] if "print_freq" in kwargs else 1
        use_replay_buffer = kwargs["use_replay_buffer"] if "use_replay_buffer" in kwargs else False

        Y_train, Y_test, Z_train, Z_test, Z_train_aux, Z_test_aux = dataset
        self.cmodel.initial_fit(Y_train, Z_train, c_lr_init,
                                num_epochs=c_epochs_init,
                                batch_size=c_nbatch,
                                print_freq=print_freq)
        # ------ decision loss of Z train/test: theoretical optimum ----
        print("\n ---- Computing optimal solutions for train")
        Z_true_obj = self.bb_problem.eval(Z_train, Z_train, aux_data=Z_train_aux)
        print("\n ---- Computing optimal solutions for test")
        Z_true_obj_test = self.bb_problem.eval(Z_test, Z_test, aux_data=Z_test_aux)
        # ------------------------------------

        log_dict = {"dl_tr": [], "dl_te": [],
                    "regret_tr": [], "regret_te": [],
                    "lancer_loss_tr": [], "lancer_loss_te": [],
                    "dl_tr_opt": Z_true_obj, "dl_te_opt": Z_true_obj_test}
        for itr in range(n_iter):
            print("\n ---- Running solver for train set -----")
            Z_pred = self.cmodel.predict(Y_train)
            f_hat = self.bb_problem.eval(Z_pred, Z_train, aux_data=Z_train_aux)
            self.logger(log_dict, itr, Z_pred, f_hat, dataset)

            # if true, adding model evaluations to the "replay buffer"
            if itr == 0 or not use_replay_buffer:
                Z_pred_4lancer = np.array(Z_pred)
                Z_true_4lancer = np.array(Z_train)
                f_hat_4lancer = np.array(f_hat)
            else:
                Z_pred_4lancer = np.vstack((Z_pred_4lancer, Z_pred))
                Z_true_4lancer = np.vstack((Z_true_4lancer, Z_train))
                f_hat_4lancer = np.vstack((f_hat_4lancer, f_hat))

            # Step over w: learning LANCER
            self.learn_w(Z_pred_4lancer, Z_true_4lancer, f_hat_4lancer,
                         max_iter=lancer_max_iter, batch_size=lancer_nbatch, print_freq=print_freq)

            # Step over theta: learning target model c
            self.learn_theta(Y_train, Z_train,
                             max_iter=c_max_iter, batch_size=c_nbatch, print_freq=print_freq)

        # saving the final results
        Z_pred = self.cmodel.predict(Y_train)
        f_hat = self.bb_problem.eval(Z_pred, Z_train, aux_data=Z_train_aux)
        self.logger(log_dict, n_iter, Z_pred, f_hat, dataset)
        return log_dict

    def logger(self, log_dict, itr, Z_pred, f_hat, dataset):
        _, Y_test, Z_train, Z_test, _, Z_test_aux = dataset
        Z_pred_test = self.cmodel.predict(Y_test)

        print("\n ---- Running solver for test set -----")
        f_hat_test = self.bb_problem.eval(Z_pred_test, Z_test, aux_data=Z_test_aux)

        lancer_loss = self.lancer_model.predict(Z_pred, Z_train)
        lancer_loss_te = self.lancer_model.predict(Z_pred_test, Z_test)

        regret_train = dfl_utils.norm_regret(f_hat, log_dict["dl_tr_opt"])
        regret_test = dfl_utils.norm_regret(f_hat_test, log_dict["dl_te_opt"])

        print("\n******* Iter:", itr, "| Train: DL =", f_hat.mean(), ", DL opt =", log_dict["dl_tr_opt"].mean(),
              ", Regret =", regret_train * 100, "%")
        print("\t\t\tTest: DL =", f_hat_test.mean(), ", DL opt =", log_dict["dl_te_opt"].mean(),
              ", Regret =", regret_test * 100, "%\n")
        # print("==== F errors: Train =", lancer_loss.mean(), " Test =", lancer_loss_te.mean())
        log_dict["dl_tr"].append(f_hat.mean()), log_dict["dl_te"].append(f_hat_test.mean())
        log_dict["regret_tr"].append(regret_train * 100), log_dict["regret_te"].append(regret_test * 100)
        log_dict["lancer_loss_tr"].append(lancer_loss.mean()), log_dict["lancer_loss_te"].append(lancer_loss_te.mean())
