import os
import time
from pathlib import Path
import random
import pickle
from HMOGPLV.utils import plot_gp, plot_orig
from HMOGPLV.utils import NMSE_test_bar, MNLP, MSE_test_bar_normall
import pandas as pd
import matplotlib.pyplot as plt
import gpflow
import HMOGPLV
import HMOGPLV.models as MODEL
from gpflow.config import default_float
import numpy as np
import tensorflow as tf
from HMOGPLV.utils import run_adam_fulldata, save_plot, LMC_data_set, LMCsum_data,new_format_for_X_Y, LMCsum_data_Missing_part_of_one_output_in_Whole, new_format_for_X_Y_Missing_part_of_one_output_in_Whole
from gpflow.utilities import parameter_dict
from gpflow.ci_utils import ci_niter ## for the number of training
from HMOGPLV.models import SHGP_replicated_within_data, SVGP_MOGP, SVGP_MOGP_sum
from gpflow.inducing_variables import InducingPoints, SeparateIndependentInducingVariables
from HMOGPLV.kernels import lmc_kernel
import GPy


from tensorflow import keras
from tensorflow.keras import layers

class Trainer(object):
    def __init__(self, X_list_missing, X_list_missing_test, Y_list_missing, Y_list_missing_test,X_all_outputs_with_replicates,Y_list,training_id, test_id, seed_index, cfg, num_queries, **kwargs):
        '''
        :param X_list_missing: Training input data set
        :param X_list_missing_test: Test input data set
        :param Y_list_missing: Training output data set
        :param Y_list_missing_test: Test output data set
        :param X_all_outputs_with_replicates:All training input data set, we mainly use this variable in Missing data set for plot
        :param Y_list: All training output data set
        :param training_id: The training index is used in Missing data set for plot
        :param test_id: The test index is used in Missing data set for plot. Sometime it can be used for output index.
        :param seed_index: The index for different repetition in cross-validation
        :param cfg: the configure parameters for all our model
        :param kwargs: potential parameters
        '''
        self.X_list_missing = X_list_missing
        self.X_list_missing_test = X_list_missing_test
        self.Y_list_missing = Y_list_missing
        self.Y_list_missing_test = Y_list_missing_test
        self.X_all_outputs_with_replicates = X_all_outputs_with_replicates
        self.Y_list = Y_list
        self.training_id = training_id
        self.test_id = test_id
        self.seed_index = seed_index
        self.cfg = cfg
        ### Synthetis Data parameter
        self.num_replicates = self.cfg.SYN.NUM_REPLICATES
        self.num_data_each_replicate = self.cfg.SYN.NUM_DATA_IN_REPLICATES
        self.D = self.cfg.SYN.NUM_OUTPUTS
        self.train_percentage = self.cfg.SYN.TRAIN_PERCENTAGE
        ## Model Parameters
        self.gap = self.cfg.MODEL.GAP  ## This is for the inducing variables
        self.Q = self.cfg.MODEL.Q
        ## Optimization Step parameter
        self.Training_step = self.cfg.OPTIMIZATION.TRAINING_NUM_EACH_STEP
        ## Path
        self.my_path = self.cfg.PATH.SAVING_GENERAL
        self.Our_path_loss = self.cfg.PATH.LOSS
        self.Our_path_parameter = self.cfg.PATH.PARAMETERS
        self.Our_path_plot = self.cfg.PATH.PLOT
        self.Our_path_result = cfg.PATH.RESULT

        ## Misc options
        self.Num_repetition = self.cfg.MISC.NUM_REPETITION
        self.Model_name = self.cfg.MISC.MODEL_NAME
        self.Data_name = self.cfg.MISC.DATA_NAME
        self.Data_spec = self.cfg.MISC.DATA_SPEC
        self.Mr = self.cfg.MISC.MR
        self.Experiment_type = self.cfg.MISC.EXPERIMENTTYPE
        self.variance_lower = self.cfg.MISC.VARIANCE_BOUND
        self.Num_ker = self.cfg.MISC.NUM_KERNEL
        self.query_num = num_queries
        random.seed(int(self.seed_index))
        tf.random.set_seed(int(self.seed_index))

    def set_up_model(self):
        random.seed(int(self.seed_index))
        tf.random.set_seed(int(self.seed_index))

        '''
        We set up the model. In each prediciton, we only set up one model
        :return: m_test, data_set, x_input_all_index
        '''
        #print('number_of_outputs', self.D)
        if self.Model_name == 'HMOGPLV':
            ## Our model
            m_test, data_set, x_input_all_index = self.set_up_HMOGPLV()
        elif self.Model_name == 'HGPInd':
            ## Single output Gaussian processes with inducing variables
            m_test, data_set, x_input_all_index = self.set_up_HGPInd()
        elif self.Model_name == 'LMC':
            ## LMC model: consider all replicas in the same output as one output
            m_test, data_set, x_input_all_index = self.set_up_LMC()
        elif self.Model_name == 'LMC2':
            ## LMC model: consider each replica as each output
            m_test, data_set, x_input_all_index = self.set_up_LMC2()
        elif self.Model_name == 'LMC3':
            ## LMC model: consider each replica as each output and LMC3 run on one output
            m_test, data_set, x_input_all_index = self.set_up_LMC2()
        elif self.Model_name == 'LMCsum':
            ## LMC model: consider each replica as each output and LMC3 run on one output
            m_test, data_set, x_input_all_index = self.set_up_LMCsum()
        elif self.Model_name == 'HGP':
            ## Single ouptut Gaussian processes (James' paper)
            m_test, data_set, x_input_all_index = self.set_up_HGP()
        elif self.Model_name == 'DNN':
            m_test, data_set, x_input_all_index = self.set_up_DNN()
        elif self.Model_name == 'SGP':
            ## Single ouptut Gaussian processes
            m_test, data_set, x_input_all_index = self.set_up_SGP()
        elif self.Model_name == 'SGP2':
            ## Single ouptut Gaussian processes
            m_test, data_set, x_input_all_index = self.set_up_SGP2()
        elif self.Model_name == 'DHGP':
            ## Deep hierarchial kernel in Gaussian processes (James' paper)
            m_test, data_set, x_input_all_index = self.set_up_DHGP()
        elif self.Model_name == 'LVMOGP':
            ## LVMOGP model: consider all replicas in the same output as one output (Dai's paper)
            m_test, data_set, x_input_all_index = self.set_up_LVMOGP()
        elif self.Model_name == 'LVMOGP2':
            ## LVMOGP model: consider each replica as each output (Dai's paper)
            m_test, data_set, x_input_all_index = self.set_up_LVMOGP2()
        elif self.Model_name == 'LVMOGP3':
            ## LVMOGP model: consider each replica as each output LVMOGP3 run on one output (Dai's paper)
            m_test, data_set, x_input_all_index = self.set_up_LVMOGP2()
        else:
            print('use correct model')
        return m_test, data_set, x_input_all_index

    def optimization(self, m_test, data_set, x_input_all_index, i=None):
        random.seed(int(self.seed_index))
        tf.random.set_seed(int(self.seed_index))
        '''
        This function is used for optimizing a model
        This function is used for optimizing a model
        :param m_test: the model is used
        :param data_set: the data set for the model
        :param x_input_all_index: input data set with index. We use N = x_input_all_index.shape[0]
        :return:
        '''
        max_run = self.Training_step
        #print(f"self.Model_name", self.Model_name)

        if self.Model_name == 'HMOGPLV' or self.Model_name == 'LMC' or self.Model_name == 'LMCsum' or self.Model_name == 'LMC2' or self.Model_name == 'LMC3':
            ### Adam optimization in tensorflow ###
            N = x_input_all_index.shape[0]
            maxiter = ci_niter(max_run)
            a = time.time()
            logf = run_adam_fulldata(m_test, maxiter, data_set, N, N)
            b = time.time()
            Total_time = b - a
        elif self.Model_name == 'HGPInd':
            ### L-BFGS-B in Scipy for GPflow code ###
            opt = gpflow.optimizers.Scipy()
            logf = []
            def callback(step, variables, values):
                if step % 100 == 0:
                    obj = -m_test.training_loss().numpy()
                    print(step, obj)
                    logf.append(obj)
            a = time.time()
            opt_log = opt.minimize(m_test.training_loss, m_test.trainable_variables, step_callback=callback,
                                   options=dict(maxiter=max_run), compile=True)
            b = time.time()
            Total_time = b - a

        elif self.Model_name == 'HGP' or self.Model_name == 'DHGP':
            if i != None:
                random.seed(int(i))
                tf.random.set_seed(int(i))
            ### L-BFGS-B in Scipy for GPy code ###
            a = time.time()
            m_test.optimize(messages=1, max_iters=self.Training_step, bfgs_factor=1e3) #, max_iters=self.Training_step,ftol=1e13*np.finfo(float).eps,  # Function tolerance gtol=1e-6,
            #print(self.Training_step)
            b = time.time()
            Total_time = b - a
            logf = [0, 0, 0, 0 ,0]

        elif self.Model_name == 'DNN':
            a = time.time()
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
            m_test.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
            m_test.fit(data_set, x_input_all_index, epochs=max_run, verbose=0, batch_size=data_set.shape[0])
            b = time.time()
            Total_time = b - a
            logf = [0, 0, 0, 0 ,0]

        elif self.Model_name == 'SGP' or self.Model_name == 'SGP2':
            ### L-BFGS-B in Scipy for GPy code ###
            a = time.time()
            m_test.optimize_restarts(5, robust=True)
            b = time.time()
            Total_time = b - a
            logf = [0, 0, 0, 0 ,0]

        elif self.Model_name == 'LVMOGP' or self.Model_name == 'LVMOGP2' or self.Model_name == 'LVMOGP3':
            ### L-BFGS-B in Scipy for GPy code ###
            a = time.time()
            m_test.optimize_auto(max_iters=max_run)
            b = time.time()
            Total_time = b - a
            logf = [0, 0, 0, 0 ,0]

        return Total_time, logf, m_test

    def save_elbo(self, logf):
        random.seed(int(self.seed_index))
        tf.random.set_seed(int(self.seed_index))
        '''
        Saving elbo for checking whether converge for those model that is build in tensorflow
        '''
        ## make a floder
        newpath_loss = self.my_path + self.Our_path_loss
        if not os.path.exists(newpath_loss):
            os.makedirs(newpath_loss)
        ## save eblo
        np.savetxt(newpath_loss + '/' + self.Experiment_type + self.Data_name + '-%i Output' % self.D
                   + ' with %i replicates' % self.num_replicates + 'loss' + self.Model_name +
                   '-%ith-Run.txt' % self.seed_index, logf, fmt='%d')
        ## plot elbo
        plt.figure(figsize=(20, 9))
        plt.plot(logf[1:])
        plt.xlabel("Training setp"), plt.ylabel("Elbo")
        plt.title("Elbo")
        plt.savefig(newpath_loss + '/' + self.Experiment_type + self.Data_name + '-%i Output' % self.D
                    + ' with %i replicates' % self.num_replicates + ' ELBO' + self.Model_name
                    + '-%ith-Run.png' % self.seed_index, format='png', bbox_inches='tight')
        plt.close()
    def save_parameter(self, m_test):
        '''
        save the parameter of model that is built in tensorflow
        '''
        ## make a floder
        newpath_parameter = self.my_path + self.Our_path_parameter
        if not os.path.exists(newpath_parameter):
            os.makedirs(newpath_parameter)
        ## save parameters
        params_dict = parameter_dict(m_test)
        for p in params_dict:
            params_dict[p] = params_dict[p].numpy()
        np.savez(newpath_parameter + f"/{self.Data_name}-{self.Experiment_type}-{self.D}Output-"
                                     f"{self.num_replicates}Replicates{self.num_replicates}-output-"
                                     f"{self.gap}gap{self.train_percentage}-trainingpercentage-"
                                     f"{self.Q}-Q-"
                                     f"{self.Model_name}-%ith-Missing-Run.npz" % self.seed_index, **params_dict)


    def plot_index(self, d):
        '''
        Finding indexs for test and training in each output with experiment type: Train_test_in_each_replica
        '''
        idx_test_d = [self.X_list_missing_test[d][:, -1] == i for i in range(self.num_replicates)]
        idx_train_d = [self.X_list_missing[d][:, -1] == i for i in range(self.num_replicates)]
        return idx_test_d, idx_train_d

    def plot_train_test(self, fig, ax, d, idx_train_d, r, idx_test_d, newpath_plot, index):
        '''
        Plot with experiment type: Train_test_in_each_replica
        '''
        if self.Model_name == 'HGP' or self.Model_name == 'HGPInd' or self.Model_name == 'SGP' or self.Model_name == 'LMC3' or self.Model_name == 'LVMOGP3':
            ax.plot(self.X_list_missing[:, :-1][idx_train_d[r]], self.Y_list_missing[idx_train_d[r]], 'ks', mew=5.2,
                    ms=4, label='Train')
            ax.plot(self.X_list_missing_test[:, :-1][idx_test_d[r]], self.Y_list_missing_test[idx_test_d[r]], 'rd',
                    mew=8.2, ms=4, label='Test')
        else:
            ax.plot(self.X_list_missing[d][:, :-1][idx_train_d[r]],
                    self.Y_list_missing[d][idx_train_d[r]], 'ks', mew=5.2, ms=4, label='Train')
            ax.plot(self.X_list_missing_test[d][:, :-1][idx_test_d[r]],
                    self.Y_list_missing_test[d][idx_test_d[r]], 'rd', mew=8.2, ms=4, label='Test')
        plt.title('%i-th output ' % d + '%ith replicates' % r)
        ax.legend()
        if 'Multilingual' in self.Data_spec:
            import re
            src_match = re.search(r'src([a-zA-Z]+)', self.Data_spec)
            tgt_match = re.search(r'tgt([a-zA-Z]+)', self.Data_spec)
            src_lang = src_match.group(1) if src_match else None
            tgt_lang = tgt_match.group(1) if tgt_match else None

            if src_lang != 'x':
                lang = src_lang
                lang_flag = 'tgt'
                to = 'src'

            elif tgt_lang != 'x':
                lang = tgt_lang
                lang_flag = 'src'
                to = 'tgt'

            embd_dict = {0: '175M', 1: '615M', 2: 'big'}
            langs = ['en', 'id', 'jv', 'ms', 'ta', 'tl']
            lang_dict = {}

            count = 0
            for l in langs:
                if l != lang:
                    lang_dict[count] = l
                    count += 1
        elif "Embed2" in self.Data_spec:
            embd_dict = {0: 512, 1: 960, 2: 1600}
            layer_dict = {0: 8, 1: 12, 2: 24, 3: 48}
        else:
            embd_dict = {0: 512, 1: 768, 2: 960, 3: 1024, 4: 1600}
            layer_dict = {0: 8, 1: 10, 2: 12, 3: 24, 4: 32, 5: 48}

        if 'Multilingual' in self.Data_spec:
            new_name = f'{self.Data_name}_{to}_{lang}'
        else:
            new_name = f'{self.Data_name}'

        save_plot(fig, 'pred', newpath_plot, self.Experiment_type, new_name, self.D, index, d, self.Model_name,
                  self.Q, self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition, self.seed_index)

    def save_plot_aistats(self, m_test):
        '''
        This function plot the prediction with
        '''
        ## save path
        newpath_plot = self.my_path + self.Our_path_plot
        if self.Data_name == 'GUSTO':
            x_raw_plot = np.arange(0, 5, 0.2)[:, None]
        elif self.Data_name == 'Gene':
            if 'Multilingual' in self.Data_spec:
                x_raw_plot = np.arange(0, 20, 0.2)[:, None]
            elif 'Bilingual' in self.Data_spec:
                x_raw_plot = np.arange(0, 20, 0.2)[:, None]
            else:
                x_raw_plot = np.arange(0, 11, 0.2)[:, None]

        Num_plot = x_raw_plot.shape[0]
        if self.Model_name == 'HMOGPLV':  ## HMOGPLV
            ## create the x for prediction
            XX_plot = []
            for r in range(self.num_replicates):
                XX_plot.append(np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)])
            X_r = np.vstack(XX_plot)
            ## prediction
            mean, variance = m_test.predict_f(X_r)
            mu, var = mean.numpy(), variance.numpy() + m_test.Heter_GaussianNoise.numpy()
            ## plot
            for d in range(self.D):
                idx_test_d, idx_train_d = self.plot_index(d)
                plt.figure(figsize=(20, 9))
                ax = plt.subplot(1, self.num_replicates, 1)
                for r in range(self.num_replicates):
                    ax = plt.subplot(1, self.num_replicates, r + 1)
                    mu_on = mu[Num_plot * r:Num_plot * (r + 1), d][:, None]
                    var_on = var[Num_plot * r:Num_plot * (r + 1), d][:, None]
                    plot_gp(x_raw_plot, mu_on, var_on)
                    self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

        elif self.Model_name == 'LMCsum':
            x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre = LMCsum_data(self.X_list_missing,
                                                                                               self.Y_list_missing,
                                                                                               self.X_list_missing_test,
                                                                                               self.Y_list_missing_test,
                                                                                               self.num_replicates,
                                                                                               self.D)
            for r in range(self.num_replicates):
                x_test_all = x_test_all_pre[r]
                y_test_all = y_test_all_pre[r]
                x_train_all = x_train_all_pre[r]
                y_train_all = y_train_all_pre[r]
                plt.figure(figsize=(20, 9))
                for d in range(self.D):
                    ax = plt.subplot(1, self.D, d+1)
                    X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
                    mu, var = m_test.predict_f(X_r)
                    var = var + m_test.likelihood.likelihoods[d].variance
                    plot_gp(x_raw_plot, mu, var)

                    index_replica_test = x_test_all[:, -1][:, None] == d
                    index_replica_train = x_train_all[:, -1][:, None] == d

                    ax.plot(x_train_all[:, :-1][index_replica_train.squeeze()],
                                y_train_all[:, :-1][index_replica_train.squeeze()], 'ks', mew=5.2, ms=4, label='Train')
                    ax.plot(x_test_all[:, :-1][index_replica_test.squeeze()],
                                y_test_all[:, :-1][index_replica_test.squeeze()], 'rd', mew=8.2, ms=4, label='Test')
                    plt.title('%i-th output' % d + ' %i-th replicate' % r)
                    ax.legend()
                    save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates, self.D, self.Model_name, self.Q,
                                  self.train_percentage, r, self.gap, self.Num_repetition, self.seed_index)

        # elif self.Model_name == 'LMC':
        #     # We assume all replica in the same output as one output.
        #     idx_train, idx_test, x_train_all, x_test_all, y_train_all, y_test_all = LMC_data_set(self.D, self.Y_list_missing, self.X_list_missing,
        #                                                                                              self.num_replicates, self.Y_list_missing_test,
        #                                                                                              self.X_list_missing_test)
        #     for d in range(self.D):
        #         plt.figure(figsize=(20, 9))
        #         for r in range(self.num_replicates):
        #             ax = plt.subplot(1, self.num_replicates, r + 1)
        #             X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
        #             mu, var = m_test.predict_f(X_r)
        #             var = var + m_test.likelihood.likelihoods[d].variance
        #             plot_gp(x_raw_plot, mu, var)
        #             ax.plot(x_train_all[:, :-1][idx_train[r + d * self.num_replicates]],
        #                         y_train_all[:, :-1][idx_train[r + d * self.num_replicates]], 'ks', mew=5.2, ms=4, label='Train')
        #             ax.plot(x_test_all[:, :-1][idx_test[r + d * self.num_replicates]],
        #                         y_test_all[:, :-1][idx_test[r + d * self.num_replicates]], 'rd', mew=8.2, ms=4, label='Test')
        #             plt.title('%i-th output' % d + ' %i-th replicate' % r)
        #             ax.legend()
        #             save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates, d, self.Model_name, self.Q,
        #                           self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition, self.seed_index)

        elif self.Model_name == 'HGPInd' or self.Model_name == 'HGP' or self.Model_name == 'SGP':
            d = self.test_id
            ## Consider one by one ouptut: we consider all the same replica as the same input
            ## Index for training and test
            idx_train_d = [self.X_list_missing[:, -1] == i for i in range(self.num_replicates)]
            idx_test_d = [self.X_list_missing_test[:, -1] == i for i in range(self.num_replicates)]
            plt.figure(figsize=(20, 9))
            for r in range(self.num_replicates):
                ax = plt.subplot(1, self.num_replicates + 1, r + 2)
                if self.Model_name == 'HGPInd':
                    X_r = np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)]
                    mu, var = m_test.predict_y(X_r)
                    mu, var = mu.numpy(), var.numpy()
                elif self.Model_name == 'HGP':
                    X_r = np.c_[x_raw_plot, (r+1) * np.ones_like(x_raw_plot)]
                    mu, var = m_test.predict(X_r)
                elif self.Model_name == 'SGP':
                    X_r = x_raw_plot
                    mu, var = m_test.predict(X_r)
                plot_gp(x_raw_plot, mu, var)
                self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

        elif self.Model_name == 'LVMOGP':
            # We assume all replica in the same output as one output.
            mu, var = m_test.predict(x_raw_plot)
            for d in range(self.D):
                idx_test_d, idx_train_d = self.plot_index(d)
                plt.figure(figsize=(20, 9))
                ax = plt.subplot(1, self.num_replicates, 1)
                for r in range(self.num_replicates):
                    ax = plt.subplot(1, self.num_replicates, r + 1)
                    mu_on = mu[:, d][:, None]
                    var_on = var[:, d][:, None]
                    plot_gp(x_raw_plot, mu_on, var_on)
                    self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

        elif self.Model_name == 'DHGP':
            for d in range(self.D):
                ## create x for ploting
                XX_plot = []
                for r in range(self.num_replicates):
                    XX_plot.append(np.c_[x_raw_plot, d * np.ones_like(x_raw_plot), r * np.ones_like(x_raw_plot) + d * self.num_replicates + 1])
                X_r = np.vstack(XX_plot)
                mu, var = m_test.predict(X_r)
                idx_test_d, idx_train_d = self.plot_index(d)
                plt.figure(figsize=(20, 9))
                ax = plt.subplot(1, self.num_replicates, 1)
                for r in range(self.num_replicates):
                    ax = plt.subplot(1, self.num_replicates, r + 1)
                    mu_on = mu[Num_plot * r:Num_plot * (r + 1)]
                    var_on = var[Num_plot * r:Num_plot * (r + 1)]
                    plot_gp(x_raw_plot, mu_on, var_on)
                    self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

    def save_plot_synthetic(self, m_test, run_i):
        '''
        This function plot the prediction for synthetic data set. It plot in two main types: Train_test_in_each_replica and others
        '''
        ## save path
        newpath_plot = self.my_path + self.Our_path_plot
        if self.Data_name == 'Synthetic_different_input':
            x_raw_plot = np.arange(0, 10, 0.2)[:, None]
        elif self.Data_name == 'Gene':
            if 'Bilingual' in self.Data_spec:
                x_raw_plot = np.arange(0, 20, 0.2)[:, None]
            elif 'Multilingual' in self.Data_spec:
                x_raw_plot = np.arange(0, 20, 0.2)[:, None]
            else:
                x_raw_plot = np.arange(0, 11, 0.2)[:, None]
        elif self.Data_name == 'MOCAP9':
            x_raw_plot = np.arange(-2, 2, 0.01)[:, None]
        # x_raw_plot = np.arange(0, 10, 0.2)[:, None]

        Num_plot = x_raw_plot.shape[0]
        if self.Experiment_type == 'Train_test_in_each_replica':
            if self.Model_name == 'HMOGPLV': ## HMOGPLV
                ## create the x for prediction
                XX_plot = []
                for r in range(self.num_replicates):
                    XX_plot.append(np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)])
                X_r = np.vstack(XX_plot)
                ## prediction
                mean, variance = m_test.predict_f(X_r)
                mu, var = mean.numpy(), variance.numpy() + m_test.Heter_GaussianNoise.numpy()
                ## plot
                for d in range(self.D):
                    idx_test_d, idx_train_d = self.plot_index(d)
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, self.num_replicates, 1)
                    fig, axes = plt.subplots(1, self.num_replicates, figsize=(20, 9))
                    for r in range(self.num_replicates):
                        ax = axes[r] #plt.subplot(1, self.num_replicates, r + 1)
                        ax.set_ylim([-3, 10])
                        mu_on = mu[Num_plot * r:Num_plot * (r + 1), d][:, None]
                        var_on = var[Num_plot * r:Num_plot * (r + 1), d][:, None]
                        plot_gp(x_raw_plot, mu_on, var_on, ax=ax)
                        self.plot_train_test(fig, ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

            elif self.Model_name == 'HGPInd' or self.Model_name == 'HGP' or self.Model_name == 'SGP':
                d = self.test_id
                ## Consider one by one ouptut: we consider all the same replica as the same input
                ## Index for training and test
                idx_train_d = [self.X_list_missing[:, -1] == i for i in range(self.num_replicates)]
                idx_test_d = [self.X_list_missing_test[:, -1] == i for i in range(self.num_replicates)]
                plt.figure(figsize=(20, 9))
                for r in range(self.num_replicates):

                    ax = plt.subplot(1, self.num_replicates + 1, r + 2)
                    ax.set_ylim([-3, 10])
                    if self.Model_name == 'HGPInd':
                        X_r = np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)]
                        mu, var = m_test.predict_y(X_r)
                        mu, var = mu.numpy(), var.numpy()
                    elif self.Model_name == 'HGP':
                        X_r = np.c_[x_raw_plot, (r+1) * np.ones_like(x_raw_plot)]
                        mu, var = m_test.predict(X_r)
                    elif self.Model_name == 'SGP':
                        X_r = x_raw_plot
                        mu, var = m_test.predict(X_r)
                    plot_gp(x_raw_plot, mu, var)
                    self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)
            elif self.Model_name == 'SGP2':
                plt.figure(figsize=(20, 9))
                ax = plt.subplot(1, 1, 1)
                mu, var = m_test.predict(x_raw_plot)
                plot_gp(x_raw_plot, mu, var)
                ax.plot(self.X_list_missing[:, :-1], self.Y_list_missing, 'ks', mew=5.2, ms=4, label='Train')
                ax.plot(self.X_list_missing_test[:, :-1], self.Y_list_missing_test, 'rd', mew=8.2, ms=4, label='Test')
                plt.title('%i-th output ' % self.test_id + '%ith replicates' % self.seed_index)
                ax.legend()
                save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.test_id, self.test_id, self.Model_name,
                      self.Q, self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition, self.seed_index)

            elif self.Model_name == 'DHGP':
                for d in range(self.D):
                    ## create x for ploting
                    XX_plot = []
                    for r in range(self.num_replicates):
                        XX_plot.append(np.c_[x_raw_plot, d * np.ones_like(x_raw_plot), r * np.ones_like(x_raw_plot) + d * self.num_replicates + 1])
                    X_r = np.vstack(XX_plot)
                    mu, var = m_test.predict(X_r)
                    idx_test_d, idx_train_d = self.plot_index(d)
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, self.num_replicates, 1)
                    for r in range(self.num_replicates):
                        ax = plt.subplot(1, self.num_replicates, r + 1)
                        mu_on = mu[Num_plot * r:Num_plot * (r + 1)]
                        var_on = var[Num_plot * r:Num_plot * (r + 1)]
                        plot_gp(x_raw_plot, mu_on, var_on)
                        self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

            elif self.Model_name == 'LMC':
                # We assume all replica in the same output as one output.
                idx_train, idx_test, x_train_all, x_test_all, y_train_all, y_test_all = LMC_data_set(self.D, self.Y_list_missing, self.X_list_missing,
                                                                                                     self.num_replicates, self.Y_list_missing_test,
                                                                                                     self.X_list_missing_test)
                for d in range(self.D):
                    plt.figure(figsize=(20, 9))
                    for r in range(self.num_replicates):
                        ax = plt.subplot(1, self.num_replicates, r + 1)
                        X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
                        mu, var = m_test.predict_f(X_r)
                        var = var + m_test.likelihood.likelihoods[d].variance
                        plot_gp(x_raw_plot, mu, var)
                        ax.plot(x_train_all[:, :-1][idx_train[r + d * self.num_replicates]],
                                y_train_all[:, :-1][idx_train[r + d * self.num_replicates]], 'ks', mew=5.2, ms=4, label='Train')
                        ax.plot(x_test_all[:, :-1][idx_test[r + d * self.num_replicates]],
                                y_test_all[:, :-1][idx_test[r + d * self.num_replicates]], 'rd', mew=8.2, ms=4, label='Test')
                        plt.title('%i-th output' % d + ' %i-th replicate' % r)
                        ax.legend()
                        save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates, d, self.Model_name, self.Q,
                                  self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition, self.seed_index)
            elif self.Model_name == 'LMCsum':
                x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre = LMCsum_data(self.X_list_missing,
                                                                                               self.Y_list_missing,
                                                                                               self.X_list_missing_test,
                                                                                               self.Y_list_missing_test,
                                                                                               self.num_replicates,
                                                                                               self.D)
                for r in range(self.num_replicates):
                    x_test_all = x_test_all_pre[r]
                    y_test_all = y_test_all_pre[r]
                    x_train_all = x_train_all_pre[r]
                    y_train_all = y_train_all_pre[r]
                    plt.figure(figsize=(20, 9))
                    for d in range(self.D):
                        ax = plt.subplot(1, self.D, d+1)
                        X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
                        mu, var = m_test.predict_f(X_r)
                        var = var + m_test.likelihood.likelihoods[d].variance
                        plot_gp(x_raw_plot, mu, var)

                        index_replica_test = x_test_all[:, -1][:, None] == d
                        index_replica_train = x_train_all[:, -1][:, None] == d

                        ax.plot(x_train_all[:, :-1][index_replica_train.squeeze()],
                                y_train_all[:, :-1][index_replica_train.squeeze()], 'ks', mew=5.2, ms=4, label='Train')
                        ax.plot(x_test_all[:, :-1][index_replica_test.squeeze()],
                                y_test_all[:, :-1][index_replica_test.squeeze()], 'rd', mew=8.2, ms=4, label='Test')
                        plt.title('%i-th output' % d + ' %i-th replicate' % r)
                        ax.legend()
                        save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates, self.D, self.Model_name, self.Q,
                                  self.train_percentage, r, self.gap, self.Num_repetition, self.seed_index)

            elif self.Model_name == 'LMC2':
                # We assume each replica as each output.
                for d in range(self.D):
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, 1, 1)
                    X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
                    idx_test_d, idx_train_d = self.plot_index(d)
                    mu, var = m_test.predict_f(X_r)
                    var = var + m_test.likelihood.likelihoods[d].variance
                    plot_gp(x_raw_plot, mu, var)
                    self.plot_train_test(ax, d, idx_train_d, self.test_id, idx_test_d, newpath_plot, self.test_id)
            elif self.Model_name == 'LMC3':
                d = self.test_id
                # We assume each replica as each output and we only prediction for one output
                idx_test_d = [self.X_list_missing_test[:, -1] == i for i in range(self.num_replicates)]
                idx_train_d = [self.X_list_missing[:, -1] == i for i in range(self.num_replicates)]
                for r in range(self.num_replicates):
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, 1, 1)
                    X_r = np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)]
                    mu, var = m_test.predict_f(X_r)
                    var = var + m_test.likelihood.likelihoods[r].variance
                    plot_gp(x_raw_plot, mu, var)
                    self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)
            elif self.Model_name == 'LVMOGP':
                # We assume all replica in the same output as one output.
                mu, var = m_test.predict(x_raw_plot)
                for d in range(self.D):
                    idx_test_d, idx_train_d = self.plot_index(d)
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, self.num_replicates, 1)
                    for r in range(self.num_replicates):
                        ax = plt.subplot(1, self.num_replicates, r + 1)
                        mu_on = mu[:, d][:, None]
                        var_on = var[:, d][:, None]
                        plot_gp(x_raw_plot, mu_on, var_on)
                        self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)
            elif self.Model_name == 'LVMOGP2':
                # We assume each replica as each output.
                mu, var = m_test.predict(x_raw_plot)
                for d in range(self.D):
                    idx_test_d, idx_train_d = self.plot_index(d)
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, 1, 1)
                    mu_on = mu[:, d][:, None]
                    var_on = var[:, d][:, None]
                    plot_gp(x_raw_plot, mu_on, var_on)
                    self.plot_train_test(ax, d, idx_train_d, self.test_id, idx_test_d, newpath_plot, self.test_id)
            elif self.Model_name == 'LVMOGP3':
                d = self.test_id
                # We assume each replica as each output and we only prediction for one output
                idx_test_d = [self.X_list_missing_test[:, -1] == i for i in range(self.num_replicates)]
                idx_train_d = [self.X_list_missing[:, -1] == i for i in range(self.num_replicates)]
                mu, var = m_test.predict(x_raw_plot)
                for r in range(self.num_replicates):
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, 1, 1)
                    mu_on = mu[:, r][:, None]
                    var_on = var[:, r][:, None]
                    plot_gp(x_raw_plot, mu_on, var_on)
                    self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)
        elif self.Experiment_type == 'Missing_part_of_one_output_in_Whole':
            if self.Model_name == 'HMOGPLV': ## HMOGPLV
                ## create the x for prediction
                XX_plot = []
                for r in range(self.num_replicates):
                    XX_plot.append(np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)])
                X_r = np.vstack(XX_plot)
                ## prediction
                mean, variance = m_test.predict_f(X_r)
                mu, var = mean.numpy(), variance.numpy() + m_test.Heter_GaussianNoise.numpy()
                ## plot

                d = self.D - 1
                idx_test_d = [self.X_list_missing_test[0][:, -1] == i for i in range(self.num_replicates)]
                idx_train_d = [self.X_list_missing[d][:, -1] == i for i in range(self.num_replicates)]
                plt.figure(figsize=(20, 9))
                ax = plt.subplot(1, self.num_replicates, 1)
                fig, axes = plt.subplots(1, self.num_replicates, figsize=(20, 9))
                for r in range(self.num_replicates):
                    ax = axes[r]  #plt.subplot(1, self.num_replicates, r + 1)
                    ax.set_ylim([-3, 10])
                    mu_on = mu[Num_plot * r:Num_plot * (r + 1), d][:, None]
                    var_on = var[Num_plot * r:Num_plot * (r + 1), d][:, None]
                    plot_gp(x_raw_plot, mu_on, var_on, ax=ax)

                    ax.plot(self.X_list_missing[d][:, :-1][idx_train_d[r]],
                            self.Y_list_missing[d][idx_train_d[r]], 'ks', mew=5.2, ms=4, label='Train')
                    ax.plot(self.X_list_missing_test[0][:, :-1][idx_test_d[r]],
                            self.Y_list_missing_test[0][idx_test_d[r]], 'rd', mew=8.2, ms=4, label='Test')
                    plt.title('%i-th output ' % d + '%i-th replicates' % r)
                    ax.legend()

                    if 'Multilingual' in self.Data_spec:
                        import re
                        src_match = re.search(r'src([a-zA-Z]+)', self.Data_spec)
                        tgt_match = re.search(r'tgt([a-zA-Z]+)', self.Data_spec)
                        src_lang = src_match.group(1) if src_match else None
                        tgt_lang = tgt_match.group(1) if tgt_match else None

                        if src_lang != 'x':
                            lang = src_lang
                            lang_flag = 'tgt'
                            to = 'src'

                        elif tgt_lang != 'x':
                            lang = tgt_lang
                            lang_flag = 'src'
                            to = 'tgt'

                        embd_dict = {0: '175M', 1: '615M', 2: 'big'}
                        langs = ['en', 'id', 'jv', 'ms', 'ta', 'tl']
                        lang_dict = {}

                        count = 0
                        for l in langs:
                            if l != lang:
                                lang_dict[count] = l
                                count += 1
                    elif "Embed2" in self.Data_spec:
                        embd_dict = {0: 512, 1: 960, 2: 1600}
                        layer_dict = {0: 8, 1: 12, 2: 24, 3: 48}
                    else:
                        embd_dict = {0: 512, 1: 768, 2: 960, 3: 1024, 4: 1600}
                        layer_dict = {0: 8, 1: 10, 2: 12, 3: 24, 4: 32, 5: 48}

                    if 'Multilingual' in self.Data_spec:
                        new_name = f'{self.Data_name}_{to}_{lang}'
                    else:
                        new_name = f'{self.Data_name}'

                    save_plot(fig, 'pred', newpath_plot, self.Experiment_type, new_name, self.D, self.num_replicates, d, self.Model_name,
                              self.Q, self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition,
                              self.seed_index)

                    # self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)
        elif 'Extrapolate' in self.Experiment_type:

            import re
            experiment_type = self.Experiment_type
            match = re.search(r'\d+', experiment_type)
            if match:
                extr_n = int(match.group())

            if self.Model_name == 'HMOGPLV': ## HMOGPLV
                ## create the x for prediction
                XX_plot = []
                for r in range(self.num_replicates):
                    XX_plot.append(np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)])
                X_r = np.vstack(XX_plot)
                ## prediction
                mean, variance = m_test.predict_f(X_r)
                mu, var = mean.numpy(), variance.numpy() + m_test.Heter_GaussianNoise.numpy()
                ## plot
                for d in range(self.D):
                    idx_train_d = [self.X_list_missing[d][:, -1] == i for i in range(self.num_replicates)]

                    X_all_outputs_orig = self.X_all_outputs_with_replicates[0][:, :-1][0:self.num_data_each_replicate:1]

                    XX_plot_comp = []
                    Num_comp = X_all_outputs_orig.shape[0]
                    for r2 in range(self.num_replicates):
                        XX_plot_comp.append(np.c_[X_all_outputs_orig, r2 * np.ones_like(X_all_outputs_orig)])
                    X_r_comp = np.vstack(XX_plot_comp)
                    mean_comp, variance_comp = m_test.predict_f(X_r_comp)
                    mu_comp, var_comp = mean_comp.numpy(), variance_comp.numpy() + m_test.Heter_GaussianNoise.numpy()

                    with open('z_score_dict.pkl', 'rb') as f:
                        z_score_dict = pickle.load(f)

                    if 'Multilingual' in self.Data_spec:
                        import re
                        src_match = re.search(r'src([a-zA-Z]+)', self.Data_spec)
                        tgt_match = re.search(r'tgt([a-zA-Z]+)', self.Data_spec)
                        src_lang = src_match.group(1) if src_match else None
                        tgt_lang = tgt_match.group(1) if tgt_match else None

                        if src_lang != 'x':
                            lang = src_lang
                            lang_flag = 'tgt'
                            to = 'src'

                        elif tgt_lang != 'x':
                            lang = tgt_lang
                            lang_flag = 'src'
                            to = 'tgt'

                        embd_dict = {0: '175M', 1: '615M', 2: 'big'}
                        langs = ['en', 'id', 'jv', 'ms', 'ta', 'tl']
                        lang_dict = {}

                        count = 0
                        for l in langs:
                            if l != lang:
                                lang_dict[count] = l
                                count += 1

                    elif "Embed2" in self.Data_spec:
                        embd_dict = {0: 512, 1: 960, 2: 1600}
                        layer_dict = {0: 8, 1: 12, 2: 24, 3: 48}
                    else:
                        embd_dict = {0: 512, 1: 768, 2: 960, 3: 1024, 4: 1600}
                        layer_dict = {0: 8, 1: 10, 2: 12, 3: 24, 4: 32, 5: 48}

                    if 'Multilingual' in self.Data_spec:
                        new_name = f'_{to}_{lang}'
                    else:
                        new_name = f'{self.Data_name}'

                    X_all_outputs_orig = np.exp(X_all_outputs_orig).astype(int)
                    idx_test_d = [self.X_list_missing_test[0][:, -1] == i for i in range(self.num_replicates)]

                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, self.num_replicates, 1)
                    fig, axes = plt.subplots(1, self.num_replicates, figsize=(20, 9))
                    fig_orig, axes_orig = plt.subplots(1, self.num_replicates, figsize=(20, 9))
                    multilingual_error_dict = {}

                    if "Multilingual" in self.Data_spec:
                        embd_dict = {0: '175M', 1: '615M', 2: 'big'}
                        lang_dict = {0: 'en', 1: 'id', 2: 'jv', 3: 'ms', 4: 'ta', 5: 'tl'}

                        for ii in np.arange(len(embd_dict)):
                            for jj in np.arange(len(embd_dict)):
                                multilingual_error_dict[f"{ii}-{jj}"] = []

                    if d == self.D - 1:
                        RMSE_vec = []
                        MSE_vec = []
                        MAE_vec = []
                        MNLPD_vec = []

                        for r in range(self.num_replicates):
                         
                            #print(f"Starting replicate {r}")
                            ax = axes[r]  #plt.subplot(1, self.num_replicates, r + 1)
                            ax.set_ylim([-3, 10])
                            ax_orig = axes_orig[r]  # plt.subplot(1, self.num_replicates, r + 1)
                            ax_orig.set_ylim([0, 50])

                            mu_on = mu[Num_plot * r:Num_plot * (r + 1), d][:, None]
                            var_on = var[Num_plot * r:Num_plot * (r + 1), d][:, None]
                            plot_gp(x_raw_plot, mu_on, var_on, ax=ax)

                            mu_on_comp = mu_comp[Num_comp * r:Num_comp * (r + 1), d][:, None]
                            var_on_comp = var_comp[Num_comp * r:Num_comp * (r + 1), d][:, None]

                            if 'Layer' in self.Data_spec:
                                mu_z, std_z = z_score_dict[f'{layer_dict[d]}']
                            else:
                                mu_z, std_z = z_score_dict[f'{embd_dict[d]}']

                            mu_comp_orig = (mu_on_comp * std_z) + mu_z
                            var_comp_orig = var_on_comp * (std_z ** 2)
                            #Plot prediction in orig space

                            plot_orig(X_all_outputs_orig, mu_comp_orig, var_comp_orig, ax=ax_orig)
                            Y_list_orig = (self.Y_list_missing_test[0][idx_test_d[r]] * std_z) + mu_z

                            RMSE = np.round(self.rmse(Y_list_orig, mu_comp_orig[-extr_n:]), 4)
                            MSE = np.round(self.mse(Y_list_orig, mu_comp_orig[-extr_n:]), 4)
                            MAE = np.round(self.mae(Y_list_orig, mu_comp_orig[-extr_n:]), 4)
                            MNLPD = np.round(MNLP(Y_list_orig, mu_comp_orig[-extr_n:], var_comp_orig[-extr_n:]), 4)

                            RMSE_vec.append(RMSE)
                            MSE_vec.append(MSE)
                            MAE_vec.append(MAE)
                            MNLPD_vec.append(MNLPD)

                            multilingual_error_dict[f"{d}-{r}"] = [RMSE, MSE, MAE, MNLPD]

                            ax.plot(self.X_list_missing[d][:, :-1][idx_train_d[r]],
                                    self.Y_list_missing[d][idx_train_d[r]], 'ks', mew=5.2, ms=4, label='Train')
                            ax.plot(self.X_list_missing_test[0][:, :-1][idx_test_d[r]],
                                    self.Y_list_missing_test[0][idx_test_d[r]], 'rd', mew=8.2, ms=4, label='Test')

                            ax_orig.plot(np.exp(self.X_list_missing[d][:, :-1][idx_train_d[r]]),
                                         (self.Y_list_missing[d][idx_train_d[r]] * std_z) + mu_z, 'ks', mew=5.2, ms=4, label='Train')
                            ax_orig.plot(np.exp(self.X_list_missing_test[0][:, :-1][idx_test_d[r]]),
                                         (self.Y_list_missing_test[0][idx_test_d[r]] * std_z) + mu_z,
                                         'rd', mew=8.2, ms=4, label='Test\n'
                                                                 f'RMSE={RMSE}\n'
                                                                 f'MSE={MSE}\n '
                                                                 f'MAE={MAE}\n'
                                                                 f'NLPD={MNLPD}')

                            final_loss = (self.Y_list_missing_test[0][idx_test_d[r]] * std_z) + mu_z
                            final_loss = final_loss[-1][0]

                            ax_orig.text(np.exp(self.X_list_missing_test[0][:, :-1][idx_test_d[r]])[-1],
                                         final_loss, f"{final_loss:.4f}",
                                         fontsize=24, color='red', verticalalignment='bottom',
                                         horizontalalignment='right')
                            ax.legend()
                            ax_orig.legend()

                            if 'Embed' in self.Data_spec:
                                ax.set_title('n_embed= %i ' % embd_dict[d] + 'n_layers= %i' % layer_dict[r])
                                ax.legend()
                                ax_orig.legend()
                                ax_orig.set_title('n_embed= %i ' % embd_dict[d] + 'n_layers= %i' % layer_dict[r])

                            elif 'Layer' in self.Data_spec:
                                ax.set_title('n_layers= %i' % layer_dict[d] + 'n_embed= %i ' % embd_dict[r] )
                                ax.legend()
                                ax_orig.legend()
                                ax_orig.set_title('n_layers= %i' % layer_dict[d] + 'n_embed= %i ' % embd_dict[r])
                            else:
                                ax.set_title(f'Model= {embd_dict[d]}; ' + f'{lang_flag}= {lang_dict[r]}, {to}={lang}')
                                ax.legend()
                                ax_orig.legend()
                                ax_orig.set_title(f'Model= {embd_dict[d]}; ' + f'{lang_flag}= {lang_dict[r]}, {to}={lang}')
                            path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                            save_plot(fig, 'pred', path_, self.Experiment_type, new_name, self.D, self.num_replicates, d, self.Model_name,
                                      self.Q, self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition,
                                      self.seed_index)

                            save_plot(fig_orig, 'orig', path_, self.Experiment_type, new_name, self.D, self.num_replicates, d, self.Model_name,
                                      self.Q, self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition,
                                      self.seed_index)
                        path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                        with open(f"{path_}/error_metrics_{self.Data_spec}_{self.Experiment_type}.txt", "a") as file:

                            file.write(
                                f"{np.round(np.mean(RMSE_vec), 2)}, "
                                f"{np.round(np.mean(MSE_vec), 2)}, "
                                f"{np.round(np.mean(MAE_vec), 2)}, "
                                f"{np.round(np.mean(MNLPD_vec), 2)}\n")
                    else:
                        for r in range(self.num_replicates):
                            #print(f"Starting replicate {r} for d {d}")
                            ax = axes[r]  # plt.subplot(1, self.num_replicates, r + 1)
                            ax.set_ylim([-3, 10])
                            ax_orig = axes_orig[r]  # plt.subplot(1, self.num_replicates, r + 1)
                            ax_orig.set_ylim([0, 50])

                            mu_on = mu[Num_plot * r:Num_plot * (r + 1), d][:, None]
                            var_on = var[Num_plot * r:Num_plot * (r + 1), d][:, None]
                            plot_gp(x_raw_plot, mu_on, var_on, ax=ax)

                            mu_on_comp = mu_comp[Num_comp * r:Num_comp * (r + 1), d][:, None]
                            var_on_comp = var_comp[Num_comp * r:Num_comp * (r + 1), d][:, None]

                            if 'Layer' in self.Data_spec:
                                mu_z, std_z = z_score_dict[f'{layer_dict[d]}']
                            else:
                                mu_z, std_z = z_score_dict[f'{embd_dict[d]}']

                            mu_comp_orig = (mu_on_comp * std_z) + mu_z
                            var_comp_orig = var_on_comp * (std_z ** 2)
                            # Plot prediction in orig space

                            plot_orig(X_all_outputs_orig, mu_comp_orig, var_comp_orig, ax=ax_orig)


                            ax.plot(self.X_list_missing[d][:, :-1][idx_train_d[r]],
                                    self.Y_list_missing[d][idx_train_d[r]], 'ks', mew=5.2, ms=4, label='Train')

                            ax_orig.plot(np.exp(self.X_list_missing[d][:, :-1][idx_train_d[r]]),
                                         (self.Y_list_missing[d][idx_train_d[r]] * std_z) + mu_z, 'ks', mew=5.2, ms=4,
                                         label='Train')

                            Y_list_orig = (self.Y_list_missing[d][idx_train_d[r]] * std_z) + mu_z
                            final_loss = Y_list_orig[-1][0]

                            ax_orig.text(np.exp(self.X_list_missing[d][:, :-1][idx_train_d[r]])[-1][0],
                                         final_loss, f"{final_loss:.4f}",
                                         fontsize=24, color='red', verticalalignment='bottom',
                                         horizontalalignment='right')

                            ax.set_title(f'Model= {embd_dict[d]}; ' + f'{lang_flag}= {lang_dict[r]}, {to}={lang}')
                            ax.legend()
                            ax_orig.legend()
                            ax_orig.set_title(f'Model= {embd_dict[d]}; ' + f'{lang_flag}= {lang_dict[r]}, {to}={lang}')
                            path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                            save_plot(fig, 'pred', path_, self.Experiment_type, new_name, self.D,
                                      self.num_replicates, d, self.Model_name,
                                      self.Q, self.train_percentage, self.num_data_each_replicate, self.gap,
                                      self.Num_repetition,
                                      self.seed_index)

                            save_plot(fig_orig, 'orig', path_, self.Experiment_type, new_name, self.D,
                                      self.num_replicates, d, self.Model_name,
                                      self.Q, self.train_percentage, self.num_data_each_replicate, self.gap,
                                      self.Num_repetition,
                                      self.seed_index)
                    path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                    with open(f'{path_}/multilingual_error_dict_{run_i}.pkl', 'wb') as f:
                        pickle.dump(multilingual_error_dict, f)

            if self.Model_name == 'DHGP':  ## HMOGPLV

                for d in range(self.D):
                    XX_plot = []
                    for r in range(self.num_replicates):
                        XX_plot.append(np.c_[x_raw_plot, d * np.ones_like(x_raw_plot), r * np.ones_like(x_raw_plot) + d * self.num_replicates + 1])
                    X_r = np.vstack(XX_plot)
                    mu, var = m_test.predict(X_r)

                    idx_train_d = [self.X_list_missing[d][:, -1] == i for i in range(self.num_replicates)]

                    X_all_outputs_orig = self.X_all_outputs_with_replicates[0][:, :-1][0:self.num_data_each_replicate:1]

                    XX_plot_comp = []
                    Num_comp = X_all_outputs_orig.shape[0]
                    for r2 in range(self.num_replicates):
                        XX_plot_comp.append(
                            np.c_[X_all_outputs_orig, d * np.ones_like(X_all_outputs_orig), r2 * np.ones_like(
                                X_all_outputs_orig) + d * self.num_replicates + 1])
                    X_r_comp = np.vstack(XX_plot_comp)
                    mu_comp, var_comp = m_test.predict(X_r_comp)

                    with open('z_score_dict.pkl', 'rb') as f:
                        z_score_dict = pickle.load(f)

                    if 'Multilingual' in self.Data_spec:
                        import re
                        src_match = re.search(r'src([a-zA-Z]+)', self.Data_spec)
                        tgt_match = re.search(r'tgt([a-zA-Z]+)', self.Data_spec)
                        src_lang = src_match.group(1) if src_match else None
                        tgt_lang = tgt_match.group(1) if tgt_match else None

                        if src_lang != 'x':
                            lang = src_lang
                            lang_flag = 'tgt'
                            to = 'src'

                        elif tgt_lang != 'x':
                            lang = tgt_lang
                            lang_flag = 'src'
                            to = 'tgt'

                        embd_dict = {0: '175M', 1: '615M', 2: 'big'}
                        langs = ['en', 'id', 'jv', 'ms', 'ta', 'tl']
                        lang_dict = {}

                        count = 0
                        for l in langs:
                            if l != lang:
                                lang_dict[count] = l
                                count += 1

                    elif "Embed2" in self.Data_spec:
                        embd_dict = {0: 512, 1: 960, 2: 1600}
                        layer_dict = {0: 8, 1: 12, 2: 24, 3: 48}
                    else:
                        embd_dict = {0: 512, 1: 768, 2: 960, 3: 1024, 4: 1600}
                        layer_dict = {0: 8, 1: 10, 2: 12, 3: 24, 4: 32, 5: 48}

                    if 'Multilingual' in self.Data_spec:
                        new_name = f'_{to}_{lang}'
                    else:
                        new_name = f'{self.Data_name}'

                    X_all_outputs_orig = np.exp(X_all_outputs_orig).astype(int)

                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, self.num_replicates, 1)
                    fig, axes = plt.subplots(1, self.num_replicates, figsize=(20, 9))
                    fig_orig, axes_orig = plt.subplots(1, self.num_replicates, figsize=(20, 9))
                    multilingual_error_dict = {}

                    if "Multilingual" in self.Data_spec:
                        embd_dict = {0: '175M', 1: '615M', 2: 'big'}
                        lang_dict = {0: 'en', 1: 'id', 2: 'jv', 3: 'ms', 4: 'ta', 5: 'tl'}

                        for ii in np.arange(len(embd_dict)):
                            for jj in np.arange(len(embd_dict)):
                                multilingual_error_dict[f"{ii}-{jj}"] = []

                    if d == self.D - 1:
                        idx_test_d = [self.X_list_missing_test[0][:, -1] == i for i in range(self.num_replicates)]

                        RMSE_vec = []
                        MSE_vec = []
                        MAE_vec = []
                        MNLPD_vec = []

                        for r in range(self.num_replicates):

                            #print(f"Starting replicate {r}")
                            ax = axes[r]  # plt.subplot(1, self.num_replicates, r + 1)
                            ax.set_ylim([-3, 10])
                            ax_orig = axes_orig[r]  # plt.subplot(1, self.num_replicates, r + 1)
                            ax_orig.set_ylim([0, 50])

                            mu_on = mu[Num_plot * r:Num_plot * (r + 1), 0][:, None]
                            var_on = var[Num_plot * r:Num_plot * (r + 1), 0][:, None]
                            plot_gp(x_raw_plot, mu_on, var_on, ax=ax)

                            mu_on_comp = mu_comp[Num_comp * r:Num_comp * (r + 1), 0][:, None]
                            var_on_comp = var_comp[Num_comp * r:Num_comp * (r + 1), 0][:, None]

                            if 'Layer' in self.Data_spec:
                                mu_z, std_z = z_score_dict[f'{layer_dict[d]}']
                            else:
                                mu_z, std_z = z_score_dict[f'{embd_dict[d]}']

                            mu_comp_orig = (mu_on_comp * std_z) + mu_z
                            var_comp_orig = var_on_comp * (std_z ** 2)
                            # Plot prediction in orig space

                            plot_orig(X_all_outputs_orig, mu_comp_orig, var_comp_orig, ax=ax_orig)
                            Y_list_orig = (self.Y_list_missing_test[0][idx_test_d[r]] * std_z) + mu_z

                            RMSE = np.round(self.rmse(Y_list_orig, mu_comp_orig[-extr_n:]), 4)
                            MSE = np.round(self.mse(Y_list_orig, mu_comp_orig[-extr_n:]), 4)
                            MAE = np.round(self.mae(Y_list_orig, mu_comp_orig[-extr_n:]), 4)
                            MNLPD = np.round(MNLP(Y_list_orig, mu_comp_orig[-extr_n:], var_comp_orig[-extr_n:]), 4)

                            RMSE_vec.append(RMSE)
                            MSE_vec.append(MSE)
                            MAE_vec.append(MAE)
                            MNLPD_vec.append(MNLPD)

                            multilingual_error_dict[f"{d}-{r}"] = [RMSE, MSE, MAE, MNLPD]

                            ax.plot(self.X_list_missing[d][:, :-1][idx_train_d[r]],
                                    self.Y_list_missing[d][idx_train_d[r]], 'ks', mew=5.2, ms=4, label='Train')
                            ax.plot(self.X_list_missing_test[0][:, :-1][idx_test_d[r]],
                                    self.Y_list_missing_test[0][idx_test_d[r]], 'rd', mew=8.2, ms=4, label='Test')

                            ax_orig.plot(np.exp(self.X_list_missing[d][:, :-1][idx_train_d[r]]),
                                         (self.Y_list_missing[d][idx_train_d[r]] * std_z) + mu_z, 'ks', mew=5.2, ms=4,
                                         label='Train')
                            ax_orig.plot(np.exp(self.X_list_missing_test[0][:, :-1][idx_test_d[r]]),
                                         (self.Y_list_missing_test[0][idx_test_d[r]] * std_z) + mu_z,
                                         'rd', mew=8.2, ms=4, label='Test\n'
                                                                    f'RMSE={RMSE}\n'
                                                                    f'MSE={MSE}\n '
                                                                    f'MAE={MAE}\n'
                                                                    f'NLPD={MNLPD}')

                            final_loss = (self.Y_list_missing_test[0][idx_test_d[r]] * std_z) + mu_z
                            final_loss = final_loss[-1][0]

                            ax_orig.text(np.exp(self.X_list_missing_test[0][:, :-1][idx_test_d[r]])[-1],
                                         final_loss, f"{final_loss:.4f}",
                                         fontsize=24, color='red', verticalalignment='bottom',
                                         horizontalalignment='right')
                            ax.legend()
                            ax_orig.legend()

                            if 'Embed' in self.Data_spec:
                                ax.set_title('n_embed= %i ' % embd_dict[d] + 'n_layers= %i' % layer_dict[r])
                                ax.legend()
                                ax_orig.legend()
                                ax_orig.set_title('n_embed= %i ' % embd_dict[d] + 'n_layers= %i' % layer_dict[r])
                            if 'Layer' in self.Data_spec:
                                ax.set_title( 'n_layers= %i' % layer_dict[d] + 'n_embed= %i ' % embd_dict[r])
                                ax.legend()
                                ax_orig.legend()
                                ax_orig.set_title( 'n_layers= %i' % layer_dict[d] + 'n_embed= %i ' % embd_dict[r])
                            else:
                                ax.set_title(f'Model= {embd_dict[d]}; ' + f'{lang_flag}= {lang_dict[r]}, {to}={lang}')
                                ax.legend()
                                ax_orig.legend()
                                ax_orig.set_title(
                                    f'Model= {embd_dict[d]}; ' + f'{lang_flag}= {lang_dict[r]}, {to}={lang}')
                            path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                            save_plot(fig, 'pred', path_, self.Experiment_type, new_name, self.D, self.num_replicates,
                                      d, self.Model_name,
                                      self.Q, self.train_percentage, self.num_data_each_replicate, self.gap,
                                      self.Num_repetition,
                                      self.seed_index)

                            save_plot(fig_orig, 'orig', path_, self.Experiment_type, new_name, self.D,
                                      self.num_replicates, d, self.Model_name,
                                      self.Q, self.train_percentage, self.num_data_each_replicate, self.gap,
                                      self.Num_repetition,
                                      self.seed_index)
                        path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                        with open(f"{path_}/error_metrics_{self.Data_spec}_{self.Experiment_type}.txt", "a") as file:
                            file.write(
                                f"{np.round(np.mean(RMSE_vec), 2)}, "
                                f"{np.round(np.mean(MSE_vec), 2)}, "
                                f"{np.round(np.mean(MAE_vec), 2)}, "
                                f"{np.round(np.mean(MNLPD_vec), 2)}\n")
                    else:
                        for r in range(self.num_replicates):
                            #print(f"Starting replicate {r} for d {d}")
                            ax = axes[r]  # plt.subplot(1, self.num_replicates, r + 1)
                            ax.set_ylim([-3, 10])
                            ax_orig = axes_orig[r]  # plt.subplot(1, self.num_replicates, r + 1)
                            ax_orig.set_ylim([0, 50])

                            mu_on = mu[Num_plot * r:Num_plot * (r + 1), 0][:, None]
                            var_on = var[Num_plot * r:Num_plot * (r + 1), 0][:, None]
                            plot_gp(x_raw_plot, mu_on, var_on, ax=ax)

                            mu_on_comp = mu_comp[Num_comp * r:Num_comp * (r + 1), 0][:, None]
                            var_on_comp = var_comp[Num_comp * r:Num_comp * (r + 1), 0][:, None]

                            if 'Layer' in self.Data_spec:
                                mu_z, std_z = z_score_dict[f'{layer_dict[d]}']
                            else:
                                mu_z, std_z = z_score_dict[f'{embd_dict[d]}']

                            mu_comp_orig = (mu_on_comp * std_z) + mu_z
                            var_comp_orig = var_on_comp * (std_z ** 2)
                            # Plot prediction in orig space

                            plot_orig(X_all_outputs_orig, mu_comp_orig, var_comp_orig, ax=ax_orig)

                            ax.plot(self.X_list_missing[d][:, :-1][idx_train_d[r]],
                                    self.Y_list_missing[d][idx_train_d[r]], 'ks', mew=5.2, ms=4, label='Train')

                            ax_orig.plot(np.exp(self.X_list_missing[d][:, :-1][idx_train_d[r]]),
                                         (self.Y_list_missing[d][idx_train_d[r]] * std_z) + mu_z, 'ks', mew=5.2, ms=4,
                                         label='Train')

                            Y_list_orig = (self.Y_list_missing[d][idx_train_d[r]] * std_z) + mu_z
                            final_loss = Y_list_orig[-1][0]

                            ax_orig.text(np.exp(self.X_list_missing[d][:, :-1][idx_train_d[r]])[-1][0],
                                         final_loss, f"{final_loss:.4f}",
                                         fontsize=24, color='red', verticalalignment='bottom',
                                         horizontalalignment='right')

                            ax.set_title(f'Model= {embd_dict[d]}; ' + f'{lang_flag}= {lang_dict[r]}, {to}={lang}')
                            ax.legend()
                            ax_orig.legend()
                            ax_orig.set_title(f'Model= {embd_dict[d]}; ' + f'{lang_flag}= {lang_dict[r]}, {to}={lang}')
                            path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                            save_plot(fig, 'pred', path_, self.Experiment_type, new_name, self.D,
                                      self.num_replicates, d, self.Model_name,
                                      self.Q, self.train_percentage, self.num_data_each_replicate, self.gap,
                                      self.Num_repetition,
                                      self.seed_index)

                            save_plot(fig_orig, 'orig', path_, self.Experiment_type, new_name, self.D,
                                      self.num_replicates, d, self.Model_name,
                                      self.Q, self.train_percentage, self.num_data_each_replicate, self.gap,
                                      self.Num_repetition,
                                      self.seed_index)
                    path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                    with open(f'{path_}/multilingual_error_dict_{run_i}.pkl', 'wb') as f:
                        pickle.dump(multilingual_error_dict, f)
        else:
            #######################################
            ### Plot for the Missing data type ####
            #######################################
            if self.Model_name == 'SGP':
                d = self.test_id
                idx_train_d = [self.X_all_outputs_with_replicates[d][:, -1] == i for i in range(self.num_replicates)]
                plt.figure(figsize=(20, 9))
                ax = plt.subplot(1, self.num_replicates, 1)
                for r in range(self.num_replicates):
                    ax = plt.subplot(1, self.num_replicates, r + 1)
                    # find the mean and variance
                    mu_on, var_on = m_test.predict(x_raw_plot)

                    # plot for different experiment type
                    plot_gp(x_raw_plot, mu_on, var_on)
                    if r == (d + self.seed_index) % self.num_replicates:
                        ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]], self.Y_list[d][idx_train_d[r]], 'rd', mew=8.2, ms=4, label='Missing')
                    else:
                        ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]], self.Y_list[d][idx_train_d[r]], 'ks', mew=5.2, ms=4, label='Train')
                    plt.title('%i-th output ' % d + '%ith replicates' % r)
                    ax.legend()
                    save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates, d,
                              self.Model_name, self.Q, self.train_percentage, self.num_data_each_replicate,
                              self.gap, self.Num_repetition, self.seed_index)
            else:
                if self.Model_name == 'HMOGPLV':
                    XX_plot = []
                    for r in range(self.num_replicates):
                        XX_plot.append(np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)])
                    X_r = np.vstack(XX_plot)
                    mean, variance = m_test.predict_f(X_r)

                    mu, var = mean.numpy(), variance.numpy() + m_test.Heter_GaussianNoise.numpy()
                    # Compute for comparison


                elif self.Model_name == 'LVMOGP' or self.Model_name == 'LVMOGP3':
                    mu, var = m_test.predict(x_raw_plot)

                if self.Model_name == 'LMCsum':

                    x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre = LMCsum_data(self.X_list_missing,
                                                                                                   self.Y_list_missing,
                                                                                                   self.X_list_missing_test,
                                                                                                   self.Y_list_missing_test,
                                                                                                   self.num_replicates,
                                                                                                   self.D)
                    for r in range(self.num_replicates):
                        x_test_all = x_test_all_pre[r]
                        y_test_all = y_test_all_pre[r]
                        x_train_all = x_train_all_pre[r]
                        y_train_all = y_train_all_pre[r]
                        plt.figure(figsize=(50, 9))
                        for d in range(self.D):
                            ax = plt.subplot(1, self.D, d + 1)
                            X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
                            mu, var = m_test.predict_f(X_r)
                            var = var + m_test.likelihood.likelihoods[d].variance
                            plot_gp(x_raw_plot, mu, var)

                            index_replica_test = x_test_all[:, -1][:, None] == d
                            index_replica_train = x_train_all[:, -1][:, None] == d

                            ax.plot(x_train_all[:, :-1][index_replica_train.squeeze()],
                                    y_train_all[:, :-1][index_replica_train.squeeze()], 'ks', mew=5.2, ms=4,
                                    label='Train')
                            ax.plot(x_test_all[:, :-1][index_replica_test.squeeze()],
                                    y_test_all[:, :-1][index_replica_test.squeeze()], 'rd', mew=8.2, ms=4, label='Test')
                            plt.title('%i-th output' % d + ' %i-th replicate' % r)
                            ax.legend()
                            save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates,
                                      self.D, self.Model_name, self.Q,
                                      self.train_percentage, r, self.gap, self.Num_repetition, self.seed_index)
                else:
                    fig_sl, ax_sl = plt.subplots()
                    rep_int_init = [self.num_replicates - 1]

                    RMSE_vec = []
                    MSE_vec = []
                    MAE_vec = []
                    MNLPD_vec = []
                    RMSEp_vec = []
                    MSEp_vec = []
                    MAEp_vec = []
                    MNLPDp_vec = []
                    RMSEl_vec = []
                    MSEl_vec = []
                    MAEl_vec = []
                    MNLPDl_vec = []

                    if 'Embed' in self.Data_spec:
                        if 'active' in self.Experiment_type:
                            embed_path = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}_{self.query_num}\Embed"
                        else:
                            embed_path = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}\Embed"
                        folder_embed = f"{embed_path}/Embed_{run_i}"
                        os.makedirs(folder_embed, exist_ok=True)

                    elif 'Layer' in self.Data_spec:
                        if 'active' in self.Experiment_type:
                            layer_path = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}_{self.query_num}\Layer"
                        else:
                            layer_path = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}\Layer"
                        folder_layer = f"{layer_path}/Layer_{run_i}"
                        os.makedirs(folder_layer, exist_ok=True)

                    bilingual_error_dict = {}
                    if "Bilingual" in self.Data_spec:
                        embd_dict = {0: 'en', 1: 'id', 2: 'jv', 3: 'ms', 4: 'ta', 5: 'tl'}
                        lang_dict = {0: 'en', 1: 'id', 2: 'jv', 3: 'ms', 4: 'ta', 5: 'tl'}

                        for ii in np.arange(len(embd_dict)):
                            for jj in np.arange(len(embd_dict)):
                                bilingual_error_dict[f"{ii}-{jj}"] = []

                    multilingual_error_dict = {}
                    if "Multilingual" in self.Data_spec:
                        embd_dict = {0: '175M', 1: '615M', 2: 'big'}
                        lang_dict = {0: 'en', 1: 'id', 2: 'jv', 3: 'ms', 4: 'ta', 5: 'tl'}

                        for ii in np.arange(len(embd_dict)):
                            for jj in np.arange(len(embd_dict)):
                                multilingual_error_dict[f"{ii}-{jj}"] = []

                    if 'active' in self.Experiment_type:
                        uncertainty_dicts = {}

                    for d in range(self.D):


                        if self.Model_name == 'DHGP':
                            XX_plot = []
                            for r in range(self.num_replicates):
                                XX_plot.append(np.c_[x_raw_plot, d * np.ones_like(x_raw_plot), r * np.ones_like(
                                    x_raw_plot) + d * self.num_replicates + 1])
                            X_r = np.vstack(XX_plot)
                            mu, var = m_test.predict(X_r)



                        if self.Experiment_type == 'Missing_Triangle_1':
                            idx_train_d = []
                            for r in range(self.num_replicates):
                                if d == self.D - 1:
                                    Index_no_missing = self.X_all_outputs_with_replicates[d][:, -1] != (self.num_replicates - 1)
                                else:
                                    Index_no_missing = np.ones_like(self.X_all_outputs_with_replicates[d][:, -1]) == True
                                idx_train_d.append(Index_no_missing)

                        elif 'Missing_Quad' in self.Experiment_type:
                            import re
                            s = self.Experiment_type
                            d_match = re.search(r'd(\d+)', s)
                            r_match = re.search(r'r(\d+)', s)
                            d_number = int(d_match.group(1)) if d_match else 0
                            r_number = int(r_match.group(1)) if r_match else 0
                            rep_int = list(range(self.num_replicates))
                            rep_int = rep_int[-r_number:]
                            d_int = list(range(self.D))
                            d_int = d_int[-d_number:]
                            idx_train_d = []

                            for d_ in range(self.D):

                                Index_no_missing = np.ones_like(self.X_all_outputs_with_replicates[d][:, -1],
                                                                dtype=bool)
                                if d_ in d_int:
                                    Index_no_missing[np.isin(self.X_all_outputs_with_replicates[d][:, -1],
                                                             list(rep_int))] = False
                                else:
                                    Index_no_missing[np.isin(self.X_all_outputs_with_replicates[d][:, -1],
                                                             list(rep_int))] = True
                                idx_train_d.append(Index_no_missing)

                        elif 'active' in self.Experiment_type:

                            if "main" in self.Experiment_type:
                                path = r'0_get_data\AL\Active_main\query.txt'
                            elif "random" in self.Experiment_type:
                                path = r'0_get_data\AL\Active_random\query_random.txt'
                            elif "largest" in self.Experiment_type:
                                path = r'0_get_data\AL\Active_largest\query_largest.txt'
                            elif "smallest" in self.Experiment_type:
                                path = r'0_get_data\AL\Active_smallest\query_smallest.txt'

                            data = pd.read_csv(path, sep=",")

                            d_r_dict = {}
                            for i in range(self.D):
                                d_r_dict[i] = []
                            # append all rs for corresponding ds
                            # ------------------------------------
                            for i in range(len(data)):

                                r_ = int(data.loc[i]['r'])
                                d_ = int(data.loc[i]['d'])

                                d_r_dict[d_].append(r_)
                            for i in range(self.D):
                                d_r_dict[i] = list(set(d_r_dict[i]))

                            idx_train_d = []
                            d_r_dict_missing = {}
                            for d_, r_list in d_r_dict.items():
                                r_list_avail = r_list
                                r_list_missing = [0, 1, 2, 3, 4, 5]
                                for i in r_list_avail:
                                    if i in r_list_missing:
                                        r_list_missing.remove(i)
                                d_r_dict_missing[d_] = r_list_missing
                                rep_int = r_list_missing
                                Index_no_missing = np.ones_like(self.X_all_outputs_with_replicates[d][:, -1],
                                                                dtype=bool)

                                Index_no_missing[np.isin(self.X_all_outputs_with_replicates[d][:, -1],
                                                         list(rep_int))] = False

                                idx_train_d.append(Index_no_missing)

                        elif 'Missing_Tri_' in self.Experiment_type:
                            import re
                            s = self.Experiment_type
                            d_match = re.search(r'd(\d+)', s)
                            d_number = int(d_match.group(1)) if d_match else 0
                            # r_number = d_number
                            # rep_int = list(range(0, self.num_replicates))
                            # rep_int = rep_int[-r_number:]
                            d_int = list(range(0, self.D))
                            d_int = d_int[-d_number:]
                            rep_int = [self.num_replicates-1]
                            idx_train_d = []

                            for d_ in range(self.D):
                                #print(d_, d_int, self.D)

                                Index_no_missing = np.ones_like(self.X_all_outputs_with_replicates[d][:, -1],
                                                                dtype=bool)
                                if d_ in d_int:
                                    Index_no_missing[np.isin(self.X_all_outputs_with_replicates[d][:, -1],
                                                             list(rep_int))] = False
                                    if len(rep_int) < self.num_replicates:
                                        rep_int.append(rep_int[-1] - 1)
                                else:
                                    Index_no_missing[np.isin(self.X_all_outputs_with_replicates[d][:, -1],
                                                             list(rep_int))] = True
                                idx_train_d.append(Index_no_missing)

                        elif 'Bilingual' in self.Data_spec:
                            with open('dict_num.pkl', 'rb') as f:
                                dict_num = pickle.load(f)
                            idx_train_d = [self.X_all_outputs_with_replicates[d][:, -1] != i for i in range(self.num_replicates)]
                        else:
                            idx_train_d = [self.X_all_outputs_with_replicates[d][:, -1] == i for i in range(self.num_replicates)]

                        plt.figure(figsize=(20, 9))

                        fig, axes = plt.subplots(1, self.num_replicates, figsize=(20, 9))
                        fig_orig, axes_orig = plt.subplots(1, self.num_replicates, figsize=(20, 9))

                        for r in range(self.num_replicates):
                            # ax = plt.subplots(1, self.num_replicates, r + 1)
                            ax = axes[r]
                            ax_orig = axes_orig[r]
                            if 'Multilingual' in self.Data_spec:
                                ax.set_ylim([-5, 5])
                                ax_orig.set_ylim([0, 50])
                            if 'Bilingual' in self.Data_spec:
                                ax.set_ylim([-5, 5])
                                ax_orig.set_ylim([0, 100])
                            else:
                                ax.set_ylim([-3, 10])
                                ax_orig.set_ylim([0, 15])


                            XX_plot_comp = []

                            if self.Experiment_type == 'Missing_Triangle_1':
                                X_all_outputs_orig = self.X_all_outputs_with_replicates[d][:, :-1][0:int(len(idx_train_d[0])/self.num_replicates):1].squeeze(-1)

                            elif 'Missing_Quad' in self.Experiment_type or 'active' in self.Experiment_type or 'Missing_Tri_' in self.Experiment_type:
                                X_all_outputs_orig = self.X_all_outputs_with_replicates[d][:, :-1][0:int(len(idx_train_d[0])/self.num_replicates):1].squeeze(-1)

                            elif 'Bilingual' in self.Data_spec:
                                idx_size = [self.X_all_outputs_with_replicates[d][:, -1] != i for i in
                                               range(self.num_replicates)]
                                X_all_outputs_orig = self.X_all_outputs_with_replicates[d][:, :-1][~idx_size[r]].squeeze(-1)

                            else:
                                X_all_outputs_orig = self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]].squeeze(-1)

                            if 'Bilingual' in self.Data_spec:
                                if self.Model_name == 'HMOGPLV':
                                    Num_comp = []
                                    for r2 in range(self.num_replicates):
                                            X_orig = self.X_all_outputs_with_replicates[d][:, :-1][~idx_size[r2]].squeeze(-1)
                                            Num_comp.append(X_orig.shape[0])
                                            #X_orig = np.tile(X_orig, self.D)
                                            XX_plot_comp.append(np.c_[X_orig, r2 * np.ones_like(X_orig)])
                                    X_r_comp = np.vstack(XX_plot_comp)

                                elif self.Model_name == 'DHGP':

                                    Num_comp = []
                                    for r2 in range(self.num_replicates):
                                        X_orig = self.X_all_outputs_with_replicates[d][:, :-1][
                                            ~idx_size[r2]].squeeze(-1)
                                        Num_comp.append(X_orig.shape[0])
                                        # X_orig = np.tile(X_orig, self.D)
                                        XX_plot_comp.append(
                                            np.c_[np.c_[X_orig, d * np.ones_like(X_orig), r2 * np.ones_like(
                                                X_orig) + d * self.num_replicates + 1]])
                                    X_r_comp = np.vstack(XX_plot_comp)

                            else:
                                if self.Model_name == 'HMOGPLV':
                                    Num_comp = X_all_outputs_orig.shape[0]
                                    for r2 in range(self.num_replicates):
                                        XX_plot_comp.append(np.c_[X_all_outputs_orig, r2 * np.ones_like(X_all_outputs_orig)])
                                    X_r_comp = np.vstack(XX_plot_comp)

                                elif self.Model_name == 'DHGP':
                                    Num_comp = X_all_outputs_orig.shape[0]
                                    for r2 in range(self.num_replicates):
                                        XX_plot_comp.append(np.c_[X_all_outputs_orig, d * np.ones_like(
                                            X_all_outputs_orig), r2 * np.ones_like(
                                            X_all_outputs_orig) + d * self.num_replicates + 1])
                                    X_r_comp = np.vstack(XX_plot_comp)

                            if self.Model_name == 'HMOGPLV':
                                mean_comp, variance_comp = m_test.predict_f(X_r_comp)
                                mu_comp, var_comp = mean_comp.numpy(), variance_comp.numpy() + m_test.Heter_GaussianNoise.numpy()

                            elif self.Model_name == 'DHGP':
                                mean_comp, variance_comp = m_test.predict(X_r_comp)
                                mu_comp, var_comp = mean_comp, variance_comp

                            # find the mean and variance
                            if self.Model_name == 'HMOGPLV':
                                mu_on = mu[Num_plot * r:Num_plot * (r + 1), d][:, None]
                                var_on = var[Num_plot * r:Num_plot * (r + 1), d][:, None]

                                if 'Bilingual' in self.Data_spec:
                                    start_point = 0
                                    i = 0

                                    for i in np.arange(0, r+1):
                                        if i == 0:
                                            pass
                                        else:
                                            start_point += Num_comp[i-1]
                                    end_point = start_point + Num_comp[r]

                                    mu_on_comp = mu_comp[start_point:end_point, d][:, None]
                                    var_on_comp = var_comp[start_point:end_point, d][:, None]
                                else:
                                    mu_on_comp = mu_comp[Num_comp * r:Num_comp * (r + 1), d][:, None]
                                    var_on_comp = var_comp[Num_comp * r:Num_comp * (r + 1), d][:, None]

                            if self.Model_name == 'DHGP':
                                #print(d, r)
                                mu_on = mu[Num_plot * r:Num_plot * (r + 1), 0][:, None]
                                var_on = var[Num_plot * r:Num_plot * (r + 1), 0][:, None]

                                if 'Bilingual' in self.Data_spec:
                                    start_point = 0
                                    i = 0

                                    for i in np.arange(0, r + 1):
                                        if i == 0:
                                            pass
                                        else:
                                            start_point += Num_comp[i - 1]
                                    end_point = start_point + Num_comp[r]

                                    mu_on_comp = mu_comp[start_point:end_point, 0][:, None]
                                    var_on_comp = var_comp[start_point:end_point, 0][:, None]

                                else:
                                    mu_on_comp = mu_comp[Num_comp * r:Num_comp * (r + 1), 0][:, None]
                                    var_on_comp = var_comp[Num_comp * r:Num_comp * (r + 1), 0][:, None]

                            elif self.Model_name == 'HGPInd':
                                X_r = np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)]
                                mu, var = m_test.predict_y(X_r)
                                mu_on, var_on = mu.numpy(), var.numpy()
                            elif self.Model_name == 'LMC':
                                X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
                                mu_on, var_on = m_test.predict_f(X_r)
                                var_on = var_on + m_test.likelihood.likelihoods[d].variance
                            elif self.Model_name == 'LMC3':
                                X_r = np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)]
                                mu_on, var_on = m_test.predict_f(X_r)
                                var_on = var_on + m_test.likelihood.likelihoods[r].variance
                            elif self.Model_name == 'LVMOGP':
                                mu_on = mu[:, d][:, None]
                                var_on = var[:, d][:, None]
                            elif self.Model_name == 'LVMOGP3':
                                mu_on = mu[:, r][:, None]
                                var_on = var[:, r][:, None]
                            elif self.Model_name == 'HGP':
                                X_r = np.c_[x_raw_plot, (r + 1) * np.ones_like(x_raw_plot)]
                                mu_on, var_on = m_test.predict(X_r)

                            # plot for different experiment types

                            plot_gp(x_raw_plot, mu_on, var_on, ax=ax)

                            if 'Embed' in self.Data_spec:
                                with open(r'0_get_data\dict_of_C.pkl', 'rb') as f:
                                    dict_of_C = pickle.load(f)

                            if 'Layer' in self.Data_spec:
                                with open(r'0_get_data\dict_of_C.pkl', 'rb') as f:
                                    dict_of_C = pickle.load(f)

                            if 'Multilingual' in self.Data_spec:
                                import re
                                src_match = re.search(r'src([a-zA-Z]+)', self.Data_spec)
                                tgt_match = re.search(r'tgt([a-zA-Z]+)', self.Data_spec)
                                src_lang = src_match.group(1) if src_match else None
                                tgt_lang = tgt_match.group(1) if tgt_match else None

                                if src_lang != 'x':
                                    lang = src_lang
                                    lang_flag = 'tgt'
                                    to = 'src'

                                elif tgt_lang != 'x':
                                    lang = tgt_lang
                                    lang_flag = 'src'
                                    to = 'tgt'

                                embd_dict = {0: '175M', 1: '615M', 2: 'big'}
                                langs = ['en', 'id', 'jv', 'ms', 'ta', 'tl']
                                lang_dict = {}

                                count = 0
                                for l in langs:
                                    if l != lang:
                                        lang_dict[count] = l
                                        count += 1

                            elif 'Bilingual' in self.Data_spec:
                                embd_dict = {0: 'en', 1: 'id', 2: 'jv', 3: 'ms', 4: 'ta', 5: 'tl'}
                                lang_dict = {0: 'en', 1: 'id', 2: 'jv', 3: 'ms', 4: 'ta', 5: 'tl'}



                            elif "Embed2" in self.Data_spec:
                                embd_dict = {0: 512, 1: 960, 2: 1600}
                                layer_dict = {0: 8, 1: 12, 2: 24, 3: 48}
                            else:
                                embd_dict = {0: 512, 1: 768, 2: 960, 3: 1024, 4: 1600}
                                layer_dict = {0: 8, 1: 10, 2: 12, 3: 24, 4: 32, 5: 48}

                            with open('z_score_dict.pkl', 'rb') as f:
                                z_score_dict = pickle.load(f)

                            if 'Layer' in self.Data_spec:
                                mu_z, std_z = z_score_dict[f'{layer_dict[d]}']
                            else:
                                mu_z, std_z = z_score_dict[f'{embd_dict[d]}']

                            mu_orig = (mu_on * std_z) + mu_z
                            var_orig = var_on * (std_z**2)
                            mu_comp_orig = (mu_on_comp * std_z) + mu_z
                            var_comp_orig = var_on_comp * (std_z ** 2)

                            # orig signal
                            Y_list_orig = (self.Y_list[d] * std_z) + mu_z


                            if self.Experiment_type == 'Missing_Triangle_1':
                                Y_list_orig = Y_list_orig[r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1]
                                X_all_outputs_orig = np.exp(self.X_all_outputs_with_replicates[d][:, :-1][0:int(len(idx_train_d[0])/self.num_replicates):1])

                            elif 'Missing_Quad' in self.Experiment_type or 'active' in self.Experiment_type or 'Missing_Tri_' in self.Experiment_type:
                                Y_list_orig = Y_list_orig[r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1]
                                X_all_outputs_orig = np.exp(self.X_all_outputs_with_replicates[d][:, :-1][0:int(len(idx_train_d[0])/self.num_replicates):1])


                            elif 'Bilingual' in self.Data_spec:
                                Y_list_orig = Y_list_orig[~idx_size[r]]
                                X_all_outputs_orig = np.exp(self.X_all_outputs_with_replicates[d][:, :-1][~idx_size[r]])
                            else:
                                Y_list_orig = Y_list_orig[idx_train_d[r]]
                                X_all_outputs_orig = np.exp(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]])


                            RMSE = np.round(self.rmse(Y_list_orig, mu_comp_orig), 4)
                            MSE = np.round(self.mse(Y_list_orig, mu_comp_orig), 4)
                            MAE = np.round(self.mae(Y_list_orig, mu_comp_orig), 4)
                            MNLPD = np.round(MNLP(Y_list_orig, mu_comp_orig, var_comp_orig), 4)

                            RMSEp = np.round(self.rmse(Y_list_orig[-3:], mu_comp_orig[-3:]), 4)
                            MSEp = np.round(self.mse(Y_list_orig[-3:], mu_comp_orig[-3:]), 4)
                            MAEp = np.round(self.mae(Y_list_orig[-3:], mu_comp_orig[-3:]), 4)
                            MNLPDp = np.round(MNLP(Y_list_orig[-3:], mu_comp_orig[-3:], var_comp_orig[-3:]), 4)

                            RMSEl = np.round(self.rmse(Y_list_orig[-1:], mu_comp_orig[-1:]), 4)
                            MSEl = np.round(self.mse(Y_list_orig[-1:], mu_comp_orig[-1:]), 4)
                            MAEl = np.round(self.mae(Y_list_orig[-1:], mu_comp_orig[-1:]), 4)
                            MNLPDl = np.round(MNLP(Y_list_orig[-1:], mu_comp_orig[-1:], var_comp_orig[-1:]), 4)

                            if 'Embed' in self.Data_spec:
                                C_vec = [dict_of_C[f'{embd_dict[d]}-{layer_dict[r]}'][i] for i in
                                         X_all_outputs_orig.astype(int).squeeze(-1)]
                            if 'Layer' in self.Data_spec:
                                C_vec = [dict_of_C[f'{embd_dict[r]}-{layer_dict[d]}'][i] for i in
                                         X_all_outputs_orig.astype(int).squeeze(-1)]

                            if 'Embed' in self.Data_spec:
                                if self.Experiment_type == 'Missing_Triangle_1':
                                    if idx_train_d[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1][-1] == False and d == (self.D - 1):
                                        ax_sl.plot(C_vec, mu_comp_orig, 'b--', linewidth=0.8, label='Test Prediction')
                                        file_path = os.path.join(folder_embed, f"Test_pred_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                    else:
                                        ax_sl.plot(C_vec, mu_comp_orig, color='dodgerblue', linewidth=0.8, label='Train Prediction')
                                        file_path = os.path.join(folder_embed,
                                                                 f"Train_pred_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])

                                elif 'Missing_Quad' in self.Experiment_type:

                                    if idx_train_d[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1][-1] == False:

                                        ax_sl.plot(C_vec, mu_comp_orig, 'b--', linewidth=0.8, label='Test Prediction')
                                        file_path = os.path.join(folder_embed,
                                                                 f"Test_pred_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                    else:
                                        ax_sl.plot(C_vec, mu_comp_orig, color='dodgerblue', linewidth=0.8, label='Train Prediction')
                                        file_path = os.path.join(folder_embed,
                                                                 f"Train_pred_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")

                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])

                                elif 'active' in self.Experiment_type:
                                    r_list_missing = d_r_dict_missing[d]

                                    if r in r_list_missing:
                                        ax_sl.plot(C_vec, mu_comp_orig, 'b--', linewidth=0.8, label='Test Prediction')
                                        file_path = os.path.join(folder_embed,
                                                                 f"Test_pred_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                    else:
                                        ax_sl.plot(C_vec, mu_comp_orig, color='dodgerblue', linewidth=0.8, label='Train Prediction')
                                        file_path = os.path.join(folder_embed,
                                                                 f"Train_pred_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")

                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                elif 'Missing_Tri_' in self.Experiment_type:
                                    if r in rep_int_init and d in d_int:
                                        if idx_train_d[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1][-1] == False and d in d_int:
                                            ax_sl.plot(C_vec, mu_comp_orig, 'b--', linewidth=0.8, label='Test Prediction')
                                            file_path = os.path.join(folder_embed,
                                                                     f"Test_pred_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                            np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                    else:
                                        ax_sl.plot(C_vec, mu_comp_orig, color='dodgerblue', linewidth=0.8,
                                                   label='Train Prediction')
                                        file_path = os.path.join(folder_embed,
                                                                 f"Train_pred_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                else:
                                    if idx_train_d[r][0] == True:
                                        ax_sl.plot(C_vec, mu_comp_orig, 'b--', linewidth=0.8, label='Test Prediction')
                                        file_path = os.path.join(folder_embed,
                                                                 f"Test_pred_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                    else:
                                        ax_sl.plot(C_vec, mu_comp_orig, color='dodgerblue', linewidth=0.8, label='Train Prediction')
                                        file_path = os.path.join(folder_embed,
                                                                 f"Train_pred_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])

                            if 'Layer' in self.Data_spec:
                                if self.Experiment_type == 'Missing_Triangle_1':
                                    if idx_train_d[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1][-1] == False and d == (self.D - 1):
                                        ax_sl.plot(C_vec, mu_comp_orig, 'b--', linewidth=0.8, label='Test Prediction')
                                        file_path = os.path.join(folder_layer, f"Test_pred_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                    else:
                                        ax_sl.plot(C_vec, mu_comp_orig, color='dodgerblue', linewidth=0.8, label='Train Prediction')
                                        file_path = os.path.join(folder_layer,
                                                                 f"Train_pred_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])

                                elif 'Missing_Quad' in self.Experiment_type:

                                    if idx_train_d[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1][-1] == False:

                                        ax_sl.plot(C_vec, mu_comp_orig, 'b--', linewidth=0.8, label='Test Prediction')
                                        file_path = os.path.join(folder_layer,
                                                                 f"Test_pred_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                    else:
                                        ax_sl.plot(C_vec, mu_comp_orig, color='dodgerblue', linewidth=0.8, label='Train Prediction')
                                        file_path = os.path.join(folder_layer,
                                                                 f"Train_pred_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")

                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])

                                elif 'active' in self.Experiment_type:
                                    print('To Do')
                                    # r_list_missing = d_r_dict_missing[d]
                                    #
                                    # if r in r_list_missing:
                                    #     ax_sl.plot(C_vec, mu_comp_orig, 'b--', linewidth=0.8, label='Test Prediction')
                                    #     file_path = os.path.join(folder_embed,
                                    #                              f"Test_pred_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                    #     np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                    # else:
                                    #     ax_sl.plot(C_vec, mu_comp_orig, color='dodgerblue', linewidth=0.8, label='Train Prediction')
                                    #     file_path = os.path.join(folder_embed,
                                    #                              f"Train_pred_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                    #
                                    #     np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                elif 'Missing_Tri_' in self.Experiment_type:
                                    if r in rep_int_init and d in d_int:
                                        if idx_train_d[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1][-1] == False and d in d_int:
                                            ax_sl.plot(C_vec, mu_comp_orig, 'b--', linewidth=0.8, label='Test Prediction')
                                            file_path = os.path.join(folder_layer,
                                                                     f"Test_pred_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")
                                            np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                    else:
                                        ax_sl.plot(C_vec, mu_comp_orig, color='dodgerblue', linewidth=0.8,
                                                   label='Train Prediction')
                                        file_path = os.path.join(folder_layer,
                                                                 f"Train_pred_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                else:
                                    if idx_train_d[r][0] == True:
                                        ax_sl.plot(C_vec, mu_comp_orig, 'b--', linewidth=0.8, label='Test Prediction')
                                        file_path = os.path.join(folder_layer,
                                                                 f"Test_pred_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])
                                    else:
                                        ax_sl.plot(C_vec, mu_comp_orig, color='dodgerblue', linewidth=0.8, label='Train Prediction')
                                        file_path = os.path.join(folder_layer,
                                                                 f"Train_pred_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")
                                        np.save(file_path, [C_vec, mu_comp_orig.squeeze()])

                            #plot_orig(np.exp(x_raw_plot), mu_orig, var_orig, ax=ax_orig)
                            plot_orig(X_all_outputs_orig, mu_comp_orig, var_comp_orig, ax=ax_orig)

                            if d == self.seed_index and r == (
                                    d + self.seed_index) % self.num_replicates and self.Experiment_type == 'Missing_One_replica_in_Whole':
                                ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]], self.Y_list[d][idx_train_d[r]], 'rd',
                                        mew=8.2, ms=4, label='Missing')
                            elif d == self.seed_index and r == (
                                    d + self.seed_index) % self.num_replicates and self.Experiment_type == 'Missing_part_of_one_replica_in_Whole':
                                ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]][self.training_id],
                                        self.Y_list[d][idx_train_d[r]][self.training_id], 'ks', mew=5.2, ms=4, label='Train')
                                ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]][self.test_id],
                                        self.Y_list[d][idx_train_d[r]][self.test_id], 'rd', mew=8.2, ms=4, label='Missing')



                            if self.Experiment_type == 'Missing_One_replica_in_each_output':

                                if 'Bilingual' in self.Data_spec:
                                    if r in dict_num[d]:

                                        ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][~idx_size[r]],self.Y_list[d][~idx_size[r]], 'rd',mew=8.2, ms=2, label='Missing')

                                        ax_orig.plot(X_all_outputs_orig, Y_list_orig, 'rd',
                                                     mew=8.2, ms=2, label=f'Missing\n '
                                                                          f'RMSE={RMSE}\n'
                                                                          f'MSE={MSE}\n '
                                                                          f'MAE={MAE}\n'
                                                                          f'NLPD={MNLPD}')
                                        if r != d:
                                            bilingual_error_dict[f"{d}-{r}"] = [RMSE, MSE, MAE, MNLPD,
                                                                                RMSEp, MSEp, MAEp, MNLPDp,
                                                                                RMSEl, MSEl, MAEl, MNLPDl]
                                             
                                            RMSE_vec.append(RMSE)
                                            MSE_vec.append(MSE)
                                            MAE_vec.append(MAE)
                                            MNLPD_vec.append(MNLPD)
                                            RMSEp_vec.append(RMSEp)
                                            MSEp_vec.append(MSEp)
                                            MAEp_vec.append(MAEp)
                                            MNLPDp_vec.append(MNLPDp)
                                            RMSEl_vec.append(RMSEl)
                                            MSEl_vec.append(MSEl)
                                            MAEl_vec.append(MAEl)
                                            MNLPDl_vec.append(MNLPDl)
                                        final_loss = Y_list_orig[-1][0]
                                        ax_orig.text(X_all_outputs_orig[-1], final_loss, f"{final_loss:.4f}",
                                                     fontsize=24, color='red', verticalalignment='bottom',
                                                     horizontalalignment='right')


                                    else:
                                        ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][~idx_size[r]],
                                                self.Y_list[d][~idx_size[r]], 'ks',
                                                mew=5.2, ms=2, label='Train')
                                        ax_orig.plot(X_all_outputs_orig,
                                                     Y_list_orig, 'ks',
                                                     mew=5.2, ms=2, label='Train')
                                        final_loss = Y_list_orig[-1][0]
                                        ax_orig.text(X_all_outputs_orig[-1], final_loss,
                                                     f"{final_loss:.4f}",
                                                     fontsize=24, color='red', verticalalignment='bottom',
                                                     horizontalalignment='right')


                                else:
                                    if r == (d + self.seed_index) % self.num_replicates:
                                        ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]],
                                                self.Y_list[d][idx_train_d[r]], 'rd',
                                                mew=8.2, ms=2, label='Missing')

                                        ax_orig.plot(X_all_outputs_orig, Y_list_orig, 'rd',
                                                     mew=8.2, ms=2, label=f'Missing\n '
                                                                          f'RMSE={RMSE}\n'
                                                                          f'MSE={MSE}\n '
                                                                          f'MAE={MAE}\n'
                                                                          f'NLPD={MNLPD}')
                                         
                                        RMSE_vec.append(RMSE)
                                        MSE_vec.append(MSE)
                                        MAE_vec.append(MAE)
                                        MNLPD_vec.append(MNLPD)
                                        RMSEp_vec.append(RMSEp)
                                        MSEp_vec.append(MSEp)
                                        MAEp_vec.append(MAEp)
                                        MNLPDp_vec.append(MNLPDp)
                                        RMSEl_vec.append(RMSEl)
                                        MSEl_vec.append(MSEl)
                                        MAEl_vec.append(MAEl)
                                        MNLPDl_vec.append(MNLPDl)

                                        if 'Embed' in self.Data_spec:
                                            ax_sl.plot(C_vec, Y_list_orig, 'k', linewidth=0.8,
                                                       label=f'Test Ground Truth', zorder=-1)
                                            file_path = os.path.join(folder_embed,
                                                                     f"Test_GT_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                            np.save(file_path, [C_vec, Y_list_orig.squeeze()])

                                        if 'Layer' in self.Data_spec:
                                            ax_sl.plot(C_vec, Y_list_orig, 'k', linewidth=0.8,
                                                       label=f'Test Ground Truth', zorder=-1)
                                            file_path = os.path.join(folder_layer,
                                                                     f"Test_GT_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")
                                            np.save(file_path, [C_vec, Y_list_orig.squeeze()])

                                        final_loss = Y_list_orig[-1][0]

                                        ax_orig.text(X_all_outputs_orig[-1], final_loss, f"{final_loss:.4f}",
                                                     fontsize=24, color='red', verticalalignment='bottom',
                                                     horizontalalignment='right')

                                    else:
                                        ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]], self.Y_list[d][idx_train_d[r]], 'ks',
                                                mew=5.2, ms=2, label='Train')
                                        ax_orig.plot(X_all_outputs_orig,
                                                Y_list_orig, 'ks',
                                                mew=5.2, ms=2, label='Train')
                                        final_loss = Y_list_orig[-1][0]
                                        ax_orig.text(X_all_outputs_orig[-1], final_loss,
                                                     f"{final_loss:.4f}",
                                                     fontsize=24, color='red', verticalalignment='bottom',
                                                     horizontalalignment='right')

                            elif self.Experiment_type == 'Missing_Triangle_1':
                                if d == (self.D - 1) and r == self.num_replicates-1:


                                    ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1],
                                            self.Y_list[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1],
                                            'rd', mew=8.2, ms=2, label='Missing')

                                    ax_orig.plot(X_all_outputs_orig, Y_list_orig, 'rd',
                                                 mew=8.2, ms=2, label=f'Missing\n '
                                                                      f'RMSE={RMSE}\n'
                                                                      f'MSE={MSE}\n '
                                                                      f'MAE={MAE}\n'
                                                                      f'NLPD={MNLPD}')
                                    RMSE_vec.append(RMSE)
                                    MSE_vec.append(MSE)
                                    MAE_vec.append(MAE)
                                    MNLPD_vec.append(MNLPD)
                                    RMSEp_vec.append(RMSEp)
                                    MSEp_vec.append(MSEp)
                                    MAEp_vec.append(MAEp)
                                    MNLPDp_vec.append(MNLPDp)
                                    RMSEl_vec.append(RMSEl)
                                    MSEl_vec.append(MSEl)
                                    MAEl_vec.append(MAEl)
                                    MNLPDl_vec.append(MNLPDl)

                                    if 'Embed' in self.Data_spec:
                                        ax_sl.plot(C_vec, Y_list_orig, 'b', linewidth=0.8,
                                                   label=f'Test Ground Truth')
                                        file_path = os.path.join(folder_embed,
                                                                 f"Test_GT_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                        np.save(file_path, [C_vec, Y_list_orig.squeeze()])
                                    if 'Layer' in self.Data_spec:
                                        ax_sl.plot(C_vec, Y_list_orig, 'b', linewidth=0.8,
                                                   label=f'Test Ground Truth')
                                        file_path = os.path.join(folder_layer,
                                                                 f"Test_GT_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")
                                        np.save(file_path, [C_vec, Y_list_orig.squeeze()])

                                    final_loss = Y_list_orig[-1][0]

                                    ax_orig.text(X_all_outputs_orig[-1], final_loss, f"{final_loss:.4f}",
                                                 fontsize=24, color='red', verticalalignment='bottom',
                                                 horizontalalignment='right')

                                else:


                                    ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1],
                                            self.Y_list[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1], 'ks',
                                            mew=5.2, ms=2, label='Train')
                                    ax_orig.plot(X_all_outputs_orig,
                                                 Y_list_orig, 'ks',
                                                 mew=5.2, ms=2, label='Train')
                                    final_loss = Y_list_orig[-1][0]
                                    ax_orig.text(X_all_outputs_orig[-1], final_loss,
                                                 f"{final_loss:.4f}",
                                                 fontsize=24, color='red', verticalalignment='bottom',
                                                 horizontalalignment='right')

                            elif 'Missing_Quad' in self.Experiment_type:
                                if d in d_int and r in rep_int:
                                    ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1],
                                            self.Y_list[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1], 'rd',
                                            mew=8.2, ms=2, label='Missing')

                                    ax_orig.plot(X_all_outputs_orig, Y_list_orig, 'rd',
                                                 mew=8.2, ms=2, label=f'Missing\n '
                                                                      f'RMSE={RMSE}\n'
                                                                      f'MSE={MSE}\n '
                                                                      f'MAE={MAE}\n'
                                                                      f'NLPD={MNLPD}')
                                    RMSE_vec.append(RMSE)
                                    MSE_vec.append(MSE)
                                    MAE_vec.append(MAE)
                                    MNLPD_vec.append(MNLPD)
                                    RMSEp_vec.append(RMSEp)
                                    MSEp_vec.append(MSEp)
                                    MAEp_vec.append(MAEp)
                                    MNLPDp_vec.append(MNLPDp)
                                    RMSEl_vec.append(RMSEl)
                                    MSEl_vec.append(MSEl)
                                    MAEl_vec.append(MAEl)
                                    MNLPDl_vec.append(MNLPDl)

                                    if 'Embed' in self.Data_spec:
                                        ax_sl.plot(C_vec, Y_list_orig, 'b', linewidth=0.8,
                                                   label=f'Test Ground Truth')
                                        file_path = os.path.join(folder_embed,
                                                                 f"Test_GT_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                        np.save(file_path, [C_vec, Y_list_orig.squeeze()])
                                    if 'Layer' in self.Data_spec:
                                        ax_sl.plot(C_vec, Y_list_orig, 'b', linewidth=0.8,
                                                   label=f'Test Ground Truth')
                                        file_path = os.path.join(folder_layer,
                                                                 f"Test_GT_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")
                                        np.save(file_path, [C_vec, Y_list_orig.squeeze()])

                                    final_loss = Y_list_orig[-1][0]

                                    ax_orig.text(X_all_outputs_orig[-1], final_loss, f"{final_loss:.4f}",
                                                 fontsize=24, color='red', verticalalignment='bottom',
                                                 horizontalalignment='right')

                                else:
                                    ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1],
                                            self.Y_list[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1], 'ks',
                                            mew=5.2, ms=2, label='Train')
                                    ax_orig.plot(X_all_outputs_orig,
                                                 Y_list_orig, 'ks',
                                                 mew=5.2, ms=2, label='Train')
                                    final_loss = Y_list_orig[-1][0]
                                    ax_orig.text(X_all_outputs_orig[-1], final_loss,
                                                 f"{final_loss:.4f}",
                                                 fontsize=24, color='red', verticalalignment='bottom',
                                                 horizontalalignment='right')

                            elif 'active' in self.Experiment_type:
                                r_list_missing = d_r_dict_missing[d]

                                if r in r_list_missing:

                                    ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1],
                                            self.Y_list[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1], 'rd',
                                            mew=8.2, ms=2, label='Missing')

                                    ax_orig.plot(X_all_outputs_orig, Y_list_orig, 'rd',
                                                 mew=8.2, ms=2, label=f'Missing\n '
                                                                      f'RMSE={RMSE}\n'
                                                                      f'MSE={MSE}\n '
                                                                      f'MAE={MAE}\n'
                                                                      f'NLPD={MNLPD}')
                                    RMSE_vec.append(RMSE)
                                    MSE_vec.append(MSE)
                                    MAE_vec.append(MAE)
                                    MNLPD_vec.append(MNLPD)
                                    RMSEp_vec.append(RMSEp)
                                    MSEp_vec.append(MSEp)
                                    MAEp_vec.append(MAEp)
                                    MNLPDp_vec.append(MNLPDp)
                                    RMSEl_vec.append(RMSEl)
                                    MSEl_vec.append(MSEl)
                                    MAEl_vec.append(MAEl)
                                    MNLPDl_vec.append(MNLPDl)

                                    if 'active' in self.Experiment_type:
                                        uncertainty_dicts[f'{r}-{d}'] = [mu_comp_orig]


                                    if 'Embed' in self.Data_spec:
                                        ax_sl.plot(C_vec, Y_list_orig, 'b', linewidth=0.8,
                                                   label=f'Test Ground Truth')
                                        file_path = os.path.join(folder_embed,
                                                                 f"Test_GT_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                        np.save(file_path, [C_vec, Y_list_orig.squeeze()])

                                    if 'Layer' in self.Data_spec:
                                        ax_sl.plot(C_vec, Y_list_orig, 'b', linewidth=0.8,
                                                   label=f'Test Ground Truth')
                                        file_path = os.path.join(folder_layer,
                                                                 f"Test_GT_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")
                                        np.save(file_path, [C_vec, Y_list_orig.squeeze()])

                                    final_loss = Y_list_orig[-1][0]

                                    ax_orig.text(X_all_outputs_orig[-1], final_loss, f"{final_loss:.4f}",
                                                 fontsize=24, color='red', verticalalignment='bottom',
                                                 horizontalalignment='right')

                                else:
                                    ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1],
                                            self.Y_list[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1], 'ks',
                                            mew=5.2, ms=2, label='Train')
                                    ax_orig.plot(X_all_outputs_orig,
                                                 Y_list_orig, 'ks',
                                                 mew=5.2, ms=2, label='Train')
                                    final_loss = Y_list_orig[-1][0]
                                    ax_orig.text(X_all_outputs_orig[-1], final_loss,
                                                 f"{final_loss:.4f}",
                                                 fontsize=24, color='red', verticalalignment='bottom',
                                                 horizontalalignment='right')


                            elif 'Missing_Tri_' in self.Experiment_type:

                                import re
                                s = self.Experiment_type
                                d_match = re.search(r'd(\d+)', s)
                                d_number = int(d_match.group(1)) if d_match else 0
                                # r_number = d_number
                                # rep_int = list(range(0, self.num_replicates))
                                # rep_int = rep_int[-r_number:]
                                d_int = list(range(0, self.D))
                                d_int = d_int[-d_number:]


                                if d in d_int and r in rep_int_init:
                                    #print('d', d, 'r', r, 'rep_int_init', rep_int_init)

                                    ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1],
                                            self.Y_list[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1], 'rd',
                                            mew=8.2, ms=2, label='Missing')

                                    ax_orig.plot(X_all_outputs_orig, Y_list_orig, 'rd',
                                                 mew=8.2, ms=2, label=f'Missing\n '
                                                                      f'RMSE={RMSE}\n'
                                                                      f'MSE={MSE}\n '
                                                                      f'MAE={MAE}\n'
                                                                      f'NLPD={MNLPD}')
                                     
                                    RMSE_vec.append(RMSE)
                                    MSE_vec.append(MSE)
                                    MAE_vec.append(MAE)
                                    MNLPD_vec.append(MNLPD)
                                    RMSEp_vec.append(RMSEp)
                                    MSEp_vec.append(MSEp)
                                    MAEp_vec.append(MAEp)
                                    MNLPDp_vec.append(MNLPDp)
                                    RMSEl_vec.append(RMSEl)
                                    MSEl_vec.append(MSEl)
                                    MAEl_vec.append(MAEl)
                                    MNLPDl_vec.append(MNLPDl)

                                    if 'Embed' in self.Data_spec:
                                        ax_sl.plot(C_vec, Y_list_orig, 'b', linewidth=0.8,
                                                   label=f'Test Ground Truth')
                                        file_path = os.path.join(folder_embed,
                                                                 f"Test_GT_{self.Experiment_type}_{embd_dict[d]}_{layer_dict[r]}.npy")
                                        np.save(file_path, [C_vec, Y_list_orig.squeeze()])

                                    if 'Layer' in self.Data_spec:
                                        ax_sl.plot(C_vec, Y_list_orig, 'b', linewidth=0.8,
                                                   label=f'Test Ground Truth')
                                        file_path = os.path.join(folder_layer,
                                                                 f"Test_GT_{self.Experiment_type}_{embd_dict[r]}_{layer_dict[d]}.npy")
                                        np.save(file_path, [C_vec, Y_list_orig.squeeze()])

                                    final_loss = Y_list_orig[-1][0]

                                    ax_orig.text(X_all_outputs_orig[-1], final_loss, f"{final_loss:.4f}",
                                                 fontsize=24, color='red', verticalalignment='bottom',
                                                 horizontalalignment='right')

                                    if len(rep_int_init) < self.num_replicates and r == rep_int_init[0]:
                                        rep_int_init.append(rep_int_init[-1] - 1)

                                else:
                                    ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1],
                                            self.Y_list[d][r*int(len(idx_train_d[0])/self.num_replicates):r*int(len(idx_train_d[0])/self.num_replicates)+int(len(idx_train_d[0])/self.num_replicates):1], 'ks',
                                            mew=5.2, ms=2, label='Train')

                                    ax_orig.plot(X_all_outputs_orig,
                                                 Y_list_orig, 'ks',
                                                 mew=5.2, ms=2, label='Train')
                                    final_loss = Y_list_orig[-1][0]
                                    ax_orig.text(X_all_outputs_orig[-1], final_loss,
                                                 f"{final_loss:.4f}",
                                                 fontsize=24, color='red', verticalalignment='bottom',
                                                 horizontalalignment='right')
                            elif r == (d + self.seed_index) % self.num_replicates and self.Experiment_type == 'Missing_part_of_one_replica_in_each_ouput':
                                ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]][self.training_id[d]],
                                        self.Y_list[d][idx_train_d[r]][self.training_id[d]], 'ks', mew=5.2, ms=4, label='Train')
                                ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]][self.test_id[d]],
                                        self.Y_list[d][idx_train_d[r]][self.test_id[d]], 'rd', mew=8.2, ms=4, label='Test')

                                multilingual_error_dict[f"{d}-{r}"] = [RMSE, MSE, MAE, MNLPD,
                                                                    RMSEp, MSEp, MAEp, MNLPDp,
                                                                    RMSEl, MSEl, MAEl, MNLPDl]
                                RMSE_vec.append(RMSE)
                                MSE_vec.append(MSE)
                                MAE_vec.append(MAE)
                                MNLPD_vec.append(MNLPD)
                                RMSEp_vec.append(RMSEp)
                                MSEp_vec.append(MSEp)
                                MAEp_vec.append(MAEp)
                                MNLPDp_vec.append(MNLPDp)
                                RMSEl_vec.append(RMSEl)
                                MSEl_vec.append(MSEl)
                                MAEl_vec.append(MAEl)
                                MNLPDl_vec.append(MNLPDl)
                            #plt.title('%i-th output ' % d + '%ith replicates' % r)
                            #embed'512', '768', '960', '1024', layer 8, 10, 24, 32

                            if 'Embed' in self.Data_spec:
                                ax.set_title('n_embed= %i ' % embd_dict[d] + 'n_layers= %i' % layer_dict[r])
                                ax.legend()
                                ax_orig.legend()
                                ax_orig.set_title('n_embed= %i ' % embd_dict[d] + 'n_layers= %i' % layer_dict[r])

                            elif 'Layer' in self.Data_spec:
                                ax.set_title('n_layers= %i' % layer_dict[d] + 'n_embed= %i ' % embd_dict[r])
                                ax.legend()
                                ax_orig.legend()
                                ax_orig.set_title('n_layers= %i' % layer_dict[d] + 'n_embed= %i ' % embd_dict[r])

                            elif 'Bilingual' in self.Data_spec:
                                ax.set_title(f'SRC= {embd_dict[d]} to TGT= {lang_dict[r]}')
                                ax.legend()
                                ax_orig.legend()
                                ax_orig.set_title(f'SRC= {embd_dict[d]} to TGT= {lang_dict[r]}')
                            else:
                                ax.set_title(f'Model= {embd_dict[d]}; ' + f'{lang_flag}= {lang_dict[r]}, {to}={lang}')
                                ax.legend()
                                ax_orig.legend()
                                ax_orig.set_title(f'Model= {embd_dict[d]}; ' + f'{lang_flag}= {lang_dict[r]}, {to}={lang}')

                        if 'Multilingual' in self.Data_spec:
                            new_name = f'{self.Data_name}_{to}_{lang}'
                        else:
                            new_name = f'{self.Data_name}'

                        if 'active' in self.Experiment_type:
                            path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}_{self.query_num}"
                        else:
                            path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                        save_plot(fig, 'pred', path_, self.Experiment_type, new_name, self.D, self.num_replicates, d, self.Model_name, self.Q,
                                  self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition, self.seed_index)
                        save_plot(fig_orig, 'orig', path_, self.Experiment_type, new_name, self.D, self.num_replicates, d, self.Model_name, self.Q,
                                  self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition, self.seed_index)

                        if 'Embed' in self.Data_spec:
                            ax_sl.set_title(f"Validation Loss over Compute", fontsize=9)
                            ax_sl.grid(True, linestyle="--", alpha=0.6)
                            ax_sl.set_xlabel("Compute")
                            ax_sl.set_ylabel("Validation Loss")
                            ax_sl.set_xscale('log')
                            ax_sl.set_yscale('log')

                            handles, labels = ax_sl.get_legend_handles_labels()
                            unique_labels = dict(zip(labels, handles))  # Keep only the first occurrence

                            # Show legend with unique labels
                            ax_sl.legend(unique_labels.values(), unique_labels.keys())
                            if 'active' in self.Experiment_type:
                                path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}_{self.query_num}"

                            else:
                                path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                            fig_sl.savefig(os.path.join(f"{path_}", f'C_plot_{self.Experiment_type}_{run_i}.png'), format='png', bbox_inches='tight')
                            plt.close()

                        if 'Layer' in self.Data_spec:

                            ax_sl.set_title(f"Validation Loss over Compute", fontsize=9)
                            ax_sl.grid(True, linestyle="--", alpha=0.6)
                            ax_sl.set_xlabel("Compute")
                            ax_sl.set_ylabel("Validation Loss")
                            ax_sl.set_xscale('log')
                            ax_sl.set_yscale('log')

                            handles, labels = ax_sl.get_legend_handles_labels()
                            unique_labels = dict(zip(labels, handles))  # Keep only the first occurrence

                            # Show legend with unique labels
                            ax_sl.legend(unique_labels.values(), unique_labels.keys())
                            if 'active' in self.Experiment_type:
                                path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}_{self.query_num}"

                            else:
                                path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                            fig_sl.savefig(os.path.join(f"{path_}", f'C_plot_{self.Experiment_type}_{run_i}.png'), format='png', bbox_inches='tight')
                            plt.close()

                    if 'active' in self.Experiment_type:
                        path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}_{self.query_num}"

                    else:
                        path_ = self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                    with open(f"{path_}/error_metrics_{self.Data_spec}_{self.Experiment_type}.txt", "a") as file:
                         
                        file.write(
                            f"{np.round(np.mean(RMSE_vec), 2)}, "
                            f"{np.round(np.mean(MSE_vec), 2)}, "
                            f"{np.round(np.mean(MAE_vec), 2)}, "
                            f"{np.round(np.mean(MNLPD_vec), 2)}, "
                            f"{np.round(np.mean(RMSEp_vec), 2)}, "
                            f"{np.round(np.mean(MSEp_vec), 2)}, "
                            f"{np.round(np.mean(MAEp_vec), 2)}, "
                            f"{np.round(np.mean(MNLPDp_vec), 2)}, "
                            f"{np.round(np.mean(RMSEl_vec), 2)}, "
                            f"{np.round(np.mean(MSEl_vec), 2)}, "
                            f"{np.round(np.mean(MAEl_vec), 2)}, "
                            f"{np.round(np.mean(MNLPDl_vec), 2)} \n")

                    if 'Bilingual' in self.Data_spec:
                        path_= self.my_path+rf"\{self.Model_name}\{self.Data_spec}_{self.Experiment_type}"

                        with open(f'{path_}/bilingual_error_dict_{run_i}.pkl', 'wb') as f:
                            pickle.dump(bilingual_error_dict, f)

                    if 'active' in self.Experiment_type:
                        if "main" in self.Experiment_type:
                            file_path = r'0_get_data\AL\Active_main/uncertainty_dicts'
                        elif "random" in self.Experiment_type:
                            file_path = r'0_get_data\AL\Active_random/uncertainty_dicts'
                        elif "largest" in self.Experiment_type:
                            file_path = r'0_get_data\AL\Active_largest/uncertainty_dicts'
                        elif "smallest" in self.Experiment_type:
                            file_path = r'0_get_data\AL\Active_smallest/uncertainty_dicts'

                        path_folder = Path(file_path)
                        path_folder.mkdir(parents=True, exist_ok=True)

                        with open(f'{file_path}/uncertainty_dicts_{run_i}.pkl', 'wb') as f:
                            pickle.dump(uncertainty_dicts, f)

    def prediction_for_model(self, m_test):
        '''
        This function is used for prediction for test data set and calculation for evaluation metric
        :param m_test:
        :return:
        '''
        # evaluation metric
        NMSE_test_bar_missing_replicates = []
        MNLP_missing_replicates = []
        Global_mean_pre = []
        Global_variance_pre = []
        Global_test_pre = []
        if self.Experiment_type == 'Missing_One_replica_in_Whole' or self.Experiment_type == 'Missing_part_of_one_replica_in_Whole':
            for d in range(self.D):
                if d == self.seed_index:
                    if self.Model_name == 'HMOGPLV':
                        mean_missing, variance_missing = m_test.predict_f(self.X_list_missing_test[0])
                        mean_missing, variance_missing = mean_missing.numpy(), variance_missing.numpy() + m_test.Heter_GaussianNoise.numpy()
                        mean_one_missing = mean_missing[:, d][:, None]
                        variance_one_missing = variance_missing[:, d][:, None]
                    elif self.Model_name == 'DHGP':

                        x_test = np.hstack((self.X_list_missing_test[d][:, :-1], self.X_list_missing_test[d][:, -1][:, None] * 0 + d,
                                            self.X_list_missing_test[d][:, -1][:, None] * 0))


                        mean_one_missing, variance_one_missing = m_test.predict(x_test)
                    elif self.Model_name == 'HGPInd':
                        mean_missing, variance_missing = m_test.predict_y(self.X_list_missing_test[0])
                        mean_one_missing, variance_one_missing = mean_missing.numpy(), variance_missing.numpy()
                    elif self.Model_name == 'LMC':
                        X_pre = np.c_[self.X_list_missing_test[0][:, :-1], np.ones_like(self.X_list_missing_test[0][:, :-1]) * d]
                        mean_missing, variance_missing = m_test.predict_f(X_pre)
                        mean_one_missing = mean_missing.numpy()
                        variance_one_missing = variance_missing.numpy() + m_test.likelihood.likelihoods[d].variance.numpy()
                    Y_one_missing = self.Y_list_missing_test[0]
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
        elif self.Experiment_type == 'Missing_One_replica_in_each_output' or self.Experiment_type == 'Missing_part_of_one_replica_in_each_ouput':
            if self.Model_name == 'SGP':
                x_test = self.X_list_missing_test[:, :-1]
                mean_one_missing, variance_one_missing = m_test.predict(x_test)
                Y_one_missing = self.Y_list_missing_test
                Global_mean_pre.append(mean_one_missing)
                Global_variance_pre.append(variance_one_missing)
                Global_test_pre.append(Y_one_missing)
                NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

            elif self.Model_name == 'DNN':
                x_test = self.X_list_missing_test[:, :-1]
                mean_one_missing = m_test.predict(x_test)
                variance_one_missing = abs(mean_one_missing)
                Y_one_missing = self.Y_list_missing_test
                Global_mean_pre.append(mean_one_missing)
                Global_variance_pre.append(variance_one_missing)
                Global_test_pre.append(Y_one_missing)
                NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

            elif self.Model_name == 'LMC3' or self.Model_name == 'LVMOGP3':
                Indexoutput_test = []
                x_all_test = []
                y_all_test = []
                for r in range(self.num_replicates):
                    x_output_d_r = []
                    y_output_d_r = []
                    Indexoutput_d_r = []
                    for d in range(self.D):
                        index_r = self.X_list_missing_test[d][:, -1] == r
                        x_output_d_r.append(self.X_list_missing_test[d][index_r][:, :-1])
                        y_output_d_r.append(self.Y_list_missing_test[d][index_r])
                        Indexoutput_d_r.append(np.ones_like(self.Y_list_missing_test[d][index_r]) * r)
                    x_all_test.append(np.vstack(x_output_d_r))
                    y_all_test.append(np.vstack(y_output_d_r))
                    Indexoutput_test.append(np.vstack(Indexoutput_d_r))
                Indexoutput_test = np.vstack(Indexoutput_test)
                x_all_test = np.vstack(x_all_test)
                x_input_all_index = np.hstack((x_all_test, Indexoutput_test))
                y_all_test = np.vstack(y_all_test)
                idx_test = [x_input_all_index[:, -1] == i for i in range(self.num_replicates)]
                R = []
                for i in range(self.num_replicates):
                    if True in idx_test[i]:
                        R.append(i)

                ## Prediction
                if self.Model_name == 'LMC3':
                    mean_test, var_test = m_test.predict_f(x_input_all_index)
                    mean_test, var_test = mean_test.numpy(), var_test.numpy()
                elif self.Model_name == 'LVMOGP3':
                    mean_test, var_test = m_test.predict(x_all_test)


                for r in R:
                    if self.Model_name == 'LMC3':
                        mean_one_missing = mean_test[idx_test[r]]
                        variance_one_missing = var_test[idx_test[r]] + m_test.likelihood.likelihoods[r].variance.numpy()
                    elif self.Model_name == 'LVMOGP3':
                        mean_one_missing = mean_test[idx_test[r], r][:, None]
                        variance_one_missing = var_test[idx_test[r], r][:, None]
                    Y_one_missing = y_all_test[idx_test[r]]
                    # Evalutaion metric
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

            else:
                for d in range(self.D):
                    if self.Model_name == 'HMOGPLV':
                        mean_missing, variance_missing = m_test.predict_f(self.X_list_missing_test[d])
                        mean_missing, variance_missing = mean_missing.numpy(), variance_missing.numpy() + m_test.Heter_GaussianNoise.numpy()
                        mean_one_missing = mean_missing[:, d][:, None]
                        variance_one_missing = variance_missing[:, d][:, None]
                    elif self.Model_name == 'HGPInd':
                        mean_missing, variance_missing = m_test.predict_y(self.X_list_missing_test[d])
                        mean_one_missing, variance_one_missing = mean_missing.numpy(), variance_missing.numpy()
                    elif self.Model_name == 'LMC':
                        X_pre = np.c_[self.X_list_missing_test[d][:, :-1], np.ones_like(self.X_list_missing_test[d][:, :-1]) * d]

                        mean_missing, variance_missing = m_test.predict_f(X_pre)
                        mean_one_missing = mean_missing.numpy()
                        variance_one_missing = variance_missing.numpy() + m_test.likelihood.likelihoods[d].variance.numpy()
                    elif self.Model_name == 'LMCsum':
                        X_pre = np.c_[self.X_list_missing_test[d][:, :-1], np.ones_like(self.X_list_missing_test[d][:, :-1]) * d]
                        mean_missing, variance_missing = m_test.predict_f(X_pre)
                        mean_one_missing = mean_missing.numpy()
                        variance_one_missing = variance_missing.numpy() + m_test.likelihood.likelihoods[d].variance.numpy()
                    elif self.Model_name == 'LVMOGP':
                        mean_test, variance_test = m_test.predict(self.X_list_missing_test[d][:, :-1])
                        mean_one_missing = mean_test[:, d][:, None]
                        variance_one_missing = variance_test[:, d][:, None]
                    elif self.Model_name == 'HGP':
                        x_test = np.hstack((self.X_list_missing_test[d][:, :-1], self.X_list_missing_test[d][:, -1][:, None] + 1))
                        mean_one_missing, variance_one_missing = m_test.predict(x_test)
                    elif self.Model_name == 'DHGP':

                        x_test = np.hstack((self.X_list_missing_test[d][:, :-1], self.X_list_missing_test[d][:, -1][:, None] * 0 + d,
                                            self.X_list_missing_test[d][:, -1][:, None] * 0))


                        mean_one_missing, variance_one_missing = m_test.predict(x_test)

                    Y_one_missing = self.Y_list_missing_test[d]
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
        elif self.Experiment_type == 'Missing_Triangle_1':
            for d in range(self.D):
                if d == self.D-1:
                    if self.Model_name == 'HMOGPLV':
                        mean_missing, variance_missing = m_test.predict_f(self.X_list_missing_test[0])
                        mean_missing, variance_missing = mean_missing.numpy(), variance_missing.numpy() + m_test.Heter_GaussianNoise.numpy()
                        mean_one_missing = mean_missing[:, d][:, None]
                        variance_one_missing = variance_missing[:, d][:, None]

                    Y_one_missing = self.Y_list_missing_test[0]
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

        elif 'Missing_Quad' in self.Experiment_type:
            import re
            s = self.Experiment_type
            d_match = re.search(r'd(\d+)', s)
            d_number = int(d_match.group(1)) if d_match else 0
            d_int = list(range(0, self.D))
            d_int = d_int[-d_number:]

            for d in range(d_number):
                if self.Model_name == 'HMOGPLV':
                    mean_missing, variance_missing = m_test.predict_f(self.X_list_missing_test[d])
                    mean_missing, variance_missing = mean_missing.numpy(), variance_missing.numpy() + m_test.Heter_GaussianNoise.numpy()
                    mean_one_missing = mean_missing[:, d][:, None]
                    variance_one_missing = variance_missing[:, d][:, None]
                elif self.Model_name == 'DHGP':

                    x_test = np.hstack((self.X_list_missing_test[d][:, :-1], self.X_list_missing_test[d][:, -1][:, None] * 0 + d,
                                        self.X_list_missing_test[d][:, -1][:, None] * 0))


                    mean_one_missing, variance_one_missing = m_test.predict(x_test)
                Y_one_missing = self.Y_list_missing_test[d]
                Global_mean_pre.append(mean_one_missing)
                Global_variance_pre.append(variance_one_missing)
                Global_test_pre.append(Y_one_missing)
                NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

        elif 'active' in self.Experiment_type:
            for d in range(self.D):
                if self.Model_name == 'HMOGPLV':
                    mean_missing, variance_missing = m_test.predict_f(self.X_list_missing_test[d])
                    mean_missing, variance_missing = mean_missing.numpy(), variance_missing.numpy() + m_test.Heter_GaussianNoise.numpy()
                    mean_one_missing = mean_missing[:, d][:, None]
                    variance_one_missing = variance_missing[:, d][:, None]

                elif self.Model_name == 'DHGP':

                    x_test = np.hstack((self.X_list_missing_test[d][:, :-1], self.X_list_missing_test[d][:, -1][:, None] * 0 + d,
                                        self.X_list_missing_test[d][:, -1][:, None] * 0))


                    mean_one_missing, variance_one_missing = m_test.predict(x_test)

                Y_one_missing = self.Y_list_missing_test[d]
                if Y_one_missing.size == 0:
                    pass
                else:
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)

                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
        elif 'Missing_Tri_' in self.Experiment_type:
            import re
            s = self.Experiment_type
            d_match = re.search(r'd(\d+)', s)
            d_number = int(d_match.group(1)) if d_match else 0
            # r_number = d_number
            # rep_int = list(range(0, self.num_replicates))
            # rep_int = rep_int[-r_number:]
            d_int = list(range(0, self.D))
            d_int = d_int[-d_number:]
            rep_int = [len(d_int)]

            for d in range(d_number):
                if self.Model_name == 'HMOGPLV':
                    mean_missing, variance_missing = m_test.predict_f(self.X_list_missing_test[d])
                    mean_missing, variance_missing = mean_missing.numpy(), variance_missing.numpy() + m_test.Heter_GaussianNoise.numpy()
                    mean_one_missing = mean_missing[:, d][:, None]
                    variance_one_missing = variance_missing[:, d][:, None]
                elif self.Model_name == 'DHGP':
                    x_test = np.hstack(
                        (self.X_list_missing_test[d][:, :-1], self.X_list_missing_test[d][:, -1][:, None] * 0 + d,
                         self.X_list_missing_test[d][:, -1][:, None] * 0))

                    mean_one_missing, variance_one_missing = m_test.predict(x_test)

                Y_one_missing = self.Y_list_missing_test[d]
                Global_mean_pre.append(mean_one_missing)
                Global_variance_pre.append(variance_one_missing)
                Global_test_pre.append(Y_one_missing)
                NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
        elif self.Experiment_type == 'Missing_part_of_one_output_in_Whole' or 'Extrapolate' in self.Experiment_type:
            if self.Model_name == 'HMOGPLV':
                d = self.D-1
                idx_test_d = [self.X_list_missing_test[0][:, -1] == i for i in range(self.num_replicates)]
                mean_test, variance_test = m_test.predict_f(self.X_list_missing_test[0])
                mean_test, variance_test = mean_test.numpy(), variance_test.numpy() + m_test.Heter_GaussianNoise.numpy()
                for r_test in range(self.num_replicates):
                    mean_one_missing = mean_test[idx_test_d[r_test], d][:, None]
                    variance_one_missing = variance_test[idx_test_d[r_test], d][:, None]
                    if len(variance_one_missing) == 0:
                        continue
                    Y_one_missing = self.Y_list_missing_test[0][idx_test_d[r_test]]
                    # Evalutaion metric
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
            elif self.Model_name == 'DHGP':
                d = self.D-1
                idx_test_d = [self.X_list_missing_test[0][:, -1] == i for i in range(self.num_replicates)]
                x_test = np.hstack(
                    (self.X_list_missing_test[0][:, :-1], self.X_list_missing_test[0][:, -1][:, None] * 0 + 0,
                self.X_list_missing_test[0][:, -1][:, None] * 0))
                mean_test, variance_test = m_test.predict(x_test)
                mean_test, variance_test = mean_test, variance_test

                for r_test in range(self.num_replicates):
                    mean_one_missing = mean_test[idx_test_d[r_test], 0][:, None]
                    variance_one_missing = variance_test[idx_test_d[r_test], 0][:, None]
                    if len(variance_one_missing) == 0:
                        continue
                    Y_one_missing = self.Y_list_missing_test[0][idx_test_d[r_test]]
                    # Evalutaion metric
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

            elif self.Model_name == 'LMCsum':

                d = self.D -1
                x_test_missing = self.X_list_missing_test[0]
                x_test_all = np.c_[x_test_missing[:,0][:,None], np.ones_like(x_test_missing[:, -1][:, None]) * d]
                y_test_all = self.Y_list_missing_test[0]
                ## Prediction
                mean_test, var_test = m_test.predict_f(x_test_all)
                mean_test, var_test = mean_test.numpy(), var_test.numpy()

                mean_one_missing = mean_test
                variance_one_missing = var_test + m_test.likelihood.likelihoods[d].variance.numpy()
                Y_one_missing = y_test_all

                Global_mean_pre.append(mean_one_missing)
                Global_variance_pre.append(variance_one_missing)
                Global_test_pre.append(Y_one_missing)
                NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

        elif self.Experiment_type == 'Train_test_in_each_replica':
            if self.Model_name == 'HMOGPLV':
                for d in range(self.D):
                    idx_test_d = [self.X_list_missing_test[d][:, -1] == i for i in range(self.num_replicates)]
                    mean_test, variance_test = m_test.predict_f(self.X_list_missing_test[d])
                    mean_test, variance_test = mean_test.numpy(), variance_test.numpy() + m_test.Heter_GaussianNoise.numpy()
                    NMSE_GUSTO_pred = []
                    NMSE_GUSTO_real = []

                    for r_test in range(self.num_replicates):
                        mean_one_missing = mean_test[idx_test_d[r_test], d][:, None]
                        variance_one_missing = variance_test[idx_test_d[r_test], d][:, None]
                        if len(variance_one_missing) == 0:
                            continue
                        Y_one_missing = self.Y_list_missing_test[d][idx_test_d[r_test]]
                        # Evalutaion metric
                        Global_mean_pre.append(mean_one_missing)
                        Global_variance_pre.append(variance_one_missing)
                        Global_test_pre.append(Y_one_missing)
                        if self.Data_name == 'GUSTO':
                            NMSE_GUSTO_real.append(Y_one_missing)
                            NMSE_GUSTO_pred.append(mean_one_missing)
                            # NMSE_test_bar_missing_replicates.append(MSE_test_bar_normall(Y_one_missing, mean_one_missing))
                        else:
                            NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                        MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
                    if self.Data_name == 'GUSTO':
                        NMSE_GUSTO_real_d_output = np.vstack(NMSE_GUSTO_real)
                        NMSE_GUSTO_pred_d_output = np.vstack(NMSE_GUSTO_pred)
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar(NMSE_GUSTO_real_d_output, NMSE_GUSTO_pred_d_output))


            elif self.Model_name == 'LVMOGP':
                for d in range(self.D):
                    NMSE_GUSTO_pred = []
                    NMSE_GUSTO_real = []
                    idx_test_d = [self.X_list_missing_test[d][:, -1] == i for i in range(self.num_replicates)]
                    mean_test, variance_test = m_test.predict(self.X_list_missing_test[d][:, :-1])
                    for r_test in range(self.num_replicates):
                        mean_one_missing = mean_test[idx_test_d[r_test], d][:, None]
                        variance_one_missing = variance_test[idx_test_d[r_test], d][:, None]
                        if len(variance_one_missing) == 0:
                            continue
                        Y_one_missing = self.Y_list_missing_test[d][idx_test_d[r_test]]
                        # Evalutaion metric
                        Global_mean_pre.append(mean_one_missing)
                        Global_variance_pre.append(variance_one_missing)
                        Global_test_pre.append(Y_one_missing)
                        if self.Data_name == 'GUSTO':
                            NMSE_GUSTO_real.append(Y_one_missing)
                            NMSE_GUSTO_pred.append(mean_one_missing)
                            # NMSE_test_bar_missing_replicates.append(MSE_test_bar_normall(Y_one_missing, mean_one_missing))
                        else:
                            NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                        MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
                    if self.Data_name == 'GUSTO':
                        NMSE_GUSTO_real_d_output = np.vstack(NMSE_GUSTO_real)
                        NMSE_GUSTO_pred_d_output = np.vstack(NMSE_GUSTO_pred)
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar(NMSE_GUSTO_real_d_output, NMSE_GUSTO_pred_d_output))
            elif self.Model_name == 'LVMOGP2':
                for d in range(self.D):
                    idx_test_d = [self.X_list_missing_test[d][:, -1] == i for i in range(self.num_replicates)]
                    mean_test, variance_test = m_test.predict(self.X_list_missing_test[d][:, :-1])
                    mean_one_missing = mean_test[idx_test_d[self.test_id], d][:, None]
                    variance_one_missing = variance_test[idx_test_d[self.test_id], d][:, None]
                    Y_one_missing = self.Y_list_missing_test[d][idx_test_d[self.test_id]]
                    # Evalutaion metric
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
            elif self.Model_name == 'DHGP':
                for d in range(self.D):
                    NMSE_GUSTO_pred = []
                    NMSE_GUSTO_real = []
                    idx_test_d = [self.X_list_missing_test[d][:, -1] == i for i in range(self.num_replicates)]
                    Indexoutput_d = np.ones_like(self.Y_list_missing_test[d]) * d
                    Index_replica_d = self.X_list_missing_test[d][:, -1][:, None] + 1 + d * self.num_replicates
                    x_all_d = np.hstack((np.vstack(self.X_list_missing_test[d])[:, :-1], Indexoutput_d))
                    X_d = np.hstack((x_all_d, Index_replica_d))
                    mean_test, variance_test = m_test.predict(X_d)
                    for r_test in range(self.num_replicates):
                        mean_one_missing = mean_test[idx_test_d[r_test]]
                        variance_one_missing = variance_test[idx_test_d[r_test]]
                        if len(variance_one_missing) == 0:
                            continue
                        Y_one_missing = self.Y_list_missing_test[d][idx_test_d[r_test]]
                        # Evalutaion metric
                        Global_mean_pre.append(mean_one_missing)
                        Global_variance_pre.append(variance_one_missing)
                        Global_test_pre.append(Y_one_missing)
                        if self.Data_name == 'GUSTO':
                            NMSE_GUSTO_real.append(Y_one_missing)
                            NMSE_GUSTO_pred.append(mean_one_missing)
                            # NMSE_test_bar_missing_replicates.append(MSE_test_bar_normall(Y_one_missing, mean_one_missing))
                        else:
                            NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                        MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
                    if self.Data_name == 'GUSTO':
                        NMSE_GUSTO_real_d_output = np.vstack(NMSE_GUSTO_real)
                        NMSE_GUSTO_pred_d_output = np.vstack(NMSE_GUSTO_pred)
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar(NMSE_GUSTO_real_d_output, NMSE_GUSTO_pred_d_output))
            elif self.Model_name == 'LMCsum':
                idx_train, idx_test, x_train_all, x_test_all, y_train_all, y_test_all = LMC_data_set(self.D, self.Y_list_missing, self.X_list_missing,
                                                                                                     self.num_replicates, self.Y_list_missing_test,
                                                                                                     self.X_list_missing_test)
                ## Prediction
                mean_test, var_test = m_test.predict_f(x_test_all)
                mean_test, var_test = mean_test.numpy(), var_test.numpy()
                for d in range(self.D):
                    NMSE_GUSTO_pred = []
                    NMSE_GUSTO_real = []
                    for r in range(self.num_replicates):
                        mean_one_missing = mean_test[idx_test[r + d * self.num_replicates]]
                        variance_one_missing = var_test[idx_test[r + d * self.num_replicates]] + m_test.likelihood.likelihoods[d].variance.numpy()
                        Y_one_missing = y_test_all[:, :-1][idx_test[r + d * self.num_replicates]]
                        if len(variance_one_missing) == 0:
                            continue
                        # Evalutaion metric
                        Global_mean_pre.append(mean_one_missing)
                        Global_variance_pre.append(variance_one_missing)
                        Global_test_pre.append(Y_one_missing)
                        if self.Data_name == 'GUSTO':
                            NMSE_GUSTO_real.append(Y_one_missing)
                            NMSE_GUSTO_pred.append(mean_one_missing)
                            # NMSE_test_bar_missing_replicates.append(MSE_test_bar_normall(Y_one_missing, mean_one_missing))
                        else:
                            NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                        MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
                    if self.Data_name == 'GUSTO':
                        NMSE_GUSTO_real_d_output = np.vstack(NMSE_GUSTO_real)
                        NMSE_GUSTO_pred_d_output = np.vstack(NMSE_GUSTO_pred)
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar(NMSE_GUSTO_real_d_output, NMSE_GUSTO_pred_d_output))

            elif self.Model_name == 'LMC2':
                Indexoutput_test = []
                x_test_all = []
                y_test_all = []
                for d in range(self.D):
                    index_r = self.X_list_missing_test[d][:, -1] == self.test_id
                    x_test_all.append(self.X_list_missing_test[d][index_r][:, :-1])
                    y_test_all.append(self.Y_list_missing_test[d][index_r])
                    Indexoutput_test.append(np.ones_like(self.Y_list_missing_test[d][index_r]) * d)
                Indexoutput_test = np.vstack(Indexoutput_test)
                x_test_all = np.vstack(x_test_all)
                y_test_all = np.vstack(y_test_all)
                x_test_all_index = np.hstack((x_test_all, Indexoutput_test))
                ## Prediction
                mean_test, var_test = m_test.predict_f(x_test_all_index)
                mean_test, var_test = mean_test.numpy(), var_test.numpy()
                idx_test = [x_test_all_index[:, -1] == i for i in range(self.D)]
                for d in range(self.D):
                    mean_one_missing = mean_test[idx_test[d]]
                    variance_one_missing = var_test[idx_test[d]] + m_test.likelihood.likelihoods[d].variance.numpy()
                    Y_one_missing = y_test_all[idx_test[d]]
                    # Evalutaion metric
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
            elif self.Model_name == 'LMCsum':
                x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre = LMCsum_data(self.X_list_missing,
                                                                                               self.Y_list_missing,
                                                                                               self.X_list_missing_test,
                                                                                               self.Y_list_missing_test,
                                                                                               self.num_replicates,
                                                                                               self.D)
                ## Prediction
                for r in range(self.num_replicates):
                    x_test_all = x_test_all_pre[r]
                    y_test_all = y_test_all_pre[r]

                    mean_test, var_test = m_test.predict_f(x_test_all)
                    mean_test, var_test = mean_test.numpy(), var_test.numpy()
                    for d in range(self.D):
                        index_replica = x_test_all[:, -1][:, None] == d
                        mean_one_missing = mean_test[index_replica.squeeze()]
                        variance_one_missing = var_test[index_replica.squeeze()] + m_test.likelihood.likelihoods[d].variance.numpy()
                        Y_one_missing = y_test_all[:, :-1][index_replica.squeeze()]
                        if len(variance_one_missing) == 0:
                            continue
                        # Evalutaion metric
                        Global_mean_pre.append(mean_one_missing)
                        Global_variance_pre.append(variance_one_missing)
                        Global_test_pre.append(Y_one_missing)
                        # if self.Data_name == 'GUSTO':
                        #     NMSE_test_bar_missing_replicates.append(MSE_test_bar_normall(Y_one_missing, mean_one_missing))
                        # else:
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                        MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))


            elif self.Model_name == 'HGPInd' or self.Model_name == 'HGP' or self.Model_name == 'SGP' or self.Model_name == 'DNN' or self.Model_name == 'LMC3' or self.Model_name == 'LVMOGP3':
                ## Finding the index for each replicate
                idx_test_d = [self.X_list_missing_test[:, -1] == i for i in range(self.num_replicates)]
                ## Prediction

                if self.Model_name == 'HGPInd':
                    mean_test, variance_test = m_test.predict_y(self.X_list_missing_test)
                    mean_test, variance_test = mean_test.numpy(), variance_test.numpy()
                elif self.Model_name == 'HGP':
                    x_test = np.hstack((self.X_list_missing_test[:, :-1], self.X_list_missing_test[:, -1][:, None] + 1))
                    mean_test, variance_test = m_test.predict(x_test)
                elif self.Model_name == 'SGP':
                    x_test = np.vstack(self.X_list_missing_test[:, :-1])
                    mean_test, variance_test = m_test.predict(x_test)
                elif self.Model_name == 'DNN':
                    x_test = np.vstack(self.X_list_missing_test[:, :-1])
                    mean_test = m_test.predict(x_test)
                    variance_test = abs(mean_test)
                elif self.Model_name == 'LMC3':
                    mean_test, variance_test = m_test.predict_f(self.X_list_missing_test)
                    mean_test, variance_test = mean_test.numpy(), variance_test.numpy()
                elif self.Model_name == 'LVMOGP3':
                    x_test = np.vstack(self.X_list_missing_test[:, :-1])
                    mean_test, variance_test = m_test.predict(x_test)
                NMSE_GUSTO_pred = []
                NMSE_GUSTO_real = []
                for r_test in range(self.num_replicates):
                    ## This the mean prediction for the output with the replicate
                    if self.Model_name == 'LMC3':
                        mean_one_missing = mean_test[idx_test_d[r_test]]
                        variance_one_missing = variance_test[idx_test_d[r_test]] + m_test.likelihood.likelihoods[r_test].variance.numpy()
                    elif self.Model_name == 'LVMOGP3':
                        mean_one_missing = mean_test[idx_test_d[r_test], r_test][:, None]
                        variance_one_missing = variance_test[idx_test_d[r_test], r_test][:, None]
                    else:
                        mean_one_missing = mean_test[idx_test_d[r_test]]
                        variance_one_missing = variance_test[idx_test_d[r_test]]
                    Y_one_missing = self.Y_list_missing_test[idx_test_d[r_test]]
                    # Evalutaion metric
                    if len(variance_one_missing) == 0:
                        continue
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    if self.Data_name == 'GUSTO':
                        NMSE_GUSTO_real.append(Y_one_missing)
                        NMSE_GUSTO_pred.append(mean_one_missing)
                        # NMSE_test_bar_missing_replicates.append(MSE_test_bar_normall(Y_one_missing, mean_one_missing))
                    else:
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    # NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
                if self.Data_name == 'GUSTO':
                    NMSE_GUSTO_real_d_output = np.vstack(NMSE_GUSTO_real)
                    NMSE_GUSTO_pred_d_output = np.vstack(NMSE_GUSTO_pred)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(NMSE_GUSTO_real_d_output, NMSE_GUSTO_pred_d_output))
            elif self.Model_name == 'SGP2':
                x_test = self.X_list_missing_test[:, :-1]
                mean_one_missing, variance_one_missing = m_test.predict(x_test)
                Y_one_missing = self.Y_list_missing_test
                Global_mean_pre.append(mean_one_missing)
                Global_variance_pre.append(variance_one_missing)
                Global_test_pre.append(Y_one_missing)
                NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

        if len(Global_mean_pre) == 0:
            Global_mean = 0
            Global_variance = 0
            Global_test = 0
            Glo_NMSE_test = 0
            Glo_MNLP = 0
        else:
            Global_mean = np.vstack(Global_mean_pre)
            Global_variance = np.vstack(Global_variance_pre)
            Global_test = np.vstack(Global_test_pre)
            Glo_NMSE_test = NMSE_test_bar(Global_test, Global_mean)
            Glo_MNLP = MNLP(Global_test, Global_mean, Global_variance)
        return Glo_NMSE_test, Glo_MNLP, NMSE_test_bar_missing_replicates, MNLP_missing_replicates

    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def mse(self, y_true, y_pred):
        return (np.mean((y_true - y_pred) ** 2))

    def mae(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred)))

    def save_result(self, Glo_NMSE_test_all, Glo_MNLP_all, NMSE_test_bar_missing_replicates_all, MNLP_missing_replicates_all,Time_all):
        '''
        In this function, we want to save all the evaluation metric into a folder
        '''
        Glo_NMSE_test_mean = np.mean(Glo_NMSE_test_all)
        Glo_NMSE_test_std = np.std(Glo_NMSE_test_all)
        Glo_MNLP_mean = np.mean(Glo_MNLP_all)
        Glo_MNLP_std = np.std(Glo_MNLP_all)
        Mean_missing_NMSE_test_bar = np.mean(NMSE_test_bar_missing_replicates_all)
        Std_missing_NMSE_test_bar = np.std(NMSE_test_bar_missing_replicates_all)
        Mean_missing_MNLP = np.mean(MNLP_missing_replicates_all)
        Std_missing_MNLP = np.std(MNLP_missing_replicates_all)

        Result_performance_measure = pd.DataFrame({'Model': [self.Model_name],
                                                   'Total time': [Time_all],
                                                   'Mean of NMSE_test_bar for missing replicates': [
                                                       Mean_missing_NMSE_test_bar],
                                                   'Std of NMSE_test_bar for missing replicates': [
                                                       Std_missing_NMSE_test_bar],
                                                   'Mean of MNLP for missing replicates': [Mean_missing_MNLP],
                                                   'Std of MNLP for missing replicates': [Std_missing_MNLP],
                                                   'Mean of Global NMSE_test for missing replicates': [
                                                       Glo_NMSE_test_mean],
                                                   'Std of Global NMSE_test for missing replicates': [
                                                       Glo_NMSE_test_std],
                                                   'Mean of Global MNLP for missing replicates': [Glo_MNLP_mean],
                                                   'Std of Global MNLP for missing replicates': [Glo_MNLP_std], })

        ## Save our result
        newpath_result = self.my_path + self.Our_path_result
        ## We make a floder
        if not os.path.exists(newpath_result):
            os.makedirs(newpath_result)
        Result_name = self.Experiment_type + self.Model_name + self.Data_name + '-%i-outputs' % self.D + '%i-replicates' % self.num_replicates + '-%i-Q' % self.Q + '-%f-train_percentage' % self.train_percentage + \
                      '-%i-num_data_each_replicate' % self.num_data_each_replicate + '-%i-gap' % self.gap + '-%i-Num_repetition' % self.Num_repetition + '-%i-Num_Inducing_outputs' % self.Mr + '.csv'
        Result_performance_measure.to_csv(newpath_result + Result_name)

    def set_up_HMOGPLV(self):
        '''
        Here we set up HMOGPLV model.
        '''
        ## data set

        Indexoutput = []
        for d in range(self.D):
            Indexoutput.append(np.ones_like(self.Y_list_missing[d]) * d)
        Indexoutput = np.vstack(Indexoutput)
        x_all = np.vstack(self.X_list_missing)

        x_input_all_index = tf.cast(np.hstack((x_all, Indexoutput)), dtype=tf.float64)
        Y_all = np.vstack(self.Y_list_missing)
        indexD = x_input_all_index[..., -1].numpy()
        data_set = (x_input_all_index, Y_all)

        ## model
        lengthscales2 = tf.convert_to_tensor([1.0] * self.Q, dtype=default_float())
        kernel_row = gpflow.kernels.RBF(lengthscales=lengthscales2)
        total_replicated = self.num_replicates
        # kern_upper_R = gpflow.kernels.RBF()
        # kern_lower_R = gpflow.kernels.Matern52()
        kern_upper_R = gpflow.kernels.Matern32()
        kern_lower_R = gpflow.kernels.Matern32()
        k_hierarchy_outputs = HMOGPLV.kernels.Hierarchial_kernel_replicated(kernel_g=kern_upper_R,
                                                                            kernel_f=[kern_lower_R],
                                                                            total_replicated=total_replicated)
        # if self.Data_name == 'GUSTO':
            # Z = x_all[:3*self.num_replicates]
            # Z = x_all[::self.gap].copy()
        # else:

        Z = x_all[::self.gap].copy()
        Mc = Z.shape[0]
        Xr_dim = self.Q
        Mr = self.Mr
        Heter_GaussianNoise = np.full(self.D, 1)
        m_test = MODEL.HMOGP_prior_outputs_kronecker_product_Missing_speed_up(kernel=k_hierarchy_outputs,
                                                                              kernel_row=kernel_row,
                                                                              Heter_GaussianNoise=Heter_GaussianNoise,
                                                                              Xr_dim=Xr_dim,
                                                                              Z=Z,
                                                                              num_inducing=(Mc, Mr), indexD=indexD,
                                                                              Initial_parameter='GP',
                                                                              variance_lowerbound=self.variance_lower,
                                                                              x_all=x_all, y_all=Y_all)
        return m_test, data_set, x_input_all_index

    def set_up_HGPInd(self):
        '''
        Here we set up the HGPInd model.
        '''
        ## data set
        x_all = np.vstack(self.X_list_missing)
        Y_all = np.vstack(self.Y_list_missing)
        ## model
        Z = x_all[::self.gap].copy()
        Total_num_replicates = self.num_replicates
        kern_upper_HGP_Inducing = gpflow.kernels.Matern32()
        kern_lower_HGP_Inducing = gpflow.kernels.Matern32()
        k_hierarchy = HMOGPLV.kernels.Hierarchial_kernel_replicated(kernel_g=kern_upper_HGP_Inducing,
                                                                    kernel_f=[kern_lower_HGP_Inducing],
                                                                    total_replicated=Total_num_replicates)
        m_test = SHGP_replicated_within_data(kernel=k_hierarchy, inducing_variable=Z, data=(x_all, Y_all), mean_function=None)
        return m_test, 0, 0

    def set_up_HGP(self):
        '''
        Here we set up the HGP model.
        '''

        ## data set
        if self.Experiment_type == 'Train_test_in_each_replica':
            x_all = np.hstack((self.X_list_missing[:, :-1], self.X_list_missing[:, -1][:, None]+1))
        elif self.Experiment_type == 'Missing_One_replica_in_each_output':
            xx = []
            for d in range(self.D):
                xx.append(np.hstack((self.X_list_missing[d][:, :-1], self.X_list_missing[d][:, -1][:, None]+1)))
            x_all = np.vstack(xx)
        elif self.Experiment_type == 'Missing_Tri_d1':
            xx = []
            for d in range(self.D):
                xx.append(np.hstack((self.X_list_missing[d][:, :-1], self.X_list_missing[d][:, -1][:, None]+1)))
            x_all = np.vstack(xx)
        Y_all = np.vstack(self.Y_list_missing)

        ## model
        kern_upper = GPy.kern.Matern32(input_dim=1, active_dims=[0], name='upper')
        kern_lower = GPy.kern.Matern32(input_dim=1, active_dims=[0], name='lower')
        k_hierarchy = GPy.kern.Hierarchical(kernels=[kern_upper, kern_lower])
        m_test = GPy.models.GPRegression(X=x_all, Y=Y_all, kernel=k_hierarchy)
        return m_test, 0, 0

    def set_up_DNN(self):
        '''
        Here we set up the HGP model.
        '''

        ## data set
        x_all = self.X_list_missing[:, :-1]

        Y_all = np.vstack(self.Y_list_missing).squeeze()

        ## model
        m_test = keras.Sequential([
            layers.Dense(200, activation='relu', input_shape=[x_all.shape[1]]),
            layers.Dense(200, activation='relu'),
            layers.Dense(1)
        ])

        return m_test, x_all , Y_all

    def set_up_SGP(self):
        '''
        Here we set up the SGP model where we consider all replicas in the same output as one output.
        '''
        ## data set
        x_all = np.vstack(self.X_list_missing[:, :-1])
        Y_all = np.vstack(self.Y_list_missing)
        ## model
        k = GPy.kern.RBF(input_dim=1, active_dims=[0], name="rbf")
        m_test = GPy.models.GPRegression(X=x_all, Y=Y_all, kernel=k)
        return m_test, 0, 0

    def set_up_SGP2(self):
        '''
        Here we set up the SGP model where we consider all replicas in the same output as one output.
        '''
        ## data set
        x_all = self.X_list_missing[:, :-1]
        Y_all = self.Y_list_missing
        ## model
        k = GPy.kern.RBF(input_dim=1, active_dims=[0], name="rbf")
        m_test = GPy.models.GPRegression(X=x_all, Y=Y_all, kernel=k)
        return m_test, 0, 0

    def set_up_DHGP(self):
        '''
        Here we set up the DHGP model.
        '''
        # data set
        Indexoutput = []
        Index_replica = []
        for d in range(self.D):
            Indexoutput.append(np.ones_like(self.Y_list_missing[d]) * d)
            Index_replica.append(self.X_list_missing[d][:, -1][:, None] + 1 + d*self.num_replicates)
        Indexoutput = np.vstack(Indexoutput)
        if self.Experiment_type == 'Missing_One_replica_in_each_output':
            Index_replica = np.vstack(Index_replica)
            total_index_replica = self.D * self.num_replicates
            j = 1
            for i in range(total_index_replica+1):
                if sum(Index_replica == i):
                    Index_replica[Index_replica == i] = j
                    j += 1
        else:
            Index_replica = np.vstack(Index_replica)


        x_all = np.hstack((np.vstack(self.X_list_missing)[:, :-1], Indexoutput))
        X = np.hstack((x_all, Index_replica))
        Y = np.vstack(self.Y_list_missing)
        # model
        k_cluster = GPy.kern.Matern32(input_dim=1, active_dims=[0], name='cluster')
        k_gene = GPy.kern.Matern32(input_dim=1, active_dims=[0], name='gene')
        k_replicate = GPy.kern.Matern32(input_dim=1, active_dims=[0], name='replicate')
        k_hierarchy = GPy.kern.Hierarchical([k_cluster, k_gene, k_replicate])
        m_test = GPy.models.GPRegression(X, Y, kernel=k_hierarchy)
        return m_test, 0, 0

    def set_up_LMC(self):
        '''
        Here we set up the LMC model where we consider all replicas in the same output as one output.
        '''

        # data set
        if self.Experiment_type == 'Train_test_in_each_replica':
            _, _, x_input_all_index, _, Y_all, _ = LMC_data_set(self.D, self.Y_list_missing, self.X_list_missing, self.num_replicates,
                                                                self.Y_list_missing_test, self.X_list_missing_test)
        else:
            Indexoutput = []
            for d in range(self.D):
                Indexoutput.append(np.ones_like(self.Y_list_missing[d]) * d)
            Indexoutput = np.vstack(Indexoutput)

            x_all = np.vstack(self.X_list_missing)[:, :-1]
            x_input_all_index = np.hstack((x_all, Indexoutput))
            Y_all = np.hstack((np.vstack(self.Y_list_missing), Indexoutput))
        data_set = (x_input_all_index, Y_all)

        # model
        Z = x_input_all_index[::self.gap, 0][:, None].copy()
        ks = [gpflow.kernels.RBF() for _ in range(self.Num_ker)]
        L = len(ks)
        Zs = [Z.copy() for _ in range(L)]
        iv_list = [InducingPoints(Z) for Z in Zs]
        iv = SeparateIndependentInducingVariables(iv_list)
        N = x_input_all_index.shape[0]
        likelihood_gp = [gpflow.likelihoods.Gaussian() for _ in range(self.D)]
        kern = lmc_kernel(self.D, ks)
        lik = gpflow.likelihoods.SwitchedLikelihood(likelihood_gp)
        m_test = SVGP_MOGP(kernel=kern, likelihood=lik, inducing_variable=iv, num_latent_gps=L, num_data=N)

        return m_test, data_set, x_input_all_index

    def set_up_LMC2(self):
        '''
        Here we set up the LMC2 model where we consider each replica as each output.
        '''
        # data set
        Indexoutput = []
        x_all = []
        y_all = []
        if self.Model_name == 'LMC2':
            for d in range(self.D):
                index_r = self.X_list_missing[d][:, -1] == self.test_id
                x_all.append(self.X_list_missing[d][index_r][:, :-1])
                y_all.append(self.Y_list_missing[d][index_r])
                Indexoutput.append(np.ones_like(self.Y_list_missing[d][index_r]) * d)
        elif self.Model_name == 'LMC3' and self.Experiment_type == 'Train_test_in_each_replica':
            for r in range(self.num_replicates):
                index_r = self.X_list_missing[:, -1] == r
                x_all.append(self.X_list_missing[index_r][:, :-1])
                y_all.append(self.Y_list_missing[index_r])
                Indexoutput.append(np.ones_like(self.Y_list_missing[index_r]) * r)
        elif self.Model_name == 'LMC3' and self.Experiment_type == 'Missing_One_replica_in_each_output':
            for r in range(self.num_replicates):
                x_output_d_r = []
                y_output_d_r = []
                Indexoutput_d_r = []
                for d in range(self.D):
                    index_r = self.X_list_missing[d][:, -1] == r
                    x_output_d_r.append(self.X_list_missing[d][index_r][:, :-1])
                    y_output_d_r.append(self.Y_list_missing[d][index_r])
                    Indexoutput_d_r.append(np.ones_like(self.Y_list_missing[d][index_r]) * r)
                x_all.append(np.vstack(x_output_d_r))
                y_all.append(np.vstack(y_output_d_r))
                Indexoutput.append(np.vstack(Indexoutput_d_r))

        Indexoutput = np.vstack(Indexoutput)
        x_all = np.vstack(x_all)
        x_input_all_index = np.hstack((x_all, Indexoutput))
        y_all = np.vstack(y_all)
        Y_all = np.hstack((y_all, Indexoutput))
        data_set = (x_input_all_index, Y_all)

        # model
        Z = x_input_all_index[::self.gap, 0][:, None].copy()
        ks = [gpflow.kernels.RBF() for _ in range(self.Num_ker)]
        L = len(ks)
        Zs = [Z.copy() for _ in range(L)]
        iv_list = [InducingPoints(Z) for Z in Zs]
        iv = SeparateIndependentInducingVariables(iv_list)
        N = x_input_all_index.shape[0]
        if self.Model_name == 'LMC3':
            likelihood_gp = [gpflow.likelihoods.Gaussian() for _ in range(self.num_replicates)]
            kern = lmc_kernel(self.num_replicates, ks)
        elif self.Model_name == 'LMC2':
            likelihood_gp = [gpflow.likelihoods.Gaussian() for _ in range(self.D)]
            kern = lmc_kernel(self.D, ks)
        lik = gpflow.likelihoods.SwitchedLikelihood(likelihood_gp)
        m_test = SVGP_MOGP(kernel=kern, likelihood=lik, inducing_variable=iv, num_latent_gps=L, num_data=N)
        return m_test, data_set, x_input_all_index


    def set_up_LMCsum(self):
        '''
        Here we set up the LMC2 model where we consider each replica as each output.
        '''


        if self.Experiment_type == 'Missing_part_of_one_output_in_Whole':
            x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre = LMCsum_data_Missing_part_of_one_output_in_Whole(self.X_list_missing,
                                                                                           self.Y_list_missing,
                                                                                           self.X_list_missing_test,
                                                                                           self.Y_list_missing_test,
                                                                                           self.num_replicates,
                                                                                           self.D)
            x_input_all_index, All_X_test, Y_all, All_Y_test = new_format_for_X_Y_Missing_part_of_one_output_in_Whole(x_train_all_pre, x_test_all_pre,
                                                                                  y_train_all_pre, y_test_all_pre,
                                                                                  self.num_replicates)

        else:
            # data set
            x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre = LMCsum_data(self.X_list_missing,
                                                                                           self.Y_list_missing,
                                                                                           self.X_list_missing_test,
                                                                                           self.Y_list_missing_test,
                                                                                           self.num_replicates,
                                                                                           self.D)
            #
            x_input_all_index, All_X_test, Y_all, All_Y_test = new_format_for_X_Y(x_train_all_pre, x_test_all_pre,y_train_all_pre, y_test_all_pre, self.num_replicates)




        data_set = (x_input_all_index, Y_all)

        # model
        Z = x_input_all_index[::self.gap, 0][:, None].copy()
        ks = [gpflow.kernels.RBF() for _ in range(self.Num_ker)]
        L = len(ks)
        Zs = [Z.copy() for _ in range(L)]
        iv_list = [InducingPoints(Z) for Z in Zs]
        iv = SeparateIndependentInducingVariables(iv_list)
        N = x_input_all_index.shape[0]
        likelihood_gp = [gpflow.likelihoods.Gaussian() for _ in range(self.D)]
        kern = lmc_kernel(self.D, ks)
        lik = gpflow.likelihoods.SwitchedLikelihood(likelihood_gp)
        m_test = SVGP_MOGP_sum(kernel=kern, likelihood=lik, inducing_variable=iv, num_latent_gps=L, num_data=N, num_replicates=self.num_replicates)



        return m_test, data_set, x_input_all_index



    def set_up_LVMOGP(self):
        '''
        Here we set up the LVMOGP model where we consider all replicas in the same output as one output.
        '''

        # data set
        Indexoutput = []
        for d in range(self.D):
            Indexoutput.append(np.ones_like(self.Y_list_missing[d]) * d)
        Indexoutput = np.vstack(Indexoutput)
        indexD = Indexoutput.squeeze()
        x_all = np.vstack(self.X_list_missing)[:, :-1]
        Y_all = np.vstack(self.Y_list_missing)
        data_set = (x_all, Y_all)
        Mc = x_all[::self.gap].shape[0]
        ## model
        m_test = GPy.models.GPMultioutRegressionMD(x_all, Y_all, indexD, Xr_dim=self.Q, kernel_row=GPy.kern.RBF(self.Q, ARD=True),
                                              num_inducing=(Mc, self.Mr), init='GP')
        return m_test, data_set, x_all

    def set_up_LVMOGP2(self):
        '''
        Here we set up the LVMOGP2 model where we consider each replica data as each output.
        '''

        # data set
        Indexoutput = []
        x_all = []
        y_all = []
        if self.Model_name == 'LVMOGP2':
            for d in range(self.D):
                index_r = self.X_list_missing[d][:, -1] == self.test_id
                x_all.append(self.X_list_missing[d][index_r][:, :-1])
                y_all.append(self.Y_list_missing[d][index_r])
                Indexoutput.append(np.ones_like(self.Y_list_missing[d][index_r]) * d)
        elif self.Model_name == 'LVMOGP3' and self.Experiment_type == 'Train_test_in_each_replica':
            for r in range(self.num_replicates):
                index_r = self.X_list_missing[:, -1] == r
                x_all.append(self.X_list_missing[index_r][:, :-1])
                y_all.append(self.Y_list_missing[index_r])
                Indexoutput.append(np.ones_like(self.Y_list_missing[index_r]) * r)
        elif self.Model_name == 'LVMOGP3' and self.Experiment_type == 'Missing_One_replica_in_each_output':
            for r in range(self.num_replicates):
                x_output_d_r = []
                y_output_d_r = []
                Indexoutput_d_r = []
                for d in range(self.D):
                    index_r = self.X_list_missing[d][:, -1] == r
                    x_output_d_r.append(self.X_list_missing[d][index_r][:, :-1])
                    y_output_d_r.append(self.Y_list_missing[d][index_r])
                    Indexoutput_d_r.append(np.ones_like(self.Y_list_missing[d][index_r]) * r)
                x_all.append(np.vstack(x_output_d_r))
                y_all.append(np.vstack(y_output_d_r))
                Indexoutput.append(np.vstack(Indexoutput_d_r))

        Indexoutput = np.vstack(Indexoutput)
        indexD = Indexoutput.squeeze()
        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
        data_set = (x_all, y_all)

        Mc = x_all[::self.gap].shape[0]
        ## model
        m_test = GPy.models.GPMultioutRegressionMD(x_all, y_all, indexD, Xr_dim=self.Q, kernel_row=GPy.kern.RBF(self.Q, ARD=True),
                                              num_inducing=(Mc, self.Mr), init='GP')
        return m_test, data_set, x_all


from gpflow.kernels.stationaries import Matern32
from gpflow.kernels import IsotropicStationary

class MaskedMatern32(Matern32):
    def __init__(self, variance=1.0, lengthscale=1.0, **kwargs):
        """
        A Matern 3/2 kernel with the diagonal always masked (set to zero).

        :param variance: Kernel variance.
        :param lengthscale: Kernel lengthscale.
        """
        super().__init__(**kwargs)

    def K(self, X, X2=None):
        K_base = super().K(X, X2)

        if X2 is None:  # Square covariance matrix
            #mask_diag = 1 - tf.eye(tf.shape(K_base)[0], dtype=K_base.dtype)
            return K_base #* mask_diag
        else:
            return K_base  # No masking for cross-covariances

    def K_diag(self, X):
        return tf.zeros(tf.shape(X)[0], dtype=tf.float64)  # Always mask diagonal

class FullyMaskedMatern32(Matern32):
    def __init__(self, variance=1.0, lengthscale=1.0, **kwargs):
        """
        A Matern 3/2 kernel where the entire covariance matrix is masked (set to zero).

        :param variance: Kernel variance.
        :param lengthscale: Kernel lengthscale.
        """
        super().__init__(**kwargs)

    def K(self, X, X2=None):
        """
        Returns a zero matrix of the same shape as the expected covariance matrix.
        """
        shape_X = tf.shape(X)[0]
        shape_X2 = shape_X if X2 is None else tf.shape(X2)[0]
        return tf.zeros((shape_X, shape_X2), dtype=tf.float64)  # Ensure correct shape

    def K_diag(self, X):
        """
        Returns a zero vector for the diagonal elements.
        """
        return tf.zeros(tf.shape(X)[0], dtype=tf.float64)