                                    #################################################
                                    ### We build our own utilize based on GPflow. ###
                                    #################################################

##### we import from gpflow and tensorflow #####
import os
import tensorflow as tf
from gpflow import default_float
from gpflow.utilities import to_default_float
from collections.abc import Iterable
from HMOGPLV import covariances
from gpflow.expectations import expectation
from HMOGPLV.amc_parser import parse_amc
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
from gpflow.ci_utils import ci_niter ## for the number of training
import gpflow
import random
from gpflow.utilities import parameter_dict
import HMOGPLV
import GPy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tf_keras as keras
os.environ["TF_USE_LEGACY_KERAS"] = "True"

                                    #### This is used in the MC
def ndiag_mc_updated(funcs, S: int, Fmu, Fvar, logspace: bool = False, epsilon=None, **Ys):
    """
    Computes N Gaussian expectation integrals of one or more functions
    using Monte Carlo samples. The Gaussians must be independent.

    :param funcs: the integrand(s):
        Callable or Iterable of Callables that operates elementwise
    :param S: number of Monte Carlo sampling points
    :param Fmu: array/tensor
    :param Fvar: array/tensor
    :param logspace: if True, funcs are the log-integrands and this calculates
        the log-expectation of exp(funcs)
    :param **Ys: arrays/tensors; deterministic arguments to be passed by name

    Fmu, Fvar, Ys should all have same shape, with overall size `N`
    :return: shape is the same as that of the first Fmu
    """

    ######################################################################################################################
    ############ I change the  N, D = Fmu.shape[0], Fvar.shape[1] to N, D = tf.shape(Fmu)[0], tf.shape(Fvar)[1] ##########
    ######################################################################################################################
    ############ Thus, it can compile to the tensorflow in static graph. E.g. we can use tf.function.  ###################
    ######################################################################################################################

    N, D = tf.shape(Fmu)[0], tf.shape(Fvar)[1]

    if epsilon is None:
        epsilon = tf.random.normal((S, N, D), dtype=default_float())

    mc_x = Fmu[None, :, :] + tf.sqrt(Fvar[None, :, :]) * epsilon
    mc_Xr = tf.reshape(mc_x, (S * N, D))

    for name, Y in Ys.items():
        D_out = Y.shape[1]
        # we can't rely on broadcasting and need tiling
        mc_Yr = tf.tile(Y[None, ...], [S, 1, 1])  # [S, N, D]_out
        Ys[name] = tf.reshape(mc_Yr, (S * N, D_out))  # S * [N, _]out

    def eval_func(func):
        feval = func(mc_Xr, **Ys)
        feval = tf.reshape(feval, (S, N, -1))
        if logspace:
            log_S = tf.math.log(to_default_float(S))
            return tf.reduce_logsumexp(feval, axis=0) - log_S  # [N, D]
        else:
            return tf.reduce_mean(feval, axis=0)

    if isinstance(funcs, Iterable):
        return [eval_func(f) for f in funcs]
    else:
        return eval_func(funcs)



                    #####################################################################################
                    ########################## For building model #######################################
                    #####################################################################################
def tdot_tensorflow(a):
    """
    This function is to calculate matrix product :aa^T
    """
    return tf.linalg.matmul(a, a, transpose_b=True)


def backsub_both_sides_tensorflow(L, X, transpose='left'):
    """
    Return L^-T * X * L^-1, assumuing X is symmetrical and L is lower cholesky
    """
    if transpose == 'left':
        tmp = tf.linalg.triangular_solve(L, X, lower=True, adjoint=True)
        return tf.transpose(tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True, adjoint=True))
    else:
        tmp = tf.linalg.triangular_solve(L, X, lower=True, adjoint=False)
        return tf.transpose(tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True, adjoint=False))

def backsub_both_sides_tensorflow_3DD(L,X):
    """
    Return L^-1 * X * L^-T, assumuing X [N,M,M] and L is lower cholesky
    """
    LInv_X = tf.linalg.triangular_solve(L, X, lower=True)
    LInv = tf.linalg.inv(L)
    return tf.linalg.matmul(LInv_X, LInv, transpose_b=True)

def gatherPsiStat_sum(kern, X, Z, uncertain_inputs):
    '''
    ### In this function, we calculate the tf.reduce_sum for psi0; psi1
    :param kern: kernel function
    :param X: input
    :param Z: inducing input
    :param uncertain_inputs: check wheter X is uncertain input
    :return: psi0, psi1, psi2
    '''

    if uncertain_inputs:
        psi0 = tf.reduce_sum(expectation(X, kern))
        psi1 = expectation(X, (kern, Z))
        psi2 = tf.reduce_sum(
                expectation(
                X, (kern, Z), (kern, Z)
                ),
                axis=0,
                )
    else:
        psi0 = tf.reduce_sum(kern.K_diag(X))
        Kmns = covariances.Kuf(Z, kern, X)
        psi1 = tf.transpose(Kmns)
        psi2 = tdot_tensorflow(tf.transpose(psi1))

    return psi0, psi1, psi2

def gatherPsiStat(kern, X, Z, uncertain_inputs):
    '''
    ### In this function, we calculate the for psi0; psi1
    :param kern: kernel function
    :param X: input
    :param Z: inducing input
    :param uncertain_inputs: check wheter X is uncertain input
    :return: psi0, psi1, psi2
    '''
    if uncertain_inputs:
        psi0 = expectation(X, kern) ## Done
        psi1 = expectation(X, (kern, Z)) ## Done
        psi2 = expectation(X, (kern, Z), (kern, Z)) ## Done
    else:
        psi0 = kern.K_diag(X) ## Done
        Kmns = covariances.Kuf(Z, kern, X) ## Done
        psi1 = tf.transpose(Kmns) ## Done
        psi2 = psi1[:,:,None]*psi1[:,None,:] ## Done

    return psi0, psi1, psi2

    ############################################################################
    ########################## Optimizer #######################################
    ############################################################################

def run_adam_fulldata(model, iterations,data,N,minibatch_size, Moreoutput = False):
    """
    Utility function running the Adam optimizer
    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    if Moreoutput == True:
        train_iter = iter(data.batch(minibatch_size))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat()
        train_iter = iter(train_dataset.batch(N))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = keras.optimizers.Adam(1e-2)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 100 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
            print(step, elbo)
    return logf


## changing here
def optimize_model_with_scipy_sparseGPMD_speed_up(model):
    '''
    This model is used for training SparseGPMD
    :param model:
    :return:
    '''
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
        model.training_loss,
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": ci_niter(1)},
        compile=False,
    )

    ############################################################################
    ########################## Evaluation Metrics ##############################
    ############################################################################

def NMSE_test_bar(y_test, y_pre_mu):
    # y_test_mean = np.mean(y_test)
    # up = ((y_test - y_pre_mu) ** 2).mean()
    # down = ((y_test - y_test_mean) ** 2).mean()
    # return up/down
    return 1 - r2_score(y_test, y_pre_mu)

def MSE_test_bar_normall(y_test, y_pre_mu):
    # y_test_mean = np.mean(y_test)
    up = ((y_test - y_pre_mu) ** 2).mean()
    # down = ((y_test - y_test_mean) ** 2).mean()
    return up
    # return 1 - r2_score(y_test, y_pre_mu)


def MNLP(y_test, y_pre_mu, y_pre_var):
    Negative_lpd = 0.5 * np.log(2*np.pi) + 0.5 * np.log(y_pre_var) + 0.5 * np.square(y_test - y_pre_mu)/y_pre_var
    return Negative_lpd.mean()

    ############################################################################
    ################## Initialize HMOGPLV with same input ######################
    ############################################################################

def Initialize_HMOGP_No_Missing(x_raw, num_replicates, y, Q, D, Z):

    ## Apply Sparse GP
    # Z = x_raw[::gap].copy()
    Mc = Z.shape[0]
    total_replicated = num_replicates
    kern_upper_R = gpflow.kernels.Matern32()
    kern_lower_R = gpflow.kernels.Matern32()
    k_hierarchy_outputs = HMOGPLV.kernels.Hierarchial_kernel_replicated(kernel_g=kern_upper_R,
                                                                            kernel_f=[kern_lower_R],
                                                                            total_replicated=total_replicated)

    # initialization of inducing input locations (M random points from the training inputs)
    m = gpflow.models.GPRFITC(data=(x_raw, y), kernel=k_hierarchy_outputs
                                  , inducing_variable=Z)
    m.likelihood.variance.assign(0.01)
    MAXITER = ci_niter(10000)

    def optimize_model_with_scipy(model):
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(
            model.training_loss,
            variables=model.trainable_variables,
            method="l-bfgs-b",
            options={"disp": True, "maxiter": MAXITER},
            )

    optimize_model_with_scipy(m)
    m_current_dict = parameter_dict(m)  ## We find the current model parameters
    Y_t = tf.transpose(m.predict_f(m_current_dict['.inducing_variable.Z'])[0])

    ## Apply BGPLVM
    Xr_dim = Q
    Mr = D
    from GPy.util.linalg import jitchol
    from GPy.models import SparseGPRegression, BayesianGPLVM
    m_lvm = BayesianGPLVM(Y_t.numpy(), Xr_dim, kernel=GPy.kern.RBF(Xr_dim, ARD=True), num_inducing=Mr)
    m_lvm.likelihood.variance[:] = m_lvm.Y.var() * 0.01
    m_lvm.optimize(max_iters=10000)

    lengthscales2 = tf.convert_to_tensor(m_lvm.kern.lengthscale, dtype=default_float())
    variance2 = tf.convert_to_tensor(m_lvm.kern.variance, dtype=default_float())
    kernel_row_R = gpflow.kernels.RBF(lengthscales=lengthscales2, variance=variance2)

    Z_row = m_lvm.Z.values.copy()
    X_row_mean = m_lvm.X.mean.values.copy()
    Xvariance_row = m_lvm.X.variance.values

    qU_mean = m_lvm.posterior.mean.T.copy()
    # qU_var_row_W = jitchol(m_lvm.posterior.covariance)
    qU_var_row_W = jitchol(m_lvm.posterior.covariance)
    m_current_dict = parameter_dict(m)  ## model parameters of m
    Z = m_current_dict['.inducing_variable.Z']
    k_hierarchy_outputs.kernel_g.variance.assign(m_current_dict['.kernel.kernel_g.variance'])
    k_hierarchy_outputs.kernel_g.lengthscales.assign(m_current_dict['.kernel.kernel_g.lengthscales'])
    k_hierarchy_outputs.kernels[0].lengthscales.assign(m_current_dict['.kernel.kernels[0].lengthscales'])
    k_hierarchy_outputs.kernels[0].variance.assign(m_current_dict['.kernel.kernels[0].variance'])
    qU_var_col_W = m.predict_f(m_current_dict['.inducing_variable.Z'])[1].numpy()

    return k_hierarchy_outputs, kernel_row_R, Z, Z_row, X_row_mean, Xvariance_row, qU_mean, qU_var_row_W, qU_var_col_W

    ##############################################
    ################## Plot ######################
    ##############################################

def plot_gp(x, mu, var, ax):
    ax.plot(x, mu, color='blue', lw=0.8)
    ax.fill_between(
        x[:, 0],
        (mu - 2 * np.sqrt(var))[:, 0],
        (mu + 2 * np.sqrt(var))[:, 0],
        color='dodgerblue',
        alpha=0.1,
    )

def plot_orig(x, mu, var, ax):
    ax.plot(x, mu, color='blue', lw=0.8)
    loss = mu[-1][0]
    ax.text(x[-1], mu[-1], f"{loss:.4f}",
                 fontsize=24, color='blue', verticalalignment='top',
                 horizontalalignment='right')
    ax.fill_between(
        x[:, 0],
        (mu - 2 * np.sqrt(var))[:, 0],
        (mu + 2 * np.sqrt(var))[:, 0],
        color='dodgerblue',
        alpha=0.1,
    )
def save_plot(fig, flag, newpath_plot, Experiment_type, Data_name, D, num_replicates, d, Model_name, Q, train_percentage, num_data_each_replicate, gap, Num_repetition, seed_index):
    if not os.path.exists(newpath_plot):
        os.makedirs(newpath_plot)

    filename = (f"{flag}_{Experiment_type}{Data_name}-{D} Output with {num_replicates} replicates "
                f"{d}-th_Output{Model_name}Missing-{Q}-Q-{train_percentage:.6f}-train_percentage "
                f"{num_data_each_replicate}-num_data_each_replicate-{gap}-gap "
                f"{Num_repetition}-Num_repetition-{seed_index}-seed-{D}-th-Run.png")

    fig.savefig(os.path.join(newpath_plot, filename), format='png', bbox_inches='tight')
    plt.close()
    # plt.savefig(
    #     newpath_plot + '/' + Experiment_type + Data_name + '-%i Output' % D + 'with %i replicates' % num_replicates + ' %ith_Output' % d + Model_name +
    #     'Missing' + '-%i-Q' % Q + '-%f-train_percentage' % train_percentage + '-%i-num_data_each_replicate' % num_data_each_replicate + '-%i-gap' % gap +
    #     '-%i-Num_repetition' % Num_repetition + ' %iseed' % seed_index + '-%i-th-Run',
    #     format='PNG',
    #     bbox_inches='tight')

def LMC_data_set(D, Y_list_missing, X_list_missing,num_replicates,Y_list_missing_test, X_list_missing_test ):
    '''
    We create the data set for the LMC where we assume all replica in the same output as one output.
    :return:
    idx_train: the index for all replica in training data set
    idx_test: the index for all replica in test data set
    x_train_all: all training input data set
    x_test_all: all test input data set
    y_train_all: all training output data set
    y_test_all: all test output data set
    '''
    ## Train and test data
    x_train_all_pre = []
    x_test_all_pre = []
    y_train_all_pre = []
    y_test_all_pre = []
    for d in range(D):
        y_train_all_pre.append(np.c_[np.c_[Y_list_missing[d], np.ones_like(X_list_missing[d][:, -1][:, None]) * d],
                                    X_list_missing[d][:, -1][:, None] + d * num_replicates])
        y_test_all_pre.append(np.c_[np.c_[Y_list_missing_test[d], np.ones_like(X_list_missing_test[d][:, -1][:, None]) * d],
                                    X_list_missing_test[d][:, -1][:, None] + d * num_replicates])
        x_train_all_pre.append(np.c_[X_list_missing[d][:, 0][:, None], np.ones_like(X_list_missing[d][:, 0][:, None]) * d])
        x_test_all_pre.append(np.c_[X_list_missing_test[d][:, 0][:, None], np.ones_like(X_list_missing_test[d][:, 0][:, None]) * d])
    x_train_all = np.vstack(x_train_all_pre)
    x_test_all = np.vstack(x_test_all_pre)
    y_train_all_three = np.vstack(y_train_all_pre)
    y_test_all_three = np.vstack(y_test_all_pre)
    ## Finding the index for each replicate
    idx_train = [y_train_all_three[:, -1] == i for i in range(num_replicates * D)]
    idx_test = [y_test_all_three[:, -1] == i for i in range(num_replicates * D)]
    y_train_all = y_train_all_three[:, :-1]
    y_test_all = y_test_all_three[:, :-1]
    return idx_train, idx_test, x_train_all, x_test_all, y_train_all, y_test_all



def LMCsum_data(X_list_missing,Y_list_missing,X_list_missing_test,Y_list_missing_test, num_replicates,D):

    #### Here we return training and test dataset for X and Y. E.g X_list_missing is a training list for training input dataset.
    #### Each element in the training list is the same index replica for each output.

    ## Train and test data
    x_train_all_pre = []
    x_test_all_pre = []
    y_train_all_pre = []
    y_test_all_pre = []
    for rep in range(num_replicates):
        X_list_training = []
        Y_list_training = []

        X_list_test = []
        Y_list_test = []

        for d in range(D):
            ## Training
            index_appendix = X_list_missing[d][:, -1][:, None] == rep
            X_d_r = X_list_missing[d][index_appendix.squeeze()]
            index_d = np.ones_like(X_d_r[:, -1][:, None]) * d
            X_d_r_with_index = np.c_[X_d_r[:,:-1], index_d]
            X_list_training.append(X_d_r_with_index)

            Y_d_r = Y_list_missing[d][index_appendix.squeeze()]
            Y_d_r_with_index = np.c_[Y_d_r, index_d]
            Y_list_training.append(Y_d_r_with_index)

            ### Testing
            index_appendix_test = X_list_missing_test[d][:, -1][:, None] == rep
            X_d_r_test = X_list_missing_test[d][index_appendix_test.squeeze()]
            index_d_test = np.ones_like(X_d_r_test[:, -1][:, None]) * d
            X_d_r_test_with_index = np.c_[X_d_r_test[:,:-1], index_d_test]
            X_list_test.append(X_d_r_test_with_index)

            Y_d_r_test = Y_list_missing_test[d][index_appendix_test.squeeze()]
            Y_d_r_test_with_index = np.c_[Y_d_r_test, index_d_test]
            Y_list_test.append(Y_d_r_test_with_index)


        X_r = np.vstack(X_list_training)
        Y_r = np.vstack(Y_list_training)
        X_r_test = np.vstack(X_list_test)
        Y_r_test = np.vstack(Y_list_test)

        y_train_all_pre.append(Y_r)
        x_train_all_pre.append(X_r)
        x_test_all_pre.append(X_r_test)
        y_test_all_pre.append(Y_r_test)
    return x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre

def LMCsum_data_Missing_part_of_one_output_in_Whole(X_list_missing,Y_list_missing,X_list_missing_test,Y_list_missing_test, num_replicates,D):

    ## for our final rebuttal
    ## Train and test data
    x_train_all_pre = []
    x_test_all_pre = 1
    y_train_all_pre = []
    y_test_all_pre = 1
    for rep in range(num_replicates):
        X_list_training = []
        Y_list_training = []

        for d in range(D):
            ## Training
            index_appendix = X_list_missing[d][:, -1][:, None] == rep
            X_d_r = X_list_missing[d][index_appendix.squeeze()]
            index_d = np.ones_like(X_d_r[:, -1][:, None]) * d
            X_d_r_with_index = np.c_[X_d_r[:,:-1], index_d]
            X_list_training.append(X_d_r_with_index)

            Y_d_r = Y_list_missing[d][index_appendix.squeeze()]
            Y_d_r_with_index = np.c_[Y_d_r, index_d]
            Y_list_training.append(Y_d_r_with_index)



        X_r = np.vstack(X_list_training)
        Y_r = np.vstack(Y_list_training)

        y_train_all_pre.append(Y_r)
        x_train_all_pre.append(X_r)
    return x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre

def new_format_for_X_Y(x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre,num_replicates):
    x_full_list_train = []
    x_full_list_test = []
    y_full_list_train = []
    y_full_list_test = []

    for r in range(num_replicates):
        index_r_train = np.ones_like(x_train_all_pre[r][:, -1][:, None]) * r
        index_r_test = np.ones_like(x_test_all_pre[r][:, -1][:, None]) * r

        X_full_train_with_index = np.c_[x_train_all_pre[r], index_r_train]
        X_full_test_with_index = np.c_[x_test_all_pre[r], index_r_test]

        Y_full_train_with_index = np.c_[y_train_all_pre[r], index_r_train]
        Y_full_test_with_index = np.c_[y_test_all_pre[r], index_r_test]

        x_full_list_train.append(X_full_train_with_index)
        x_full_list_test.append(X_full_test_with_index)

        y_full_list_train.append(Y_full_train_with_index)
        y_full_list_test.append(Y_full_test_with_index)

    All_Y_train = np.vstack(y_full_list_train)
    All_X_train = np.vstack(x_full_list_train)
    All_Y_test = np.vstack(y_full_list_test)
    All_X_test = np.vstack(x_full_list_test)
    return All_X_train, All_X_test, All_Y_train, All_Y_test

def new_format_for_X_Y_Missing_part_of_one_output_in_Whole(x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre,num_replicates):
    x_full_list_train = []
    y_full_list_train = []

    for r in range(num_replicates):
        index_r_train = np.ones_like(x_train_all_pre[r][:, -1][:, None]) * r

        X_full_train_with_index = np.c_[x_train_all_pre[r], index_r_train]

        Y_full_train_with_index = np.c_[y_train_all_pre[r], index_r_train]

        x_full_list_train.append(X_full_train_with_index)

        y_full_list_train.append(Y_full_train_with_index)

    All_Y_train = np.vstack(y_full_list_train)
    All_X_train = np.vstack(x_full_list_train)
    All_Y_test = 1
    All_X_test = 1
    return All_X_train, All_X_test, All_Y_train, All_Y_test