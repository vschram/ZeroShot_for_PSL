                                #################################################
                                ### We build our own model based on GPflow. #####
                                #################################################

##### we import from gpflow and tensorflow #####
import gpflow
import tensorflow as tf
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData, BayesianModel
from gpflow.models.svgp import SVGP
from gpflow.models.gpr import GPR
from gpflow.mean_functions import MeanFunction
from gpflow.kernels import Kernel
from gpflow.models.util import data_input_to_tensor
import numpy as np
from gpflow.models.util import inducingpoint_wrapper
from gpflow.config import default_float, default_jitter
from typing import Optional, TypeVar, Union
OutputData = Union[tf.Tensor]
Data = TypeVar("Data", RegressionData, InputData, OutputData)
import time
from gpflow.probability_distributions import DiagonalGaussian, ProbabilityDistribution
from gpflow.base import Parameter
from gpflow.utilities import positive, to_default_float, triangular
from gpflow.expectations import expectation
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin
from HMOGPLV import covariances
from HMOGPLV.utils import tdot_tensorflow, backsub_both_sides_tensorflow_3DD,gatherPsiStat_sum, gatherPsiStat, backsub_both_sides_tensorflow
from HMOGPLV.conditionals import lmc_conditional_mogp
### We import GPy only for prediction; Since we use tensorflow to optimize our model.
### When we use prediction, we do not need to use tensorflow. In this case, it is really convenient for mainly use the code from Dai
import GPy
from GPy.util.linalg import dtrtrs
from HMOGPLV.utils import optimize_model_with_scipy_sparseGPMD_speed_up
from gpflow.inducing_variables import InducingPoints







###############################################################################################################################################################################
###############################################################################################################################################################################


########################################################################################################
##### Here we build the kernel for Jame's paper only for layer hierarchial GP with inducing points #####
########################################################################################################

class SHGP_replicated_within_data(GPModel, InternalDataTrainingLossMixin):
    '''
    This model is for James paper with inducing points
    '''
    def __init__(
            self,
            kernel,
            inducing_variable,
            *,
            data: RegressionData,
            noise_variance: float = 1.0,
            mean_function=None,
            num_latent_gps: int = 1,
            q_mu=None,
            q_sqrt=None,
            num_data=None,
    ):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        self.data = data_input_to_tensor(data)
        self.num_data = num_data
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        # init variational parameters
        num_inducing = self.inducing_variable.num_inducing

        # Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        q_mu = np.zeros((num_inducing, self.num_latent_gps)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]
        ones = np.eye(num_inducing, dtype=default_float()) if q_sqrt is None else q_sqrt
        self.q_sqrt = Parameter(ones, transform=triangular())  # [M, P]


    def Calculate_KL_q_U_p_U(self, Kuu):
        '''
        Here we build the funciton to calculate the KL divergence: KL(q(U)||p(U))
        '''

        Lu_p = tf.linalg.cholesky(Kuu)
        qU_mean = self.q_mu
        Lu_q = self.q_sqrt
        alpha = tf.linalg.triangular_solve(Lu_p, qU_mean, lower=True)

        # Mahalanobis term: μqᵀ Kuu⁻¹ μq
        mahalanobis = tf.reduce_sum(tf.square(alpha))

        # Constant term:  M
        constant = -tf.cast(tf.shape(qU_mean)[0], dtype=default_float())

        # Log-determinant of the covariance of S:
        logdet_qcov = - tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(Lu_q))))

        # Log-determinant of the covariance of Kuu:
        logdet_pcov= tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(Lu_p))))

        # Trace term: tr(Kuu⁻¹ S)
        LpiLq = tf.linalg.triangular_solve(Lu_p, Lu_q, lower=True)
        trace = tf.reduce_sum(tf.square(LpiLq))

        twoKL = 0.5* (trace + mahalanobis + constant + logdet_pcov +logdet_qcov)
        # print(twoKL)
        return twoKL

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = self.data

        ## calculate the kl q(u)||p(u)
        Kuu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        kl = self.Calculate_KL_q_U_p_U(Kuu)

        ## Predictive mean and variance
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)

        ## This can be stochastic for input data points
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)

        ## calculate the elbo
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt

        Kmm = covariances.Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())  # [M, M]
        Kmn = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)  # [M, N]
        Knn = self.kernel(Xnew, full_cov=full_cov)  # (N,)

        Lm = tf.linalg.cholesky(Kmm)
        A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [M,N]
        Af = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)  # [M,N] Kmm ^ (-1)Kmn

        ## Mean Kfu Kuu^{-1} q_mu
        mu = tf.linalg.matmul(Af, q_mu, transpose_a=True)  # [..., N, R]

        ## Variance Kff - Kfu Kuu^{-1} Kuf + Kfu Kuu^{-1} Su Kuu^{-1} Kuf (cholesky(Su) = q_sqrt)
        L_sqrtT_Af = tf.linalg.matmul(q_sqrt, Af, transpose_a=True)  # [M, N]
        S = tf.reduce_sum(tf.square(L_sqrtT_Af), -2)  ## The diagnoal of Kfu Kuu^(-1) Su Kuu^(-1) Kuf
        var = Knn - tf.reduce_sum(tf.square(A), -2) + S
        return mu + self.mean_function(Xnew), tf.expand_dims(var, 1)


###############################################################################################################################################################################
###############################################################################################################################################################################

                                ### We build our own hierarchial GP model with prior correlation between outputs ###

#######################################################################################
#### We build our own hierarchical GP model with prior correlation between outputs ####
#######################################################################################

class HMOGP_prior_outputs(BayesianModel, ExternalDataTrainingLossMixin):
    """
    Hierarchical Multi-output Gaussian Processes with Latent Variable with replicated data set. where the input kernel Kx is a hierarchical kernel whileas the ouput kernel Kh is normal kernel.
    :param kernel: Kernel for the input X.
    :param kernel_latent: a gpflow kernel for the GP of the latent space ** defaults to RBF-ARD **
    :param Heter_GaussianNoise: The different Gaussian noise for different output.
    :type Heter_GaussianNoise: numpy.ndarray (D,) D is the number of outputs.
    :param X_latent_dim: The dimension of the latent variable.
    :param Z: the inducing points for input X.
    :param Z_latent: inducing inputs for the latent space
    :type Z_latent: numpy.ndarray or None
    :param X_latent_mean: the initial value of the mean of the variational posterior distribution of points in the latent space
    :type X_latent_mean: numpy.ndarray or None
    :param X_latent_variance: the initial value of the variance of the variational posterior distribution of points in the latent space
    :type X_latent_variance: numpy.ndarray or None
    :param num_inducing:M . M is the number of inducing points for GP of individual output dimensions. M is the number of inducing points for the latent space.
    :type num_inducing: int
    :param mean_function: This is mean funciton. However, we seldom use it.
    :param X_latent_prior_mean: The prior mean for the latent variable H. We usually assume it is 0.
    :param X_latent_prior_var: The prior variance for the latent variable H. We usually assume it is 1.
    """
    def __init__(self,
                 kernel, ## Kernel for the input X.
                 kernel_latent, ## Kernel for the latent variable H.
                 Heter_GaussianNoise, ## The different Gaussian noise for different output.
                 X_latent_dim, ## The dimension of the latent variable.
                 Z, ## the inducing points for input X.
                 Z_latent, ## The inducing point for the latent variable H.
                 X_latent_mean, ## The mean of the inducing variable H.
                 X_latent_variance, ## The variance of the inducing variable H.
                 total_replicates, ## The number of total replicates.
                 index_replicates, ## The index for index_replicates with respect to outputs
                 num_outputs, ## The number of inducing points.
                 mean_function=None, ## This is mean funciton. However, we seldom use it.
                 X_latent_prior_mean = None, ## The prior mean for the latent variable H. We usually assume it is 0.
                 X_latent_prior_var = None, ## The prior variance for the latent variable H. We usually assume it is 1.
                 num_latent_gps: int = 1,
                 **kwargs):

        super().__init__(**kwargs)


        ### set up kernel and likelihood
        self.kernel = kernel
        self.kernel_latent = kernel_latent
        self.Heter_GaussianNoise = Parameter(Heter_GaussianNoise, transform=positive())  ## the nosie variable in likelihood for each output
        self.num_latent_gps = num_latent_gps

        num_latent_data,_ = X_latent_mean.shape
        self.num_latent_data = num_latent_data
        self.index_replicates = index_replicates

        self.num_outputs = num_outputs
        self.total_replicates = total_replicates
        ## set up uncertainty input
        self.X_latent_mean = Parameter(X_latent_mean)
        self.X_latent_variance = Parameter(X_latent_variance, transform=positive())
        self.X_latent_variable_H = DiagonalGaussian(self.X_latent_mean, self.X_latent_variance)

        # deal with parameters for the prior mean variance of X
        if X_latent_prior_mean is None:
            X_latent_prior_mean = tf.zeros((self.num_latent_data, X_latent_dim), dtype=default_float())
        if X_latent_prior_var is None:
            X_latent_prior_var = tf.ones((self.num_latent_data, X_latent_dim))
        self.X_prior_mean = tf.convert_to_tensor(np.atleast_1d(X_latent_prior_mean), dtype=default_float())
        self.X_prior_var = tf.convert_to_tensor(np.atleast_1d(X_latent_prior_var), dtype=default_float())

        ## set up inducing variables
        self.Z = inducingpoint_wrapper(Z) ### this is the inducing point for real input X
        self.Z_latent = inducingpoint_wrapper(Z_latent) ### This is inducing point for the latent function h


        ## Initialize and set up the mean and variance for Q_U
        num_inducing_data_Z = self.Z.num_inducing
        qU_mean = np.zeros((num_inducing_data_Z,self.num_latent_gps))## It is a vector
        qU_var = np.eye(num_inducing_data_Z, dtype=default_float())

        self.qU_mean = Parameter(qU_mean,dtype=default_float())
        self.qU_var = Parameter(qU_var,transform=triangular())


    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)


    def elbo(self, data: RegressionData) -> tf.Tensor:
        '''
        We calculate elbo for the model
        :param data: X,Y where X is the input data and Y is the output data
        :return: ELBO
        '''
        X, Y = data
        KL_H = self.Calculate_KL_q_H_p_H()
        part_loglikelihood = self.Computed_part_log_likehood(X, Y)
        return part_loglikelihood - KL_H

    def Calculate_KL_q_H_p_H(self):
        '''
        We calculate the KL_H[q(H) || p(H)]
        :return:
        '''
        dX_data_var = (
            self.X_latent_variance
            if self.X_latent_variance.shape.ndims == 2
            else tf.linalg.diag_part(self.X_latent_variance)
        )
        NQ = to_default_float(tf.size(self.X_latent_mean))
        KL_H = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var))
        KL_H += 0.5 * tf.reduce_sum(tf.math.log(self.X_prior_var))
        KL_H -= 0.5 * NQ
        KL_H += 0.5 * tf.reduce_sum(
            (tf.square(self.X_latent_mean - self.X_prior_mean) + dX_data_var) / self.X_prior_var
        )
        return KL_H


    def Calculate_KL_q_U_p_U(self,Kuu):
        '''
        Here we build the funciton to calculate the KL divergence: KL(q(U)||p(U))
        '''
        Lu_p = tf.linalg.cholesky(Kuu)
        qU_mean = self.qU_mean
        Lu_q = self.qU_var
        alpha = tf.linalg.triangular_solve(Lu_p, qU_mean, lower=True)

        # Mahalanobis term: μqᵀ Kuu⁻¹ μq
        mahalanobis = tf.reduce_sum(tf.square(alpha))

        # Constant term:  M
        constant = -tf.cast(tf.shape(qU_mean)[0], dtype=default_float())

        # Log-determinant of the covariance of S:
        logdet_qcov = - tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(Lu_q))))

        # Log-determinant of the covariance of Kuu:
        logdet_pcov= tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(Lu_p))))

        # Trace term: tr(Kuu⁻¹ S)
        LpiLq = tf.linalg.triangular_solve(Lu_p, Lu_q, lower=True)
        trace = tf.reduce_sum(tf.square(LpiLq))

        twoKL = 0.5* (trace + mahalanobis + constant + logdet_pcov +logdet_qcov)

        return twoKL


    def Computed_part_log_likehood(self, X, Y):
        '''
        In this function, we compute the part of log-likelihood ($\mathcal{F}$ in the ELBO)
        :param X: X is real input; X has two extra columns. The last one is the index for outputs. The last 2 is the index for replicates.
        :param Y: Y is a three columns matrix where the second column have the index for each replicates, the third column is the index for outputs
        :return: $\mathcal{F}$ in the ELBO
        '''

        ## the inverse of the noise distribution
        Variance_noise = self.Heter_GaussianNoise

        X_latent_variable_H = self.X_latent_variable_H
        Z_latent = self.Z_latent
        Z = self.Z

        ## Calculate the K value where we consider the uncertain input and real input
        uncertain_inputs_H = isinstance(X_latent_variable_H, ProbabilityDistribution)
        Real_inputs_X = isinstance(X, ProbabilityDistribution)

        psi0_latent_H, psi1_latent_H, psi2_latent_H = gatherPsiStat(self.kernel_latent, X_latent_variable_H, Z_latent, uncertain_inputs_H)
        psi0_real_X, psi1_real_X, psi2_real_X = gatherPsiStat(self.kernel, X, Z, Real_inputs_X)

        ## psi0 is the diagnoal of Kff
        ## psi1 is Knm
        ## psi2 \psi_{2}^{n, m, m^{\prime}}=E_{q(X)}\left[k\left(Z_{m}, X_{n}\right) k\left(X_{n}, Z_{m^{\prime}}\right)\right]

        # ======================================================================
        # Compute Common Components
        # ======================================================================
        ## Mean and variance of q(U:)
        qU_mean = self.qU_mean
        qU_var_full = tdot_tensorflow(self.qU_var)

        ## Compute the Kuu
        Kuu_h = tf.identity(covariances.Kuu(Z_latent, self.kernel_latent, jitter=default_jitter()))
        Kuu_x = tf.identity(covariances.Kuu(Z, self.kernel, jitter=default_jitter()))
        Index_outputs_Z = self.Z.Z[..., -1]
        ## The Index is sorted
        _, _, counts_outputs = tf.unique_with_counts(tf.sort(Index_outputs_Z))
        ## we change Kuu_h format in order to have a pointwise with Kuu_x
        Kuu_h_final = tf.repeat(tf.repeat(Kuu_h, repeats=counts_outputs, axis=0), repeats=counts_outputs, axis=1)
        Kuu = Kuu_h_final * Kuu_x

        # Compute KL_U[q(U:) || p(U:]
        KL_U = self.Calculate_KL_q_U_p_U(Kuu)


        ## The following can help us to find corresponding element for each replicates
        ## Since we aleady order the Kernel, we order the Y here
        ## This is the index for each replicates
        index_y_replicates = Y[:, -2]
        index_y_replicates = tf.cast(index_y_replicates, tf.int32)
        index_y_output = Y[:, -1]
        index_y_output = tf.cast(index_y_output, tf.int32)
        Y = Y[..., :-2]

        argsy_allreplicates = tf.dynamic_partition(Y, index_y_replicates, self.total_replicates)
        index_replicated_sorted = tf.cast(tf.sort(index_y_replicates), tf.int32)

        ### Here we divided all the psi into different part with respect to replicates
        psi0_real_X_alloutputs = tf.dynamic_partition(psi0_real_X, index_replicated_sorted, self.total_replicates)
        psi1_real_X_alloutputs = tf.dynamic_partition(psi1_real_X, index_replicated_sorted, self.total_replicates)
        psi2_real_X_alloutputs = tf.dynamic_partition(psi2_real_X, index_replicated_sorted, self.total_replicates)


        mid_res = {
            'psi0_latent_H': psi0_latent_H,
            'psi1_latent_H': psi1_latent_H,
            'psi2_latent_H': psi2_latent_H,
            'psi0_real_X_alloutputs': psi0_real_X_alloutputs,
            'psi1_real_X_alloutputs': psi1_real_X_alloutputs,
            'psi2_real_X_alloutputs': psi2_real_X_alloutputs
        }

        # ======================================================================
        # Compute log-likelihood
        # ======================================================================
        Index_cluster_outputs = tf.dynamic_partition(index_y_replicates, index_y_output, self.num_outputs)
        logL_tensorflow = tf.cast([[0.]],tf.float64)

        ## We compute each output in $\mathcal{F}$ and combine it together.
        # for d in range(self.num_outputs):
        #     d_count_replicate, _ = tf.unique(Index_cluster_outputs[d])
        #     for r in range(tf.shape(d_count_replicate)[0]):
        #         logL_tensorflow += self.inference_dr(tf.cast(d_count_replicate[r],tf.float64), d, Variance_noise,
        #                                              argsy_allreplicates, mid_res, Kuu, qU_mean, qU_var_full)
        # return logL_tensorflow - KL_U

        for d in range(self.num_outputs):
            for r in range(len(self.index_replicates[d])):
                logL_tensorflow += self.inference_dr(self.index_replicates[d][r], d, Variance_noise,
                                                     argsy_allreplicates, mid_res, Kuu, qU_mean, qU_var_full)
        return logL_tensorflow - KL_U


    def inference_dr(self, r, d, Variance_noise, argsy_alloutputs, mid_res, Kuu, qU_mean, qU_var_full):
        '''
        Compute each output in $\mathcal{F}$
        :param d: The d-th output
        :param r: The r-th replicates
        :param Variance_noise: The Variance noise
        :param Y: the output
        :param mid_res: a dictionary that include elements in order to compute.
        :return:
        '''
        # r = 1
        Y_dr = argsy_alloutputs[r]
        # r_index = tf.cast(r, tf.int32)
        # Y_dr = tf.gather(argsy_alloutputs,r_index)
        ## The number of data point in r-th replicates in d-th output and the likelihood variable
        N_dr = tf.shape(Y_dr)[0]
        Variance_noise_d = Variance_noise[d]



        ## We choose element for each kernel function so that we will build the kernel function later
        psi0_latent_H, psi1_latent_H, psi2_latent_H = mid_res['psi0_latent_H'], mid_res['psi1_latent_H'], mid_res['psi2_latent_H']
        psi0_real_X_alloutputs, psi1_real_X_alloutputs, psi2_real_X_alloutputs = mid_res['psi0_real_X_alloutputs'], mid_res['psi1_real_X_alloutputs'], mid_res['psi2_real_X_alloutputs']

        ## We choose the value for the d-th output
        psi0_latent_H_d_ouput = psi0_latent_H[d] ## This is the diagnoal of K_uu_h (M_u_h,)
        psi1_latent_H_d_ouput = psi1_latent_H[d:d+1] ## [N,M_u_h]: N is the number of the data points in d-output
        psi2_latent_H_d_ouput = psi2_latent_H[d] ## [N,M,M_u_h]: N is the number of the data points in d-output

        psi0_real_X_d_ouput_r_replicate = psi0_real_X_alloutputs[r] ## This is the diagnoal of K_uu_X (N_dr,)
        psi1_real_X_d_ouput_r_replicate =  psi1_real_X_alloutputs[r] ## [N_dr,M_u_x]
        psi2_real_X_d_ouput_r_replicate =  psi2_real_X_alloutputs[r] ## [N_dr,M_u_x, M_u_x]

        # psi0_real_X_d_ouput_r_replicate = tf.gather(psi0_real_X_alloutputs, r_index) ## This is the diagnoal of K_uu_X (N_dr,)
        # psi1_real_X_d_ouput_r_replicate =  tf.gather(psi1_real_X_alloutputs,r_index) ## [N_dr,M_u_x]
        # psi2_real_X_d_ouput_r_replicate =  tf.gather(psi2_real_X_alloutputs,r_index) ## [N_dr,M_u_x, M_u_x]

        ## Calculate the psi0_dr, psi1_dr, psi2_dr
        psi0_dr = tf.reduce_sum(psi0_latent_H_d_ouput * psi0_real_X_d_ouput_r_replicate) ## A scalar

        Index_outputs_Z = self.Z.Z[..., -1]
        ## The Index is sorted
        _, _, counts_outputs = tf.unique_with_counts(tf.sort(Index_outputs_Z))
        psi1_dr = tf.repeat(psi1_latent_H_d_ouput, repeats=counts_outputs,axis=1) * psi1_real_X_d_ouput_r_replicate  ## [Nr,M_u_x]
        psi2_dr = tf.repeat(tf.repeat(psi2_latent_H_d_ouput, repeats=counts_outputs, axis=0),repeats=counts_outputs, axis=1) * psi2_real_X_d_ouput_r_replicate ## [Nr, M_u_x, M_u_x]
        psi2_dr_mm = tf.reduce_sum(psi2_dr, axis=0)


        Lu = tf.linalg.cholesky(Kuu)
        Lu_inv = tf.linalg.inv(Lu)
        # Kuu_inv = tf.linalg.inv(Kuu)
        M_MT_plus_Qvar = tf.linalg.matmul(qU_mean,qU_mean,transpose_b=True) + qU_var_full
        Constant_d = tf.cast(- N_dr, dtype=default_float()) / 2. * (tf.math.log(2. * tf.cast(np.pi, dtype=default_float())) + tf.math.log(Variance_noise_d)) - tf.math.reduce_sum(tf.math.square(Y_dr)) / (2. * Variance_noise_d)

        Lu_inv_psi2_d_mm = tf.linalg.triangular_solve(Lu, psi2_dr_mm, lower=True)
        LuInvpsi2_d_mm_Lu_inv_T = backsub_both_sides_tensorflow_3DD(Lu, psi2_dr_mm)

        Lu_invT_LuInvpsi2_d_mm_Lu_inv_T = tf.linalg.matmul(Lu_inv, LuInvpsi2_d_mm_Lu_inv_T, transpose_a=True)
        Lu_inv_M_MT_plus_Qvar = tf.linalg.matmul(Lu_inv,M_MT_plus_Qvar)

        Y_d_T_psi1_d = tf.linalg.matmul(Y_dr, psi1_dr, transpose_a=True)
        Lu_inv_M = tf.linalg.triangular_solve(Lu, qU_mean, lower=True)

        LogL_A = - tf.linalg.trace(tf.linalg.matmul(Lu_invT_LuInvpsi2_d_mm_Lu_inv_T, Lu_inv_M_MT_plus_Qvar)) \
                 + 2.0 * tf.linalg.matmul(tf.linalg.matmul(Y_d_T_psi1_d,Lu_inv,transpose_b=True),Lu_inv_M) \
                 - psi0_dr + tf.linalg.trace(tf.linalg.matmul(Lu_inv,Lu_inv_psi2_d_mm, transpose_a=True))
        logL_d = Constant_d + tf.cast(LogL_A, dtype=default_float()) / (2. * Variance_noise_d)

        return logL_d


    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        ## Xnew have two extra columns. The last one is the index for outputs. The last 2 is the index for replicates.
        ## The ind_replicate means the index of the replicate. When we use the index for replicates, we will add it into the Hierarchical kernel.
        ind_output = Xnew[...,-1]
        ind_replicate = Xnew[...,-2]
        d, _ = tf.unique(ind_output)
        d = tf.cast(d[0],dtype=tf.int32)

        ## Calculate the K value where we consider the uncertain input and real input
        uncertain_inputs_H = isinstance(self.X_latent_variable_H, ProbabilityDistribution)
        Real_inputs_X = isinstance(Xnew, ProbabilityDistribution)

        psi0_latent_H, psi1_latent_H, psi2_latent_H = gatherPsiStat(self.kernel_latent, self.X_latent_variable_H, self.Z_latent, uncertain_inputs_H)
        psi0_real_X_r_replicate_d_output, psi1_real_X_r_replicate_d_output, psi2_real_X_r_replicate_d_output = gatherPsiStat(self.kernel, Xnew, self.Z, Real_inputs_X)

        psi0_latent_H_d_ouput = psi0_latent_H[d] ## This is the diagnoal of K_uu_h (M_u_h,)
        psi1_latent_H_one_output = psi1_latent_H[d:d + 1]
        psi2_latent_H_one_output = psi2_latent_H[d]

        ## Calculate the psi0_dr, psi1_dr, psi2_dr or Compute Kfu and Kff for only one output

        Index_outputs_Z = self.Z.Z[..., -1]
        ## The Index is sorted
        _, _, counts_outputs = tf.unique_with_counts(tf.sort(Index_outputs_Z))
        ## Kfu
        psi1_dr = tf.repeat(psi1_latent_H_one_output, repeats=counts_outputs, axis=1) * psi1_real_X_r_replicate_d_output  ## [N,M_u_x]
        ## Kff
        psi2_dr = tf.repeat(tf.repeat(psi2_latent_H_one_output, repeats=counts_outputs, axis=0),repeats=counts_outputs, axis=1) * psi2_real_X_r_replicate_d_output ## [N, M_u_x, M_u_x]


        ## Mean and variance of q(U:)
        qU_mean = self.qU_mean
        qU_var_full = tdot_tensorflow(self.qU_var)

        ## Compute the Kuu (Kuu is for all outputs)
        Kuu_h = tf.identity(covariances.Kuu(self.Z_latent, self.kernel_latent, jitter=default_jitter()))
        Kuu_x = tf.identity(covariances.Kuu(self.Z, self.kernel, jitter=default_jitter()))
        ## we change Kuu_h format in order to have a pointwise with Kuu_x
        Kuu_h_final = tf.repeat(tf.repeat(Kuu_h, repeats=counts_outputs, axis=0), repeats=counts_outputs, axis=1)
        Kuu = Kuu_h_final * Kuu_x

        Lu = tf.linalg.cholesky(Kuu)
        Lu_inv = tf.linalg.inv(Lu)
        ## Prediction one output in each time
        Kfu_one = psi1_dr


        #########################
        #### Predictive Mean ####
        #########################

        LuInvqUmean = tf.linalg.triangular_solve(Lu, qU_mean, lower=True)
        mu_predict = tf.linalg.matmul(tf.linalg.matmul(Kfu_one, Lu_inv), LuInvqUmean)

        #############################
        #### Predictive Variance ####
        #############################
        ## variance 1
        var1 = psi0_latent_H_d_ouput * psi0_real_X_r_replicate_d_output #(N,)]
        ## variance 2
        LuInv_Psi2_d_N_Lu_inv_T = tf.linalg.matmul(tf.linalg.matmul(Lu_inv, psi2_dr), Lu_inv, transpose_b=True) #  lu_inv Phi_D Lu_inv_T: Here only calculate the ## [N, M_u_x, M_u_x]
        var2 = tf.linalg.trace(LuInv_Psi2_d_N_Lu_inv_T)
        ## variance 3
        LuInv_SigmaU_Lu_inv_T = tf.linalg.matmul(tf.linalg.matmul(Lu_inv, qU_var_full), Lu_inv, transpose_b=True)
        var3 = tf.linalg.trace(tf.linalg.matmul(LuInv_SigmaU_Lu_inv_T,LuInv_Psi2_d_N_Lu_inv_T))
        ## variance 4
        MM = tf.linalg.matmul(qU_mean, qU_mean,transpose_b=True)
        LuInv_MM_Lu_inv_T = tf.linalg.matmul(tf.linalg.matmul(Lu_inv, MM), Lu_inv, transpose_b=True)
        KBB = Kfu_one[:,:,None] * Kfu_one[:,None,:]
        LuInv_KBB_Lu_inv_T = tf.linalg.matmul(tf.linalg.matmul(Lu_inv, KBB), Lu_inv, transpose_b=True)
        minuse_part = LuInv_Psi2_d_N_Lu_inv_T - LuInv_KBB_Lu_inv_T
        var4 = tf.linalg.trace(tf.linalg.matmul(LuInv_MM_Lu_inv_T,minuse_part))

        ## Final variance
        var_predic = var1 - var2 + var3 + var4

        return mu_predict, var_predic


###################################################################################################################
#### We build our own hierarchical GP model with prior correlation between outputs and using Kronecker product ####
###################################################################################################################

## We using Kronecker product like the Dai's paper.
class HMOGP_prior_outputs_kronecker(GPModel, ExternalDataTrainingLossMixin):
    """
    This model heavily depends on the Dai's missing model: Gaussian Process model for multi-output regression without missing data

    ## For real inputs
    :param Z: inducing inputs

    ## For latent inputs
    :param Z_row: inducing inputs for the latent space
    :param X_row: the initial value of the mean of the variational posterior distribution of points in the latent space
    :param Xvariance_row: the initial value of the variance of the variational posterior distribution of points in the latent space
    :type Xvariance_row: numpy.ndarray or None

    :param num_inducing: a tuple (M, Mr). M is the number of inducing points for GP of individual output dimensions. Mr is the number of inducing points for the latent space.
    :type num_inducing: (int, int)  Mr is the total number of inducing points for all the replicates
    :param int qU_var_r_W_dim: the dimensionality of the covariance of q(U) for the latent space. If it is smaller than the number of inducing points, it represents a low-rank parameterization of the covariance matrix.
    :param int qU_var_c_W_dim: the dimensionality of the covariance of q(U) for the GP regression. If it is smaller than the number of inducing points, it represents a low-rank parameterization of the covariance matrix.
    """

    def __init__(self,
                 kernel,
                 likelihood,
                 kernel_row,
                 Z,
                 Z_row,
                 X_row_mean,
                 Xvariance_row,
                 num_inducing,
                 qU_mean,
                 qU_var_col_W,
                 qU_var_row_W,
                 qU_var_r_W_dim=None,
                 qU_var_c_W_dim=None,
                 mean_function=None,
                 X_prior_mean = None,
                 X_prior_var = None,
                 **kwargs):
        num_data, num_latent_gps = X_row_mean.shape

        super().__init__(kernel,likelihood,num_latent_gps=num_latent_gps,**kwargs)

        ### set up kernel and likelihood
        self.kernel = kernel
        self.likelihood = likelihood
        self.kern_row = kernel_row
        self.num_data = num_data

        self.Mr, self.Mc, self.Qr, self.Qc =  Z_row.shape[0], Z.shape[0], Z_row.shape[1], Z.shape[1]

        ## set up uncertainty input and inducing variables
        self.X_row_mean = Parameter(X_row_mean)
        self.Xvariance_row = Parameter(Xvariance_row, transform=positive())
        self.X_row = DiagonalGaussian(self.X_row_mean, self.Xvariance_row)
        self.Z = inducingpoint_wrapper(Z) ### this is the inducing point for real input X
        self.Z_row = inducingpoint_wrapper(Z_row) ### This is inducing point for the latent function h

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = tf.zeros((self.num_data, self.num_latent_gps), dtype=default_float())
        if X_prior_var is None:
            X_prior_var = tf.ones((self.num_data, self.num_latent_gps))

        self.X_prior_mean = tf.convert_to_tensor(np.atleast_1d(X_prior_mean), dtype=default_float())
        self.X_prior_var = tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float())


        ## Initialize and set up the mean and variance for Q_U
        # qU_mean = np.ones(num_inducing)
        # qU_var_col_W = np.random.randn(num_inducing[0],num_inducing[0] if qU_var_c_W_dim is None else qU_var_c_W_dim)*0.01
        qU_var_col_diag = np.full(num_inducing[0],1e-5)
        # qU_var_col_diag = np.full(num_inducing[0],0)
        # qU_var_row_W = np.random.randn(num_inducing[1],num_inducing[1] if qU_var_r_W_dim is None else qU_var_r_W_dim)*0.01
        qU_var_row_diag = np.full(num_inducing[1],1e-5)
        # qU_var_row_diag = np.full(num_inducing[1],0)

        self.qU_mean = Parameter(qU_mean)
        self.qU_var_c_W = Parameter(qU_var_col_W)
        self.qU_var_c_diag = Parameter(qU_var_col_diag, transform=positive())
        # self.qU_var_c_diag = qU_var_col_diag
        self.qU_var_r_W = Parameter(qU_var_row_W)
        self.qU_var_r_diag = Parameter(qU_var_row_diag, transform=positive())
        # self.qU_var_r_diag = qU_var_row_diag

        self.posterior = None

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        X, Y = data
        # KL[q(H) || p(H)]
        dX_data_var = (
            self.Xvariance_row
            if self.Xvariance_row.shape.ndims == 2
            else tf.linalg.diag_part(self.Xvariance_row)
        )
        NQ = to_default_float(tf.size(self.X_row_mean))
        KL = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var))
        KL += 0.5 * tf.reduce_sum(tf.math.log(self.X_prior_var))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(self.X_row_mean - self.X_prior_mean) + dX_data_var) / self.X_prior_var
        )

        part_loglikelihood, self.posterior = self.Computed_part_log_likehood(X,Y)

        return part_loglikelihood - KL
        # return tf.reduce_sum(var_exp) * scale - kl


    def Computed_part_log_likehood(self,Xc,Y):
        '''
        The idea of this function is from the inference function in Dai's paper
        :param Xc: Xc is real input for the replicated dataset, The last column correpsonds for the index for each replicates
        Xc = [[  X_11     0],
                .
                .
              [ X_1N     0],
                .
                .
              [ X_R1     R],
                .
                .
              [ X_RN     R ] ]

        :param Y: Y is a matrix and each column corresponding to a output with replicates

         Y = [[ Y_11,1     Y_D1,1],
                .
                .
              [ Y_11,N     Y_D1,N],
                .
                .
              [ Y_1R,1     Y_DR,1],
                .
                .
              [ Y_1R,N     Y_DR,N] ]
        '''
        qU_var_c = tdot_tensorflow(self.qU_var_c_W) + tf.linalg.diag(self.qU_var_c_diag)
        # qU_var_c = tdot_tensorflow(self.qU_var_c_W)
        qU_var_r = tdot_tensorflow(self.qU_var_r_W) + tf.linalg.diag(self.qU_var_r_diag)
        # qU_var_r = tdot_tensorflow(self.qU_var_r_W)

        N, D = tf.shape(Y)[0], tf.shape(Y)[1]
        Mr, Mc, Qr, Qc= self.Mr, self.Mc, self.Qr, self.Qc

        Xr = self.X_row
        Zr = self.Z_row
        Zc = self.Z
        beta = 1. / self.likelihood.variance
        ## Calculate the k valu
        uncertain_inputs_r = isinstance(Xr, ProbabilityDistribution)
        uncertain_inputs_c = isinstance(Xc, ProbabilityDistribution)

        ### This is for latent input H
        psi0_r, psi1_r, psi2_r = gatherPsiStat_sum(self.kern_row, Xr, Zr, uncertain_inputs_r)
        ### This is for real input T
        psi0_c, psi1_c, psi2_c = gatherPsiStat_sum(self.kernel, Xc, Zc, uncertain_inputs_c)

        # ======================================================================
        # Compute Common Components
        # ======================================================================

        ## The Kuu_r is for the latent inputs.
        Kuu_r = tf.identity(covariances.Kuu(Zr, self.kern_row,
                               jitter=default_jitter()))
        Lr_tensorflow = tf.linalg.cholesky(Kuu_r)
        ## The Kuu_c has the Hierarchical structure
        Kuu_c = tf.identity(covariances.Kuu(Zc, self.kernel,
                               jitter=default_jitter()))
        Lc_tensorflow = tf.linalg.cholesky(Kuu_c)

        mu, Sr, Sc = self.qU_mean, qU_var_r, qU_var_c
        LSr_tensorflow = tf.linalg.cholesky(Sr)
        LSc_tensorflow = tf.linalg.cholesky(Sc)


        LcInvMLrInvT_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, tf.transpose(
            tf.linalg.triangular_solve(Lr_tensorflow, tf.transpose(mu), lower=True)), lower=True)

        LcInvPsi2_cLcInvT_tensorflow = backsub_both_sides_tensorflow_3DD(Lc_tensorflow, psi2_c)
        LrInvPsi2_rLrInvT_tensorflow = backsub_both_sides_tensorflow_3DD(Lr_tensorflow, psi2_r)

        LcInvLSc_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, LSc_tensorflow, lower=True)
        LrInvLSr_tensorflow = tf.linalg.triangular_solve(Lr_tensorflow, LSr_tensorflow, lower=True)


        LcInvScLcInvT_tensorflow = tdot_tensorflow(LcInvLSc_tensorflow)
        LrInvSrLrInvT_tensorflow = tdot_tensorflow(LrInvLSr_tensorflow)

        LcInvPsi1_cT_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, tf.transpose(psi1_c), lower=True)
        LrInvPsi1_rT_tensorflow = tf.linalg.triangular_solve(Lr_tensorflow, tf.transpose(psi1_r), lower=True)

        tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT_tensorflow = tf.math.reduce_sum(LrInvPsi2_rLrInvT_tensorflow * LrInvSrLrInvT_tensorflow)
        tr_LcInvPsi2_cLcInvT_LcInvScLcInvT_tensorflow = tf.math.reduce_sum(LcInvPsi2_cLcInvT_tensorflow * LcInvScLcInvT_tensorflow)

        tr_LrInvSrLrInvT_tensorflow = tf.math.reduce_sum(tf.math.square(LrInvLSr_tensorflow))
        tr_LcInvScLcInvT_tensorflow = tf.math.reduce_sum(tf.math.square(LcInvLSc_tensorflow))
        tr_LrInvPsi2_rLrInvT_tensorflow = tf.linalg.trace(LrInvPsi2_rLrInvT_tensorflow)
        tr_LcInvPsi2_cLcInvT_tensorflow = tf.linalg.trace(LcInvPsi2_cLcInvT_tensorflow)

        # ======================================================================
        # Compute log-likelihood
        # ======================================================================

        ### This log-likelihood have not included the KL(q(H)|p(H))
        logL_A_tensorflow = - tf.math.reduce_sum(tf.math.square(Y)) \
                            - tf.math.reduce_sum(tf.linalg.matmul(tf.linalg.matmul(LcInvMLrInvT_tensorflow, LcInvPsi2_cLcInvT_tensorflow, transpose_a=True),LcInvMLrInvT_tensorflow) * LrInvPsi2_rLrInvT_tensorflow) \
                            - tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT_tensorflow * tr_LcInvPsi2_cLcInvT_LcInvScLcInvT_tensorflow \
                            + 2 * tf.math.reduce_sum(Y * tf.linalg.matmul(tf.linalg.matmul(LcInvPsi1_cT_tensorflow, LcInvMLrInvT_tensorflow, transpose_a=True), LrInvPsi1_rT_tensorflow))\
                            - psi0_c * psi0_r + tr_LrInvPsi2_rLrInvT_tensorflow * tr_LcInvPsi2_cLcInvT_tensorflow


        logL_tensorflow = tf.cast(- N * D, dtype=tf.float64)/2. * (tf.math.log(2.*tf.cast(np.pi, dtype = tf.float64))-tf.math.log(beta)) + beta/2.* tf.cast(logL_A_tensorflow,dtype = tf.float64)\
                          - Mc * (tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lr_tensorflow))) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSr_tensorflow)))) \
                          - Mr * (tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lc_tensorflow))) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSc_tensorflow))))\
                          - tf.math.reduce_sum(tf.math.square(LcInvMLrInvT_tensorflow))/2. - tr_LrInvSrLrInvT_tensorflow * tr_LcInvScLcInvT_tensorflow/2. + Mr*Mc/2.

        post = OwnPosteriorMultioutput(LcInvMLrInvT= LcInvMLrInvT_tensorflow, LcInvScLcInvT=LcInvScLcInvT_tensorflow,
                LrInvSrLrInvT=LrInvSrLrInvT_tensorflow, Lr=Lr_tensorflow, Lc=Lc_tensorflow, kern_r=self.kern_row, Xr=Xr, Zr=Zr,Z=Zc,kernel=self.kernel)

        return logL_tensorflow, post


    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        ## This is raw predict without niose variable
        ## This f is only work for no complie since we have numpy() in the raw_predict
        ## However it work. The only different is that if we use complie (@tf.function), it would be super fast.
        mu, var = self.posterior.raw_predict_tf(Xnew=Xnew)
        return mu + self.mean_function(Xnew), var




class OwnPosteriorMultioutput():
    '''
    This class is only for calculate the prediction.
    The keep the parameter from our model. Then we can calculate the prediction based on those parameters.
    '''
    def __init__(self,LcInvMLrInvT, LcInvScLcInvT, LrInvSrLrInvT, Lr, Lc, kern_r, Xr, Zr,Z, kernel):
        self.LcInvMLrInvT = LcInvMLrInvT
        self.LcInvScLcInvT = LcInvScLcInvT
        self.LrInvSrLrInvT = LrInvSrLrInvT
        self.Lr = Lr
        self.Lc = Lc
        self.kern_r = kern_r
        self.Xr = Xr
        self.Zr = Zr
        self.Z = Z
        self.kernel = kernel

    def raw_predict_tf(self, Xnew):
        # N = Xnew
        Kmns_new = covariances.Kuf(self.Z, self.kernel, Xnew)
        psi1_c = tf.transpose(Kmns_new)
        psi2_c_n = psi1_c[:, :, None] * psi1_c[:, None, :]
        psi0_c = self.kernel.K_diag(Xnew)
        Lc = self.Lc
        Xr = self.Xr
        Zr = self.Zr
        Lr = self.Lr
        LrInvSrLrInvT = self.LrInvSrLrInvT
        LcInvScLcInvT = self.LcInvScLcInvT
        # LcInvMLrInvT = tf.identity(self.LcInvMLrInvT)
        LcInvMLrInvT = self.LcInvMLrInvT
        LcInvPsi1_cT_tf = tf.linalg.triangular_solve(Lc, tf.transpose(psi1_c), lower=True)

        psi0_r = expectation(Xr, self.kern_r)
        psi1_r = expectation(Xr, (self.kern_r, Zr))
        psi2_r_n = expectation(Xr, (self.kern_r, Zr), (self.kern_r, Zr))

        LrInvPsi1_rT_tf = tf.linalg.triangular_solve(Lr, tf.transpose(psi1_r), lower=True)

        woodbury_vector = tf.linalg.matmul(LcInvMLrInvT, LrInvPsi1_rT_tf)

        mu = tf.linalg.matmul(tf.transpose(LcInvPsi1_cT_tf), woodbury_vector)

        LrInvPsi2_r_nLrInvT_tf = backsub_both_sides_tensorflow_3DD(Lr, psi2_r_n)

        tr_LrInvPsi2_r_nLrInvT_tf = tf.linalg.trace(LrInvPsi2_r_nLrInvT_tf)

        LcInvPsi2_c_NLcInvT_tensorflow = backsub_both_sides_tensorflow_3DD(Lc, psi2_c_n)


        tr_LrInvPsi2_r_nLrInvT_LrInvSrLrInvT_tf = tf.linalg.trace(tf.linalg.matmul(LrInvSrLrInvT, LrInvPsi2_r_nLrInvT_tf))
        tr_LcInvPsi2_c_nLcInvT_LcInvScLcInvT_tf = tf.linalg.trace(tf.linalg.matmul(LcInvScLcInvT, LcInvPsi2_c_NLcInvT_tensorflow))


        tmp_tf = LrInvPsi2_r_nLrInvT_tf - tf.transpose(LrInvPsi1_rT_tf)[:, :, None] * tf.transpose(LrInvPsi1_rT_tf)[:, None, :]
        LcInv = tf.linalg.inv(Lc)
        psi1_c_LcInvT_LcInv_M_LrInvT = tf.linalg.matmul(tf.linalg.matmul(psi1_c, LcInv, transpose_b=True), LcInvMLrInvT)


        var1_tf = tf.transpose(tf.linalg.diag_part(tf.linalg.matmul(tf.linalg.matmul(psi1_c_LcInvT_LcInv_M_LrInvT, tmp_tf),psi1_c_LcInvT_LcInv_M_LrInvT,transpose_b=True)))

        var2_tf = psi0_c[:, None] * psi0_r[None, :]

        var3_tf = tr_LrInvPsi2_r_nLrInvT_tf[None, :] * tf.reduce_sum(tf.math.square(LcInvPsi1_cT_tf), axis=0)[:, None]

        var4_tf = tr_LrInvPsi2_r_nLrInvT_LrInvSrLrInvT_tf * tr_LcInvPsi2_c_nLcInvT_LcInvScLcInvT_tf[:,None]

        var_tf = var1_tf + var2_tf - var3_tf + var4_tf
        var_tf = var_tf.numpy()
        var_tf[var_tf<1e-6] = 1e-6
        var_tf = tf.convert_to_tensor(var_tf, dtype=default_float())
        return mu, var_tf


#####################################################################################################################################
#### We build our own hierarchical GP model with prior correlation between outputs and using Kronecker product with Missing data ####
#####################################################################################################################################

class HMOGP_prior_outputs_kronecker_product_Missing_speed_up(BayesianModel, ExternalDataTrainingLossMixin):

    """
    This model heavily depends on the Dai's missing model: Gaussian Process model for multi-output regression with missing data

    One of the difference between Dai's missing model is that this model use the hierarchical kernel for the real inputs. In this case,
    we calculate the \phi_d for this model by find the X_d first. E.g., we find the X_d based the index for X. Then we caluclate the \phi_d
    However, in the Dai's missing model, they calculate the \phi based on whole X. After that, they calculated \phi_d based on index.

    X have two extra column for the index. The last column is the index for outputs. The second last column is the index for the replicated.
    E.g. X=[[x11,0,0],
            ...
            [x1N,R,0],
            ...
            [xD1,0,D],
            ...
            [xDN,R,D]]
    :param Y: output observations, each column corresponding to an output dimension.
    :type Y: numpy.ndarray

    We keep the indexD in this model but we may do not need it. And we have hte indexD in the Dai's missing model
    :param indexD: the array containing the index of output dimension for each data point
    :type indexD: numpy.ndarray

    :param kernel_row: a GPy kernel for the GP of the latent space ** defaults to RBF **
    :param Z_row: inducing inputs for the latent space
    :type Z_row: numpy.ndarray or None
    :param X_row: the initial value of the mean of the variational posterior distribution of points in the latent space
    :type X_row: numpy.ndarray or None
    :param Xvariance_row: the initial value of the variance of the variational posterior distribution of points in the latent space
    :type Xvariance_row: numpy.ndarray or None
    :param num_inducing: a tuple (M, Mr). M is the number of inducing points for GP of individual output dimensions. Mr is the number of inducing points for the latent space.
    :type num_inducing: (int, int)
    :param int qU_var_r_W_dim: the dimensionality of the covariance of q(U) for the latent space. If it is smaller than the number of inducing points, it represents a low-rank parameterization of the covariance matrix.
    :param int qU_var_c_W_dim: the dimensionality of the covariance of q(U) for the GP regression. If it is smaller than the number of inducing points, it represents a low-rank parameterization of the covariance matrix.
    """
    def __init__(self,
                 kernel,
                 kernel_row,
                 Heter_GaussianNoise,
                 Xr_dim,
                 Z,
                 # Z_row,
                 # X_row_mean,
                 # Xvariance_row,
                 num_inducing,
                 indexD,
                 Initial_parameter,
                 variance_lowerbound=1e-6,
                 x_all = None,
                 y_all = None,
                 mean_function=None,
                 X_prior_mean = None,
                 X_prior_var = None,
                 **kwargs):

        if Initial_parameter == 'GP':
            from HMOGPLV.models import SparseGPMD
            from gpflow.utilities import parameter_dict
            from GPy.util.linalg import jitchol
            from GPy.models import SparseGPRegression, BayesianGPLVM
            ## we use SparseGPMD
            # m_sgp = SparseGPMD(data=(x_all, y_all), kernel=kernel, inducing_variable=Z, indexD=indexD,
            #                    noise=0.001)

            m_sgp = SparseGPMD(data=(x_all, y_all), kernel=kernel, inducing_variable=Z, indexD=indexD,
                               noise=0.01)
            a_HMOGPLV_missing_time = time.time()
            optimize_model_with_scipy_sparseGPMD_speed_up(m_sgp)
            b_HMOGPLV_missing_time = time.time()
            Training_time = b_HMOGPLV_missing_time - a_HMOGPLV_missing_time
            print('Time')
            print(Training_time)
            m_current_dict = parameter_dict(m_sgp)
            Y_t = []
            D = int(np.max(indexD))+1
            for d in range(D):
                Y_t.append(tf.transpose(m_sgp.predict_f(m_current_dict['.inducing_variable.Z'], d)[0]).numpy())
            Y_t = np.vstack(Y_t)

            ## we use Bayesian GPLVM in GPy
            Mr = num_inducing[1]
            m_lvm = BayesianGPLVM(Y_t, Xr_dim, kernel=GPy.kern.RBF(Xr_dim, ARD=True), num_inducing=Mr)
            # m_lvm.likelihood.variance[:] = m_lvm.Y.var() * 0.1
            m_lvm.likelihood.variance[:] = m_lvm.Y.var() * 0.01
            print('B-GPLVM')
            m_lvm.optimize(max_iters=1)
            print('Finish Optimization in B-GPLVM')

            ## Initialize the parameters
            # From Bayesian GP-LVM
            lengthscales2 = tf.convert_to_tensor(m_lvm.kern.lengthscale, dtype=default_float())
            variance2 = tf.convert_to_tensor(m_lvm.kern.variance, dtype=default_float())
            kernel_row = gpflow.kernels.RBF(lengthscales=lengthscales2, variance=variance2)

            Z_row = m_lvm.Z.values.copy()
            X_row_mean = m_lvm.X.mean.values.copy()
            Xvariance_row = m_lvm.X.variance.values

            ## From Sparse Gaussian Processes
            Z = m_current_dict['.inducing_variable.Z']
            kernel.kernel_g.variance.assign(m_current_dict['.kernel.kernel_g.variance'])
            kernel.kernel_g.lengthscales.assign(m_current_dict['.kernel.kernel_g.lengthscales'])
            kernel.kernels[0].lengthscales.assign(m_current_dict['.kernel.kernels[0].lengthscales'])
            kernel.kernels[0].variance.assign(m_current_dict['.kernel.kernels[0].variance'])
            qU_var_col_W = []
            for d in range(D):
                qU_var_col_W.append(tf.transpose(m_sgp.predict_f(m_current_dict['.inducing_variable.Z'], d)[1]).numpy())
            qU_var_col_W = np.vstack(qU_var_col_W).T

            qU_mean = m_lvm.posterior.mean.T.copy()
            qU_var_row_W = jitchol(m_lvm.posterior.covariance)
        else:
            from HMOGPLV.models import SparseGPMD
            from gpflow.utilities import parameter_dict
            from GPy.util.linalg import jitchol
            from GPy.models import SparseGPRegression, BayesianGPLVM
            ## we use SparseGPMD
            # m_sgp = SparseGPMD(data=(x_all, y_all), kernel=kernel, inducing_variable=Z, indexD=indexD,
            #                    noise=0.001)
            m_sgp = SparseGPMD(data=(x_all, y_all), kernel=kernel, inducing_variable=Z, indexD=indexD,
                               noise=0.01)
            a_HMOGPLV_missing_time = time.time()
            optimize_model_with_scipy_sparseGPMD_speed_up(m_sgp)
            b_HMOGPLV_missing_time = time.time()
            Training_time = b_HMOGPLV_missing_time - a_HMOGPLV_missing_time
            print('Time')
            print(Training_time)
            m_current_dict = parameter_dict(m_sgp)
            Y_t = []
            D = int(np.max(indexD))+1
            for d in range(D):
                Y_t.append(tf.transpose(m_sgp.predict_f(m_current_dict['.inducing_variable.Z'], d)[0]).numpy())
            Y_t = np.vstack(Y_t)


            ## From Sparse Gaussian Processes
            Z = m_current_dict['.inducing_variable.Z']
            kernel.kernel_g.variance.assign(m_current_dict['.kernel.kernel_g.variance'])
            kernel.kernel_g.lengthscales.assign(m_current_dict['.kernel.kernel_g.lengthscales'])
            kernel.kernels[0].lengthscales.assign(m_current_dict['.kernel.kernels[0].lengthscales'])
            kernel.kernels[0].variance.assign(m_current_dict['.kernel.kernels[0].variance'])
            qU_var_col_W = []
            for d in range(D):
                qU_var_col_W.append(tf.transpose(m_sgp.predict_f(m_current_dict['.inducing_variable.Z'], d)[1]).numpy())
            qU_var_col_W = np.vstack(qU_var_col_W).T

            ## Initialize the parameters
            # m_lvm = BayesianGPLVM(Y_t, Xr_dim, kernel=GPy.kern.RBF(Xr_dim, ARD=True), num_inducing=Mr)
            # m_lvm.likelihood.variance[:] = m_lvm.Y.var() * 0.1
            # m_lvm.likelihood.variance[:] = m_lvm.Y.var() * 0.01
            # m_lvm.optimize(max_iters=2)
            # From Bayesian GP-LVM
            # lengthscales2 = tf.convert_to_tensor(m_lvm.kern.lengthscale, dtype=default_float())
            # variance2 = tf.convert_to_tensor(m_lvm.kern.variance, dtype=default_float())
            # kernel_row = gpflow.kernels.RBF(lengthscales=lengthscales2, variance=variance2)
            # Z_row = m_lvm.Z.values.copy()
            # X_row_mean = m_lvm.X.mean.values.copy()
            # Xvariance_row = m_lvm.X.variance.values

            # qU_mean = m_lvm.posterior.mean.T.copy()
            # qU_var_row_W = jitchol(m_lvm.posterior.covariance)

            ### we do not use the Bayesian GPLVM to initialise parameters
            # lengthscales2 = tf.convert_to_tensor(1, dtype=default_float())
            # variance2 = tf.convert_to_tensor(0.1 , dtype=default_float())
            # kernel_row = gpflow.kernels.RBF(lengthscales=lengthscales2, variance=variance2)
            kernel_row = gpflow.kernels.RBF()

            X_row_mean = np.random.randn(D, Xr_dim)
            Xvariance_row = np.ones((D, Xr_dim)) * 0.01
            Z_row = X_row_mean[np.random.permutation(X_row_mean.shape[0])[:num_inducing[1]]].copy()


            # qU_mean = np.zeros((num_inducing[0], num_inducing[1]))
            qU_mean = np.random.randn(num_inducing[0], num_inducing[1])

            a = np.random.randn(num_inducing[1])[:, None]
            d = a.dot(a.T)
            d += np.eye(d.shape[0]) * 1e-6
            qU_var_row_W = jitchol(d)

##########################################################
            # Mr = num_inducing[1]
            # m_lvm = BayesianGPLVM(Y_t, Xr_dim, kernel=GPy.kern.RBF(Xr_dim, ARD=True), num_inducing=Mr)
            # # m_lvm.likelihood.variance[:] = m_lvm.Y.var() * 0.1
            # m_lvm.likelihood.variance[:] = m_lvm.Y.var() * 0.01
            # m_lvm.optimize(max_iters=2)

            # lengthscales2 = tf.convert_to_tensor(m_lvm.kern.lengthscale, dtype=default_float())
            # variance2 = tf.convert_to_tensor(m_lvm.kern.variance, dtype=default_float())
            # kernel_row = gpflow.kernels.RBF(lengthscales=lengthscales2, variance=variance2)

            # Z_row = m_lvm.Z.values.copy()
            # X_row_mean = m_lvm.X.mean.values.copy()
            # Xvariance_row = m_lvm.X.variance.values

            #
            # qU_mean = m_lvm.posterior.mean.T.copy()
            # qU_var_row_W = jitchol(m_lvm.posterior.covariance)



            print("Please initalize the parameter")

        super().__init__(**kwargs)
        num_data, num_latent_gps = X_row_mean.shape

        ## The different with No missing data
        self.output_dim = int(np.max(indexD))+1 ## the output dimension
        self.indexD = indexD ## index of the Output

        self.Heter_GaussianNoise = Parameter(Heter_GaussianNoise, transform=positive(lower=variance_lowerbound))  ## the likelihood for each output
        # self.Heter_GaussianNoise = Heter_GaussianNoise ## the likelihood for each output

        ### set up kernel and likelihood
        self.kernel = kernel
        self.kern_row = kernel_row
        self.num_data = num_data

        ## check this tomorrow
        self.Mr, self.Mc, self.Qr, self.Qc =  Z_row.shape[0], Z.shape[0], Z_row.shape[1], Z.shape[1]

        ## set up uncertainty input and inducing variables
        self.X_row_mean = Parameter(X_row_mean)
        self.Xvariance_row = Parameter(Xvariance_row, transform=positive())
        # The hidden variable H:
        self.X_row = DiagonalGaussian(self.X_row_mean, self.Xvariance_row)

        self.Xr_dim = Xr_dim ### We may use this one later
        self.Z = inducingpoint_wrapper(Z) ### this is the inducing point for real input X
        self.Z_row = inducingpoint_wrapper(Z_row) ### This is inducing point for the latent function h

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = tf.zeros((self.num_data, Xr_dim), dtype=default_float())
        if X_prior_var is None:
            X_prior_var = tf.ones((self.num_data, Xr_dim))

        self.X_prior_mean = tf.convert_to_tensor(np.atleast_1d(X_prior_mean), dtype=default_float())
        self.X_prior_var = tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float())



        ## Initialize and set up the mean and variance for Q_U
        # qU_mean = np.ones(num_inducing)
        # qU_var_col_W = np.random.randn(num_inducing[0],num_inducing[0] if qU_var_c_W_dim is None else qU_var_c_W_dim)*0.01
        qU_var_col_diag = np.full(num_inducing[0],1e-5)
        # qU_var_row_W = np.random.randn(num_inducing[1],num_inducing[1] if qU_var_r_W_dim is None else qU_var_r_W_dim)*0.01
        qU_var_row_diag = np.full(num_inducing[1],1e-5)

        self.qU_mean = Parameter(qU_mean)
        self.qU_var_c_W = Parameter(qU_var_col_W)
        self.qU_var_c_diag = Parameter(qU_var_col_diag, transform=positive())
        self.qU_var_r_W = Parameter(qU_var_row_W)
        self.qU_var_r_diag = Parameter(qU_var_row_diag, transform=positive())

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        X, Y = data

        # KL[q(H) || p(H)]
        dX_data_var = (
            self.Xvariance_row
            if self.Xvariance_row.shape.ndims == 2
            else tf.linalg.diag_part(self.Xvariance_row)
        )
        NQ = to_default_float(tf.size(self.X_row_mean))
        KL = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var))
        KL += 0.5 * tf.reduce_sum(tf.math.log(self.X_prior_var))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(self.X_row_mean - self.X_prior_mean) + dX_data_var) / self.X_prior_var
        )

        part_loglikelihood = self.Computed_part_log_likehood(X, Y)
        return part_loglikelihood - KL

    def Computed_part_log_likehood(self,Xc,Y):
        '''
        The idea of this function is from the inference function in Dai's paper
        :param Xc: Xc is real input
        :param Y: Y is a vector [Y1, Y2, Y3] and we have index for each output
        :return:
        '''
        qU_var_c = tdot_tensorflow(self.qU_var_c_W) + tf.linalg.diag(self.qU_var_c_diag)

        qU_var_r = tdot_tensorflow(self.qU_var_r_W) + tf.linalg.diag(self.qU_var_r_diag)

        # N, D, Mr, Mc, Qr, Qc = Y.shape[0], output_dim,Zr.shape[0], Zc.shape[0], Zr.shape[1], Zc.shape[1]
        N, D = tf.shape(Y)[0], self.output_dim
        Mr, Mc, Qr, Qc= self.Mr, self.Mc, self.Qr, self.Qc

        Xr = self.X_row
        Zr = self.Z_row
        Zc = self.Z

        beta = 1. / self.Heter_GaussianNoise

        ## Calculate the k value
        uncertain_inputs_r = isinstance(Xr, ProbabilityDistribution)
        uncertain_inputs_c = isinstance(Xc, ProbabilityDistribution)

        psi0_r, psi1_r, psi2_r = gatherPsiStat(self.kern_row, Xr, Zr, uncertain_inputs_r)

        # ======================================================================
        # Compute Common Components
        # ======================================================================

        Kuu_r = tf.identity(covariances.Kuu(Zr, self.kern_row,
                               jitter=default_jitter()))
        Lr_tensorflow = tf.linalg.cholesky(Kuu_r)


        Kuu_c = tf.identity(covariances.Kuu(Zc, self.kernel,
                               jitter=default_jitter()))
        Lc_tensorflow = tf.linalg.cholesky(Kuu_c)

        mu, Sr, Sc = self.qU_mean, qU_var_r, qU_var_c
        LSr_tensorflow = tf.linalg.cholesky(Sr)
        LSc_tensorflow = tf.linalg.cholesky(Sc)

        LcInvMLrInvT_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, tf.transpose(tf.linalg.triangular_solve(Lr_tensorflow, tf.transpose(mu), lower=True)), lower=True)
        LcInvLSc_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, LSc_tensorflow, lower=True)
        LrInvLSr_tensorflow = tf.linalg.triangular_solve(Lr_tensorflow, LSr_tensorflow, lower=True)

        LcInvScLcInvT_tensorflow = tdot_tensorflow(LcInvLSc_tensorflow)
        LrInvSrLrInvT_tensorflow = tdot_tensorflow(LrInvLSr_tensorflow)

        tr_LrInvSrLrInvT_tensorflow = tf.math.reduce_sum(tf.math.square(LrInvLSr_tensorflow))
        tr_LcInvScLcInvT_tensorflow = tf.math.reduce_sum(tf.math.square(LcInvLSc_tensorflow))

        mid_res = {
            'psi0_r': psi0_r,
            'psi1_r': psi1_r,
            'psi2_r': psi2_r,
            'Lr_tensorflow':Lr_tensorflow,
            'Lc_tensorflow':Lc_tensorflow,
            'LcInvMLrInvT_tensorflow': LcInvMLrInvT_tensorflow,
            'LcInvScLcInvT_tensorflow': LcInvScLcInvT_tensorflow,
            'LrInvSrLrInvT_tensorflow': LrInvSrLrInvT_tensorflow,
        }

        # ======================================================================
        # Compute log-likelihood
        # ======================================================================
        index_x_outputs = tf.cast(Xc[:, -1], tf.int32)
        x_input_with_replicates = Xc[..., :-1]
        x_allreplicates = tf.dynamic_partition(x_input_with_replicates, index_x_outputs, self.output_dim)

        logL_tensorflow = 0.
        for d in range(self.output_dim):
            psi0_c, psi1_c, psi2_c = gatherPsiStat(self.kernel, x_allreplicates[d], Zc, uncertain_inputs_c)

            logL_tensorflow += self.inference_d(d, beta, Y, self.indexD, psi0_c, psi1_c, psi2_c, mid_res, uncertain_inputs_r, uncertain_inputs_c, Mr, Mc)

        logL_tensorflow += - Mc * (tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lr_tensorflow))) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSr_tensorflow)))) \
                - Mr * (tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lc_tensorflow))) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSc_tensorflow))))\
                - tf.math.reduce_sum(tf.math.square(LcInvMLrInvT_tensorflow))/2. - tr_LrInvSrLrInvT_tensorflow * tr_LcInvScLcInvT_tensorflow/2. + Mr*Mc/2.

        return logL_tensorflow

    def inference_d(self, d, beta, Y, indexD, psi0_c, psi1_c, psi2_c, mid_res,
                    uncertain_inputs_r, uncertain_inputs_c, Mr, Mc):

        idx_d = indexD==d
        Y = Y[idx_d]
        N, D = tf.shape(Y)[0], 1
        beta = beta[d]

        psi0_r, psi1_r, psi2_r = mid_res['psi0_r'], mid_res['psi1_r'], mid_res['psi2_r']
        psi0_r, psi1_r, psi2_r = psi0_r[d], psi1_r[d:d+1], psi2_r[d]

        # psi0_c, psi1_c, psi2_c = mid_res['psi0_c'], mid_res['psi1_c'], mid_res['psi2_c']
        psi0_c, psi1_c, psi2_c = tf.reduce_sum(psi0_c), psi1_c, tf.reduce_sum(psi2_c, axis=0,)

        Lr_tensorflow = mid_res['Lr_tensorflow']
        Lc_tensorflow = mid_res['Lc_tensorflow']
        LcInvMLrInvT_tensorflow = mid_res['LcInvMLrInvT_tensorflow']
        LcInvScLcInvT_tensorflow = mid_res['LcInvScLcInvT_tensorflow']
        LrInvSrLrInvT_tensorflow = mid_res['LrInvSrLrInvT_tensorflow']

        LcInvPsi2_cLcInvT_tensorflow = backsub_both_sides_tensorflow(Lc_tensorflow, psi2_c, 'right')
        LrInvPsi2_rLrInvT_tensorflow = backsub_both_sides_tensorflow(Lr_tensorflow, psi2_r, 'right')

        LcInvPsi1_cT_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, tf.transpose(psi1_c), lower=True)
        LrInvPsi1_rT_tensorflow = tf.linalg.triangular_solve(Lr_tensorflow, tf.transpose(psi1_r), lower=True)

        tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT_tensorflow = tf.math.reduce_sum(LrInvPsi2_rLrInvT_tensorflow * LrInvSrLrInvT_tensorflow)
        tr_LcInvPsi2_cLcInvT_LcInvScLcInvT_tensorflow = tf.math.reduce_sum(LcInvPsi2_cLcInvT_tensorflow * LcInvScLcInvT_tensorflow)

        tr_LrInvPsi2_rLrInvT_tensorflow = tf.linalg.trace(LrInvPsi2_rLrInvT_tensorflow)
        tr_LcInvPsi2_cLcInvT_tensorflow = tf.linalg.trace(LcInvPsi2_cLcInvT_tensorflow)

        ### This log-likelihood have not included the KL(q(H)|p(H))
        logL_A_tensorflow = - tf.math.reduce_sum(tf.math.square(Y)) \
                            - tf.math.reduce_sum(tf.linalg.matmul(tf.linalg.matmul(LcInvMLrInvT_tensorflow, LcInvPsi2_cLcInvT_tensorflow, transpose_a=True),LcInvMLrInvT_tensorflow) * LrInvPsi2_rLrInvT_tensorflow) \
                            - tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT_tensorflow * tr_LcInvPsi2_cLcInvT_LcInvScLcInvT_tensorflow \
                            + 2 * tf.math.reduce_sum(Y * tf.linalg.matmul(tf.linalg.matmul(LcInvPsi1_cT_tensorflow, LcInvMLrInvT_tensorflow, transpose_a=True), LrInvPsi1_rT_tensorflow))\
                            - psi0_c * psi0_r + tr_LrInvPsi2_rLrInvT_tensorflow * tr_LcInvPsi2_cLcInvT_tensorflow


        logL_tensorflow = tf.cast(- N * D, dtype=default_float()) / 2. * (tf.math.log(2. * tf.cast(np.pi, dtype=default_float())) - tf.math.log(beta)) + beta / 2. * tf.cast(logL_A_tensorflow, dtype=default_float())

        return logL_tensorflow


    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:

        qU_var_c = tdot_tensorflow(self.qU_var_c_W) + tf.linalg.diag(self.qU_var_c_diag)
        qU_var_r = tdot_tensorflow(self.qU_var_r_W) + tf.linalg.diag(self.qU_var_r_diag)
        Xr = self.X_row
        Zr = self.Z_row
        Zc = self.Z
        Kuu_r = tf.identity(covariances.Kuu(Zr, self.kern_row, jitter=default_jitter()))
        Lr_tensorflow = tf.linalg.cholesky(Kuu_r)
        Kuu_c = tf.identity(covariances.Kuu(Zc, self.kernel, jitter=default_jitter()))
        Lc_tensorflow = tf.linalg.cholesky(Kuu_c)
        mu, Sr, Sc = self.qU_mean, qU_var_r, qU_var_c
        LSr_tensorflow = tf.linalg.cholesky(Sr)
        LSc_tensorflow = tf.linalg.cholesky(Sc)
        LcInvMLrInvT_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, tf.transpose(tf.linalg.triangular_solve(Lr_tensorflow, tf.transpose(mu), lower=True)), lower=True)
        LcInvLSc_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, LSc_tensorflow, lower=True)
        LrInvLSr_tensorflow = tf.linalg.triangular_solve(Lr_tensorflow, LSr_tensorflow, lower=True)
        LcInvScLcInvT_tensorflow = tdot_tensorflow(LcInvLSc_tensorflow)
        LrInvSrLrInvT_tensorflow = tdot_tensorflow(LrInvLSr_tensorflow)
        ## set up parameters
        Kmns_new = covariances.Kuf(self.Z, self.kernel, Xnew)
        psi1_c = tf.transpose(Kmns_new)
        psi2_c_n = psi1_c[:, :, None] * psi1_c[:, None, :]
        psi0_c = self.kernel.K_diag(Xnew)
        Lr = Lr_tensorflow
        Lc = Lc_tensorflow
        LcInvMLrInvT = LcInvMLrInvT_tensorflow
        LcInvScLcInvT = LcInvScLcInvT_tensorflow
        LrInvSrLrInvT = LrInvSrLrInvT_tensorflow
        LcInvPsi1_cT_tf = tf.linalg.triangular_solve(Lc, tf.transpose(psi1_c), lower=True)
        psi0_r = expectation(Xr, self.kern_row)
        psi1_r = expectation(Xr, (self.kern_row, Zr))
        psi2_r_n = expectation(Xr, (self.kern_row, Zr), (self.kern_row, Zr))
        LrInvPsi1_rT_tf = tf.linalg.triangular_solve(Lr, tf.transpose(psi1_r), lower=True)
        woodbury_vector = tf.linalg.matmul(LcInvMLrInvT, LrInvPsi1_rT_tf)

        ## calculate mu
        mu = tf.linalg.matmul(tf.transpose(LcInvPsi1_cT_tf), woodbury_vector)
        LrInvPsi2_r_nLrInvT_tf = backsub_both_sides_tensorflow_3DD(Lr, psi2_r_n)
        tr_LrInvPsi2_r_nLrInvT_tf = tf.linalg.trace(LrInvPsi2_r_nLrInvT_tf)
        LcInvPsi2_c_NLcInvT_tensorflow = backsub_both_sides_tensorflow_3DD(Lc, psi2_c_n)
        tr_LrInvPsi2_r_nLrInvT_LrInvSrLrInvT_tf = tf.linalg.trace(
            tf.linalg.matmul(LrInvSrLrInvT, LrInvPsi2_r_nLrInvT_tf))
        tr_LcInvPsi2_c_nLcInvT_LcInvScLcInvT_tf = tf.linalg.trace(
            tf.linalg.matmul(LcInvScLcInvT, LcInvPsi2_c_NLcInvT_tensorflow))

        tmp_tf = LrInvPsi2_r_nLrInvT_tf - tf.transpose(LrInvPsi1_rT_tf)[:, :, None] * tf.transpose(LrInvPsi1_rT_tf)[:, None, :]
        LcInv = tf.linalg.inv(Lc)
        psi1_c_LcInvT_LcInv_M_LrInvT = tf.linalg.matmul(tf.linalg.matmul(psi1_c, LcInv, transpose_b=True), LcInvMLrInvT)
        ## calculate variance
        var1_tf = tf.transpose(tf.linalg.diag_part(
            tf.linalg.matmul(tf.linalg.matmul(psi1_c_LcInvT_LcInv_M_LrInvT, tmp_tf), psi1_c_LcInvT_LcInv_M_LrInvT,
                             transpose_b=True)))
        var2_tf = psi0_c[:, None] * psi0_r[None, :]

        var3_tf = tr_LrInvPsi2_r_nLrInvT_tf[None, :] * tf.reduce_sum(tf.math.square(LcInvPsi1_cT_tf), axis=0)[:, None]

        var4_tf = tr_LrInvPsi2_r_nLrInvT_LrInvSrLrInvT_tf * tr_LcInvPsi2_c_nLcInvT_LcInvScLcInvT_tf[:, None]

        var_tf = var1_tf + var2_tf - var3_tf + var4_tf
        var_tf = var_tf.numpy()
        var_tf[var_tf < 1e-6] = 1e-6
        var_tf = tf.convert_to_tensor(var_tf, dtype=default_float())
        return mu, var_tf



###############################################################################################################################################################################
###############################################################################################################################################################################

                                                              ### We rewrite some existed models ###

###################################
#### Rewrite Zhenwen's code #######
###################################

class GPMultioutRegression(GPModel, ExternalDataTrainingLossMixin):
    """
    Gaussian Process model for multi-output regression without missing data

    Trying to rewrite the Zhenwen's paper into Tensorflow format. The main difference is that this format will auto-differentiation..
    For convenience, I will use the same notation as Zhenwen have done in GPy:https://gpy.readthedocs.io/en/deploy/_modules/GPy/models/gp_multiout_regression.html

    This is an implementation of Latent Variable Multiple Output Gaussian Processes (LVMOGP) in [Dai et al. 2017].

    Zhenwen Dai, Mauricio A. Alvarez and Neil D. Lawrence. Efficient Modeling of Latent Information in Supervised Learning using Gaussian Processes. In NIPS, 2017.

    :param int Xr_dim: the dimensionality of a latent space, in which output dimensions are embedded in
    :param Z: inducing inputs
    :param Z_row: inducing inputs for the latent space
    :param X_row: the initial value of the mean of the variational posterior distribution of points in the latent space
    :param Xvariance_row: the initial value of the variance of the variational posterior distribution of points in the latent space
    :type Xvariance_row: numpy.ndarray or None

    :param num_inducing: a tuple (M, Mr). M is the number of inducing points for GP of individual output dimensions. Mr is the number of inducing points for the latent space.
    :type num_inducing: (int, int)
    :param int qU_var_r_W_dim: the dimensionality of the covariance of q(U) for the latent space. If it is smaller than the number of inducing points, it represents a low-rank parameterization of the covariance matrix.
    :param int qU_var_c_W_dim: the dimensionality of the covariance of q(U) for the GP regression. If it is smaller than the number of inducing points, it represents a low-rank parameterization of the covariance matrix.
    :param str init: the choice of initialization: 'GP' or 'rand'. With 'rand', the model is initialized randomly. With 'GP', the model is initialized through a protocol as follows: (1) fits a sparse GP (2) fits a BGPLVM based on the outcome of sparse GP (3) initialize the model based on the outcome of the BGPLVM.
    :param str name: the name of the model
    """

    def __init__(self,
                 kernel,
                 likelihood,
                 kernel_row,
                 Xr_dim,
                 Z,
                 Z_row,
                 X_row_mean,
                 Xvariance_row,
                 num_inducing,
                 qU_mean,
                 qU_var_col_W,
                 qU_var_row_W,
                 qU_var_r_W_dim=None,
                 qU_var_c_W_dim=None,
                 mean_function=None,
                 X_prior_mean = None,
                 X_prior_var = None,
                 **kwargs):
        num_data, num_latent_gps = X_row_mean.shape

        super().__init__(kernel,likelihood,num_latent_gps=num_latent_gps,**kwargs)

        ## the number of inducing points should be same (Ignore this part)
        # assert num_inducing[0] == num_inducing[1]

        ### set up kernel and likelihood
        self.kernel = kernel
        self.likelihood = likelihood
        self.kern_row = kernel_row
        self.num_data = num_data

        self.Mr, self.Mc, self.Qr, self.Qc =  Z_row.shape[0], Z.shape[0], Z_row.shape[1], Z.shape[1]

        # f64 = lambda x: np.array(x, dtype=np.float64)
        # positive_with_min = lambda: tfp.bijectors.AffineScalar(shift=f64(1e-4))(tfp.bijectors.Softplus())

        ## set up uncertainty input and inducing variables
        self.X_row_mean = Parameter(X_row_mean)
        self.Xvariance_row = Parameter(Xvariance_row, transform=positive())
        self.X_row = DiagonalGaussian(self.X_row_mean, self.Xvariance_row)
        self.Xr_dim = Xr_dim ### We may use this one later
        self.Z = inducingpoint_wrapper(Z) ### this is the inducing point for real input X
        self.Z_row = inducingpoint_wrapper(Z_row) ### This is inducing point for the latent function h

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = tf.zeros((self.num_data, self.num_latent_gps), dtype=default_float())
        if X_prior_var is None:
            X_prior_var = tf.ones((self.num_data, self.num_latent_gps))

        self.X_prior_mean = tf.convert_to_tensor(np.atleast_1d(X_prior_mean), dtype=default_float())
        self.X_prior_var = tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float())

        # ## Initialize and set up the mean and variance for Q_U
        # qU_mean = np.ones(num_inducing)
        # qU_var_col_W = np.random.randn(num_inducing[0],num_inducing[0] if qU_var_c_W_dim is None else qU_var_c_W_dim)*0.01
        qU_var_col_diag = np.full(num_inducing[0],1e-5)
        # qU_var_row_W = np.random.randn(num_inducing[1],num_inducing[1] if qU_var_r_W_dim is None else qU_var_r_W_dim)*0.01
        qU_var_row_diag = np.full(num_inducing[1],1e-5)

        self.qU_mean = Parameter(qU_mean)
        self.qU_var_c_W = Parameter(qU_var_col_W)
        self.qU_var_c_diag = Parameter(qU_var_col_diag, transform=positive())
        self.qU_var_r_W = Parameter(qU_var_row_W)
        self.qU_var_r_diag = Parameter(qU_var_row_diag, transform=positive())

        self.posterior = None

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        X, Y = data

        # KL[q(H) || p(H)]
        dX_data_var = (
            self.Xvariance_row
            if self.Xvariance_row.shape.ndims == 2
            else tf.linalg.diag_part(self.Xvariance_row)
        )
        NQ = to_default_float(tf.size(self.X_row_mean))
        KL = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var))
        KL += 0.5 * tf.reduce_sum(tf.math.log(self.X_prior_var))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(self.X_row_mean - self.X_prior_mean) + dX_data_var) / self.X_prior_var
        )

        part_loglikelihood, self.posterior = self.Computed_part_log_likehood(X,Y)
        # if self.num_data is not None:
        #     num_data = tf.cast(self.num_data, kl.dtype)
        #     minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
        #     scale = num_data / minibatch_size
        # else:
        #     scale = tf.cast(1.0, kl.dtype)
        return part_loglikelihood - KL
        # return tf.reduce_sum(var_exp) * scale - kl


    def Computed_part_log_likehood(self,Xc,Y):
        '''
        The idea of this function is from the inference function in Dai's paper
        :param Xc: Xc is real input
        :param Y: Y is a matrix [Y1, Y2, Y3] and each column corresponding to a output
        :return:
        '''
        qU_var_c = tdot_tensorflow(self.qU_var_c_W) + tf.linalg.diag(self.qU_var_c_diag)

        qU_var_r = tdot_tensorflow(self.qU_var_r_W) + tf.linalg.diag(self.qU_var_r_diag)

        # N, D, Mr, Mc, Qr, Qc = Y.shape[0], Y.shape[1], self.Z_row.shape[0], self.Z.shape[0], self.Z_row.shape[1], self.Z.shape[1]
        N, D = tf.shape(Y)[0], tf.shape(Y)[1]
        Mr, Mc, Qr, Qc= self.Mr, self.Mc, self.Qr, self.Qc

        Xr = self.X_row
        Zr = self.Z_row
        Zc = self.Z
        beta = 1. / self.likelihood.variance
        ## Calculate the k valu
        uncertain_inputs_r = isinstance(Xr, ProbabilityDistribution)
        uncertain_inputs_c = isinstance(Xc, ProbabilityDistribution)
        psi0_r, psi1_r, psi2_r = gatherPsiStat_sum(self.kern_row, Xr, Zr, uncertain_inputs_r)
        psi0_c, psi1_c, psi2_c = gatherPsiStat_sum(self.kernel, Xc, Zc, uncertain_inputs_c)

        # ======================================================================
        # Compute Common Components
        # ======================================================================

        Kuu_r = tf.identity(covariances.Kuu(Zr, self.kern_row,
                               jitter=default_jitter()))
        # Kuu_r = tf.identity(self.kern_row.K(Zr))
        # diag.add(Kuu_r, self.const_jitter) ## If we need it, we will do it.
        Lr_tensorflow = tf.linalg.cholesky(Kuu_r)


        Kuu_c = tf.identity(covariances.Kuu(Zc, self.kernel,
                               jitter=default_jitter()))
        # Kuu_c = tf.identity(self.kernel.K(Zc))
        # diag.add(Kuu_c, self.const_jitter)  ## If we need it, we will do it.
        Lc_tensorflow = tf.linalg.cholesky(Kuu_c)

        mu, Sr, Sc = self.qU_mean, qU_var_r, qU_var_c
        LSr_tensorflow = tf.linalg.cholesky(Sr)
        LSc_tensorflow = tf.linalg.cholesky(Sc)

        LcInvMLrInvT_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, tf.transpose(
            tf.linalg.triangular_solve(Lr_tensorflow, tf.transpose(mu), lower=True)), lower=True)
        LcInvPsi2_cLcInvT_tensorflow = backsub_both_sides_tensorflow_3DD(Lc_tensorflow, psi2_c)
        LrInvPsi2_rLrInvT_tensorflow = backsub_both_sides_tensorflow_3DD(Lr_tensorflow, psi2_r)
        LcInvLSc_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, LSc_tensorflow, lower=True)
        LrInvLSr_tensorflow = tf.linalg.triangular_solve(Lr_tensorflow, LSr_tensorflow, lower=True)

        LcInvScLcInvT_tensorflow = tdot_tensorflow(LcInvLSc_tensorflow)
        LrInvSrLrInvT_tensorflow = tdot_tensorflow(LrInvLSr_tensorflow)

        LcInvPsi1_cT_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, tf.transpose(psi1_c), lower=True)
        LrInvPsi1_rT_tensorflow = tf.linalg.triangular_solve(Lr_tensorflow, tf.transpose(psi1_r), lower=True)

        tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT_tensorflow = tf.math.reduce_sum(LrInvPsi2_rLrInvT_tensorflow * LrInvSrLrInvT_tensorflow)
        tr_LcInvPsi2_cLcInvT_LcInvScLcInvT_tensorflow = tf.math.reduce_sum(LcInvPsi2_cLcInvT_tensorflow * LcInvScLcInvT_tensorflow)

        tr_LrInvSrLrInvT_tensorflow = tf.math.reduce_sum(tf.math.square(LrInvLSr_tensorflow))
        tr_LcInvScLcInvT_tensorflow = tf.math.reduce_sum(tf.math.square(LcInvLSc_tensorflow))
        tr_LrInvPsi2_rLrInvT_tensorflow = tf.linalg.trace(LrInvPsi2_rLrInvT_tensorflow)
        tr_LcInvPsi2_cLcInvT_tensorflow = tf.linalg.trace(LcInvPsi2_cLcInvT_tensorflow)

        # ======================================================================
        # Compute log-likelihood
        # ======================================================================

        ### This log-likelihood have not included the KL(q(H)|p(H))
        logL_A_tensorflow = - tf.math.reduce_sum(tf.math.square(Y)) \
                            - tf.math.reduce_sum(tf.linalg.matmul(tf.linalg.matmul(LcInvMLrInvT_tensorflow, LcInvPsi2_cLcInvT_tensorflow, transpose_a=True),LcInvMLrInvT_tensorflow) * LrInvPsi2_rLrInvT_tensorflow) \
                            - tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT_tensorflow * tr_LcInvPsi2_cLcInvT_LcInvScLcInvT_tensorflow \
                            + 2 * tf.math.reduce_sum(Y * tf.linalg.matmul(tf.linalg.matmul(LcInvPsi1_cT_tensorflow, LcInvMLrInvT_tensorflow, transpose_a=True), LrInvPsi1_rT_tensorflow))\
                            - psi0_c * psi0_r + tr_LrInvPsi2_rLrInvT_tensorflow * tr_LcInvPsi2_cLcInvT_tensorflow


        logL_tensorflow = tf.cast(- N * D, dtype=tf.float64)/2. * (tf.math.log(2.*tf.cast(np.pi, dtype = tf.float64))-tf.math.log(beta)) + beta/2.* tf.cast(logL_A_tensorflow,dtype = tf.float64)\
                          - Mc * (tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lr_tensorflow))) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSr_tensorflow)))) \
                          - Mr * (tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lc_tensorflow))) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSc_tensorflow))))\
                          - tf.math.reduce_sum(tf.math.square(LcInvMLrInvT_tensorflow))/2. - tr_LrInvSrLrInvT_tensorflow * tr_LcInvScLcInvT_tensorflow/2. + Mr*Mc/2.

        post = PosteriorMultioutput(LcInvMLrInvT=LcInvMLrInvT_tensorflow, LcInvScLcInvT=LcInvScLcInvT_tensorflow,
                LrInvSrLrInvT=LrInvSrLrInvT_tensorflow, Lr=Lr_tensorflow, Lc=Lc_tensorflow, kern_r=self.kern_row, Xr=Xr, Zr=Zr,Z=Zc,kernel=self.kernel)

        return logL_tensorflow, post


    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        ## This is raw predict without niose variable
        ## This f is only work for no complie since we have numpy() in the raw_predict
        ## However it work. The only different is that if we use complie (@tf.function), it would be super fast.
        mu, var = self.posterior.raw_predict_tf(Xnew=Xnew)
        return mu + self.mean_function(Xnew), var

class PosteriorMultioutput():
    '''
    This class is only for calculate the prediction.
    The reason we build this class is that we use the class to save some notation in order to calculate the prediciton.
    The prediction is mainly from the Dai's code, I am not fully sure the mathematic part, so I re-write it for our model.
    We can change it into tensorflow format. Now I have not change it since I have not seen the values. I may change it if we want.
    '''
    def __init__(self,LcInvMLrInvT, LcInvScLcInvT, LrInvSrLrInvT, Lr, Lc, kern_r, Xr, Zr,Z, kernel):
        self.LcInvMLrInvT = LcInvMLrInvT
        self.LcInvScLcInvT = LcInvScLcInvT
        self.LrInvSrLrInvT = LrInvSrLrInvT
        self.Lr = Lr
        self.Lc = Lc
        self.kern_r = kern_r
        self.Xr = Xr
        self.Zr = Zr
        self.Z = Z
        self.kernel = kernel

    def raw_predict(self, Xnew):
        N = Xnew.shape[0]
        # psi1_c = kern.K(Xnew, pred_var) GPy version
        Kmns_new = covariances.Kuf(self.Z, self.kernel, Xnew)
        Kmns_new = Kmns_new.numpy() ## change to numpy format
        psi1_c = Kmns_new.T

        # psi0_c = kern.Kdiag(Xnew)
        psi0_c = self.kernel.K_diag(Xnew)
        psi0_c = psi0_c.numpy() ## change to numpy format

        LcInvPsi1_cT = dtrtrs(self.Lc.numpy(), psi1_c.T)[0] ## change to numpy format

        D, Mr, Mc = self.Xr.cov.numpy().shape[0], self.Zr.Z.numpy().shape[0], self.LcInvMLrInvT.numpy().shape[0] ## change to numpy format

        # psi2_r_n = self.kern_r.psi2n(self.Zr, self.Xr)
        # psi0_r = self.kern_r.psi0(self.Zr, self.Xr)
        # psi1_r = self.kern_r.psi1(self.Zr, self.Xr)
        psi0_r = expectation(self.Xr, self.kern_r)
        psi1_r = expectation(self.Xr, (self.kern_r, self.Zr))
        psi2_r_n = expectation(self.Xr, (self.kern_r, self.Zr), (self.kern_r, self.Zr))
        psi0_r = psi0_r.numpy() ## change to numpy format
        psi1_r = psi1_r.numpy() ## change to numpy format
        psi2_r_n = psi2_r_n.numpy() ## change to numpy format

        LrInvPsi1_rT = dtrtrs(self.Lr.numpy(), psi1_r.T)[0]
        woodbury_vector = self.LcInvMLrInvT.numpy().dot(LrInvPsi1_rT)

        mu = np.dot(LcInvPsi1_cT.T, woodbury_vector)

        LrInvPsi2_r_nLrInvT = dtrtrs(self.Lr.numpy(), np.swapaxes((dtrtrs(self.Lr.numpy(), psi2_r_n.reshape(D*Mr,Mr).T)[0].T).reshape(D,Mr,Mr),1,2).reshape(D*Mr,Mr).T)[0].T.reshape(D,Mr,Mr)

        tr_LrInvPsi2_r_nLrInvT = np.diagonal(LrInvPsi2_r_nLrInvT,axis1=1,axis2=2).sum(1)
        tr_LrInvPsi2_r_nLrInvT_LrInvSrLrInvT = LrInvPsi2_r_nLrInvT.reshape(D,Mr*Mr).dot(self.LrInvSrLrInvT.numpy().flat)

        tmp = LrInvPsi2_r_nLrInvT - LrInvPsi1_rT.T[:,:,None]*LrInvPsi1_rT.T[:,None,:]
        tmp = np.swapaxes(tmp.reshape(D*Mr,Mr).dot(self.LcInvMLrInvT.numpy().T).reshape(D,Mr,Mc), 1,2).reshape(D*Mc,Mr).dot(self.LcInvMLrInvT.numpy().T).reshape(D,Mc,Mc)

        var1 = (tmp.reshape(D*Mc,Mc).dot(LcInvPsi1_cT).reshape(D,Mc,N)*LcInvPsi1_cT[None,:,:]).sum(1).T
        var2 = psi0_c[:,None]*psi0_r[None,:]
        var3 = tr_LrInvPsi2_r_nLrInvT[None,:]*np.square(LcInvPsi1_cT).sum(0)[:,None]
        var4 = tr_LrInvPsi2_r_nLrInvT_LrInvSrLrInvT[None,:]* (self.LcInvScLcInvT.numpy().dot(LcInvPsi1_cT)*LcInvPsi1_cT).sum(0)[:,None]
        var = var1+var2-var3+var4
        return mu, var
    def raw_predict_tf(self, Xnew):
        # N = Xnew
        Kmns_new = covariances.Kuf(self.Z, self.kernel, Xnew)
        psi1_c = tf.transpose(Kmns_new)
        psi2_c_n = psi1_c[:, :, None] * psi1_c[:, None, :]
        psi0_c = self.kernel.K_diag(Xnew)
        Lc = self.Lc
        Xr = self.Xr
        Zr = self.Zr
        Lr = self.Lr
        LrInvSrLrInvT = self.LrInvSrLrInvT
        LcInvScLcInvT = self.LcInvScLcInvT
        LcInvMLrInvT = tf.identity(self.LcInvMLrInvT)
        LcInvPsi1_cT_tf = tf.linalg.triangular_solve(Lc, tf.transpose(psi1_c), lower=True)


        psi0_r = expectation(Xr, self.kern_r)
        psi1_r = expectation(Xr, (self.kern_r, Zr))
        psi2_r_n = expectation(Xr, (self.kern_r, Zr), (self.kern_r, Zr))

        LrInvPsi1_rT_tf = tf.linalg.triangular_solve(Lr, tf.transpose(psi1_r), lower=True)

        woodbury_vector = tf.linalg.matmul(LcInvMLrInvT, LrInvPsi1_rT_tf)

        mu = tf.linalg.matmul(tf.transpose(LcInvPsi1_cT_tf), woodbury_vector)

        LrInvPsi2_r_nLrInvT_tf = backsub_both_sides_tensorflow_3DD(Lr, psi2_r_n)

        tr_LrInvPsi2_r_nLrInvT_tf = tf.linalg.trace(LrInvPsi2_r_nLrInvT_tf)

        LcInvPsi2_c_NLcInvT_tensorflow = backsub_both_sides_tensorflow_3DD(Lc, psi2_c_n)


        tr_LrInvPsi2_r_nLrInvT_LrInvSrLrInvT_tf = tf.linalg.trace(tf.linalg.matmul(LrInvSrLrInvT, LrInvPsi2_r_nLrInvT_tf))
        tr_LcInvPsi2_c_nLcInvT_LcInvScLcInvT_tf = tf.linalg.trace(tf.linalg.matmul(LcInvScLcInvT, LcInvPsi2_c_NLcInvT_tensorflow))


        tmp_tf = LrInvPsi2_r_nLrInvT_tf - tf.transpose(LrInvPsi1_rT_tf)[:, :, None] * tf.transpose(LrInvPsi1_rT_tf)[:, None, :]
        LcInv = tf.linalg.inv(Lc)
        psi1_c_LcInvT_LcInv_M_LrInvT = tf.linalg.matmul(tf.linalg.matmul(psi1_c, LcInv, transpose_b=True), LcInvMLrInvT)


        var1_tf = tf.transpose(tf.linalg.diag_part(tf.linalg.matmul(tf.linalg.matmul(psi1_c_LcInvT_LcInv_M_LrInvT, tmp_tf),psi1_c_LcInvT_LcInv_M_LrInvT,transpose_b=True)))

        var2_tf = psi0_c[:, None] * psi0_r[None, :]

        var3_tf = tr_LrInvPsi2_r_nLrInvT_tf[None, :] * tf.reduce_sum(tf.math.square(LcInvPsi1_cT_tf), axis=0)[:, None]

        var4_tf = tr_LrInvPsi2_r_nLrInvT_LrInvSrLrInvT_tf * tr_LcInvPsi2_c_nLcInvT_LcInvScLcInvT_tf[:,None]

        var_tf = var1_tf + var2_tf - var3_tf + var4_tf
        return mu, var_tf


class GPMultioutRegressionMD(BayesianModel, ExternalDataTrainingLossMixin):

    """
    Gaussian Process model for multi-output regression with missing data

    Trying to rewrite the Zhenwen's paper into Tensorflow format. The main difference is that this format will auto-differentiation..
    For convenience, I will use the same notation as Zhenwen have done in GPy:https://gpy.readthedocs.io/en/deploy/_modules/GPy/models/gp_multiout_regression_md.html#GPMultioutRegressionMD

    This is an implementation of Latent Variable Multiple Output Gaussian Processes (LVMOGP) in [Dai et al. 2017]. This model targets at the use case, in which each output dimension is observed at a different set of inputs. The model takes a different data format: the inputs and outputs observations of all the output dimensions are stacked together correspondingly into two matrices. An extra array is used to indicate the index of output dimension for each data point. The output dimensions are indexed using integers from 0 to D-1 assuming there are D output dimensions.

    Zhenwen Dai, Mauricio A. Alvarez and Neil D. Lawrence. Efficient Modeling of Latent Information in Supervised Learning using Gaussian Processes. In NIPS, 2017.

    :param Y: output observations, each column corresponding to an output dimension.
    :type Y: numpy.ndarray
    :param indexD: the array containing the index of output dimension for each data point
    :type indexD: numpy.ndarray
    :param kernel_row: a GPy kernel for the GP of the latent space ** defaults to RBF **
    :param Z_row: inducing inputs for the latent space
    :type Z_row: numpy.ndarray or None
    :param X_row: the initial value of the mean of the variational posterior distribution of points in the latent space
    :type X_row: numpy.ndarray or None
    :param Xvariance_row: the initial value of the variance of the variational posterior distribution of points in the latent space
    :type Xvariance_row: numpy.ndarray or None
    :param num_inducing: a tuple (M, Mr). M is the number of inducing points for GP of individual output dimensions. Mr is the number of inducing points for the latent space.
    :type num_inducing: (int, int)
    :param int qU_var_r_W_dim: the dimensionality of the covariance of q(U) for the latent space. If it is smaller than the number of inducing points, it represents a low-rank parameterization of the covariance matrix.
    :param int qU_var_c_W_dim: the dimensionality of the covariance of q(U) for the GP regression. If it is smaller than the number of inducing points, it represents a low-rank parameterization of the covariance matrix.
    """
    def __init__(self,
                 kernel,
                 kernel_row,
                 Heter_GaussianNoise,
                 Xr_dim,
                 Z,
                 Z_row,
                 X_row_mean,
                 Xvariance_row,
                 num_inducing,
                 indexD,
                 qU_var_r_W_dim=None,
                 qU_var_c_W_dim=None,
                 mean_function=None,
                 X_prior_mean = None,
                 X_prior_var = None,
                 **kwargs):

        super().__init__(**kwargs)
        num_data, num_latent_gps = X_row_mean.shape

        ## The different with No missing data
        self.output_dim = int(np.max(indexD))+1 ## the output dimension
        self.indexD = indexD ## index of the Output
        self.Heter_GaussianNoise = Parameter(Heter_GaussianNoise, transform=positive())  ## the likelihood for each output


        ### set up kernel and likelihood
        self.kernel = kernel
        self.kern_row = kernel_row
        self.num_data = num_data

        ## check this tomorrow
        self.Mr, self.Mc, self.Qr, self.Qc =  Z_row.shape[0], Z.shape[0], Z_row.shape[1], Z.shape[1]

        ## set up uncertainty input and inducing variables
        self.X_row_mean = Parameter(X_row_mean)
        self.Xvariance_row = Parameter(Xvariance_row, transform=positive())
        self.X_row = DiagonalGaussian(self.X_row_mean, self.Xvariance_row)
        self.Xr_dim = Xr_dim ### We may use this one later
        self.Z = inducingpoint_wrapper(Z) ### this is the inducing point for real input X
        self.Z_row = inducingpoint_wrapper(Z_row) ### This is inducing point for the latent function h

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = tf.zeros((self.num_data, Xr_dim), dtype=default_float())
        if X_prior_var is None:
            X_prior_var = tf.ones((self.num_data, Xr_dim))

        self.X_prior_mean = tf.convert_to_tensor(np.atleast_1d(X_prior_mean), dtype=default_float())
        self.X_prior_var = tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float())



        ## Initialize and set up the mean and variance for Q_U
        qU_mean = np.ones(num_inducing)
        qU_var_col_W = np.random.randn(num_inducing[0],num_inducing[0] if qU_var_c_W_dim is None else qU_var_c_W_dim)*0.01
        qU_var_col_diag = np.full(num_inducing[0],1e-5)
        qU_var_row_W = np.random.randn(num_inducing[1],num_inducing[1] if qU_var_r_W_dim is None else qU_var_r_W_dim)*0.01
        qU_var_row_diag = np.full(num_inducing[1],1e-5)

        self.qU_mean = Parameter(qU_mean)
        self.qU_var_c_W = Parameter(qU_var_col_W)
        self.qU_var_c_diag = Parameter(qU_var_col_diag, transform=positive())
        self.qU_var_r_W = Parameter(qU_var_row_W)
        self.qU_var_r_diag = Parameter(qU_var_row_diag, transform=positive())

        self.posterior = None

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)



    def elbo(self, data: RegressionData) -> tf.Tensor:
        X, Y = data

        # KL[q(H) || p(H)]
        dX_data_var = (
            self.Xvariance_row
            if self.Xvariance_row.shape.ndims == 2
            else tf.linalg.diag_part(self.Xvariance_row)
        )
        NQ = to_default_float(tf.size(self.X_row_mean))
        KL = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var))
        KL += 0.5 * tf.reduce_sum(tf.math.log(self.X_prior_var))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(self.X_row_mean - self.X_prior_mean) + dX_data_var) / self.X_prior_var
        )

        part_loglikelihood, self.posterior = self.Computed_part_log_likehood(X,Y)
        return part_loglikelihood - KL


    def Computed_part_log_likehood(self,Xc,Y):
        '''
        The idea of this function is from the inference function in Dai's paper
        :param Xc: Xc is real input
        :param Y: Y is a matrix [Y1, Y2, Y3] and each column corresponding to a output
        :return:
        '''
        qU_var_c = tdot_tensorflow(self.qU_var_c_W) + tf.linalg.diag(self.qU_var_c_diag)

        qU_var_r = tdot_tensorflow(self.qU_var_r_W) + tf.linalg.diag(self.qU_var_r_diag)

        # N, D, Mr, Mc, Qr, Qc = Y.shape[0], output_dim,Zr.shape[0], Zc.shape[0], Zr.shape[1], Zc.shape[1]
        N, D = tf.shape(Y)[0], self.output_dim
        Mr, Mc, Qr, Qc= self.Mr, self.Mc, self.Qr, self.Qc

        Xr = self.X_row
        Zr = self.Z_row
        Zc = self.Z

        beta = 1. / self.Heter_GaussianNoise

        ## Calculate the k value
        uncertain_inputs_r = isinstance(Xr, ProbabilityDistribution)
        uncertain_inputs_c = isinstance(Xc, ProbabilityDistribution)
        psi0_r, psi1_r, psi2_r = gatherPsiStat(self.kern_row, Xr, Zr, uncertain_inputs_r)
        psi0_c, psi1_c, psi2_c = gatherPsiStat(self.kernel, Xc, Zc, uncertain_inputs_c)

        # ======================================================================
        # Compute Common Components
        # ======================================================================

        Kuu_r = tf.identity(covariances.Kuu(Zr, self.kern_row,
                               jitter=default_jitter()))
        Lr_tensorflow = tf.linalg.cholesky(Kuu_r)


        Kuu_c = tf.identity(covariances.Kuu(Zc, self.kernel,
                               jitter=default_jitter()))
        Lc_tensorflow = tf.linalg.cholesky(Kuu_c)

        mu, Sr, Sc = self.qU_mean, qU_var_r, qU_var_c
        LSr_tensorflow = tf.linalg.cholesky(Sr)
        LSc_tensorflow = tf.linalg.cholesky(Sc)

        LcInvMLrInvT_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, tf.transpose(tf.linalg.triangular_solve(Lr_tensorflow, tf.transpose(mu), lower=True)), lower=True)
        LcInvLSc_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, LSc_tensorflow, lower=True)
        LrInvLSr_tensorflow = tf.linalg.triangular_solve(Lr_tensorflow, LSr_tensorflow, lower=True)

        LcInvScLcInvT_tensorflow = tdot_tensorflow(LcInvLSc_tensorflow)
        LrInvSrLrInvT_tensorflow = tdot_tensorflow(LrInvLSr_tensorflow)

        tr_LrInvSrLrInvT_tensorflow = tf.math.reduce_sum(tf.math.square(LrInvLSr_tensorflow))
        tr_LcInvScLcInvT_tensorflow = tf.math.reduce_sum(tf.math.square(LcInvLSc_tensorflow))

        mid_res = {
            'psi0_r': psi0_r,
            'psi1_r': psi1_r,
            'psi2_r': psi2_r,
            'psi0_c': psi0_c,
            'psi1_c': psi1_c,
            'psi2_c': psi2_c,
            'Lr_tensorflow':Lr_tensorflow,
            'Lc_tensorflow':Lc_tensorflow,
            'LcInvMLrInvT_tensorflow': LcInvMLrInvT_tensorflow,
            'LcInvScLcInvT_tensorflow': LcInvScLcInvT_tensorflow,
            'LrInvSrLrInvT_tensorflow': LrInvSrLrInvT_tensorflow,
        }

        # ======================================================================
        # Compute log-likelihood
        # ======================================================================
        logL_tensorflow = 0.
        for d in range(self.output_dim):
            logL_tensorflow += self.inference_d(d, beta, Y, self.indexD, mid_res, uncertain_inputs_r, uncertain_inputs_c, Mr, Mc)

        logL_tensorflow += - Mc * (tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lr_tensorflow))) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSr_tensorflow)))) \
                - Mr * (tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lc_tensorflow))) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSc_tensorflow))))\
                - tf.math.reduce_sum(tf.math.square(LcInvMLrInvT_tensorflow))/2. - tr_LrInvSrLrInvT_tensorflow * tr_LcInvScLcInvT_tensorflow/2. + Mr*Mc/2.

        #======================================================================
        # Compute the Posterior distribution of inducing points p(u|Y)
        #======================================================================
        post = PosteriorMultioutput(LcInvMLrInvT=LcInvMLrInvT_tensorflow, LcInvScLcInvT=LcInvScLcInvT_tensorflow,
                LrInvSrLrInvT=LrInvSrLrInvT_tensorflow, Lr=Lr_tensorflow, Lc=Lc_tensorflow, kern_r=self.kern_row, Xr=Xr, Zr=Zr,Z=Zc,kernel=self.kernel)

        return logL_tensorflow, post


    def inference_d(self, d, beta, Y, indexD, mid_res, uncertain_inputs_r, uncertain_inputs_c, Mr, Mc):

        idx_d = indexD==d
        Y = Y[idx_d]
        N, D = Y.shape[0], 1
        beta = beta[d]

        psi0_r, psi1_r, psi2_r = mid_res['psi0_r'], mid_res['psi1_r'], mid_res['psi2_r']
        psi0_c, psi1_c, psi2_c = mid_res['psi0_c'], mid_res['psi1_c'], mid_res['psi2_c']
        psi0_r, psi1_r, psi2_r = psi0_r[d], psi1_r[d:d+1], psi2_r[d]
        psi0_c, psi1_c, psi2_c = tf.reduce_sum(psi0_c[idx_d]), psi1_c[idx_d], tf.reduce_sum(psi2_c[idx_d], axis=0,)

        Lr_tensorflow = mid_res['Lr_tensorflow']
        Lc_tensorflow = mid_res['Lc_tensorflow']
        LcInvMLrInvT_tensorflow = mid_res['LcInvMLrInvT_tensorflow']
        LcInvScLcInvT_tensorflow = mid_res['LcInvScLcInvT_tensorflow']
        LrInvSrLrInvT_tensorflow = mid_res['LrInvSrLrInvT_tensorflow']



        LcInvPsi2_cLcInvT_tensorflow = backsub_both_sides_tensorflow(Lc_tensorflow, psi2_c, 'right')
        LrInvPsi2_rLrInvT_tensorflow = backsub_both_sides_tensorflow(Lr_tensorflow, psi2_r, 'right')

        LcInvPsi1_cT_tensorflow = tf.linalg.triangular_solve(Lc_tensorflow, tf.transpose(psi1_c), lower=True)
        LrInvPsi1_rT_tensorflow = tf.linalg.triangular_solve(Lr_tensorflow, tf.transpose(psi1_r), lower=True)

        tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT_tensorflow = tf.math.reduce_sum(LrInvPsi2_rLrInvT_tensorflow * LrInvSrLrInvT_tensorflow)
        tr_LcInvPsi2_cLcInvT_LcInvScLcInvT_tensorflow = tf.math.reduce_sum(LcInvPsi2_cLcInvT_tensorflow * LcInvScLcInvT_tensorflow)

        tr_LrInvPsi2_rLrInvT_tensorflow = tf.linalg.trace(LrInvPsi2_rLrInvT_tensorflow)
        tr_LcInvPsi2_cLcInvT_tensorflow = tf.linalg.trace(LcInvPsi2_cLcInvT_tensorflow)

        ### This log-likelihood have not included the KL(q(H)|p(H))
        logL_A_tensorflow = - tf.math.reduce_sum(tf.math.square(Y)) \
                            - tf.math.reduce_sum(tf.linalg.matmul(tf.linalg.matmul(LcInvMLrInvT_tensorflow, LcInvPsi2_cLcInvT_tensorflow, transpose_a=True),LcInvMLrInvT_tensorflow) * LrInvPsi2_rLrInvT_tensorflow) \
                            - tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT_tensorflow * tr_LcInvPsi2_cLcInvT_LcInvScLcInvT_tensorflow \
                            + 2 * tf.math.reduce_sum(Y * tf.linalg.matmul(tf.linalg.matmul(LcInvPsi1_cT_tensorflow, LcInvMLrInvT_tensorflow, transpose_a=True), LrInvPsi1_rT_tensorflow))\
                            - psi0_c * psi0_r + tr_LrInvPsi2_rLrInvT_tensorflow * tr_LcInvPsi2_cLcInvT_tensorflow


        logL_tensorflow = tf.cast(- N * D, dtype=default_float()) / 2. * (tf.math.log(2. * tf.cast(np.pi, dtype=default_float())) - tf.math.log(beta)) + beta / 2. * tf.cast(logL_A_tensorflow, dtype=default_float())

        return logL_tensorflow


    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        ## This is raw predict without niose variable
        ## This f is only work for no complie since we have numpy() in the raw_predict
        ## However it work. The only different is that if we use complie (@tf.function), it would be super fast.
        mu, var = self.posterior.raw_predict_tf(Xnew=Xnew)
        return mu, var




class SparseGPMD(BayesianModel, InternalDataTrainingLossMixin):
    '''
    We create a model for deal with missing data. In this model, we have different inputs and outputs (X_1,...,X_D; Y_1,...,Y_D).
    We assume all outputs have the same likelihood noise.
    '''
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        inducing_variable: InducingPoints,
        *,
        mean_function: Optional[MeanFunction] = None,
        indexD,
        noise,
        **kwargs
         ):
        """
        `data`:  a tuple of (X, Y), where the inputs X has shape [N, D]
            and the outputs Y has shape [N, 1].
        `inducing_variable`:  an InducingPoints instance or a matrix of
            the pseudo inputs Z, of shape [M, 1].
        `kernel`, `mean_function` are appropriate GPflow objects
        This method only works with a Gaussian likelihood.

        """
        likelihood = gpflow.likelihoods.Gaussian(noise)
        X_data, Y_data = data_input_to_tensor(data)
        super().__init__(**kwargs)
        self.kernel = kernel
        self.data = X_data, Y_data
        self.likelihood = likelihood
        # self.num_data = X_data.shape[0]
        self.output_dim = int(np.max(indexD))+1 ## the output dimension
        self.indexD = indexD

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

    def common_terms(self, d):
        X, Y = self.data
        idx_d = self.indexD == d
        Y_data = Y[idx_d]
        X_data = X[idx_d]
        # number_data_d = X_data.shape[0]
        number_data_d = tf.shape(X_data)[0]

        num_inducing = self.inducing_variable.num_inducing
        # err = Y_data - self.mean_function(X_data)  # size [N, 1]
        err = Y_data
        Kdiag = self.kernel(X_data, full_cov=False)
        kuf = covariances.Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        Luu = tf.linalg.cholesky(kuu)  # => Luu Luu^T = kuu
        V = tf.linalg.triangular_solve(Luu, kuf)  # => V^T V = Qff = kuf^T kuu^-1 kuf
        diagQff = tf.reduce_sum(tf.square(V), 0)
        nu = Kdiag - diagQff + self.likelihood.variance

        B = tf.eye(num_inducing, dtype=default_float()) + tf.linalg.matmul(
            V / nu, V, transpose_b=True
        )
        L = tf.linalg.cholesky(B)
        beta = err / tf.expand_dims(nu, 1)  # size [N, 1]
        alpha = tf.linalg.matmul(V, beta)  # size [N, 1]
        gamma = tf.linalg.triangular_solve(L, alpha, lower=True)  # size [N, 1]
        return err, nu, Luu, L, alpha, beta, gamma, number_data_d

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.fitc_log_marginal_likelihood()

    def fitc_log_marginal_likelihood(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """

        # FITC approximation to the log marginal likelihood is
        # log ( normal( y | mean, K_fitc ) )
        # where K_fitc = Qff + diag( \nu )
        # where Qff = Kfu kuu^{-1} kuf
        # with \nu_i = Kff_{i,i} - Qff_{i,i} + \sigma^2

        # We need to compute the Mahalanobis term -0.5* err^T K_fitc^{-1} err
        # (summed over functions).

        # We need to deal with the matrix inverse term.
        # K_fitc^{-1} = ( Qff + \diag( \nu ) )^{-1}
        #            = ( V^T V + \diag( \nu ) )^{-1}
        # Applying the Woodbury identity we obtain
        #            = \diag( \nu^{-1} ) - \diag( \nu^{-1} ) V^T ( I + V \diag( \nu^{-1} ) V^T )^{-1) V \diag(\nu^{-1} )
        # Let \beta =  \diag( \nu^{-1} ) err
        # and let \alpha = V \beta
        # then Mahalanobis term = -0.5* ( \beta^T err - \alpha^T Solve( I + V \diag( \nu^{-1} ) V^T, alpha ) )

        for d in range(self.output_dim):
            err, nu, Luu, L, alpha, beta, gamma, number_data_d = self.common_terms(d)
            mahalanobisTerm = -0.5 * tf.reduce_sum(
                tf.square(err) / tf.expand_dims(nu, 1)
            ) + 0.5 * tf.reduce_sum(tf.square(gamma))

            # We need to compute the log normalizing term -N/2 \log 2 pi - 0.5 \log \det( K_fitc )

            # We need to deal with the log determinant term.
            # \log \det( K_fitc ) = \log \det( Qff + \diag( \nu ) )
            #                    = \log \det( V^T V + \diag( \nu ) )
            # Applying the determinant lemma we obtain
            #                    = \log [ \det \diag( \nu ) \det( I + V \diag( \nu^{-1} ) V^T ) ]
            #                    = \log [ \det \diag( \nu ) ] + \log [ \det( I + V \diag( \nu^{-1} ) V^T ) ]
            # constantTerm = -0.5 * number_data_d * tf.math.log(tf.constant(2.0 * np.pi, default_float()))
            number_data_d = tf.cast(number_data_d, default_float())
            constantTerm = -0.5 * number_data_d * np.log(2.0 * np.pi)
            logDeterminantTerm = -0.5 * tf.reduce_sum(tf.math.log(nu)) - tf.reduce_sum(
                tf.math.log(tf.linalg.diag_part(L))
            )

            logNormalizingTerm = constantTerm + logDeterminantTerm
            logLoss = mahalanobisTerm + logNormalizingTerm

        return logLoss

    def predict_f(self, Xnew: InputData,d, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """
        _, _, Luu, L, _, _, gamma, _ = self.common_terms(d)
        Kus = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)  # [M, N]

        w = tf.linalg.triangular_solve(Luu, Kus, lower=True)  # [M, N]

        tmp = tf.linalg.triangular_solve(tf.transpose(L), gamma, lower=False)

        # mean = tf.linalg.matmul(w, tmp, transpose_a=True) + self.mean_function(Xnew)
        mean = tf.linalg.matmul(w, tmp, transpose_a=True)
        intermediateA = tf.linalg.triangular_solve(L, w, lower=True)

        if full_cov:
            var = (
                self.kernel(Xnew)
                - tf.linalg.matmul(w, w, transpose_a=True)
                + tf.linalg.matmul(intermediateA, intermediateA, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [1, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                - tf.reduce_sum(tf.square(w), 0)
                + tf.reduce_sum(tf.square(intermediateA), 0)
            )  # [N, P]
            var = tf.tile(var[:, None], [1, 1])

        return mean, var

############################################################################################
############ Here we rebuild the kernel for Jame's paper only for layer hierarchial GP #####
############################################################################################

class SingleHGP(GPR):
    def predict_f(
        self, Xnew: InputData, general: bool = False, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)
        if general is True:

            ####### This only consider single kernel_g #############
            # X_data_input = X_data[..., :-1]
            # kmm = self.kernel.kernel_g(X_data_input)
            # knn = self.kernel.kernel_g(Xnew, full_cov=full_cov)
            # kmn = self.kernel.kernel_g(X_data_input, Xnew)
            #
            # num_data = X_data_input.shape[0]
            # # s = tf.cast(tf.linalg.diag(tf.fill([num_data], 1e-10)),dtype=tf.float64)
            # s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))
            #
            # conditional = gpflow.conditionals.base_conditional
            # f_mean_zero, f_var = conditional(
            #     kmn, kmm + s, knn, err, full_cov=full_cov, white=False
            # )  # [N, P], [N, P] or [P, N, N]
            # # f_mean_zero, f_var = conditional(
            # #     kmn, kmm, knn, err, full_cov=full_cov, white=False
            # # )  # [N, P], [N, P] or [P, N, N]
            # f_mean = f_mean_zero + self.mean_function(Xnew)
            # return f_mean, f_var
            #
            ####### This kernel_g for Kmn and hierarchial for Kmm #############

            X_data_input = X_data[..., :-1]
            kmm = self.kernel(X_data)
            knn = self.kernel.kernel_g(Xnew, full_cov=full_cov)
            kmn = self.kernel.kernel_g(X_data_input, Xnew)

            num_data = X_data_input.shape[0]
            # s = tf.cast(tf.linalg.diag(tf.fill([num_data], 1e-10)),dtype=tf.float64)
            s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))

            conditional = gpflow.conditionals.base_conditional
            f_mean_zero, f_var = conditional(
                kmn, kmm + s, knn, err, full_cov=full_cov, white=False
            )  # [N, P], [N, P] or [P, N, N]
            # f_mean_zero, f_var = conditional(
            #     kmn, kmm, knn, err, full_cov=full_cov, white=False
            # )  # [N, P], [N, P] or [P, N, N]
            f_mean = f_mean_zero + self.mean_function(Xnew)
            return f_mean, f_var



        else:
            kmm = self.kernel(X_data)
            knn = self.kernel(Xnew, full_cov=full_cov)
            kmn = self.kernel(X_data, Xnew)

            num_data = X_data.shape[0]
            s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))

            conditional = gpflow.conditionals.base_conditional
            f_mean_zero, f_var = conditional(
                kmn, kmm + s, knn, err, full_cov=full_cov, white=False
            )  # [N, P], [N, P] or [P, N, N]
            f_mean = f_mean_zero + self.mean_function(Xnew)
            return f_mean, f_var


    def predict_y(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the held-out data at the input points.
        """
        if full_cov or full_output_cov:
            # See https://github.com/GPflow/GPflow/issues/1461
            raise NotImplementedError(
                "The predict_y method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        f_mean, f_var = self.predict_f(Xnew, general=False, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)


######################################################
############# We build our LMC model #################
######################################################

class SVGP_MOGP(SVGP):
    """
    The function of this model is exact same as svgp.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = lmc_conditional_mogp(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(Xnew), var

    def predict_y_one_output(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False):
        f_mean, f_var = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        d, _ = tf.unique(Xnew[:, -1])
        d = tf.cast(d, tf.int32)
        y_var = f_var + self.likelihood.likelihoods[d.numpy().squeeze()].variance
        return f_mean, y_var



class SVGP_MOGP_sum(SVGP):
    """
    The function of this model is exact simlar to SVGP. Here we assume different output with same index replica follow a same multi-output Gaussian proir. e.g.,
    y1(x) = [y_1_1(x), y_1_2(x)]^T y2(x)= [y_2_1(x), y_2_2(x)]^T
    [y_1_1(x), y_2_1(x)]^T ~ N(0, B*K + \sigma)
    [y_1_2(x), y_2_2(x)]^T ~ N(0, B*K + \sigma)

    """

    def __init__(self, num_replicates, **kwargs):
        super().__init__(**kwargs)
        self.num_replicates = num_replicates


    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = lmc_conditional_mogp(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(Xnew), var

    def predict_y_one_output(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False):
        f_mean, f_var = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        d, _ = tf.unique(Xnew[:, -1])
        d = tf.cast(d, tf.int32)
        y_var = f_var + self.likelihood.likelihoods[d.numpy().squeeze()].variance
        return f_mean, y_var

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        ############## I need to modify the ELBO here.
        X, Y = data
        # X_list_all = []
        # Y_list_all = []
        kl = self.prior_kl()
        var_exp_sum = 0
        ind = X[:,-1]
        ind = tf.cast(ind, tf.int32)
        X_list_all = tf.dynamic_partition(X[:, :-1], ind, self.num_replicates)
        Y_list_all = tf.dynamic_partition(Y[:,:-1], ind, self.num_replicates)

        for rep in range(self.num_replicates):
            # index_appendix = X[:, -1][:, None] == rep
            # X_list_all.append(X[:, :-1][index_appendix.squeeze()])
            # Y_list_all.append(Y[:, :-1][index_appendix.squeeze()])
            # X_one_with_replica = X[:, :-1][index_appendix.squeeze()]
            # Y_one_with_replica = Y[:, :-1][index_appendix.squeeze()]
            f_mean, f_var = self.predict_f(X_list_all[rep], full_cov=False, full_output_cov=False)
            var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y_list_all[rep])
            var_exp_sum = var_exp_sum + tf.reduce_sum(var_exp)
        return var_exp_sum  - kl