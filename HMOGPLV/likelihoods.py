                        #####################################################
                        ### We build our own likelihood based on GPflow.#####
                        #####################################################
##### we import from gpflow and tensorflow #####
import tensorflow as tf
import numpy as np
from gpflow.likelihoods import ScalarLikelihood, SwitchedLikelihood,Softmax,RobustMax
from gpflow import logdensities
from gpflow.base import Parameter
from gpflow.utilities import positive
import tensorflow_probability as tfp


                            #####################################
                            #### Building new likelihood ########
                            #####################################



class SwitchedLikelihood_HierarchicalMOGP_W(SwitchedLikelihood):
    def __init__(self, likelihood_list, **kwargs):
        self.likelihoods = likelihood_list
        self.num_output = len(likelihood_list)


    def _partition_and_stitch(self, args, func_name):
        """
        args is a list of tensors, to be passed to self.likelihoods.<func_name>

        args[-1] is the 'Y' argument, which contains the indexes to self.likelihoods or outputs

        We sort Y based on args[-1], we do not need to sort for f-mean and f-variance since it is already sorted
        """
        # get the index from Y
        Y = args[-1]
        ind = Y[..., -1]
        Y = Y[..., :-1]
        ind = tf.cast(ind, tf.int32)
        Y_sorted = tf.concat(tf.dynamic_partition(Y, ind, self.num_output), axis=0)
        Ind_sort = tf.cast(tf.sort(ind), tf.int32)
        args = args[:-1]

        # split up the arguments into chunks corresponding to the relevant likelihoods
        args = [tf.dynamic_partition(X, Ind_sort, self.num_output) for X in args]
        args = zip(*args)
        arg_Y = tf.dynamic_partition(Y_sorted, Ind_sort, self.num_output)

        # # apply the likelihood-function to each section of the data
        funcs = [getattr(lik, func_name) for lik in self.likelihoods]
        results = [f(*args_i, Yi) for f, args_i, Yi in zip(funcs, args, arg_Y)]

        return results


    def variational_expectations(self, Fmu, Fvar, Y):
        return self._partition_and_stitch([Fmu, Fvar, Y], "variational_expectations")





##################################################################################################################
##################################################################################################################
##################################################################################################################
### SwithedLikelihood. It works on the Pablo's paper toy example;
### It works that you have same input but different latent parameter function

                        #####################################
                        ####            MOGP         ########
                        #####################################
class SwitchedLikelihood_MOGP(SwitchedLikelihood):
    def __init__(self, likelihood_list, num_latent_list, **kwargs):
        self.likelihoods = likelihood_list
        self.num_latent_list = num_latent_list
        self.num_latent = sum(num_latent_list)
        self.num_task = len(likelihood_list)


    def _partition_and_stitch(self, args, func_name):
        """
        args is a list of tensors, to be passed to self.likelihoods.<func_name>

        args[-1] is the 'Y' argument, which contains the indexes to self.likelihoods.

        This function splits up the args using dynamic_partition, calls the
        relevant function on the likelihoods, and re-combines the result.
        """
        # get the index from Y
        Y = args[-1]
        ind = Y[..., -1]
        ind = tf.cast(ind, tf.int32)
        num_data = tf.math.bincount(ind, minlength=self.num_task)
        num_data = tf.repeat(num_data, self.num_latent_list)
        ind_task = tf.repeat(tf.range(self.num_task), self.num_latent_list)
        ind_task = tf.repeat(ind_task, num_data)
        Y = Y[..., :-1]
        args = args[:-1]


        # split up the arguments into chunks corresponding to the relevant likelihoods
        args = [tf.dynamic_partition(X, ind_task, self.num_task) for X in args]
        args = [
            [
                tf.transpose(tf.reshape(f_t, [n_latent, -1]))
                for f_t, n_latent in zip(arg, self.num_latent_list)
            ]
            for arg in args
        ]
        #
        args = zip(*args)
        arg_Y = tf.dynamic_partition(Y, ind, self.num_task)

        # # apply the likelihood-function to each section of the data
        funcs = [getattr(lik, func_name) for lik in self.likelihoods]
        results = [f(*args_i, Yi) for f, args_i, Yi in zip(funcs, args, arg_Y)]


        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, self.num_task)
        results = tf.dynamic_stitch(partitions, results)

        return results

    def variational_expectations(self, Fmu, Fvar, Y):
        return self._partition_and_stitch([Fmu, Fvar, Y], "variational_expectations")




##################################
#### This is only for testing#####
##################################

################################################################################################
#### whether we can build the kernel in this way
class Gaussianr(ScalarLikelihood):
    r"""
    The Gaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with constant
    variance.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the
    likelihood variance by default.
    """

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

    def __init__(self, variance=1.0, variance_lower_bound=DEFAULT_VARIANCE_LOWER_BOUND, **kwargs):
        """
        :param variance: The noise variance; must be greater than
            ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """
        super().__init__(**kwargs)

        if variance <= variance_lower_bound:
            raise ValueError(
                f"The variance of the Gaussian likelihood must be strictly greater than 		{variance_lower_bound}"
            )

        self.variance = Parameter(variance, transform=positive(lower=variance_lower_bound))

    def _scalar_log_prob(self, F, Y):
        return logdensities.gaussian(Y, F, self.variance)

    def _conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def _predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def _predict_log_density(self, Fmu, Fvar, Y):
        return tf.reduce_sum(logdensities.gaussian(Y, Fmu, Fvar + self.variance), axis=-1)

    def _variational_expectations(self, Fmu, Fvar, Y):
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(self.variance)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variance,
            axis=-1,
        )







