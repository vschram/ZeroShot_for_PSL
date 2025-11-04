                                    #################################################
                                    ### We build our own kernel based on GPflow.#####
                                    #################################################
##### we import from gpflow and tensorflow #####
import tensorflow as tf
import numpy as np
from gpflow.kernels import SquaredExponential, Combination, Coregion, Kernel, Stationary,IsotropicStationary
from gpflow.config import default_float, default_jitter
from gpflow.base import Parameter
from gpflow.utilities import to_default_float, positive
from gpflow.utilities.ops import square_distance

                                                #################################
                                                #### Building new kernel ########
                                                #################################

#############################################################################
########## Hierarchial_Kernel_replicated has different replicated  ##########
#############################################################################
### This kernel is used for the Hierarchial GPs with inducing variables.
class Hierarchial_kernel_replicated(Combination):
    """
    In this function, we would like to build a hierarchical kernel.
    g(x) & \sim \mathcal{G} \mathcal{P}\left(\mathbf{0}, k_{g}\left(X, X}\right)\right)
    f(x) & \sim \mathcal{G} \mathcal{P}\left(g(X), k_{f}\left(X, X\right)\right)

    In each block, there will be a same kenel k_{f}. There k_{g} kernel is for all the output correlation
    E.g there are 2 outputs with 2 replicated data:
    X1 = X11, X12 where X12 is the second replicated data in the first output
    X2 = X21, X22 where X22 is the second replicated data in the second output
    Thus, the kernel will be
    [[ kg(X11,X11) + kf(X11, X11),     kg(X11,X12),         kg(X11,X21),                  kg(X11,X22)],
     [ kg(X12,X11),       kg(X12,X12) + kf(X12, X12),       kg(X12,X21),                  kg(X12,X22)],
     [ kg(X21,X11),       kf(X21, X12),      kg(X21,X21) + kf(X21, X21),                  kg(X21,X22)],
     [ kg(X22,X11),       kg(X22,X12),       kg(X22,X21)                    kg(X22,X22) + kf(X22, X22)]]

    When we want to make prediction for the replicate 0, we can easily use X_pre = [X_pre 0].


    """
    def __init__(self, kernel_g, kernel_f, total_replicated, name=None):
        ## We set up the kernels: global kernel and specific kernel
        self.kernel_g = kernel_g
        self.total_replicated = total_replicated ### the total number of replicated data set
        ks = []
        for k in kernel_f:
            ks.append(k)

        Combination.__init__(self, ks, name)

    def _partition_and_stitch_kernel(self, X, X2):
        if X2 is None:
            ind = X[..., -1]
            ind = tf.cast(ind, tf.int32)  ### the index is for each replicated data set.
            X = X[..., :-1]
            d, _ = tf.unique(ind)
            if tf.shape(d) == 1:
                ### Kff for one output
                return self.kernels[0].K(X) + self.kernel_g.K(X)
            else:
                # split up the arguments into chunks corresponding to different replicated data set
                args_raw = tf.dynamic_partition(X, ind,  self.total_replicated)
                args = zip(*[args_raw])
                ### calculate the Kff for each replicated data set in the specifical kernel Kf
                results = [tf.reduce_sum(self.kernels[0].K(*args_i),0) for args_i in zip(args)]
                # This is for general kernel Kg
                X_l = tf.concat(args_raw, axis=0)

                return self.block_diagonal(results) + self.kernel_g.K(X_l)

        else:
            # split up the arguments into chunks corresponding to different replicated data set in X (observation X)

            ind1 = X[..., -1]
            ind1 = tf.cast(ind1, tf.int32)
            d1, _ = tf.unique(ind1)
            # num_replicated1 = tf.shape(d1)[0]
            X1 = X[..., :-1]
            # print('num1',num_replicated1)
            args1_raw = tf.dynamic_partition(X1, ind1, self.total_replicated)
            args1 = zip(*[args1_raw])

            # split up the arguments into chunks corresponding to different inducing replicated data set in U (inducing variable U)
            ind2 = X2[..., -1]
            ind2 = tf.cast(ind2, tf.int32)
            X2 = X2[..., :-1]
            d2, _ = tf.unique(ind2)
            # num_replicated2 = tf.shape(d2)[0]
            # print('num2',num_replicated2)
            args2_raw = tf.dynamic_partition(X2, ind2, self.total_replicated)
            args2 = zip(*[args2_raw])

            # This one combine X and X2 together based on the index of replicated data set. This is for general kernel
            X_l_1 = tf.concat(args1_raw, axis=0)
            X_l_2 = tf.concat(args2_raw, axis=0)

            # apply the kernel-function to each section of the data (The tf.reduce_sum(f.K(X=args1x,X2=args2x), axis=[0,2]) is for reducing the dimension [1,M,1,M] to [M,M]
            kernel_block = [tf.reduce_sum(self.kernels[0].K(X=args1x, X2=args2x), axis=[0, 2]) for args1x, args2x in
                                    zip(args1, args2)]

            blk_diag_operatorUU = self.block_diagonal(kernel_block)

            return blk_diag_operatorUU + self.kernel_g.K(X_l_1, X_l_2)

    def K(self, X, X2=None):  # [N, N2]
        return self._partition_and_stitch_kernel(X, X2)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

    def block_diagonal(self, matrices, dtype=tf.float64):
        '''This function is used for the build the block diagonal matrix'''
        matrices = [tf.convert_to_tensor(value=matrix, dtype=dtype) for matrix in matrices]
        blocked_rows = tf.compat.v1.Dimension(0)
        blocked_cols = tf.compat.v1.Dimension(0)
        batch_shape = tf.TensorShape(None)
        for matrix in matrices:
            full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
            batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
            blocked_rows += full_matrix_shape[-2]
            blocked_cols += full_matrix_shape[-1]
        ret_columns_list = []
        for matrix in matrices:
            matrix_shape = tf.shape(input=matrix)
            ret_columns_list.append(matrix_shape[-1])
        ret_columns = tf.add_n(ret_columns_list)
        row_blocks = []
        current_column = 0
        for matrix in matrices:
            matrix_shape = tf.shape(input=matrix)
            row_before_length = current_column
            current_column += matrix_shape[-1]
            row_after_length = ret_columns - current_column
            row_blocks.append(tf.pad(
                    tensor=matrix,
                    paddings=tf.concat(
                        [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                         [(row_before_length, row_after_length)]],
                        axis=0)))
        blocked = tf.concat(row_blocks, -2)

        return blocked

class Hierarchial_kernel_replicated_masked(Combination):
    def __init__(self, kernel_g, kernel_f, total_replicated, name=None):
        self.kernel_g = kernel_g
        self.total_replicated = total_replicated  # The total number of replicated data sets
        ks = [k for k in kernel_f]

        Combination.__init__(self, ks, name)

    def _partition_and_stitch_kernel(self, X, X2=None):
        if X2 is None:
            # Handle the case where X2 is None (e.g., calculating the covariance matrix for a single set of data points)
            ind = X[..., -1]
            ind = tf.cast(ind, tf.int32)  # Index for each replicated data set
            X = X[..., :-1]  # Remove the index column
            d, _ = tf.unique(ind)

            if tf.shape(d) == 1:
                # If there's only one output, return Kf + Kg
                return self.kernels[0].K(X) + self.kernel_g.K(X)
            else:
                # Split up the arguments into chunks for each replicated data set
                args_raw = tf.dynamic_partition(X, ind, self.total_replicated)

                args = zip(*[args_raw])

                # Calculate the Kf for each replicated data set using kernel_f
                results = [tf.reduce_sum(self.kernels[0].K(*args_i), 0) for args_i in zip(args)]

                # This is for the general kernel Kg
                X_l = tf.concat(args_raw, axis=0)

                return self._mask_diagonal(self.block_diagonal(results) + self.kernel_g.K(X_l))

        else:
            # Handle the case where X2 is provided (e.g., cross-covariance calculation)
            ind1 = X[..., -1]
            ind1 = tf.cast(ind1, tf.int32)
            d1, _ = tf.unique(ind1)
            X1 = X[..., :-1]  # Remove the index column
            args1_raw = tf.dynamic_partition(X1, ind1, self.total_replicated)
            args1 = zip(*[args1_raw])

            ind2 = X2[..., -1]
            ind2 = tf.cast(ind2, tf.int32)
            X2 = X2[..., :-1]
            d2, _ = tf.unique(ind2)
            args2_raw = tf.dynamic_partition(X2, ind2, self.total_replicated)
            args2 = zip(*[args2_raw])

            # Combine X and X2 based on the index of the replicated data set
            X_l_1 = tf.concat(args1_raw, axis=0)
            X_l_2 = tf.concat(args2_raw, axis=0)

            # Apply the kernel function to each section of the data
            kernel_block = [tf.reduce_sum(self.kernels[0].K(X=args1x, X2=args2x), axis=[0, 2]) for args1x, args2x in
                            zip(args1, args2)]

            blk_diag_operatorUU = self.block_diagonal(kernel_block)

            # Mask the diagonal and return the result
            return blk_diag_operatorUU + self.kernel_g.K(X_l_1, X_l_2)

    def _mask_diagonal(self, matrix):
        """
        Masks the diagonal of the covariance matrix by setting it to zero.

        :param matrix: The input covariance matrix to mask
        :return: The matrix with diagonal elements set to zero
        """
        epsilon = 1e-1
        # Create an identity matrix of the same shape as the input matrix
        #eye = tf.eye(tf.shape(matrix)[0], dtype=matrix.dtype)
        eye = tf.eye(tf.shape(matrix)[0], dtype=matrix.dtype)
        try:
            masked_matrix = matrix - eye + epsilon * eye
            #masked_matrix = matrix * (1 - eye)
        except:
            print("**********ERROR*******************")

        # Set the diagonal elements to zero
        #return matrix * (1 - eye)
        return masked_matrix

    def K(self, X, X2=None):  # [N, N2]
        return self._partition_and_stitch_kernel(X, X2)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

    def block_diagonal(self, matrices, dtype=tf.float64):
        '''This function is used for the build the block diagonal matrix'''
        matrices = [tf.convert_to_tensor(value=matrix, dtype=dtype) for matrix in matrices]
        blocked_rows = tf.compat.v1.Dimension(0)
        blocked_cols = tf.compat.v1.Dimension(0)
        batch_shape = tf.TensorShape(None)
        for matrix in matrices:
            full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
            batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
            blocked_rows += full_matrix_shape[-2]
            blocked_cols += full_matrix_shape[-1]
        ret_columns_list = []
        for matrix in matrices:
            matrix_shape = tf.shape(input=matrix)
            ret_columns_list.append(matrix_shape[-1])
        ret_columns = tf.add_n(ret_columns_list)
        row_blocks = []
        current_column = 0
        for matrix in matrices:
            matrix_shape = tf.shape(input=matrix)
            row_before_length = current_column
            current_column += matrix_shape[-1]
            row_after_length = ret_columns - current_column
            row_blocks.append(tf.pad(
                    tensor=matrix,
                    paddings=tf.concat(
                        [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                         [(row_before_length, row_after_length)]],
                        axis=0)))
        blocked = tf.concat(row_blocks, -2)

        return blocked
######################################
########## lmc kernel   ##############
######################################
class lmc_kernel(Combination):
    '''
    LMC kernel (This is inherent from Gym). See the Kernels for Vector-Valued Functions: a Review by Mauricio etc.
    '''
    def __init__(self, output_dim, kernels, ranks=None, name=None):
        """
        A Kernel for Linear Model of Coregionalization
        """
        self.output_dim = output_dim
        if ranks is None:
            ranks = np.ones_like(kernels)

        ks = []
        self.coregs = []
        for k, r in zip(kernels, ranks):
            ### The active_dims is -1. It is the last column of X.
            coreg = Coregion(output_dim, r, active_dims=slice(-1, None))
            # coreg.kappa = default_jitter() * tf.constant(np.zeros(output_dim))  # noqa
            coreg.kappa = tf.constant(np.zeros(output_dim))
            coreg.W.assign(np.random.rand(output_dim, r))
            self.coregs.append(coreg)
            ks.append(k)

        Combination.__init__(self, ks, name)

    def Kgg(self, X, X2=None, full_cov=True): # [L, N, N2]
        if full_cov:
            if X2 is None:
                return tf.stack([coreg(X) * k(X[:, :-1]) for coreg, k in
                                 zip(self.coregs, self.kernels)], axis=0)
            return tf.stack([coreg(X, X2) * k(X[:, :-1], X2[:, :-1])
                             for coreg, k in zip(self.coregs, self.kernels)],
                            axis=0)
        ## We use coreg(X, full_cov=Flase) to calculate the B together
        return tf.stack([coreg(X, full_cov=False) * k(X[:, :-1], full_cov=False)
                         for coreg, k in zip(self.coregs, self.kernels)],
                        axis=0)

    def K(self, X, X2=None):  # [N, N2]
        return tf.reduce_sum(self.Kgg(X, X2), axis=0)

    def K_diag(self, X):  # [N]
        return tf.reduce_sum(self.Kgg(X, full_cov=False), axis=0)



#####################
#### Test kernel#####
#####################
class SE_test(SquaredExponential):
    '''
    This is only for testing whether we can build the kernel in this way
    '''
    pass



class Isotrpic_Stationary_no_variance(Kernel):

    """
    This is similar to the IsotropicStationary kernel in gpflow but no variance variable.
    """

    def __init__(self,lengthscales, **kwargs):
        """
        :param variance: the (initial) value for the variance parameter.
        :param lengthscales: the (initial) value for the lengthscale
            parameter(s), to induce ARD behaviour this must be initialised as
            an array the same length as the the number of active dimensions
            e.g. [1., 1., 1.]. If only a single value is passed, this value
            is used as the lengthscale of each dimension.
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")
        super().__init__(**kwargs)
        self.lengthscales = Parameter(lengthscales, transform=positive())
        self._validate_ard_active_dims(self.lengthscales)


    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        return self.lengthscales.shape.ndims > 0
    def scale(self, X):
        X_scaled = X / self.lengthscales if X is not None else X
        return X_scaled


    def K(self, X, X2=None):
        r2 = self.scaled_squared_euclid_dist(X, X2)
        return self.K_r2(r2)

    def K_r2(self, r2):
        if hasattr(self, "K_r"):
            # Clipping around the (single) float precision which is ~1e-45.
            r = tf.sqrt(tf.maximum(r2, 1e-36))
            return self.K_r(r)  # pylint: disable=no-member
        raise NotImplementedError

    def scaled_squared_euclid_dist(self, X, X2=None):
        """
        Returns ‖(X - X2ᵀ) / ℓ‖², i.e. the squared L₂-norm.
        """
        return square_distance(self.scale(X), self.scale(X2))


    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(tf.constant(1,dtype=tf.float64)))
        # return tf.fill(tf.shape(X)[:-1], tf.squeeze(1.0))


class Square_Exp_basic(IsotropicStationary):
    """
     This is basic kernel fro additive kernel. The kenel is k(r) = exp{-½ r²}
    """
    def K_r2(self, r2):
        return tf.exp(-0.5 * r2)


################################################
########## Kernel for fix input   ##############
################################################
class Input_H_kernel_JupyterNotebook(Kernel):
    """
    Base class for isotropic stationary kernels, i.e. kernels that only
    depend on

        r = ‖x - x'‖

    Derived classes should implement one of:

        K_r2(self, r2): Returns the kernel evaluated on r² (r2), which is the
        squared scaled Euclidean distance Should operate element-wise on r2.

        K_r(self, r): Returns the kernel evaluated on r, which is the scaled
        Euclidean distance. Should operate element-wise on r.
    """

    def __init__(self,lengthscales,number_task, **kwargs):
        """
        :param variance: the (initial) value for the variance parameter.
        :param lengthscales: the (initial) value for the lengthscale
            parameter(s), to induce ARD behaviour this must be initialised as
            an array the same length as the the number of active dimensions
            e.g. [1., 1., 1.]. If only a single value is passed, this value
            is used as the lengthscale of each dimension.
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")
        super().__init__(**kwargs)
        self.lengthscales = Parameter(lengthscales, transform=positive())
        self._validate_ard_active_dims(self.lengthscales)
        self.number_task = number_task
        W = np.random.randn(self.number_task)[:,None]
        # W = np.array([1,2])[:, None]
        self.W = Parameter(W)


    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        return self.lengthscales.shape.ndims > 0
    def scale(self, X):
        X_scaled = X / self.lengthscales if X is not None else X
        return X_scaled

    def K(self,X,X2=None):
        ### We augment W to make sure that W can suitable for each task
        if X2 is None:
            num_task = self.number_task
            ind = tf.cast(X[..., -1], tf.int32)
            Xnew = X[..., :-1]
            X_lpf = tf.dynamic_partition(Xnew, ind, num_task)
            num_data = [tf.shape(Xi)[0] for Xi in X_lpf]
            ind_W = tf.repeat(self.W, num_data)[:, None]
            r2 = self.scaled_squared_euclid_dist(ind_W)
            return self.K_r2(r2)
        else:
            num_task = self.number_task

            ind1 = tf.cast(X[..., -1], tf.int32)
            Xnew = X[..., :-1]
            X_lpf = tf.dynamic_partition(Xnew, ind1, num_task)
            num_data = [tf.shape(Xi)[0] for Xi in X_lpf]
            ind_W = tf.repeat(self.W, num_data)[:, None]

            ind2 = tf.cast(X2[..., -1], tf.int32)
            Xnew2 = X2[..., :-1]
            X_lpf2 = tf.dynamic_partition(Xnew2, ind2, num_task)
            num_data2 = [tf.shape(Xi2)[0] for Xi2 in X_lpf2]
            ind_Wtest = tf.repeat(self.W, num_data2)[:, None]
            r2 = self.scaled_squared_euclid_dist(ind_W, ind_Wtest)
            return self.K_r2(r2)

    def K_r2(self, r2):
        if hasattr(self, "K_r"):
            # Clipping around the (single) float precision which is ~1e-45.
            r = tf.sqrt(tf.maximum(r2, 1e-36))
            return self.K_r(r)  # pylint: disable=no-member
        return tf.exp(-0.5 * r2)

    def scaled_squared_euclid_dist(self, X, X2=None):
        """
        Returns ‖(X - X2ᵀ) / ℓ‖², i.e. the squared L₂-norm.
        """
        return square_distance(self.scale(X), self.scale(X2))


    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))
        # return tf.fill(tf.shape(X)[:-1], tf.squeeze(1.0))

class Input_H_kernel(IsotropicStationary):
    """
    Base class for isotropic stationary kernels, i.e. kernels that only
    depend on

        r = ‖x - x'‖

    Derived classes should implement one of:

        K_r2(self, r2): Returns the kernel evaluated on r² (r2), which is the
        squared scaled Euclidean distance Should operate element-wise on r2.

        K_r(self, r): Returns the kernel evaluated on r, which is the scaled
        Euclidean distance. Should operate element-wise on r.
    """

    def __init__(self,number_task, **kwargs):
        """
        :param variance: the (initial) value for the variance parameter.
        :param lengthscales: the (initial) value for the lengthscale
            parameter(s), to induce ARD behaviour this must be initialised as
            an array the same length as the the number of active dimensions
            e.g. [1., 1., 1.]. If only a single value is passed, this value
            is used as the lengthscale of each dimension.
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")
        super().__init__(**kwargs)
        self.number_task = number_task
        W = np.random.randn(self.number_task)[:,None]
        # W = np.array([1,2])[:, None]
        self.W = Parameter(W)
    def K(self,X,X2=None):

        ### We augment W to make sure that W can suitable for each task
        if X2 is None:
            num_task = self.number_task
            ind = tf.cast(X[..., -1], tf.int32)
            Xnew = X[..., :-1]
            X_lpf = tf.dynamic_partition(Xnew, ind, num_task)
            num_data = [tf.shape(Xi)[0] for Xi in X_lpf]
            ind_W = tf.repeat(self.W, num_data)[:, None]
            r2 = self.scaled_squared_euclid_dist(ind_W)
            return self.K_r2(r2)
        else:
            num_task = self.number_task

            ind1 = tf.cast(X[..., -1], tf.int32)
            Xnew = X[..., :-1]
            X_lpf = tf.dynamic_partition(Xnew, ind1, num_task)
            num_data = [tf.shape(Xi)[0] for Xi in X_lpf]
            ind_W = tf.repeat(self.W, num_data)[:, None]

            ind2 = tf.cast(X2[..., -1], tf.int32)
            Xnew2 = X2[..., :-1]
            X_lpf2 = tf.dynamic_partition(Xnew2, ind2, num_task)
            num_data2 = [tf.shape(Xi2)[0] for Xi2 in X_lpf2]
            ind_Wtest = tf.repeat(self.W, num_data2)[:, None]
            r2 = self.scaled_squared_euclid_dist(ind_W, ind_Wtest)
            return self.K_r2(r2)

    def K_r2(self, r2):
        if hasattr(self, "K_r"):
            # Clipping around the (single) float precision which is ~1e-45.
            r = tf.sqrt(tf.maximum(r2, 1e-36))
            return self.K_r(r)  # pylint: disable=no-member
        return tf.exp(-0.5 * r2)




####################################################
########## Single_Hierarchial_Kernel  ##############
####################################################
### We would like to use Hierarchial_Kernel_replicated to replace this kernel since this kernel predict for different replicates need to set up predict_output_index
### This kernel is used for the Hierarchial GPs no inducing version.
class Hierarchial_Kernel_replicated_no_inducing(Combination):
    """
    In this function, we would like to build a hierarchical kernel.
    g(x) & \sim \mathcal{G} \mathcal{P}\left(\mathbf{0}, k_{g}\left(x, x^{\prime}\right)\right)

    f_{1}(x)  \sim \mathcal{G} \mathcal{P}\left(g(x), k_{f}\left(x, x^{\prime}\right)\right)
    .
    .
    .
    f_{D}(x)  \sim \mathcal{G} \mathcal{P}\left(g(x), k_{f}\left(x, x^{\prime}\right)\right)

    In each block, there will be a same kenel k_{f}. There k_{g} kernel is for all the output correlation
    E.g there are 3 outputs: X1, X2, X3 so there will be three diferent kernel function kf, kf, kf
    Thus, the block diagnoal kernel will be
    [[ kg(x1,x1) + kf(x1, x1),     kg(x1,x2),                           kg(x1,x3)],
     [ kg(x2,x1),                  kg(x2,x2) + kf(x2, x2),              kg(x2,x3)],
     [ kg(x3,x1),                  kg(x3,x2),               kg(x3,x3) + kf(x3, x3)]]
    """
    def __init__(self, kernel_g, kernel_f, predict_output_index, num_outputs, name=None):

        self.kernel_g = kernel_g
        self.predict_output_index = predict_output_index
        self.num_outputs = num_outputs
        ks = []
        for k in kernel_f:
            ks.append(k)
        Combination.__init__(self, ks, name)

    def _partition_and_stitch_kernel(self, X, X2):
        # get the index from Y

        if X2 is None:
            ind = X[..., -1]
            ind = tf.cast(ind, tf.int32)
            X = X[..., :-1]
            d, _ = tf.unique(ind)
            if tf.shape(d) == 1:
                ### Kff for one output
                return self.kernels[0].K(X) + self.kernel_g.K(X)
            else:
                # num_replicated = tf.shape(d)[0]
                # split up the arguments into chunks corresponding to the relevant likelihoods
                args_raw = tf.dynamic_partition(X, ind,  self.num_outputs)
                args = zip(*[args_raw])
                results = [tf.reduce_sum(self.kernels[0].K(*args_i),0) for args_i in zip(args)]
                # This is for general kernel
                X_l = tf.concat(args_raw, axis=0)
                return self.block_diagonal(results) + self.kernel_g.K(X_l)
        else:
            if self.predict_output_index is not None:
                ## Kfu for one output, it use for prediction part; ind2 is only one number
                ind1 = X[..., -1]
                # d_output = ind1[0].astype(np.int32)
                ind1 = tf.cast(ind1, tf.int32)
                # d1, _ = tf.unique(ind1)
                # num_replicated1 = tf.shape(d1)[0]
                X1 = X[..., :-1]
                args1_raw = tf.dynamic_partition(X1, ind1,  self.num_outputs)
                # args1 = zip(*[args1_raw])

                X2 = X2[..., :-1]
                X_l_1 = tf.concat(args1_raw, axis=0)
                X_l_2 = X2
                Kfu_general = self.kernel_g.K(X_l_1, X_l_2)
                X1_corrsponding_data = args1_raw[self.predict_output_index]

                ### find all the number in the list before and after the predict_output_index
                Len_before = len(args1_raw[:self.predict_output_index])
                Len_after= len(args1_raw[(self.predict_output_index+1):])
                count_before = 0
                for i in range(Len_before):
                    count_before = count_before + len(args1_raw[i])
                count_after = 0
                for i in range(Len_after):
                    count_after = count_after + len(args1_raw[i+1+self.predict_output_index])
                result_specific_kernel_kfu = self.kernels[0].K(X=X1_corrsponding_data, X2=X2)
                size_X2 = tf.shape(X2)[0]
                first_part_zeros = tf.cast(tf.zeros([count_before, size_X2]),dtype=default_float())
                last_part_zeros = tf.cast(tf.zeros([count_after, size_X2]), dtype=default_float())
                Kfu_specific = tf.concat([first_part_zeros, result_specific_kernel_kfu, last_part_zeros],axis=0)

                # general_result = self.kernel_g.K(X_l_1, X2)

                return Kfu_specific + Kfu_general


    def K(self, X, X2=None):  # [N, N2]
        return self._partition_and_stitch_kernel(X, X2)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

    def block_diagonal(self, matrices, dtype=tf.float64):

        matrices = [tf.convert_to_tensor(value=matrix, dtype=dtype) for matrix in matrices]
        blocked_rows = tf.compat.v1.Dimension(0)
        blocked_cols = tf.compat.v1.Dimension(0)
        batch_shape = tf.TensorShape(None)
        for matrix in matrices:
            full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
            batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
            blocked_rows += full_matrix_shape[-2]
            blocked_cols += full_matrix_shape[-1]
        ret_columns_list = []
        for matrix in matrices:
            matrix_shape = tf.shape(input=matrix)
            ret_columns_list.append(matrix_shape[-1])
        ret_columns = tf.add_n(ret_columns_list)
        row_blocks = []
        current_column = 0
        for matrix in matrices:
            matrix_shape = tf.shape(input=matrix)
            row_before_length = current_column
            current_column += matrix_shape[-1]
            row_after_length = ret_columns - current_column
            row_blocks.append(tf.pad(
                    tensor=matrix,
                    paddings=tf.concat(
                        [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                         [(row_before_length, row_after_length)]],
                        axis=0)))
        blocked = tf.concat(row_blocks, -2)
        blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
        return blocked





# class ExpDec(
#     Kernel):  # Implementation based on https://github.com/sremes/nssm-gp/blob/master/nssm_gp/spectral_kernels.py
#     """"
#     Nonstationary class for the exponential kernel
#     """
#
#     def __init__(self, variance=1.0, alpha=0.8, beta=0.7):  # the derivations are valid only for one dimensional input
#         super().__init__(active_dims=[0])
#         self.variance = Parameter(variance, transform=positive())
#         self.alpha = Parameter(alpha, transform=positive())
#         self.beta = Parameter(beta, transform=positive())
#
#     def K(self, X, X2=None):
#         if X2 is None:
#             X2 = X
#         input_sum = self.matrix_sum(X, X2)
#         kernel = self.variance * ((self.beta) ** self.alpha / (input_sum + self.beta) ** self.alpha)
#         return kernel
#
#     def K_diag(self, X):
#         X2 = X
#         input_sum = self.matrix_sum(X, X2)
#         kernel = self.variance * ((self.beta) ** self.alpha / (input_sum + self.beta) ** self.alpha)
#         return np.diag(kernel)
#
#     def matrix_sum(self, x1: TensorType, x2: Optional[TensorType] = None) -> tf.Tensor:
#         if x2 == None:
#             x2 = x1
#         x1_ones = tf.ones_like(x1)
#         x2_ones = tf.ones_like(x2)
#         X1 = tf.tensordot(x1, x2_ones, axes=0)
#         X2 = tf.tensordot(x1_ones, x2, axes=0)
#         X1 = tf.squeeze(X1)
#         X2 = tf.squeeze(X2)
#         res = X1 + X2
#         return res