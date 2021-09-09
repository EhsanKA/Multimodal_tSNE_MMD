import numpy as np

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


class TSNE:
    def __init__(self, X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):

        if isinstance(no_dims, float):
            print("Error: array X should have type float.")
        if round(no_dims) != no_dims:
            print("Error: number of dimensions should be an integer.")

        # Initialize variables
        (self.no_dims, self.initial_dims, self.perplexity) = (no_dims, initial_dims, perplexity)
        self.X = X
        if X.shape[1] >= 50:
            self.X = pca(X, initial_dims).real
        (self.n, self.d) = X.shape
        self.max_iter = 1000
        self.initial_momentum = 0.5
        self.final_momentum = 0.8
        self.eta = 500
        self.min_gain = 0.01
        #         np.random.seed(seed=1)
        self.Y = np.random.randn(self.n, no_dims)
        self.dY = np.zeros((self.n, no_dims))
        self.iY = np.zeros((self.n, no_dims))
        self.gains = np.ones((self.n, no_dims))

        # Compute P-values
        P = x2p(self.X, 1e-5, self.perplexity)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        P = P * 4.  # early exaggeration
        self.P = np.maximum(P, 1e-12)
        self.embeddings_before_mmd = []
        self.embeddings_after_mmd = []
        self.ce_loss = []
        self.mmd_loss = []

    def _update(self, iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(self.Y), 1)
        num = -2. * np.dot(self.Y, self.Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(self.n), range(self.n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = self.P - Q
        for i in range(self.n):
            self.dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (self.no_dims, 1)).T * (self.Y[i, :] - self.Y), 0)

        # Perform the update
        if iter < 20:
            momentum = self.initial_momentum
        else:
            momentum = self.final_momentum
        self.gains = (self.gains + 0.2) * ((self.dY > 0.) != (self.iY > 0.)) + \
                     (self.gains * 0.8) * ((self.dY > 0.) == (self.iY > 0.))
        self.gains[self.gains < self.min_gain] = self.min_gain
        self.iY = momentum * self.iY - self.eta * (self.gains * self.dY)
        self.Y = self.Y + self.iY
        self.Y = self.Y - np.tile(np.mean(self.Y, 0), (self.n, 1))

        # Compute current value of cost function
        C = np.sum(self.P * np.log(self.P / Q))
        if (iter + 1) % 10 == 0:
            #             C = np.sum(self.P * np.log(self.P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            self.P = self.P / 4.

        self.ce_loss.append(C)


from sklearn.metrics.pairwise import rbf_kernel as rbf


def mmd(embedding1, embedding2, proximity=0.1, ratio=1):
    ker1 = rbf(embedding1, embedding1, gamma=proximity)
    ker2 = rbf(embedding2, embedding2, gamma=proximity)
    ker12 = rbf(embedding1, embedding2, gamma=proximity)
    ker21 = ker12.T
    new_embedding1 = embedding1.copy()
    new_embedding2 = embedding2.copy()
    n = embedding1.shape[0]
    m = embedding2.shape[0]
    for i in range(embedding1.shape[0]):
        for j in range(embedding1.shape[1]):
            l = ratio * (2 / (n * n)) * np.dot(ker1[i], embedding1[i, j] - embedding1[:, j])
#             print("llllllll:", l)
#             print(new_embedding1[i, j])
            new_embedding1[i, j] += l
#             break
#         break

    for i in range(embedding2.shape[0]):
        for j in range(embedding2.shape[1]):
            l = ratio * (2 / (m * m)) * np.dot(ker2[i], embedding2[i, j] - embedding2[:, j])
#             print("llllllll:", l)
#             print(new_embedding2[i, j])
            new_embedding2[i, j] += l
#             break
#         break

    for i in range(embedding1.shape[0]):
        for j in range(embedding1.shape[1]):
            l = ratio * (4 / (n * m)) * np.dot(ker12[i], embedding1[i, j] - embedding2[:, j])
#             print("llllllll:", l)
#             print(new_embedding1[i, j])
            new_embedding1[i, j] -= l
#             break
#         break

    for i in range(embedding2.shape[0]):
        for j in range(embedding2.shape[1]):
            l = ratio * (4 / (n * m)) * np.dot(ker21[i], embedding2[i, j] - embedding1[:, j])
#             print("llllllll:", l)
#             print(new_embedding2[i, j])
            new_embedding2[i, j] -= l
#             break
#         break
    mmd_loss = np.mean(ker1) + np.mean(ker2) - 2 * np.mean(ker12)
    return new_embedding1, new_embedding2, mmd_loss

import tensorflow as tf
import numpy as np
import argparse

def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]
    Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
    ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)

    # By making the `inner' dimensions of the two matrices equal to 1 using
    # broadcasting then we are essentially substracting every pair of rows
    # of x and y.
    # x will be num_samples x num_features x 1,
    # and y will be 1 x num_features x num_samples (after broadcasting).
    # After the substraction we will get a
    # num_x_samples x num_features x num_y_samples matrix.
    # The resulting dist will be of shape num_y_samples x num_x_samples.
    # and thus we need to transpose it again.
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    """Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
    Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
def t_distribution_kernel_matrix(x, y, sigmas):
    """Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
    Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    # beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)
    dist_1 = tf.add(dist, tf.constant(1., dtype=tf.float32))

    s = tf.pow(tf.reshape(dist_1, (1, -1)), -1)

    return tf.reshape(tf.reduce_sum(s, 0), tf.shape(dist))

def maximum_mean_discrepancy(x, y, kernel=t_distribution_kernel_matrix, bandwidth=1.0):

    """Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.
    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
    Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
    """
    with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x, tf.constant([bandwidth])))
        cost += tf.reduce_mean(kernel(y, y, tf.constant([bandwidth])))
        cost -= 2 * tf.reduce_mean(kernel(x, y, tf.constant([bandwidth])))

        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def tsne(X1, X2, no_dims=30, perplexity=70, max_iter=1000, mmd_coef = 1.0, mmd_radius=1.0, mmd_kernel= t_distribution_kernel_matrix):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    ts1 = TSNE(X1, no_dims=no_dims, perplexity=perplexity)
    ts2 = TSNE(X2, no_dims=no_dims, perplexity=perplexity)

    # Run iterations
    for iter in range(max_iter):

        tf_embed1 = tf.Variable(ts1.Y, dtype=tf.float32)
        tf_embed2 = tf.Variable(ts2.Y, dtype=tf.float32)
        res = tf.constant(1000.0, dtype=tf.float32)

        # if iter >100:
        with tf.GradientTape(persistent=True) as tape:
            res = maximum_mean_discrepancy(tf_embed1, tf_embed2,kernel=mmd_kernel, bandwidth=mmd_radius)

        ts1._update(iter)
        ts2._update(iter)
        ts1.embeddings_before_mmd.append(ts1.Y)
        ts2.embeddings_before_mmd.append(ts2.Y)

        # best results were with proximity=4 and reatio=5000
        # ts1.Y, ts2.Y, mmd_loss = mmd(ts1.Y, ts2.Y, proximity=8, ratio=100000)

        # if iter > 100:
        [dx1, dx2] = tape.gradient(res, [tf_embed1, tf_embed2])
        ts1.Y += mmd_coef * dx1.numpy()
        ts2.Y += mmd_coef * dx2.numpy()

        ts1.embeddings_after_mmd.append(ts1.Y)
        ts2.embeddings_after_mmd.append(ts2.Y)
        ts1.mmd_loss.append(res.numpy())
        ts2.mmd_loss.append(res.numpy())

        # if len(ts1.ce_loss) > 5:
        #     loss = ts1.ce_loss[-1] + ts2.ce_loss[-1] + ts1.mmd_loss[-1]
        #     prev_loss = ts1.ce_loss[-2] + ts2.ce_loss[-2] + ts1.mmd_loss[-2]
        #     if (np.abs(loss-prev_loss)/np.abs(prev_loss)) < 0.00005 and iter > 100:
        #         break

    # Return solution
    return ts1, ts2


def tsne_single(X, no_dims=2, perplexity=70, max_iter=1000):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    ts = TSNE(X, no_dims=no_dims, perplexity=perplexity)

    # Run iterations
    for iter in range(max_iter):

        ts._update(iter)
        ts.embeddings_before_mmd.append(ts.Y)

        ts.embeddings_after_mmd.append(ts.Y)

    return ts


import time

def generate_results(X1, X2, n_repeat=1, no_dims=30, perplexity=70, max_iter=300,
                     mmd_coef = 1.0, mmd_radius=1.0, mmd_kernel= t_distribution_kernel_matrix):
    ts1_list = []
    ts2_list = []
    time_list = []

    for i in range(n_repeat):
        print("********** replication:{} ********".format(i + 1))
        start = time.time()
        ts1, ts2 = tsne(X1, X2, no_dims=no_dims, perplexity=perplexity, max_iter=max_iter,
                        mmd_coef = mmd_coef, mmd_radius=mmd_radius, mmd_kernel= mmd_kernel)
        end = time.time()
        u1 = pca(ts1.Y, no_dims=2)
        ts1.y2d = u1
        u2 = pca(ts2.Y, no_dims=2)
        ts2.y2d = u2
        ts1_list.append(ts1)
        ts2_list.append(ts2)
        time_list.append(end - start)
        print("********** run time is:{} ********".format(end - start))

    return ts1_list, ts2_list, time_list
