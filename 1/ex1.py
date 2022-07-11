import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def create_data(x1, x2, x3):
    x4 = -4.0 * x1
    x5 = 10 * x1 + 10
    x6 = -1 * x2 / 2
    x7 = np.multiply(x2, x2)
    x8 = -1 * x3 / 10
    x9 = 2.0 * x3 + 2.0
    X = np.hstack((x1, x2, x3, x4, x5, x6, x7, x8, x9))
    return X

def pca(X):
    '''
    PCA step by step
      1. normalize matrix X
      2. compute the covariance matrix of the normalized matrix X
      3. do the eigenvalue decomposition on the covariance matrix
    Return: [d, V]
      d is the column vector containing all the corresponding eigenvalues,
      V is the matrix containing all the eigenvectors.
    If you do not remember Eigenvalue Decomposition, please review the linear
    algebra
    In this assignment, we use the ``unbiased estimator'' of covariance. You
    can refer to this website for more information
    http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.cov.html
    Actually, Singular Value Decomposition (SVD) is another way to do the
    PCA, if you are interested, you can google SVD.
    '''

    ####################################################################
    #
    # YOUR CODE HERE!
    #
    ####################################################################
    # test
    # A = np.array([[2, -4], [-1, -1]])
    # d, V = LA.eig(A)
    #
    mean_v = np.mean(X, axis=0)
    mean_v
    normal_X = X - mean_v
    cov_X = np.cov(normal_X.T)
    d, V = LA.eig(cov_X)

    pca = PCA()
    pca.fit(X)
    pV = pca.components_  # eigen vectors
    pd = pca.explained_variance_  # eigen values
    # here d is the column vector containing all the corresponding eigenvalues.
    # V is the matrix containing all the eigenvectors,
    return [d, V]

def plot_figs(X):
    """
    1. perform PCA (you can use pca(X) completed by yourself) on this matrix X
    2. plot (a) The figure of eigenvalues v.s. the order of eigenvalues. All eigenvalues in a decreasing order.
    3. plot (b) The figure of POV v.s. the order of the eigenvalues.
    """
    d, V = pca(X)
    y_values = (list(d))
    x_index = [idx +1 for idx, x in enumerate(y_values)]
    plt.xlabel('the order of eigenvalues')
    plt.ylabel('eigenvalues')
    plt.title('eigenvalues v.s. the order of eigenvalues')
    plt.plot(x_index, y_values)
    plt.show()

    eigen_sum = sum(y_values)
    pov_list = [sum(y_values[:idx+1])/eigen_sum for idx, y in enumerate(y_values)]
    plt.xlabel('the order of eigenvalues')
    plt.ylabel('POV')
    plt.title('POV v.s. the order of eigenvalues')
    plt.plot(x_index, pov_list)
    plt.show()




    ####################################################################
    #
    # YOUR CODE HERE!
    #
    ####################################################################
    return

def main():
    N = 1000
    shape = (N, 1)
    x1 = np.random.normal(0, 1, shape) # samples from normal distribution
    x2 = np.random.exponential(10.0, shape) # samples from exponential distribution
    x3 = np.random.uniform(-100, 100, shape) # uniformly sampled data points
    X = create_data(x1, x2, x3)

    print(pca(X))
    plot_figs(X)

if __name__ == '__main__':
    main()

