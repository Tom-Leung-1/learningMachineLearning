import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import imageio
from sklearn.decomposition import PCA
from numpy import linalg as LA

def load_data(digits, num):
    '''
    Loads all of the images into a data-array.

    The training data has 5000 images per digit,
    but loading that many images from the disk may take a while.  So, you can
    just use a subset of them, say 300 for training (otherwise it will take a
    long time to complete.

    Note that each image as a 28x28 grayscale image, loaded as an array and
    then reshaped into a single row-vector.

    Use the function display(row-vector) to visualize an image.

    '''
    totalsize = 0
    for digit in digits:
        totalsize += min([len(next(os.walk('train%d' % digit))[2]), num])
    print('We will load %d images' % totalsize)
    X = np.zeros((totalsize, 784), dtype = np.uint8)   #784=28*28
    for index in range(0, len(digits)):
        digit = digits[index]
        print('\nReading images of digit %d' % digit)
        for i in range(num):
            pth = os.path.join('train%d' % digit,'%05d.pgm' % i)
            # image = misc.imread(pth).reshape((1, 784)) // modified source code
            image = imageio.imread(pth).reshape((1, 784))
            X[i + index * num, :] = image
        print('\n')
    return X

def plot_mean_image(X, digits = [0]):
    ''' example on presenting vector as an image
    '''
    plt.close('all')
    meanrow = X.mean(0)
    # present the row vector as an image
    plt.imshow(np.reshape(meanrow,(28,28)))
    plt.title('Mean image of digit ' + str(digits))
    plt.gray(), plt.xticks(()), plt.yticks(()), plt.show()


def plot_eigenvectors_image(d, V):
    plt.close('all')
    fig, ax = plt.subplots(nrows=3, ncols=3)
    for x in range(9):
        eigen_vector = V[:, x]
        eigen_vector_float = eigen_vector.astype('float')
        ax[x // 3][x % 3].imshow(np.reshape(eigen_vector_float.T, (28, 28)))
        ax[x // 3][x % 3].set_title('eigen vector ' + str(x), fontsize=10)
        plt.gray(), ax[x // 3][x % 3].set_xticks(()), ax[x // 3][x % 3].set_yticks(())
    plt.savefig('eigenimages.jpg')
    plt.show()


def plot_pov(d, V):
    y_values = (list(d))
    x_index = [idx + 1 for idx, x in enumerate(y_values)]
    eigen_sum = sum(y_values)
    pov_list = [sum(y_values[:idx + 1]) / eigen_sum for idx, y in enumerate(y_values)]
    plt.xlabel('the order of eigenvalues')
    plt.ylabel('POV')
    plt.title('POV v.s. the order of eigenvalues')
    plt.plot(x_index, pov_list)
    plt.savefig('pov.jpg')
    plt.show()

def calculate_pov(d, V):
    y_values = (list(d.astype('float')))
    eigen_sum = sum(y_values)
    part_sum = 0
    count = 0
    eigen_sum = sum(y_values)
    for x in y_values:
        part_sum += x
        count += 1
        if part_sum/eigen_sum > 0.9:
            print(count)
            break

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

def main():
    digits = [0, 1, 2]
    # load handwritten images of digit 0, 1, 2 into a matrix X
    # for each digit, we just use 300 images
    # each row of matrix X represents an image
    X = load_data(digits, 300)
    # plot the mean image of these images!
    # you will learn how to represent a row vector as an image in this function
    plot_mean_image(X, digits)
    d, V = pca(X)
    plot_eigenvectors_image(d, V)
    plot_pov(d, V)
    calculate_pov(d, V)

    ####################################################################
    # you need to
    #   1. do the PCA on matrix X;
    #
    #   2. plot the eigenimages (reshape the vector to 28*28 matrix then use
    #   the function ``imshow'' in pyplot), save the images of eigenvectors
    #   which correspond to largest 9 eigenvalues. Save them in a single file
    #   ``eigenimages.jpg''.
    #
    #   3. plot the POV (the Portion of variance explained v.s. the number of
    #   components we retain), save the figure in file ``pov.jpg''
    #
    #   4. report how many dimensions are need to preserve 0.9 POV, describe
    #   your answers and your undestanding of the results in the plain text
    #   file ``description2.txt''
    #
    #   5. remember to submit file ``eigenimages.jpg'', ``pov.jpg'',
    #   ``description2.txt'' and ``ex2.py''.
    #
    # YOUR CODE HERE!
    ####################################################################


if __name__ == '__main__':
    main()
