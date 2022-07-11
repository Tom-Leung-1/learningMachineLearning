import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import math
import random
from sklearn.linear_model import LogisticRegression

def logistic_func(x):

    ####################################################################
    # YOUR CODE HERE!
    # Output: logistic(x)
    ####################################################################
    L = 1/(1+math.e**(-x))
    return L


def train(X_train, y_train, tol=10 ** -4):

    LearningRate = 0.05

    ####################################################################
    # YOUR CODE HERE!
    # Output: the weight update result [w_0, w_1, w_2, ...]
    ####################################################################
    num_features = np.shape(X_train)[1]
    num_rows = np.shape(X_train)[0]
    weights =[]
    for x in range(num_features+1):
        # weights.append(random.uniform(-1, 1))
        weights.append(0)
    update = []
    while True:
        for idx, w in enumerate(weights):
            total_sum = 0
            for x in range(0, num_rows):
                weighted_sum = 0
                for y in range(0, num_features+1):
                    if y == 0:
                        weighted_sum += weights[y]
                    else:
                        weighted_sum += weights[y]* X_train[x,y-1]
                logistic = logistic_func(weighted_sum)
                diff = y_train[x] - logistic
                if idx == 0:
                    total_sum += diff
                else:
                    total_sum += X_train[x,idx-1] * diff
            update.append(w + LearningRate * total_sum)
        if no_diff(weights, update):
            break
        else:
            weights = update
            update = []
    return np.array(weights)

def no_diff(weights, update):
    total_diff = 0
    for idx, w in enumerate(weights):
        total_diff += abs(weights[idx] - update[idx])
    if total_diff > 10 ** (-2): #need modify
        return False
    return True

def no_diff_matrix(weights, update):
    diff = np.sum(np.absolute(weights - update))
    if diff > 10 ** (-4):
        return False
    return True

def train_matrix(X_train, y_train, tol=10 ** -4):

    LearningRate = 0.05
    num_features = np.shape(X_train)[1]
    num_rows = np.shape(X_train)[0]
    new_column = np.full((num_rows, 1), 1, dtype=int)
    X_train_1 = np.append(new_column, X_train, axis=1)
    # weights = np.random.rand(num_features+1,1)
    weights = np.zeros((num_features + 1, 1))
    while True:
        Xmul = np.matmul(X_train_1, weights)
        logistics = logistic_func(Xmul)
        diff = y_train.reshape((-1, 1)) - logistics
        X_train_1_transpose = X_train_1.T
        update = np.matmul(X_train_1_transpose, diff)
        rate_update = LearningRate * update
        new_weights = weights + rate_update
        if no_diff_matrix(weights, new_weights):
            break
        else:
            weights = new_weights
    ####################################################################
    # YOUR CODE HERE!
    # Output: the weight update result [w_0, w_1, w_2, ...]
    ####################################################################

    return weights.reshape(3, )

def verify(X_train, y_train):
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    coef = clf.coef_[0]
    return coef

def predict(X_test, weights):

    ###################################################################
    # YOUR CODE HERE!
    # The predict labels of all points in test dataset.
    ####################################################################
    num_features = np.shape(X_test)[1]
    num_rows = np.shape(X_test)[0]
    new_column = np.full((num_rows, 1), 1, dtype=int)
    X_test_1 = np.append(new_column, X_test, axis=1)
    Xmul = np.matmul(X_test_1, weights)
    logistics = logistic_func(Xmul)
    predictions = logistics >= 0.5
    predictions = predictions.astype(int)
    # predictions = [x[0] for x in predictions.tolist()]
    return predictions


def plot_prediction(X_test, X_test_prediction):
    X_test1 = X_test[X_test_prediction == 0, :]
    X_test2 = X_test[X_test_prediction == 1, :]
    plt.scatter(X_test1[:, 0], X_test1[:, 1], color='red')
    plt.scatter(X_test2[:, 0], X_test2[:, 1], color='blue')
    plt.show()


# Data Generation
n_samples = 1000

centers = [(-1, -1), (5, 10)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Experiments

w = train(X_train, y_train)
w = train_matrix(X_train, y_train)
# w = verify(X_train, y_train)
# w = np.zeros((2 + 1, 1)).reshape(3, ) # test need uncomment
X_test_prediction = predict(X_test, w)
plot_prediction(X_test, X_test_prediction)
plot_prediction(X_test, y_test)

wrong = np.count_nonzero(y_test - X_test_prediction)
print('Number of wrong predictions is: ' + str(wrong))
