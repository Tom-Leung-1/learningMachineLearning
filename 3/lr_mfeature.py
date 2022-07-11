import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn
seaborn.set()
from sklearn.preprocessing import StandardScaler

def plot_graph(actual, prediction):
    plt.scatter(actual[0], actual[1], c='b', label='actual result')
    plt.scatter(prediction[0], prediction[1], c='r', label='prediction')
    plt.xlabel("standardized horsepower")
    plt.ylabel("standardized price")
    plt.title("Relationship between horsepower and price")
    plt.legend(loc="upper left")
    plt.show()

def normal_equation_theta(X, y):
    X = X.to_numpy()
    N = X.shape[0]
    y = y.to_numpy()
    ones = np.ones((N,1))
    X_ = np.append(ones, X, axis=1)
    inverse = np.linalg.inv(np.matmul(X_.T, X_))
    theta = np.matmul(np.matmul(inverse, X_.T), y)
    print("Parameter theta calculated by normal equation:", theta)
    return theta

def regression_theta(X, y):
    reg = linear_model.SGDRegressor(max_iter=1000)
    X = X.to_numpy()
    y = y.to_numpy()
    result = reg.fit(X, y)
    theta = np.append(result.intercept_, result.coef_)
    print("Parameter theta calculated by SGD", theta)
    return theta

df = pd.read_csv('imports-85.data',
            header=None,
            names=["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors",
                   "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height",
                   "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore",
                   "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"],
                na_values=("?"))
df = df.dropna()
train_df = df[["city-mpg", "horsepower", "engine-size", "peak-rpm", "price"]]
X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(train_df)
std_train_matrix = pd.DataFrame(X_train_scaled[:, range(4)], columns = ["city-mpg", "horsepower", "engine-size", "peak-rpm"])
std_vector_price = pd.DataFrame(X_train_scaled[:, 4], columns = ["price"])
theta1 = normal_equation_theta(std_train_matrix, std_vector_price)
theta2 = regression_theta(std_train_matrix, std_vector_price)