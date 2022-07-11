import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

def plot_graph(xx_group, X_test_group, gp1_label, gp2_label, title):
    plt.scatter(xx_group[0], xx_group[1], c='b', label=gp1_label)
    plt.scatter(X_test_group[0], X_test_group[1], c='r', label=gp2_label)
    plt.xlabel("real data")
    plt.ylabel("predicted data")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.show()

X_train = [[5.3], [7.2], [10.5], [14.7], [18], [20]]
y_train = [[7.5], [9.1], [13.2], [17.5], [19.3], [19.5]]

X_test = [[6], [8], [11], [22]]
y_test = [[8.3], [12.5], [15.4], [19.6]]

poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
X_train_poly = np.delete(X_train_poly,0,1)
X_test_poly = np.delete(X_test_poly,0,1)
# some code here
pr_model = LinearRegression().fit(X_train_poly, y_train)
print("Linear regression (order 5) score is:", pr_model.score(X_test_poly, y_test))


xx = np.linspace(0, 26, 100)
xx_poly = poly.transform(xx.reshape(xx.shape[0], 1))
xx_poly = np.delete(xx_poly,0,1)
yy_poly = pr_model.predict(xx_poly)
plot_graph([xx, yy_poly], [X_test, y_test], "yy_poly", "y_test", "Linear regression (order 5) result")
# some code here

ridge_model = Ridge(alpha=1, normalize=False)
ridge_model.fit(X_train_poly, y_train)
print("Ridge regression (order 5) score is:", ridge_model.score(X_test_poly, y_test))
yy_ridge = ridge_model.predict(xx_poly)
plot_graph([xx, yy_ridge], [X_test, y_test], "yy_ridge", "y_test", "Ridge regression (order 5) result")
# some code here