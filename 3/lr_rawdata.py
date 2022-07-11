import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn
seaborn.set()

def plot_graph(actual, prediction):
    plt.scatter(actual[0], actual[1], c='b', label='actual result')
    plt.scatter(prediction[0], prediction[1], c='r', label='prediction')
    plt.xlabel("Standardized horsepower")
    plt.ylabel("Standardized price")
    plt.title("Linear regression on cleaned and standardized test data")
    plt.legend(loc="upper left")
    plt.show()

df = pd.read_csv('imports-85.data',
            header=None,
            names=["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors",
                   "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height",
                   "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore",
                   "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"],
                na_values=("?"))
df = df.dropna()
train, test = train_test_split(df, test_size=0.2, random_state=1)
train_hp_price = train[["horsepower", "price"]]
test_hp_price = test[["horsepower", "price"]]
X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(train_hp_price)
X_test_scaled = X_scaler.transform(test_hp_price)
std_train_df = pd.DataFrame(X_train_scaled, columns = ['horsepower','price'])
std_test_df = pd.DataFrame(X_test_scaled, columns = ['horsepower','price'])
X_train_scaled_hp = X_train_scaled[:,0].reshape(-1,1)
X_train_scaled_price = X_train_scaled[:,1].reshape(-1,1)
X_test_scaled_hp = X_test_scaled[:,0].reshape(-1,1)
X_test_scaled_price = X_test_scaled[:,1].reshape(-1,1)

reg = linear_model.LinearRegression().fit(X_train_scaled_hp, X_train_scaled_price)
pred = reg.predict(X_test_scaled_hp)
plot_graph([X_test_scaled_hp, X_test_scaled_price], [X_test_scaled_hp, pred])


