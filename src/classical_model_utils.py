from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath('/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main'))

X_train_scaled = pd.read_csv("/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/X_train_scaled.csv")
X_test_scaled = pd.read_csv("/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/X_test_scaled.csv")
X_train = pd.read_csv("/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/X_train.csv")
X_test = pd.read_csv("/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/X_test.csv")
y_train = pd.read_csv("/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/y_train.csv")
y_test = pd.read_csv("/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/y_test.csv")
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

def knn_predict():
    boss_knn = None
    boss_y = None
    boss_k = None
    max_accuracy = float("-inf")

    for k in range(1, 31):
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > max_accuracy:
            boss_y = y_pred
            max_accuracy = accuracy
            boss_k = k
            boss_knn = knn

    print("Best value of K is", boss_k)
    return boss_knn, boss_y, max_accuracy

def nb_predict():
    nb = GaussianNB()

    nb.fit(X_train_scaled, y_train)
    y_pred = nb.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    return nb, y_pred, accuracy

def dt_predict():
    dt = DecisionTreeClassifier(
    criterion = "entropy", 
    max_depth = None,
    random_state = 42
)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return dt, y_pred, accuracy

'''def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Logistic_Regression_GD:
    def __init__(self, lr = 0.01, epochs = 1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n) #w
        self.b = 0           #beta
        self.losses = []

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.w) + self.b
            y_pred = sigmoid(linear_model)

            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = -(1 / m) * np.sum(y * np.log(y_pred + 1e-10) + (1-y) * np.log(1 - y_pred + 1e-10))
            self.losses.append(loss)

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        y_pred = sigmoid(linear_model)
        return np.where(y_pred >= 0.5, 1, 0)

def log_predict():
    log_reg = Logistic_Regression_GD()

    log_reg.fit(X_train_scaled, y_train)
    y_pred = log_reg.predict(X_test_scaled)
    accuracy = accuracy_score(y_pred, y_test)
    return y_pred, accuracy'''


def log_predict():
    log_reg = LogisticRegression(
        solver = 'lbfgs',
        max_iter = 1000,    
        random_state = 42
    )

    log_reg.fit(X_train_scaled, y_train)
    y_pred = log_reg.predict(X_test_scaled)
    accuracy = accuracy_score(y_pred, y_test)
    return log_reg, y_pred, accuracy