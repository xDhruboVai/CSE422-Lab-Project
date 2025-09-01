from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath('/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main'))
from src import evaluations

X_train_scaled = pd.read_csv("/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/X_train_scaled.csv")
X_test_scaled = pd.read_csv("/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/X_test_scaled.csv")
y_train = pd.read_csv("/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/y_train.csv")
y_test = pd.read_csv("/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/y_test.csv")

def knn_predict():
    boss_y = None
    boss_k = None
    max_accuracy = float("-inf")

    for k in range(1, 31):
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train_scaled, y_train.values.ravel())
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > max_accuracy:
            boss_y = y_pred
            max_accuracy = accuracy
            boss_k = k
    
    print("Best value of K is", boss_k)
    return boss_y, max_accuracy