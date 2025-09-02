import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/processed_dataset.csv")


def implement_KMC(df_temp, cluster_num):
  km = KMeans(n_clusters = cluster_num)
  y_predicted = km.fit_predict(df_temp[df_temp.columns])
  return km , y_predicted #returns the model and the predicted assigned cluster to each entry