from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

dir = "/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/"
y_test = pd.read_csv("/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/y_test.csv")
