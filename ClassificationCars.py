import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("school.csv")
print(df)
print(df.isnull().sum())

print(df.shape)

