import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("iris.csv")
print(df)

print(df.isnull().sum()) # this has to be done in any data sets, machine learning models do not aloe=w empties

#heat map
import seaborn as sb
plt.figure(figsize = (10,5))
sb.heatmap(df.corr(), annot=True)
plt.show()

print(df["sepalWidth"].describe())

#Box plot
df["age"].plot(kind="box", subplots=True, layout=(1,6), sharex=False, sharey=True)
plt.show()

#predicting
array= df.values
X = array[:,0:8] #0 to 7
Y = array[:, 8] #target variable / outcome

#split to train, test
#x trai and y train will be 70%
from sklearn import model_selection
#0.30 is the testing split percentage, for testing data
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size=0.30, random_state=10)

#
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train) #learning process

#now lets ask the model to predict x test, we hide y test
predictions = model.predict(X_test)
print(predictions)

print(accuracy_score(Y_test, predictions))        #accuracy_score is imported
print(classification_report(Y_test, predictions)) #classification report is imported
print(confusion_matrix(Y_test, predictions))      #imported

#new observation
newobservation = model.predict([[2,70,80,5,42,140,26,180],[2,45,80,5,42,400,26,29]])
print(newobservation)

#model improvement
#new data set
#https://modcom.co.ke/flask/DataScience/bank.csv     #paste it on your code and convert it to 0 and 1s before exposing it to ML
#pool the data, fill empty
#https://modcom.co.ke/flask/DataScience/iris.csv
#finish ML, hypothesis
