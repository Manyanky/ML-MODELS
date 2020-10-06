import pandas as pd
import matplotlib.pyplot as plt
import sklearn      #scikit-learn is a library for machine learning
#modcom.co.ke/datascience/ML

#machine learning breaks into two; supervised and unsupervised
#supervised; you provide the machine with data train it and let t predict based on that data on its own
#unsupervised learning; there is nothing to predict, no predicting, we do data clustering.

#supervised is breaks into two, classification and regression
#pima indians diabetes dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("pima-data-orig.csv")
print(df)
print(df.isnull().sum())

fig, ax = plt.subplots()
ax.hist(df['insulin'], color='#539caf', bins=30)#bins are not a must because the
 # Label the axes and provide a title
ax.set_title("Distribution  of patients insulin")
ax.set_xlabel("insulin")
ax.set_ylabel("Frequency")
plt.show()

#pie chart
df["diabetes"].replace(1, "positive", inplace=True)
df["diabetes"].replace(0, "negative", inplace=True)
fig, ax = plt.subplots()
df.groupby('diabetes').size().plot(kind='pie',autopct='%1.1f%%')
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Distribution of patients diabetes level")
plt.show()
print(df.shape)

#heat map
import seaborn as sb
plt.figure(figsize = (10,5))
sb.heatmap(df.corr(), annot=True)
plt.show()

print(df["insulin"].describe())

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

#feature selection
#from sklearn.feature_selection import SelectKBest, chi2

#best = SelectKBest(score_func=chi2, k=2)
#fit = test.fit(X, Y)
#features = var.transform(X)
#print("selected: ", features)