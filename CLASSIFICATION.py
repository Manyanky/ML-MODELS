# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# imprt pandas
df = pandas.read_csv("iris.csv")
# split to input and output variable
array = df.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7

# split X, Y into train and test
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,
test_size=validation_size, random_state=seed)
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


kfold = model_selection.KFold(n_splits=10, random_state=seed)


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn

for name, model in models:
      kfold = model_selection.KFold(n_splits=10, random_state=seed)
      cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
      print(name, cv_results.mean())


# KNeighborsClassifier scores higher than others
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train) # put your data in the model

# validate your model and get accuracy, classification report etc
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# predict a new flower
predicted_flower = knn.predict([[1.3,1.4,3.6,3.6]])
print(predicted_flower)


