import pandas as pd
import matplotlib.pyplot as plt
import sklearn

#https://modcom.co.ke/datascience/ML/
from sklearn import model_selection

df = pd.read_csv("Advertising.csv")
print(df)

import seaborn as sb
plt.figure(figsize = (9,5))
sb.heatmap(df.corr(), annot=True)
plt.show()

#hisogram
fig, ax = plt.subplots()
ax.hist(df['Newspaper'], color='#539caf', bins=30)#bins are not a must because the
 # Label the axes and provide a title
ax.set_title("Distribution  of Newspaper cost")
ax.set_xlabel("Newspaper")
ax.set_ylabel("Frequency")
plt.show()

#splitting data to X and Y
array = df.values
X = array[:, 1:4] #upto 3rd
Y = array[:, 4]   #target variable to be predicted
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size=0.30, random_state=20)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

#pick one model
model = LinearRegression()
model.fit(X_train, Y_train)  #training process

#ask the model to predict the x test while hiding the y test from the model (unseen data)
predictions = model.predict(X_test)

#in regression we always check the R squared and, absolute error squared
from sklearn.metrics import r2_score, mean_squared_error

print("R squared(%):", r2_score(Y_test, predictions))
print("mean squared error:", mean_squared_error(Y_test, predictions))

#plot the scatter with best fit line
fig. ax = plt.subplots()
ax.scatter(Y_test, predictions)
ax.plot(Y_test, Y_test)
ax.set_xlabel("Y_test values(real answers)")
ax.set_ylabel("predicted values")
plt.show()

#use the model for new observation
observation = model.predict([[400,0,0]])
print(observation)

#feature selection
from sklearn.feature_selection import SelectKBest, chi2

#best = SelectKBest(score_func=chi2, k=2)
#fit = test.fit(X, Y)
#features = var.transform(X)
