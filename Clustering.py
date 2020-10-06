#unsupervised learning
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('AirlinesCluster')
print(df.isnull().sum())
print(df.shape)

import seaborn as sb
plt.figure(figsize = (9,5))
sb.heatmap(df.corr(), annot=True)
plt.show()

#density
fig,  ax = plt.subplots()
ax.plot(df['DaysSinceEnroll'], color='#539caf', lw=2)
ax.set_ylabel("Frequency")
ax.set_xlabel("sprint")
ax.set_title("Weight Distribution")
ax.legend(loc='best')
plt.show()

#histogram
fig, ax = plt.subplots()
ax.hist(df['DaysSinceEnroll'], color='#539caf', bins=30)#bins are not a must because the
 # Label the axes and provide a title
ax.set_title("Distribution customers Enrolment days")
ax.set_xlabel("Customers enrolment days")
ax.set_ylabel("Frequency")
plt.show()

#density
fig,  ax = plt.subplots()
ax.plot(df['DaysSinceEnroll'], color='#539caf', lw=2)
ax.set_ylabel("Frequency")
ax.set_xlabel("sprint")
ax.set_title(" Distribution of Customers enrollment days")
ax.legend(loc='best')
plt.show()

pd.set_option('display.max_columns', 10)
print(df.describe())

#use KMeans to cluster data
array = df.values
X = array[:,0:7]  #no target variables so we cannot predict

#WCSS
#within cluster sum of squares - minimising clusters
#justpaste.it/26p9o

plt.figure(figsize=(10, 8))
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
     kmeans.fit(X)
     wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#put your data for cluster into 5 clusters
kmeans = KMeans(n_clusters=5).fit(X)
centronoids = kmeans.cluster_centers_
print(centronoids)

#we put centronoids in a sample dataframe
cluster = pd.DataFrame(centronoids, columns=['Balance','QualMiles','BonusMiles','BonusTrans','FlightMiles','FlightTrans','DaysSinceEnroll'])

print(cluster)

#pull out data from these clusters
results = zip(kmeans.labels_)
sortedR = sorted(results, key = lambda X: X[1])
print(sortedR)

#justpaste.it/5woh8
y_means = kmeans.fit_predict(X)
#Visualizing the clusters for k=4
plt.scatter(X[y_means==0,0],X[y_means==0,1],s=50, c='purple',label='Cluster1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=50, c='blue',label='Cluster2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],s=50, c='green',label='Cluster3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],s=50, c='cyan',label='Cluster4')
plt.scatter(X[y_means==4,0],X[y_means==4,1],s=50, c='cyan',label='Cluster4')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=200,marker='s', c='red', alpha=0.7, label='Centroids')
plt.title('Customer segments')
plt.xlabel('Annual income of customer')
plt.ylabel('Annual spend from customer on site')
plt.legend()
plt.show()

#scipy
#justpaste.it/6ov7y
#justpaste.it/5woh8

#test hypothesis
#Hypothesis is a claim which can be true or falls
#According to KQ the mean flight transaction is 56000.
#null hypothesis(Ho) - the mean(flight trans) is equal to 56000
#alternative hypothesis(H1)- our sample mean is not equal to 56000

import scipy      #install
from scipy.stats import ttest_1samp
statistics, pvalue = ttest_1samp(df['FlightTrans'], 56000)
print('p value is :', pvalue)
alpha = 0.05
if pvalue < alpha:
     print('Reject Null Hypothesis')
     print('Accept the alternative')
     print('Alternative Hypothesis(H1)-our sample mean is not equal to 56000')
else:
     print('Accept Null Hypothesis')
     print('Null Hypothesis(H0) - Our sample mean(FlightTrans) is equal to 56000')

#sample, ANOVA, Chi square
#work on something ,,, get any data set , do  A few plots
#on eithher classification, regresssion or clustering
#1 page document explaining your work and a link to your code,

# email to: joe@modcom.co.ke
#kaggle, github provides data
#use justpaste it to submit code
