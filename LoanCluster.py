import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("train.csv")

#preparing encoding ordinal columns
df["Gender"].replace("Male",0, inplace=True)
df["Gender"].replace("Female",1, inplace=True)

df["Married"].replace("Yes",0, inplace=True)
df["Married"].replace("No",1, inplace=True)

df["Education"].replace("Graduate",0, inplace=True)
df["Education"].replace("Not Graduate",1, inplace=True)

df["Self_Employed"].replace("Yes",0, inplace=True)
df["Self_Employed"].replace("No",1, inplace=True)

df["Property_Area"].replace("Urban",0, inplace=True)
df["Property_Area"].replace("Semi_Urban",1, inplace=True)
df["Property_Area"].replace("Rural",2, inplace=True)

df["Loan_Status"].replace("Y",0, inplace=True)
df["Loan_Status"].replace("N",1, inplace=True)

df["Dependents"].replace("3+",3, inplace=True)

#imputing
df["Gender"].fillna(2, inplace=True)
df["Married"].fillna(2, inplace=True)
df["Self_Employed"].fillna(2, inplace=True)
df["Dependents"].fillna(4, inplace=True)

medianLoanAmount = df["LoanAmount"].median()
df["LoanAmount"].fillna(medianLoanAmount, inplace=True)
medianLoan_Amount_Term = df["Loan_Amount_Term"].median()
df["Loan_Amount_Term"].fillna(medianLoan_Amount_Term, inplace=True)
medianCredit_History = df["Credit_History"].median()
df["Credit_History"].fillna(medianCredit_History, inplace=True)

#print(df.isnull().sum())
#print(df.describe())
#print(df["LoanAmount"].describe())

#Plotting
fig, ax = plt.subplots()
ax.scatter(df["LoanAmount"], df["Loan_Amount_Term"], s = 30, color = "red", alpha=0.75)
ax.set_xlabel("Loan Amount")
ax.set_ylabel("Loan Amount Terms")
ax.set_title(" SCATTER PLOT:Distribution of Customers Loan Amounts by their Loan  Amount Terms")
plt.show()

#Histogram
fig, ax = plt.subplots()
ax.hist(df['LoanAmount'], color='#539caf', bins=30)
ax.set_title("Distribution of Customers Loan Amount")
ax.set_xlabel("Loan Amount")
ax.set_ylabel("Frequency")
plt.show()

#density plot
fig,  ax = plt.subplots()
ax.plot(df['LoanAmount'], color='#539caf', lw=2)
ax.set_ylabel("Frequency")
ax.set_xlabel("Loan Amount")
ax.set_title("Loan Amount Distribution")
ax.legend(loc='best')
plt.show()

#stacked bar, gender vs education vs their loan amount
fig,  ax = plt.subplots()
df.groupby(['Gender', 'Education'])['LoanAmount'].mean().unstack().plot(kind='bar',
stacked=False)
ax.set_ylabel("Loan Amount")
ax.set_xlabel("Gender")
ax.set_title("Loan Amount Distribution by Gender and Education")
plt.show()

array = df.values
X = array[:,1:11]

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

kmeans = KMeans(n_clusters=5).fit(X)#number of clusters
centronoids = kmeans.cluster_centers_
print(centronoids)
#we put centronoids in a sample dataframe
cluster = pd.DataFrame(centronoids, columns=['Balance','QualMiles','BonusMiles','BonusTrans','FlightMiles','FlightTrans','DaysSinceEnroll'])

print(cluster)