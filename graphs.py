#modcom.co.ke/datascience
#modcom.co.ke/flask
#modcom.co.ke/flask/datascience/banks
#modcom.co.ke/bank
import pandas as pd

df = pd.read_csv("school.csv") #df is the dataframe
# print(df)
# print(df["StudyTime"])
print(df[["StudyTime", "Gender"]])
#narrow down
subset = df[["StudyTime", "Gender"]] #example of extracting different columns in a dataframe , maybe for plotting.
#below code shows rows and columns
print(df.shape)
#below checks and maps empty collumns
# print(df.isnull().sum())
# df.dropna(inplace = True ) #removing empties
# print(df.isnull().sum())
# print(df.shape)

#imputation, filling the empties,
df["Gender"].fillna(2, inplace=True)
df["Athlete"].fillna(2, inplace=True)
df["Smoking"].fillna(3, inplace=True)
df["Rank"].fillna(5, inplace=True)


#lets do for math
medianmath = df["Math"].median()
df["Math"].fillna(medianmath, inplace=True)

mediansprint = df["Sprint"].median()
df["Sprint"].fillna(mediansprint, inplace=True)

medianreading = df["Reading"].median()
df["Reading"].fillna(medianreading, inplace=True)

mediansleeptime = df["SleepTime"].median()
df["SleepTime"].fillna(mediansleeptime, inplace=True)

medianstudytime = df["StudyTime"].median()
df["StudyTime"].fillna(medianstudytime, inplace=True)

medianwriting = df["Writing"].median()
df["Writing"].fillna(medianwriting, inplace=True)


print(df.isnull().sum())

#===========PLOTTING===========

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(df["Math"], df["Reading"], s = 30, color = "red", alpha=0.75)
ax.set_xlabel("Mathematics")
ax.set_ylabel("Reading")
ax.set_title("Distribution of students mathematics by reading scores")
plt.show()

print(df["Reading"].describe())
print(df["Math"].describe())
fig, ax = plt.subplots()
ax.scatter(df["SleepTime"], df["StudyTime"], s = 30, color = "green", alpha=0.75)
ax.set_xlabel("Sleeping Time")
ax.set_ylabel("Study Time")
ax.set_title("Distribution of students Sleeping Time by Study Time scores")
plt.show()

#Histogram
_, ax = plt.subplots()
ax.hist(df['Sprint'], color='#539caf', bins=30)#bins are not a must because the
 # Label the axes and provide a title
ax.set_title("Distribution of Students Sprint Time")
ax.set_xlabel("Sprint")
ax.set_ylabel("Frequency")
plt.show()

fig, ax = plt.subplots()
ax.hist(df['Math'], color='#539caf', bins=30)#bins are not a must because the
 # Label the axes and provide a title
ax.set_title("Distribution of Students Math Scores")
ax.set_xlabel("Mathematics")
ax.set_ylabel("Frequency")
plt.show()

#Density
fig,  ax = plt.subplots()
ax.plot(df['Weight'], color='#539caf', lw=2)
ax.set_ylabel("Frequency")
ax.set_xlabel("sprint")
ax.set_title("Weight Distribution")
ax.legend(loc='best')
plt.show()

#PieChart
df["Rank"].replace(1, "Freshmen", inplace=True)
df["Rank"].replace(2, "Sophomore", inplace=True)
df["Rank"].replace(3, "Junior", inplace=True)
df["Rank"].replace(4, "Senior", inplace=True)
fig, ax = plt.subplots()
df.groupby('Rank').size().plot(kind='pie',autopct='%1.1f%%')
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Distribution by Rank in %")
plt.show()

#Bar Chart
fig, ax = plt.subplots()
df.groupby('Gender')['StudyTime'].mean().plot(kind='bar', color =
"#ffcccc")
ax.set_ylabel("StudyTime")
ax.set_xlabel("Gender")
ax.set_title("studyTime Distribution by Gender")
ax.legend(loc='best')
plt.show()

#stacd bar
fig,  ax = plt.subplots()
df.groupby(['Gender', 'Rank'])['Math'].mean().unstack().plot(kind='bar',
stacked=False)
ax.set_ylabel("Math")
ax.set_xlabel("Gender")
ax.set_title("Math Distribution by Gender and Rank")
plt.show()

#Test of hypothesis
#one sample t test
#two sample t test
#Angus
#chi square
#https://modcom.co.ke/datascience
#https://modcom.co.ke/flask/DataScience/banks.csv

