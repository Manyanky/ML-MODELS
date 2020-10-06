#https://modcom.co.ke/datascince/hypothesis
import pandas
import matplotlib.pyplot as plt
import scipy

#a hypoothesis is a claim on a given dataset
#we use statistical methods to prove the claim-true/false

df = pandas.read_csv("school.csv")
#paired sample t - test
 #Ho ; the paired mean of math and Eng are equal
 #H1; the paired math of Eng and Math are not Equal

medianmath = df['Math'].median()
df['Math'].fillna(medianmath, inplace=True)

medianEnglish = df['English'].median()
df['Math'].fillna(medianEnglish, inplace=True)

 #run test using paired

from scipy.stats import ttest_rel, ttest_ind
statistics, pvalue = ttest_rel(df['Math'], df['English'])
print('P value is', pvalue)

alpha = 0.95 #confidence level
# a less p value means ....reject null hypothesis
if pvalue < alpha:
    print('There is no enough evidence to support null hyppthesis')
    print('We reject the Null hypothesis')
    print('We take the null hypothesis. ')
    print('h1:  the paired means of math an english are not equal')
else:
    print('There is enough evidence to support null hyppthesis')
    print('We Accept the Null hypothesis')
    print('h1:  the paired means of math an english are  equal')

meanMath = df['Math'].mean()
meanEng = df['English'].mean()

print("English Mean is", meanEng)
print("Math Mean is", meanMath)

diff = meanEng-meanMath # diffrence
print("We can conclude that English scored higher than math by", diff)