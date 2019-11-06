# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:14:56 2019

@author: antonio.castiglione
"""

# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
import scipy.stats as stats
from sklearn.cluster import KMeans

#read data
pathScripts="C:\\Users\\antonio.castiglione\\codiciPy\\"
df=pd.read_csv(pathScripts+'\\dataset_V2.csv')

# Check to see if there are any missing values in our data set
df.isnull().any()

#manipulation birthday column and recalled age
df['age']=df['birthday'].apply(lambda x: x.split('/')[-1])
df['year']='19'
df['age']=df['year']+df['age']
df['age']=pd.to_numeric(df['age'])
df['age']=2019-df['age']
front = df['age']
df.drop(labels=['year','age','birthday'], axis=1,inplace = True)
df.insert(0,'age',  front)

#working hours is a costant, we can remove
df.drop(labels=['num_working_hours'], axis=1,inplace = True)

# Looks like about 90% of employees stayed and 10% of employees left. 
# NOTE: When performing cross validation, its important to maintain this turnover ratio
turnover_rate = df.turnover.value_counts() / 2000

##############################################################################
#CONVERT  VARIABLES IN categories
##############################################################################
df['turnover']=df['turnover'].astype('category').cat.reorder_categories(['No','Yes']).cat.codes
df['gender']=df['gender'].astype('category').cat.reorder_categories(['Female','Male']).cat.codes
df['overtime']=df['overtime'].astype('category').cat.reorder_categories(['No','Yes']).cat.codes
df['status']=df['status'].astype('category').cat.reorder_categories(['Single','Married','Divorced']).cat.codes

#convert in categories and rename some variables
df=df.rename(columns={'work_satisfaction_level':'satisfaction'})
df['satisfaction']=df['satisfaction'].astype('category').cat.reorder_categories(['low', 'medium','high','very high']).cat.codes

df=df.rename(columns={'performance_level':'performance'})
df['performance']=df['performance'].astype('category').cat.reorder_categories(['outstanding','execellent']).cat.codes

df=df.rename(columns={'workplace_requirement_level':'workplace_requirement'})
df['workplace_requirement']=df['workplace_requirement'].astype('category').cat.reorder_categories(['low', 'medium','high','very high']).cat.codes

df=df.rename(columns={'coworker_relationship_level':'coworker_relationship'})
df['coworker_relationship']=df['coworker_relationship'].astype('category').cat.reorder_categories(['average', 'good','outstanding','execellent']).cat.codes

df=df.rename(columns={'training_frequency':'training'})
df['training']=df['training'].astype('category').cat.reorder_categories(['none','rarely','frequently']).cat.codes

df=df.rename(columns={'work_travel_frequency':'travel'})   
df['travel']=df['travel'].astype('category').cat.reorder_categories(['none','low','high']).cat.codes



describe=df.describe()

# Overview of summary (Turnover V.S. Non-turnover)
turnover_Summary = df.groupby('turnover')
tsummary=turnover_Summary.mean()

#Correlation Matrix
corr = df.corr()
corr = (corr)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
# Heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr, vmax=.5,mask=mask,linewidths=.2, cmap="YlGnBu")
 #           xticklabels=corr.columns.values,
#            yticklabels=corr.columns.values)
plt.title('Heatmap of Correlation Matrix')

# Find correlations with the target and sort
correla = df.corr()['turnover'].sort_values()
print('Most Positive Correlations: \n', correla.tail(5))
print('\nMost Negative Correlations: \n', correla.head(9))


# Let's compare the means of our employee turnover satisfaction against the employee population satisfaction
emp_population_satisfaction = df['satisfaction'].mean()
emp_turnover_satisfaction = df[df['turnover']==1]['satisfaction'].mean()

print( 'The mean for the employee population is: ' + str(emp_population_satisfaction) )
print( 'The mean for the employees that had a turnover is: ' + str(emp_turnover_satisfaction) )

df[['department', 'turnover']].groupby(['department'], as_index=False).mean().sort_values(by='turnover', ascending=False)


#Let's conduct a t-test  :
staval=stats.ttest_1samp(a=  df[df['turnover']==1]['satisfaction'], popmean = emp_population_satisfaction)  
#The test result shows the test statistic "t" is equal to -4.44. This test statistic 
#tells us how much the sample mean deviates from the null hypothesis. 
#If the t-statistic lies outside the quantiles of the t-distribution corresponding to our 
#confidence level and degrees of freedom, we reject the null hypothesis. 
#We can check the quantiles with stats.t.ppf():


#If the t-statistic value we calculated above (-4.44) is outside the quantiles, 
#then we can reject the null hypothesis

degree_freedom = len(df[df['turnover']==1])

LQ = stats.t.ppf(0.025,degree_freedom)  # Left Quartile

RQ = stats.t.ppf(0.975,degree_freedom)  # Right Quartile

print ('The t-distribution left quartile range is: ' + str(LQ))
print ('The t-distribution right quartile range is: ' + str(RQ))

#The t-distribution left quartile range is: -1.9718962236316093
#The t-distribution right quartile range is: 1.9718962236316089

#T-Test score is outside the quantiles
#P-value is lower than confidence level of 5%

#salary vs turnover
# Let's compare the means of our employee turnover salary against the employee population salary
emp_population_salary = df['salary'].mean()
emp_turnover_salary = df[df['turnover']==1]['salary'].mean()

emp_population_stock_option = df['stock_option_level'].mean()
emp_turnover_stock_option = df[df['turnover']==1]['stock_option_level'].mean()


# satisfaction distribution
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'satisfaction'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'satisfaction'] , color='r',shade=True, label='turnover')
plt.title('Employee Satisfaction Distribution: 0= low, 1= medium, 2= high, 3= very high - Turnover V.S. No Turnover')

#salary distribution
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'salary'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'salary'] , color='r',shade=True, label='turnover')
plt.title('Employee Salary Distribution - Turnover V.S. No Turnover')

#stock option distribution
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'stock_option_level'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'stock_option_level'] , color='r',shade=True, label='turnover')
plt.title('Employee stock_option_level Distribution - Turnover V.S. No Turnover')

#experience distribution
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'experience'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'experience'] , color='r',shade=True, label='turnover')
plt.title('Employee experience Distribution - Turnover V.S. No Turnover')

#overtime distribution
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'overtime'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'overtime'] , color='r',shade=True, label='turnover')
plt.title('Employee overtime Distribution - Turnover V.S. No Turnover')

#time line manager distribution
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'time_with_line_manager'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'time_with_line_manager'] , color='r',shade=True, label='turnover')
plt.title('Employee time_with_line_manager Distribution - Turnover V.S. No Turnover')

#age distribution
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'age'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'age'] , color='r',shade=True, label='turnover')
plt.title('Employee age Distribution - Turnover V.S. No Turnover')

#level distribution
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'level'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'level'] , color='r',shade=True, label='turnover')
plt.title('Employee level Distribution - Turnover V.S. No Turnover')

#current position distribution
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'current_position_experience'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'current_position_experience'] , color='r',shade=True, label='turnover')
plt.title('Employee current_position_experience Distribution - Turnover V.S. No Turnover')

#status distribution
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'status'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'status'] , color='r',shade=True, label='turnover')
plt.title('Employee status Distribution - Where :0=single 1=married 2=divorced')

 
#experience in %
ax = sns.barplot(x="experience", y="experience", hue="turnover", data=df, estimator=lambda x: len(x) / len(df) * 100)
ax.set(ylabel="Percent")

###########################################
#clustering
##########################################

# Graph and create 3 clusters of Employee Turnover
kmeans = KMeans(n_clusters=3,random_state=2)
kmeans.fit(df[df.turnover==1][["salary","experience"]])

kmeans_colors = ['green' if c == 0 else 'blue' if c == 2 else 'red' for c in kmeans.labels_]

fig = plt.figure(figsize=(10, 6))
plt.scatter(x="salary",y="experience", data=df[df.turnover==1],
            alpha=0.25,color = kmeans_colors)
plt.xlabel("Salary")
plt.ylabel("Experience")
plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)
plt.title("Clusters of Employee Turnover")
plt.show()

#save df
df.to_csv(pathScripts+'\\dfTurnover1.csv',sep=',') 






