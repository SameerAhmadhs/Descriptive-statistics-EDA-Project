#!/usr/bin/env python
# coding: utf-8

# ## Descriptive statistics & Exploratory Data Analysis (EDA) Project

# In[1]:


# import libraries
import pandas as pd
import numpy as np

# import sidetable # for frequency table
# use: pip install sidetable for installation


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use('bmh')

import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# for HD visualizations
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# In[2]:


# importing banking dataset
bank_df = pd.read_excel(r'C:\Data\Banking Data.xlsx')


# In[3]:


bank_df


# ## Audjest Column name

# In[4]:


bank_df.columns


# In[5]:


list(bank_df[1:2].values)


# In[6]:


bank_df.columns = ('customer_id', 'customer_age', 'salary', 'balance', 'marital', 'job_edu',
        'targeted_before_or_not', 'default', 'housing_loan', 'loan', 'contact', 'contact_day',
        'contact_month', 'durationof_call', 'campaign', 'pdays', 'previous', 'prv_outcome',
        'response_after_call')


# In[7]:


bank_df.head()


# In[8]:





# In[24]:


bank_df.reset_index(drop=True,inplace=True)
bank_df.head()
bank_df.shape


# # Data cleaning

# In[10]:


bank_df.info()


# ## Changing data type

# In[33]:


bank_df.customer_id=bank_df.customer_id.astype(int)
bank_df.customer_age=bank_df.customer_age.astype(float)
bank_df.balance = bank_df.balance.astype(int)
bank_df.contact_day = bank_df.contact_day.astype(int)
bank_df.campaign = bank_df.campaign.astype(int)
bank_df.pdays = bank_df.pdays.astype(int)
bank_df.previous = bank_df.previous.astype(int)
bank_df.salary = bank_df.salary.astype(float)


# In[34]:


bank_df.isna().sum()


# In[38]:


# Dealing with salary cloumn to fill null values
bank_df.salary.mean()
bank_df.salary.median()
bank_df.salary.plot(kind='box')
# Because of no outliers we can fill null values with mean
bank_df.salary.replace(np.nan,bank_df.salary.mean(),inplace=True)


# In[40]:


bank_df.isna().sum() # now there is no null value in our dataframe


# In[26]:


# cheack duplicates
bank_df.duplicated().sum()


# ## Deal with nan value

# In[13]:


bank_df.isna().sum()


# In[31]:


bank_df.replace('?',np.nan,inplace=True)


# In[15]:


#1) Dealing with  null values of customer column age 
bank_df.customer_age.mean()
bank_df.customer_age.median()

bank_df.customer_age=bank_df.customer_age.fillna(bank_df.customer_age.mean())
# replacing null values with mean of customer age 


# In[16]:


bank_df.customer_age.isna().sum()


# In[17]:


#2) dealing with customer contact month column
# Filling the null values with interpolate
bank_df.contact_month.interpolate(method='ffill',limit_direction='forward',inplace=True)
bank_df.contact_month.isna().sum()


# In[18]:


bank_df.info()


# In[19]:


#3) dealing with null values of (response after call) column 
bank_df.response_after_call.astype(str).mode()


# In[20]:


bank_df.response_after_call.interpolate(method='ffill',limit_direction='forward',inplace=True)


# In[44]:


# Merging day column with month column of dataframe
bank_df['contact_Date'] = bank_df['contact_day'].astype(str)+bank_df['contact_month']


# In[45]:


bank_df.head()


# In[51]:


# Changing contact_date cloumn to dateformat
from datetime import datetime
bank_df['Con_Date'] = pd.to_datetime(bank_df.contact_Date,format='%d%b, %Y')


# In[54]:


# Droping unrequired cloumn of date
bank_df.drop(['contact_day','contact_month','contact_Date'],axis=1,inplace=True)


# In[57]:


bank_df.to_csv(r'C:\Data\Banking_cleaned_Data.csv')


# In[55]:


bank_df.info()


# In[56]:


bank_df.head()


# In[60]:


bank_df.describe()


# # Data visualization 

# ## Univarieate Analysis 

# ### 1)Histogram

# In[66]:


plt.figure(figsize=(12,5))
plt.hist(bank_df.customer_age,edgecolor = 'b', color= 'orange');


# #### Insight
# 1)We can see that maximum count of customer of age who have account in bank  is  bitween 25 to 40.
# 
# 2)After 55 age of customers,there is very minimum no of account.

# In[16]:


plt.figure(figsize=(12,5))
plt.hist(bank_df1.salary,edgecolor = 'b', color= 'orange');


# Observation= Maximum no of salary of customers are 10k-20k, 50k-70k and 1lakh-1.20lakh

# ## 2) Box plot

# In[69]:


plt.figure(figsize=(12,5))
sns.boxplot(data=bank_df,x='salary');


# observation = Range of salary is 20k to 70k and there is no outliers in salary and median of salary is 60k.

# In[71]:


plt.figure(figsize=(12,5))
sns.boxplot(data=bank_df,x='balance');


# observation = Maximum customers save there salary bitween 0 to 1500 rs and there are some customers who save large amount of there salary.

# In[70]:


plt.figure(figsize=(12,5))
sns.boxplot(data=bank_df,x='customer_age');


# observation= Age of customers bitween 32 to 47 have maximum no of account in this bank

# In[72]:


plt.figure(figsize=(12,5))
sns.boxplot(data=bank_df,x='campaign');


# In[2]:


#starting point
bank_df1 = pd.read_csv(r'C:\Data\Banking_cleaned_Data.csv')


# In[3]:


from datetime import datetime
bank_df1['Con_Date'] = pd.to_datetime(bank_df1.Con_Date)
bank_df1.drop('Unnamed: 0',axis=1,inplace=True)


# In[4]:


bank_df1.info()


# In[5]:


bank_df1.head()


# ## 3)Count plot

# In[7]:


plt.figure(figsize=(12,5))
sns.countplot(data=bank_df1,x='loan');


# Maximum customer do use to take loan from bank apprx 7000 taken loan from bank out of 35000

# In[25]:


plt.figure(figsize=(12,5))
sns.countplot(data=bank_df1,x='housing_loan');


# obseravation =1) 25000 customer take house loan out of 45211 customers
#      2) 20000 cutomer do not take house loan

# In[8]:


plt.figure(figsize=(12,5))
sns.countplot(data=bank_df1,x='marital');


# observation = 1)28k customers are maried
# 
# 2)13k customers are single 
# 
# 3)5k customers are divorced

# In[9]:


plt.figure(figsize=(12,5))
sns.countplot(data=bank_df1,x='response_after_call');


# only 8% customer responed after call out of 100%

# ## 4) scatter plot

# In[12]:


sns.scatterplot(data =bank_df1, x = 'balance', y= 'customer_age');


# In scatter plot we can see that maximum peopal save there money is below 20000 and the age of customer 
# between 30 to 60 saves more money.

# In[13]:


sns.scatterplot(data = bank_df1, x = 'salary', y= 'customer_age');


# ## 5)pair plot

# In[14]:


sns.pairplot(bank_df1);


# ## 6) Hexin plot

# In[23]:


bank_df1.plot.hexbin(x='salary',
                    y='customer_age',
                    reduce_C_function=np.sum,
                    gridsize=10)


# ## 7) Bar plot

# In[13]:


sns.barplot(data = bank_df1, x = 'marital', y = 'salary', ci = None, palette='bright');


# maximum no.  of  divorced peopal have salary above 60,000

# In[14]:


sns.barplot(data = bank_df1, x = 'marital', y = 'balance', ci = None, palette='bright');


# Maximum no of maried peopal save there salary.

# In[15]:


sns.barplot(data = bank_df1, x = 'housing_loan', y = 'salary', ci = None, palette='bright');


# ## 8) Distribution plot

# In[18]:


plt.figure(dpi = 200)
sns.displot(data = bank_df1, x = 'salary', height = 5, aspect = 15/5,kde = True);


# maximum no of customer's earning salary are  20k, 60k and 1lakh.

# In[19]:


plt.figure(dpi = 200)
sns.displot(data = bank_df1, x = 'balance', height = 5, aspect = 15/5,kde = True);


# approx 2000 customers save there salary up to 1500

# ## 9) Heat map

# In[22]:


sns.heatmap(bank_df1.corr(), cmap = 'rainbow', annot=True);


# # Statistical manipulation 

# ## 1. Find the correlation between the columns and draw the observations from it.

# In[11]:


bank_df1.corr()


# **Observation**= 1)Costomer id is directly proportional to previous day
#                  2)Maximum column are not proportonal to each other, show almost 0 relation with each other  

# ## 2. What is the mean age and duration time of the customers with respect to every column?

# In[13]:


bank_df1.describe()


# ## 3. Find the mean and median of every column response wise and draw the observations.

# In[14]:


bank_df1.mean()
bank_df1.median()


# **Observation**= In customer's balance and previous day column there is a large difference b/n mean and meadian,it means this column consist maximum no. of outliears  

# ## 4)Show that the columns are following the Normal Distribution or not, if not following try to convert it non-normal to normal.

# In[13]:


# salary 
sns.displot(data=bank_df1,x='salary',kde=True)


# In[11]:


np.mean(bank_df1.salary)
np.std(bank_df1.salary)

np.mean(bank_df1.salary)+np.std(bank_df1.salary)

np.mean(bank_df1.salary)-np.std(bank_df1.salary)


# ## Converting salary column to normal disritution 

# In[24]:


bank_df1.salary.skew()


# In[25]:


np.log(bank_df1.salary).skew()


# In[26]:


sns.displot(np.log(bank_df1.salary), kde = True);


# ## 2 Age of customer

# In[27]:


# Age
sns.displot(data=bank_df1,x='customer_age',kde=True)


# In[28]:


bank_df1.customer_age.skew()


# In[29]:


np.log(bank_df1.customer_age).skew()


# In[30]:


sns.displot(np.log(bank_df1.customer_age), kde = True);


# ## Balance column 

# In[31]:


sns.displot(data=bank_df1,x='balance',kde=True)


# In[32]:


bank_df1.balance.skew()


# highly skwed 

# ### converting to normal distribution 

# In[33]:


np.log(bank_df1.balance).skew()


# In[36]:


sns.displot(np.log(bank_df1.balance),kde=True)


# Now its looking normal distribution bell like curve

# ## 5)Find the Best features using correlation and Chi-square test.

# ### Chi-square test 
# 

# Trying to find relation between Job and house loan

# In[6]:


# Fixing job column
bank_df1['Job'] = bank_df1.job_edu.apply(lambda x: x.split(',')[0])
bank_df1['Education'] = bank_df1.job_edu.apply(lambda x: x.split(',')[1])


# In[7]:


bank_df1.head()


# In[8]:


import scipy.stats as stats


# In[9]:


data_crosstab = pd.crosstab(bank_df1.Job,bank_df1.housing_loan,margins=True,margins_name='Total')


# In[10]:


data_crosstab


# In[11]:


# significance level
alpha = 0.05

rows = bank_df1.Job.unique()
columns = bank_df1.housing_loan.unique()

print(rows, columns)


# In[12]:


# Finding chi square value
chi_square = []
for i in columns:
    for j in rows:
        O = data_crosstab[i][j]
        E = data_crosstab[i]['Total'] * data_crosstab['Total'][j] / data_crosstab['Total']['Total']
        chi_square.append((O-E)**2/E)


# In[13]:


chi_square


# In[14]:


chi_square = np.sum(chi_square)

print(chi_square)


# In[17]:



# The critical value approach
print("\n--------------------------------------------------------------------------------------")
print(" The critical value approach to hypothesis testing in the decision rule")
critical_value = stats.chi2.ppf(1-alpha, (len(rows)-1)*(len(columns)-1))

if chi_square > critical_value:
    conclusion = "Null Hypothesis is rejected."
else:
    conclusion = "Failed to reject the null hypothesis."
        
print("chisquare-score is:", chi_square, " and critical value is:", critical_value)
print(conclusion)


# We have sufficient evidence that there is realtion between job and house loan

# ## 6)Find the probabilities with respect to the job role and education with customer responses.

# In[21]:


# Making contingency table JOb with respect to response after call
contigency_table = pd.crosstab(bank_df1.Job,bank_df1.response_after_call,margins=True,margins_name='Total')


# In[22]:


contigency_table


# ### Marginal probability

# In[23]:


p_No = 39919/45211
p_yes= 5292/45211

p_adm = 5171/45211
p_bc=9732/45211
p_ent = 1487/45211
p_hm = 1240/45211
p_manag = 9458/45211
p_ret = 2264/45211
p_semp = 1579/4511
p_ser = 4154/45211
p_std = 937/45211
p_tech = 7597/45211
p_unempl = 1303/45211
p_unkno = 288/45211


# In[24]:


p_adm
p_bc
p_ent
p_hm
p_manag
p_No
p_ret
p_semp
p_ser
p_std
p_tech
p_unempl
p_unkno
p_value
p_yes


# In[25]:


# Probability of Job with respect to resopnse of call 
contigency_table/45211


# ### Probability of eduation with respect to resopnse of call 

# In[26]:


# Making contingency table JOb with respect to response after call
contigency_table = pd.crosstab(bank_df1.Education,bank_df1.response_after_call,margins=True,margins_name='Total')


# In[27]:


contigency_table


# In[28]:


# Probability
contigency_table/45211


# # Let’s check if we have any statistical patterns in the Data frame (using plots or analysis).

# In[36]:


plt.figure(figsize=(10,6),dpi=200)

plt.subplot(321)
sns.boxplot(data=bank_df1,x = 'salary')
plt.subplot(322)
sns.boxplot(data=bank_df1,x = 'balance')

plt.subplot(323)
sns.barplot(data = bank_df1, x = 'marital', y = 'salary', ci = None, palette='bright');
plt.subplot(324)
sns.barplot(data = bank_df1, x = 'marital', y = 'balance', ci = None, palette='bright');
plt.subplot(325)
sns.scatterplot(data =bank_df1, x = 'balance', y= 'customer_age');
plt.subplot(326)
sns.scatterplot(data =bank_df1, x = 'salary', y= 'customer_age');


# plt.tight_layout();


# observation = In bar plot we can see that average salry of maried costomers are  57k and saving is more than singl and divorced peopal 

# 
