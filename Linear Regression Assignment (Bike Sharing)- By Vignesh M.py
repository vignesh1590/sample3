#!/usr/bin/env python
# coding: utf-8

#    #            Linear Regression Assignment (Bike Sharing) - By Vignesh

# # Steps to Follow the Excercise
# 
# 1. Reading, Understanding & Visualizing Data
# 2. Preparing Data Modelling ( train-test split , Rescalling etc)
# 3. Train the Models
# 4. Residual Analysis
# 4. Predection & Evaluation on Test set`

# In[16]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # 1. Reading, Understanding & Visualizing Data

# In[30]:


# readng csv
df_days = pd.read_csv('day.csv',header =0)
df_days.head()


# ### Data Inspection

# In[31]:


df_days.shape


# In[32]:


df_days.info()


# In[33]:


df_days.describe()


# ### Data Cleaning
# 

# In[35]:


# checking Null Values
df_days.isnull().sum() * 100/df_days.shape[0]


# In[90]:


# Copy dataframe
df = df_days.copy()
df.shape


# In[91]:


df.drop_duplicates(inplace=True)
df.shape


# no duplicates found shape is same for df and df_days dataframe

# In[92]:


df.head()


# In[93]:


df.rename(columns={'yr':'year','mnth':'month','hum':'humidity'}, inplace=True)
df.head()


# In[94]:


# droping unwanted columns instant , dteday, casual, registered
df.drop(['instant','dteday','casual','registered'],axis=1,inplace=True)


# In[95]:


# df after droping columns
df.head()


# In[96]:


# Encoding/mapping the season column
df.season = df['season'].map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})
df.season.value_counts()



# In[97]:


# Encoding/mapping the month column
df.month = df.month.map({1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'june',7:'july',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'})
df.month.value_counts


# In[98]:


# Encoding/mapping the weekday column
df.weekday = df.weekday.map({0:'sun',1:'mon',2:'tue',3:'wed',4:'thu',5:'fri',6:'sat'})
df.weekday.value_counts


# In[99]:


# Encoding/mapping the weathersit column
df.weathersit = df.weathersit.map({1:'Clear',2:'Misty',3:'Light_snowrain',4:'Heavy_snowrain'})
df.weathersit.value_counts


# In[100]:


df.head()


# In[129]:


# Outlier Analyis, Univariate Analysis
fig, axs = plt.subplots(figsize=(20, 15))
plt.subplot(4,4,1)
plt1 =sns.boxplot(x='season', y='cnt', data =df)
plt.subplot(4,4,2)
plt2 =sns.boxplot(x='year', y='cnt', data =df)
plt.subplot(4,4,3)
plt2 =sns.boxplot(x='month', y='cnt', data =df)
plt.subplot(4,4,4)
plt2 =sns.boxplot(x='holiday', y='cnt', data =df)
plt.subplot(4,4,5)
plt2 =sns.boxplot(x='weekday', y='cnt', data =df)
plt.subplot(4,4,6)
plt2 =sns.boxplot(x='workingday', y='cnt', data =df)
plt.subplot(4,4,7)
plt2 =sns.boxplot(x='weathersit', y='cnt', data =df)

plt.tight_layout()


# In[135]:


num = ['temp','atemp','humidity','windspeed','cnt']
sns.pairplot(df, vars=num)

plt.suptitle('Pairplot of Numeric Variables', y=1.02)  # Add a title above the pairplot
pairplot = sns.pairplot(df, vars=num)
pairplot.map_diag(sns.histplot, color='blue', edgecolor='black') plt.show()


# In[148]:


# correlation between different variables Bivariate Analysis
fig, ax = plt.subplots(figsize=(10, 8))
heatmap = sns.heatmap(df[['temp','atemp','humidity','windspeed','cnt']].corr(), cmap="YlGnBu", annot=True, linewidths=0.5, linecolor='black', square=True)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize=10)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=10)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
plt.title("Correlation Heatmap")
plt.show()


# There is linear relationship between temp(1) and atemp(0.99). Both cannot be used in the model due to multicolinearity. We will decide which parameters to keep based on VIF and p-value 

# ## 2. Preparing Data Modelling ( train-test split , Rescalling etc)

# In[297]:


df.head()


# In[298]:


# Creating Dummy variable for month, weekday, weathersit and season.
months_df=pd.get_dummies(df.month,drop_first=True)
weekdays_df=pd.get_dummies(df.weekday,drop_first=True)
weathersit_df=pd.get_dummies(df.weathersit,drop_first=True)
seasons_df=pd.get_dummies(df.season,drop_first=True)


# In[299]:


# concatenating dataframes to new dataframe df_new
df_new = pd.concat([df,months_df,weekdays_df,weathersit_df,seasons_df],axis=1)


# In[300]:


df_new.drop(['season','month','weekday','weathersit'], axis = 1, inplace = True)


# In[301]:


df_new.head()


# In[302]:


# train test model import libraries
from sklearn.model_selection import train_test_split


# In[303]:


# train 70 % and test 30%
df_train, df_test = train_test_split(df_new, train_size = 0.7,test_size= 0.3 ,random_state = 100)


# In[304]:


df_new.shape


# In[305]:


# 70% of df_new is 510
df_train.shape


# In[306]:


# 30% of df_new is 219
df_test.shape


# In[307]:


df_train.head()


# In[314]:


from sklearn.preprocessing import MinMaxScaler


# In[317]:


# rescalling features
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['temp','atemp','humidity','windspeed','cnt']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[319]:


df_train.head()


# ## 3. Train the Models

# In[320]:


# Building the Linear Model

y_train = df_train.pop('cnt')
X_train = df_train


# In[321]:


y_train.head()


# In[322]:


X_train.head()


# In[323]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


# In[324]:


# Recursive feature elimination 

lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm)
rfe = rfe.fit(X_train, y_train)


# In[281]:


#list of Train Columns
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[325]:


# selecting the selected variable via RFE in col list
X_train.columns[rfe.support_]


# In[326]:


# selecting the Rejected variable via RFE in col list
X_train.columns[~rfe.support_]


# In[284]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[285]:


# Generic function to calculate VIF of variables

def getVIF(df):
    vif = pd.DataFrame()
    vif['Features'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif 


# In[327]:


col


# In[328]:


X_train[col]


# In[329]:


# dataframe with RFE selected variables

X_train_rfe = X_train[col]


# In[330]:


getVIF(X_train_rfe)


# workingday high VIF

# ## Building Multi Linear Regression Model

# In[331]:


import statsmodels.api as sm


# In[335]:


# Building 1st linear regression modela

X_train_lm_1 = sm.add_constant(X_train_rfe)
lr_1 = sm.OLS(y_train,X_train_lm_1).fit()
print(lr_1.summary())


# In[336]:


# As workingday shows high VIF values hence we can drop it
X_train_new = X_train_rfe.drop(['workingday'], axis = 1)

# Run the function to calculate VIF for the new model
getVIF(X_train_new)


# humidity shows high VIF

# In[337]:


# Building 2nd linear regression model

X_train_lm_2 = sm.add_constant(X_train_new)
lr_2 = sm.OLS(y_train,X_train_lm_2).fit()
print(lr_2.summary())


# In[338]:


# As humidity shows high VIF values hence we can drop it
X_train_new = X_train_new.drop(['humidity'], axis = 1)

# Run the function to calculate VIF for the new model
getVIF(X_train_new)


# In[339]:


# Building 3rd linear regression model

X_train_lm_3 = sm.add_constant(X_train_new)
lr_3 = sm.OLS(y_train,X_train_lm_3).fit()
print(lr_3.summary())


# In[340]:


X_train_new


# In[341]:


# As sat shows high P-value hence we can drop it
X_train_new = X_train_new.drop(['sat'], axis = 1)

# Run the function to calculate VIF for the new model
getVIF(X_train_new)


# In[342]:


# Building 4th linear regression model

X_train_lm_4 = sm.add_constant(X_train_new)
lr_4 = sm.OLS(y_train,X_train_lm_4).fit()
print(lr_4.summary())


# We can cosider the above model i.e lr_4, as it seems to have very low multicolinearity between the predictors and the p-values for all the predictors seems to be significant.
# 
# F-Statistics value of 234.8 (which is greater than 1) and the p-value of 4.60e-189  i.e almost equals to zero, states that the overall model is significant

# In[343]:


# checking params
lr_4.params


# # 4. Residual Analysis

# In[345]:


X_train_lm_4


# In[346]:


y_train_pred = lr_4.predict(X_train_lm_4)


# In[348]:


# Plot the histogram of the error terms
fig = plt.figure(figsize=(8, 6))
# Create a distribution plot of the errors
sns.distplot((y_train - y_train_pred), bins=20, hist_kws={'edgecolor': 'black'}, color='blue')
# Add a title
plt.title('Error Terms', fontsize=20)
# Label the axes
plt.xlabel('Errors', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
# Add grid lines
plt.grid()
# Show the plot
plt.show()


# Error Terms are normally distributed

# In[350]:


getVIF(X_train_new)


# In[351]:


# correlation between different variables Bivariate Analysis
fig, ax = plt.subplots(figsize=(10, 8))
heatmap = sns.heatmap(X_train_new.corr(), cmap="YlGnBu", annot=True, linewidths=0.5, linecolor='black', square=True)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize=10)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=10)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
plt.title("Correlation Heatmap")
plt.show()


# VIF values are less than 5 which is good and also there is no multicolinearity as seen from the heatmap.

# ## Applying scaling on the test dataset
# 

# In[354]:


num_vars = ['temp', 'atemp', 'humidity', 'windspeed','cnt']
df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.head()


# In[355]:


y_test = df_test.pop('cnt')
X_test = df_test


# In[356]:


y_test.head()


# In[358]:


X_test.head()


# In[359]:


col1 = X_train_new.columns

X_test = X_test[col1]

# Adding constant variable to test dataframe
X_test_lm_4 = sm.add_constant(X_test)


# In[360]:


y_pred = lr_4.predict(X_test_lm_4)


# In[362]:


from sklearn.metrics import r2_score


# ## $ Calculate_r2$

# In[363]:


r2 = r2_score(y_test, y_pred)
round(r2,4)


# In[364]:


# Model Evaluation (predict value for actual and predicted values)
fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 20) 
plt.xlabel('y_test', fontsize = 18)
plt.ylabel('y_pred', fontsize = 16) 


# In[365]:


round(lr_4.params,4)


# We can see that the equation of our best fitted line is:
# 
# $ count  =  0.2597 + (0.2340 * year ) + (- 0.1062 * holiday) + (0.4502 * temp) + (- 0.1396 * windspeed) 
#         + (-0.0704 * july) + (0.0564 * sep) + (-0.0479 * sun) + (-0.2916 * Light_snowrain) + (-0.0831 * Misty)
#         + (-0.1102 * spring) + (0.0494 * winter) $

# In[367]:


# Calculating Adjusted-R^2 value for the test dataset

adjusted_r2 = round(1-(1-r2)*(X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1),4)
print(adjusted_r2)


# In[372]:


# Visualizing the fit on the test data
# plotting a Regression plot

# Create a figure
plt.figure(figsize=(8, 6))
# Create a regression plot
sns.regplot(x=y_test, y=y_pred, ci=68, fit_reg=True, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
# Add a title
plt.title('y_test vs y_pred', fontsize=20)
# Label the axes
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_pred', fontsize=16)

# Add a legend
plt.legend(labels=["Regression Line"], loc="best", fontsize=14)

# Add grid lines
plt.grid()

# Show the plot
plt.show()


# # Comparision between Training and Testing dataset:
#     - Train dataset R^2          : 0.838
#     - Test dataset R^2           : 0.8097
#     - Train dataset Adjusted R^2 : 0.825 
#     - Test dataset Adjusted R^2  : 0.7996
# #### Demand of bikes depend on year, holiday, temp, windspeed,july, sep,sun, Light_snowrain, Misty, spring and winter.

# In[ ]:




