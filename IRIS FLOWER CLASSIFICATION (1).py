#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the Libraries

import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


#Read The Dataset

df_iris = pd.read_csv("C://Users//HP//Downloads//Iris1.csv")


# In[3]:


df_iris


# In[4]:


# Analyse the Dataset

df_iris.info()


# In[5]:


df_iris.describe()


# In[6]:


df_iris.nunique()


# In[7]:


df_iris.head()


# In[8]:


df_iris.tail()


# In[9]:


df_iris.isnull().sum()


# In[10]:


df_iris


# In[11]:


corr=df_iris.corr()
corr


# In[12]:


sns.heatmap(corr,annot=True)


# In[13]:


#Defining dependent and independent variable
x=df_iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm']]
y=df_iris['PetalWidthCm']


# In[14]:


#splitting x and y in to train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[15]:


#Train Test Split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30, random_state=42)


# In[16]:


#Import the model/Algorithm
from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[17]:


#Train model with x_train and y_train
lr.fit(x_train,y_train)


# In[18]:


#Predict with x_train and y_train
y_pred=lr.predict(x_test)


# In[19]:


#Evaluate the Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('its Mean_absolute_error-',mean_absolute_error(y_pred,y_test))
print('mean_squared_error-',mean_squared_error(y_pred,y_test))
print('r2_score-',r2_score(y_pred,y_test))
print('Model Accuracy',lr.score(x_test,y_test))


# In[20]:


x_test.columns


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
sns.scatterplot(x=x_test['PetalLengthCm'],y=y_test,label='actual data')
sns.lineplot(x=x_test['PetalLengthCm'],y=y_pred,color='red',label='predict data')
plt.legend()
plt.grid()

