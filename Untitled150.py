#!/usr/bin/env python
# coding: utf-8

# # question 01
Q1: What are missing values in a dataset? Why is it essential to handle missing values? Name some
algorithms that are not affected by missing values.
Missing values are those data points that are not present in a dataset. These values may be missing due to various reasons, such as data entry errors, equipment failure, or participant non-response.

It is essential to handle missing values in a dataset because they can lead to biased or inaccurate results. Some algorithms may also require complete data to function correctly, and missing values can cause errors or even failure of the algorithm. Moreover, ignoring missing values may result in a reduced sample size, leading to a loss of statistical power and increased variability.

Some algorithms that are not affected by missing values are decision trees, random forests, and some forms of ensemble learning. These algorithms can handle missing values by replacing them with surrogate values or using them as a separate category in the analysis. Additionally, some regularization techniques, such as Lasso and Ridge regression, can handle missing values by reducing the impact of those variables in the model.
# # question 02
Deletion: This technique involves removing the rows or columns containing missing values.

Example: Dropping all rows with missing values in a Pandas DataFrame.
# In[3]:


import pandas as pd

df = pd.read_csv('data.csv')
df_dropped = df.dropna(axis=0) # Drop rows with missing values

Imputation: This technique involves filling in the missing values with estimated or imputed values.

Example: Filling in missing values with the mean value of the column in a Pandas DataFrame.
# In[4]:


import pandas as pd

df = pd.read_csv('data.csv')
df_imputed = df.fillna(df.mean()) # Fill missing values with the mean of the column

Prediction: This technique involves using machine learning models to predict the missing values based on the other available data.

Example: Using K-Nearest Neighbors (KNN) to impute missing values in a Pandas DataFrame.
# In[5]:


import pandas as pd
from sklearn.impute import KNNImputer

df = pd.read_csv('data.csv')
imputer = KNNImputer(n_neighbors=2)
df_imputed = imputer.fit_transform(df)

Interpolation: This technique involves filling in the missing values with estimated values based on patterns in the available data.

Example: Using linear interpolation to fill in missing values in a Pandas DataFrame.
# In[6]:


import pandas as pd

df = pd.read_csv('data.csv')
df_interpolated = df.interpolate()


# Flagging: This technique involves creating a new column or indicator variable that indicates whether a value is missing or not.
# 
# Example: Creating a new column in a Pandas DataFrame to indicate whether a value is missing or not.

# In[7]:


import pandas as pd

df = pd.read_csv('data.csv')
df['is_missing'] = df.isna().astype(int) # Create new column with 1 if value is missing, 0 otherwise


# In[ ]:




