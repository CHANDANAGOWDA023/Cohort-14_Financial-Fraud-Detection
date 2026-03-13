#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv("feature_engineered.csv")
df.head()


# In[2]:


df.info()


# In[3]:


df.drop(columns=['Date', 'Time', 'DateTime'], errors='ignore', inplace=True)


# In[4]:


df.drop(columns=['Laundering_type'], errors='ignore', inplace=True)


# In[5]:


X = df.drop(columns=['Is_laundering'])
y = df['Is_laundering']

print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# In[6]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# In[7]:


print(X_train.dtypes)


# In[8]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Original training set shape:", X_train.shape)
print("Resampled training set shape:", X_train_resampled.shape)


# In[9]:


print("Original class distribution:")
print(y_train.value_counts())

print("\nBalanced class distribution after SMOTE:")
print(y_train_resampled.value_counts())


# Strategic Train/Test Splitting
# 
# In this project, a strategic dataset splitting approach was implemented to ensure reliable and unbiased model evaluation.
# 
# First, the dataset was divided into training and testing sets using an 80–20 split. Since the dataset is highly imbalanced, stratified sampling was applied to maintain the same proportion of laundering and normal transactions in both the training and testing datasets.
# 
# To address the class imbalance problem, SMOTE (Synthetic Minority Over-sampling Technique) was applied only to the training dataset to balance the minority laundering class.
# 
# This strategic splitting approach helps the model learn from balanced training data while keeping the testing dataset completely unseen, resulting in a more realistic and reliable performance evaluation.
