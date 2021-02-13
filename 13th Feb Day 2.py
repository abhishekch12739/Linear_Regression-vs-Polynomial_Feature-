#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Model vs Polynomial Regression 
# 
# Problem Statement: Predict the salary of the new empolyees comapring with their job title. 
# 
# By Abhishek Kumar 

# In[7]:


# Importing Libraries 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 


# In[16]:


# Importing Dataset 

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

print(dataset)


# In[24]:


# Training Linear Regression model 

from sklearn.linear_model import LinearRegression 

lin_reg = LinearRegression()
lin_reg.fit(x,y)

print('Training Done')


# In[30]:


# Visualizing Results of Linear Regression Model 

plt.scatter(x,y, color='green')
plt.plot(x,lin_reg.predict(x),color='blue')

plt.title('Salary Prediction (Linear Regression)')
plt.xlabel('Position Number')
plt.ylabel('Salary')

plt.show()
           


# In[40]:


# Training Polynomial Model 

from sklearn.preprocessing import PolynomialFeatures 

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)

print('Training Done')

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


# In[42]:


# Visualizing Results of Polynomial Feature Model 

plt.scatter(x,y, color='green')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')

plt.title('Salary Prediction (Polynomial Regression)')
plt.xlabel('Position Number')
plt.ylabel('Salary')

plt.show()


# In[43]:


lin_reg.predict([[6.5]])


# In[49]:


lin_reg_2.predict( poly_reg.fit_transform([[6.5]]))


# In[ ]:




