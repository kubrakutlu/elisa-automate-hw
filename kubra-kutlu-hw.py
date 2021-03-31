#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np


# In[8]:


df = pd.read_csv("/home/user/machine-learning-exercises/elisa_homework_data_users.csv",sep=";")
print(df.head())
target_cell_x = df["L1547777_x"]
target_cell_avg_user_count = df["L1547777"]
avg_user_counts = df.iloc[:,13:]
#print(avg_user_counts['L1547777'])
#print(target_cell_x)


# In[9]:


df_new = pd.concat([df["L1547777_x"],  df.iloc[:,13:]], axis=1)
df_new["L1547777_x"] = df_new["L1547777_x"].astype(str)
df_new=df_new.apply(lambda x: x.str.replace(',','.'))
print(df_new.head())
idx=df_new.columns.get_loc('L1547777')


# In[10]:


# All values of L1547777_x in training sequences are the same, where it was however supposed to go its max value and
# return back to the previous value
print(df_new.iloc[498:501,0])

# Code below finds the index where the max value is located and then prints the whole row with that index
max_x_index = df_new.iloc[498:501,0].astype(float).idxmax()
print(df_new.iloc[max_x_index])


# In[11]:


import math 
cells = list(df_new.iloc[:,1:].columns)


# In[22]:


df_new['L1547777']=df_new['L1547777'].astype(float)

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(10,10))
locs, labels = plt.yticks()  # Get the current locations and labels.
plt.yticks(np.arange(math.floor(min(df_new['L1547777'])), math.ceil(max(df_new['L1547777'])), step=1.0))  # Set label locations.
plt.scatter(df_new["L1547777_x"], df_new['L1547777'])
plt.title('Average user count in target cell L1547777 vs. parameter L1547777_x')
plt.ylabel('Average user count')
plt.xlabel('L1547777_x')
plt.savefig('target-cell.png')

plt.show()
plt.clf()


# In[12]:


cells = list(df_new.iloc[:,10:15].columns)
for cell in cells:
    df_new[cell]=df_new[cell].astype(float)
    fig = plt.figure(figsize=(10,10))
    locs, labels = plt.yticks()  # Get the current locations and labels.
    plt.yticks(np.arange(math.floor(min(df_new[cell])), math.ceil(max(df_new[cell])), step=1.0))  # Set label locations.
    c=plt.scatter(df_new["L1547777_x"], df_new[cell], label=cell)
    plt.title('Average user count in cell {} vs. parameter L1547777_x'.format(cell))
    plt.ylabel('Average user count')
    plt.xlabel('L1547777_x')
    plt.savefig('{}.png'.format(cell))
    plt.show()
    plt.clf()


# In[43]:


# Get avg values where _x changes from 32 to 26
last=df_new.iloc[75,1:]
first=df_new.iloc[76]
df_change=pd.DataFrame()
df_change['32']=last
df_change['26']=first
print(df_change)


# In[42]:


fig = plt.figure(figsize=(30,10))
plt.rcParams.update({'font.size': 14})
locs, labels = plt.yticks()  # Get the current locations and labels.
#plt.xticks(np.arange(math.floor(min(df_change['32'])), math.ceil(max(df_change['32'])), step=1.0))  # Set label locations.

plt.scatter(df_change['32'], df_change['26'])
plt.title('Average user counts of all cells when L1547777_x is 26 vs. 32'.format(cell))
plt.ylabel('Average user counts when L1547777_x is 26')
plt.xlabel('Average user counts when L1547777_x is 32')
plt.savefig('{}.png'.format("change"))
plt.show()
plt.clf()


# In[17]:


sns.pairplot(df_new.iloc[:,17:20], height = 5)


# In[ ]:




