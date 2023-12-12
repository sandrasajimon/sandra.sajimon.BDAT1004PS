#!/usr/bin/env python
# coding: utf-8

# # Q1 

# In[13]:


import pandas as pd
url='https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
users = pd.read_csv(url, delimiter='|', index_col='user_id')
users.head()
users.groupby('occupation')['age'].mean()


# In[8]:


f1=users[users['gender']=='M'].groupby('occupation').count()['gender']
f2=users.groupby('occupation').count()['gender']
r=(f1/f2)*100
r.sort_values(ascending=True)#for printing female
r.sort_values(ascending=False)#male


# In[11]:


users.groupby('occupation').age.agg([min,max])# maximum and minimum ages
users.groupby(['occupation', 'gender'])['age'].mean()#mean

#presenting the mean age
g1=users[users['gender']=='M'].groupby('occupation').count()['gender']
g2=users.groupby('occupation').count()['gender']
r=pd.DataFrame((g1/g2)*100)
r.columns=[('M')]
r


# In[19]:


#percentage
g3=users[users['gender']=='F'].groupby('occupation').count()['gender']
g2=users.groupby('occupation').count()['gender']
r2=pd.DataFrame((g3/g2)*100)
r2.columns=['F']
r2['M']=r['M']
r2


# # Q2 Euro Teams

# In[21]:


import numpy as np
import pandas as pd
url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv"
euro = pd.read_csv(url)
euro.head()


# In[30]:


euro['Goals']#number
euro['Team'].count()#count
len(euro.columns)#length
discipline=euro[['Team','Yellow Cards','Red Cards']]
discipline.head()
discipline.sort_values(by=['Red Cards','Yellow Cards'])
euro[euro['Goals']>6]['Team']
euro[euro['Team'].str.startswith('G')]['Team']
euro.iloc[:, : 7]


# In[31]:


euro.iloc[ : , : -3]


# In[32]:


euro.loc[euro.Team.isin(['England','Italy','Russia']),
         'Shooting Accuracy']


# # Q3 Housing

# In[35]:


import numpy as np
import pandas as pd
first=pd.Series(np.random.randint(1,4,100))
second=pd.Series(np.random.randint(1,3,100))
third=pd.Series(np.random.randint(10000,30000,100))
data=pd.concat([first,second,third],axis=1)
df=pd.DataFrame(data)
print(df)


# In[36]:


df.rename(columns={0:'bedrs',1:'bathrs',2:'price_sqr_meter'},inplace=True)
print(df)


# In[38]:


bigcolumn=pd.concat([first,second,third])
bigcolumn
bigcolumn.reset_index(drop=True)


# # Q4 Wind Statistics

# In[42]:


import pandas as pd
import numpy as n
data = pd.read_csv(r"C:\Users\sandr\OneDrive\Desktop\wind.csv")
data


# In[55]:


with open(r"C:\Users\sandr\Downloads\wind.txt",'r') as txt_file:
    lines = txt_file.readlines()
data1 = [line.strip().split('\t') for line in lines]
import csv

with open(r'wind1.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(data1)
data1


# # #Q5
# 

# In[60]:


import numpy as np
import pandas as pd
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
chipo = pd.read_csv(url, sep='\t')
print(chipo.head(10))


# In[65]:


len(chipo)#number of observations in the dataset
len(chipo.columns)# the number of columns in the dataset
chipo.columns#name of all the columns
chipo.index   # dataset indexed

chipo.item_name.value_counts().head(1)#most-ordered item


# In[66]:


#most ordered item in the choice_description column
chipo.choice_description.value_counts().head(1)


# In[67]:


#items were orderd in total
len(chipo.item_name.unique())


# In[68]:


chipo['item_price'].str.replace('$', '').astype(float)


# In[70]:


chipo["item_price"].dtype
revenue = (chipo['quantity'] * chipo['item_price']).sum()
print("Total revenue:", revenue)


# In[72]:


chipo['order_id'].nunique()
len(chipo["item_name"].unique())


# In[80]:


import pandas as pd
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
chipo = pd.read_csv(url, sep='\t')
selected_columns = chipo[['order_id', 'quantity', 'item_price']]
selected_columns['item_price'] = selected_columns['item_price'].apply(lambda x: float(x[1:]))
selected_columns['total_price'] = selected_columns['quantity'] * selected_columns['item_price']
total_revenue_per_order = selected_columns.groupby('order_id')['total_price'].sum()


# In[82]:


total_revenue_per_order#What is the average revenue amount per order


# # Q6

# In[85]:


url=pd.read_csv(r"C:\Users\sandr\Downloads\us-marriages-divorces-1867-2014.csv")
import matplotlib.pyplot as plt
url.head()


# In[87]:


ax=url.plot(x='Year',y=['Marriages_per_1000','Divorces_per_1000'])
ax.set_xlabel('Year')
ax.set_ylabel('Marraiges & Divorcers per capita')


# # Q7

# In[90]:


b=url[(url['Year'].isin([1900,1950,2000]))]
x=b.plot.bar(x='Year',y=['Marriages_per_1000','Divorces_per_1000'],xlabel='Year',ylabel='Marriages & Divorcers per capita')


# # Q8

# In[92]:


actor=pd.read_csv(r"C:\Users\sandr\Downloads\actor_kill_counts.csv")
actor.head()


# In[94]:


actor.shape
a=actor.set_index('Actor').sort_values('Count').plot(kind='barh')
a.set_xlabel('Kill Count')


# # Q9

# In[96]:


roman=pd.read_csv(r"C:\Users\sandr\Downloads\roman-emperor-reigns.csv")
roman.head()


# In[103]:


r1=roman[(roman['Cause_of_Death']=='Assassinated')].set_index('Emperor')
r=r1.plot(kind='pie',x='Emperor',y='Length_of_Reign',figsize=(25,25),autopct='%1.5f%%',fontsize='x-large').legend(bbox_to_anchor=(1, 0.5), loc='center left')


# # Q10

# In[106]:


import seaborn as sns
cs=pd.read_csv(r"C:\Users\sandr\Downloads\arcade-revenue-vs-cs-doctorates.csv")
cs.head()


# In[111]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.DataFrame({
    'Year': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
    'Total Arcade Revenue (billions)': [1.5, 2.8, 3.2, 3.6, 4.0, 4.3, 4.5, 4.8, 5.0, 5.2],
    'Computer Science Doctorates Awarded (US)': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
})
sns.set(style='whitegrid')
sns.lmplot(
    x='Total Arcade Revenue (billions)',
    y='Computer Science Doctorates Awarded (US)',
    data=data,
    hue='Year',
    fit_reg=False
)
plt.xlabel('Total Arcade Revenue (billions)')
plt.ylabel('Computer Science Doctorates Awarded (US)')
plt.title('Relationship between Revenue and Computer Science PhDs (2000-2009)')
plt.show()


# In[ ]:




