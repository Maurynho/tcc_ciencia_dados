#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
pa = pd.read_csv('c://fontes_dados//bcdata.sgs.24975.csv',sep=';')


# In[4]:


pa


# In[5]:


pa.isnull()


# In[6]:


pa.info()


# In[7]:


pa['ano'] = pd.to_datetime(pa['data']).dt.year
del pa['data']
pa


# In[9]:


pa.to_csv('c://fontes_dados/bases_tratadas/quantidade_postos_atendimentos.csv',sep=',',index=False)


# In[ ]:




