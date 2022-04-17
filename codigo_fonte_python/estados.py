#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install ibge')

from ibge.localidades import *
estados = Estados()


# In[5]:


estados.json()


# In[6]:


import pandas as pd

dados_estados = pd.json_normalize(estados.json())
print(dados_estados)


# In[7]:


type(dados_estados)


# In[8]:


del dados_estados['id']
del dados_estados['regiao.id']
del dados_estados['regiao.sigla']
del dados_estados['regiao.nome']

dados_estados


# In[10]:





# In[ ]:




