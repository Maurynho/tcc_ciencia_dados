#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importando biblioteca Pandas para tratar o dataset
import pandas as pd

#Carregando o dataset previamente armazenado no repositório do projeto
qtde_pa = pd.read_csv('c://fontes_dados/bases_tratadas/quantidade_postos_atendimentos.csv',sep=',')
qtde_pa


# In[4]:


qtde_pa.info()


# In[5]:


#Importanto a biblioteca matplotlib
get_ipython().system('pip install matplotlib')

from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


qtde_pa.plot(x='ano', y='valor')


# In[3]:


#Calculando a variação de queda em percentual 
#variação percentual = (Valor futuro / Valor Inicial - 1)* 100
diferenca_percentual = (qtde_pa.valor[15] / qtde_pa.valor[0] - 1) * 100
diferenca_percentual


# In[3]:


qtde_pa.to_csv('c://fontes_dados/bases_tratadas/qtde_pa.csv',sep=';')


# In[ ]:




