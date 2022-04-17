#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importando biblioteca Pandas para tratar o dataset
import pandas as pd

#Carregando o dataset previamente armazenado no repositório do projeto
qtde_cooperativas = pd.read_csv('c://fontes_dados//bases_tratadas//quantidade_cooperativas.csv',sep=';')
qtde_cooperativas


# In[3]:


qtde_cooperativas.info()


# In[8]:


#Importanto a biblioteca matplotlib
get_ipython().system('pip install matplotlib')

from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


qtde_cooperativas.plot(x='ano', y='valor')


# In[5]:


qtde_cooperativas.plot(kind='bar', x='ano', y='valor')


# In[39]:


#criando um novo dataset para avaliar o primeiro e último registro
variacao_qtde = qtde_cooperativas.loc[[0,13]]
#visualizando os dados em um gráfico de barras
variacao_qtde.plot(kind='bar', x='ano', y='valor')


# In[47]:


#Calculando a variação de queda em percentual 
#variação percentual = (Valor futuro / Valor Inicial - 1)* 100
diferenca_percentual = (qtde_cooperativas.valor[13] / qtde_cooperativas.valor[0] - 1) * 100
diferenca_percentual


# In[46]:



qtde_cooperativas.valor[0]


# In[ ]:




