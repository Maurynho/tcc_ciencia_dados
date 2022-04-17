#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Estabelecendo conexão e requisitando a página que onde os dados estão
import urllib3
url = 'https://www.br.undp.org/content/brazil/pt/home/idh0/rankings/idhm-uf-2010.html'
conexao = urllib3.PoolManager()
retorno = conexao.request('GET',url)

#Iniciando a manipulação dos dados da página
from bs4 import BeautifulSoup 
pagina = BeautifulSoup(retorno.data, 'html.parser')


# In[2]:


#Recuperando apenas a estrutura de tabela no HTML
tabela = pagina.find_all('table', class_ = 'tableizer-table')
tabela


# In[3]:


dado = []

for celulas in tabela:
    celula = celulas.find_all('td')
    for dados in celula:
        dado.append(dados.find(text=True))

dado


# In[4]:


#Importando biblioteca Pandas para converter a lista em Dataframe
import pandas as pd
dados_uf = pd.DataFrame(dado)

#Importando biblioteca numpy para ajustar os dados em uma tabela com 27 linhas por 6 colunas
import numpy as np
matriz_np = np.array(dados_uf)
matriz_ajustada = np.reshape(matriz_np, (27,6)) 

#Criando o dataframe final inserindo os títulos das colunas
estados_idh = pd.DataFrame(matriz_ajustada,columns=['rank','nome','idh_geral','idh_renda','idh_logenvidade','idh_educacao'])
estados_idh


# In[5]:


estados_idh.to_csv('C://fontes_dados///bases_tratadas/dados_idh.csv',sep=';')


# In[ ]:




