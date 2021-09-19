#!/usr/bin/env python
# coding: utf-8

# In[193]:


#Importando bibliteca zipfile para tratramento de arquivos compactados
#Descompactado o arquivo de 2012
import zipfile
zipfile.ZipFile('C://fontes_dados//operações_credito//planilha_2012.zip').extractall('C://fontes_dados//operações_credito//2012/.')


# In[ ]:


#Concatenando todos os datasets em um único Dataframe
todos = pd.concat(pd.read_csv(f'C://fontes_dados//operações_credito//2012//planilha_2012{mes}.csv',sep=';') for mes in ('06','07','08','09','10','11','12'))


# In[190]:


#carregando o Dataset de 2021 em um dataframe
import pandas as pd
jul_2012 = pd.read_csv('C://fontes_dados//operações_credito//2012//planilha_201206.csv',sep=';')


# In[191]:


jul_2012


# In[192]:


#Analisando o formato do Dataframe
jul_2012.info()


# In[194]:





# In[195]:


todos.info()


# In[157]:


todos


# In[196]:


#Removendo colunas que não interessam
del todos['sr']
del todos['cnae_secao']
del todos['cnae_subclasse']
del todos['origem']
del todos['indexador']
del todos['a_vencer_ate_90_dias']
del todos['a_vencer_de_91_ate_360_dias']
del todos['a_vencer_de_361_ate_1080_dias']
del todos['a_vencer_de_1801_ate_5400_dias']
del todos['a_vencer_acima_de_5400_dias']
del todos['vencido_acima_de_15_dias']
del todos['a_vencer_de_1081_ate_1800_dias']
del todos['carteira_inadimplida_arrastada']
del todos['ativo_problematico']


# In[197]:


#Gerando um dataframe filtrando somente dados de Cooperativas de Crédito
filtrados_2012 = todos.query('tcb=="Cooperativas"')


# In[198]:


#Removendo a coluna depois de usada para filtrar
del filtrados_2012['tcb']


# In[199]:


filtrados_2012.info()


# In[200]:


filtrados_2012['ano'] = 2012


# In[201]:


del filtrados_2012['data_base']


# In[202]:


#Transformando campo do valor em float
filtrados_2012['carteira_ativa'] = filtrados_2012['carteira_ativa'].str.replace(',','.').astype(float)


# In[203]:


filtrados_2012['numero_de_operacoes'] = filtrados_2012['numero_de_operacoes'].str.replace('<=','', regex=True)


# In[204]:


filtrados_2012['numero_de_operacoes'] = filtrados_2012['numero_de_operacoes'].astype(int)


# In[205]:


filtrados_2012


# In[206]:


#Gerando arquivo de saída
filtrados_2012.to_csv('C://fontes_dados//operações_credito//2012//filtrados.csv',sep=',',index=False)


# In[207]:



filtrados_2012['ocupacao'] = filtrados_2012['ocupacao'].str.replace('PF -','', regex=True)
filtrados_2012['porte'] = filtrados_2012['porte'].str.replace('PF -','', regex=True)
filtrados_2012['modalidade'] = filtrados_2012['modalidade'].str.replace('PF -','', regex=True)

filtrados_2012['ocupacao'] = filtrados_2012['ocupacao'].str.replace('PJ -','', regex=True)
filtrados_2012['porte'] = filtrados_2012['porte'].str.replace('PJ -','', regex=True)
filtrados_2012['modalidade'] = filtrados_2012['modalidade'].str.replace('PJ -','', regex=True)


# In[209]:


filtrados_2012.to_csv('C://fontes_dados//operações_credito//2012//teste_saida2.csv')


# In[208]:


filtrados_2012


# In[ ]:




