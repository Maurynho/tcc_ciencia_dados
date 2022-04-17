#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importando as bibliotecas
get_ipython().system('pip install tensorflow')

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
dados = pd.read_csv('C://fontes_dados//ml//tudo.csv')
dados['data_base'] = pd.to_datetime(dados['data_base'], format='%Y-%m-%d')
dados['ano'] = dados['data_base'].dt.year

del dados['Unnamed: 0']

dados.query('carteira_ativa >= 850' and 'carteira_ativa <= 1000', inplace = True)
dados


# In[9]:


import pandas as pd
dados = pd.read_csv('C://fontes_dados//ml//tudo.csv')
dados['data_base'] = pd.to_datetime(dados['data_base'], format='%Y-%m-%d')
dados['ano'] = dados['data_base'].dt.year

del dados['Unnamed: 0']

dados


# In[3]:


#Selecionando as colunas do dataset, agora sem "cliente" 
features = ['ocupacao', 'modalidade']

#y armazena a coluna alvo, o valor de empréstimos que se espera prever
y = dados.carteira_ativa

#X armazena as variáveis que serão usadas no modelo (features)
X = dados[features]


# In[6]:


#Importando o algoritmo RandomForestRegressor para processar a árvore aletório e mean_absolute_error para
#avaliar a qualidade dos dados obtidos
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#Função para separar dados de treino e de teste
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 5)

#criando o modelo com valor de aleatoriedade igual a 24
forest_modelo = RandomForestRegressor(random_state = 24)

#Treinando o modelo com dados de treino pré-selecionados
forest_modelo.fit(train_X, train_y)

#Gerando as previsões
previsoes = forest_modelo.predict(val_X)
print(mean_absolute_error(val_y, previsoes))


# In[8]:


#Avaliação prévia em comparação ao modelo de treino e os valores previstos
print("Previsões: ", previsoes[:5])
print("Alvo     : ", val_y[:5].values)


# In[10]:


from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    modelo = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state = 5)
    modelo.fit(train_X, train_y)
    preds_val = modelo.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[13]:


print(previsoes)


# In[14]:


#Importando as bibliotecas necessárias
import pandas as pd;
import numpy as np;
import seaborn as sns; sns.set();
import matplotlib.pyplot as plt;
import plotly.express as px;


# In[ ]:


fig  = px.scatter(dados, x = 'modalidade', y = 'carteira_ativa', log_x = True, width = 800)
fig.update_traces(marker = dict(size = 12, line=dict(width = 2)), selector = dict(mode = 'markers'))
fig.update_layout(title = 'Titulo')
fig.update_xaxes(title = 'Ocupacao')
fig.update_yaxes(title = 'Valores')
fig.show()


# In[ ]:




