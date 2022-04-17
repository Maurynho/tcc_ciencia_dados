#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importando as bibliotecas
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[69]:


dados = pd.read_csv('C://fontes_dados//ml//tudo.csv')


# In[70]:


##Apenas para fazer testes mais rápidos
dados = dados.query('ano >= 2019')
dados = dados.query('carteira_ativa <= 5000')
dados


# In[71]:


dados.to_csv('C://fontes_dados//ml//manipular.csv')


# In[38]:


#Mudar o tipo data
dados['data_base'] = pd.to_datetime(dados['data_base'], format='%Y-%m-%d')

dados.dropna(inplace = True)
dados.isnull().sum()

del dados['Unnamed: 0']
del dados['uf']

#Criando uma coluna com a média móvel anual dos valores
dados['media_movel_anual'] = (dados['carteira_ativa'].rolling(365).mean())/100

limite = dados[dados['carteira_ativa'] >= 100000].index
dados.drop(limite, inplace=True)

dados


# In[55]:





# In[39]:


dados.dropna(inplace = True)
dados.isnull().sum()


# In[40]:


dados.reset_index(drop=True)


# In[48]:


#Separando quantidade de linhas para treino e teste
qtd_linhas = len(dados)

qtd_linhas_treino = round(.30 * qtd_linhas)
qtd_linhas_teste = qtd_linhas - qtd_linhas_treino  
qtd_linhas_validacao = qtd_linhas -1

info = (
    f"linhas treino= 0:{qtd_linhas_treino}"
    f" linhas teste= {qtd_linhas_treino}:{qtd_linhas_treino + qtd_linhas_teste -1}"
    f" linhas validação= {qtd_linhas_validacao}"
)

info


# In[49]:


#separando as features e labels
features = dados.drop(['carteira_ativa', 'numero_de_operacoes', 'ano', 'data_base'], 1)
labels = dados['carteira_ativa']


# In[50]:


#Criando algoritmo para escolher as melhores variáveis (features) para montar o modelo

#Criando uma lista com as variáveis candidatas do dataset
variáveis_candidatas = ('carteira_ativa','media_movel_anual','numero_de_operacoes','modalidade','ocupacao','cliente')

#Estabelecendo a leitura das variáveis candidatas - features
k_best_features = SelectKBest(k='all')

#Estabelecendo pesos para o algoritmo escolher as melhores variáveis
k_best_features.fit_transform(features, labels)
k_best_features_scores = k_best_features.scores_

#Montando os pares de agrupamento estabelecidos
raw_pairs = zip(variáveis_candidatas[1:], k_best_features_scores)
ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1])))

#As melhores variáveisjá ordenadas e dentro de um objeto dicionário
k_best_features_final = dict(ordered_pairs[:15])

#Capturando as chaves do dicionário
best_features = k_best_features_final.keys()

#Imprimindo o resultado para avaliação
print ("Melhores Variáveis para o modelo:")
print (k_best_features_final)


# In[51]:


melhores_features = dados.loc[:,['modalidade','ocupacao','media_movel_anual','numero_de_operacoes']]
melhores_features


# In[52]:


#Separando os dados que serão usados para treino teste e validação
X_treino = melhores_features[ : qtd_linhas_treino]
X_teste = melhores_features[qtd_linhas_treino : qtd_linhas_treino + qtd_linhas_teste -1]

y_treino = labels[ : qtd_linhas_treino]
y_teste = labels[qtd_linhas_treino : qtd_linhas_treino + qtd_linhas_teste -1]

print( len(X_treino), len(y_treino))

print( len(X_teste), len(y_teste))


# In[53]:


# Normalizando os dados de entrada(features)

# Gerando o novo padrão
scaler = MinMaxScaler().fit(melhores_features)
features_scale = scaler.transform(melhores_features)

print("Features: ", features_scale.shape)
print(features_scale)


# In[54]:


# Normalizando os dados de entrada(features)

# Gerando o novo padrão
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_treino)  # Normalizando os dados de entrada(treinamento)
X_test_scale  = scaler.transform(X_teste)   


# In[55]:


#treinamento usando regressão linear
lr = linear_model.LinearRegression()
lr.fit(X_train_scale, y_treino)
pred = lr.predict(X_test_scale)
cd = r2_score(y_teste, pred)

f'Coeficiente de determinação:{cd * 100:.2f}'


# In[56]:


#rede neural
rn = MLPRegressor(max_iter=2000)

rn.fit(X_train_scale, y_treino)
pred= rn.predict(X_test_scale)

cd = rn.score(X_test_scale, y_teste)


f'Coeficiente de determinação:{cd * 100:.2f}'


# In[ ]:


#rede neural com ajuste hyper parameters

rn = MLPRegressor()

parameter_space = {
        'hidden_layer_sizes': [(i,) for i in list(range(1, 21))],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'], 
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

search = GridSearchCV(rn, parameter_space, n_jobs=-1, cv=5)


search.fit(X_train_scale,y_treino)
clf = search.best_estimator_
pred= search.predict(X_test_scale)

cd = search.score(X_test_scale, y_teste)

f'Coeficiente de determinação:{cd * 100:.2f}'


# In[57]:


valor_novo = features.tail(1)
valor_novo


# In[58]:


#executando a previsão


previsao=scaler.transform(valor_novo)


pred=lr.predict(previsao)

pred


# In[68]:


data_full = dados['data_base']
data = data_full.tail(1)

res_full = dados['carteira_ativa']
res = res_full.tail(1)

dados_previsao = pd.DataFrame({'data_base':data, 'real':res, 'previsao':pred})
dados_previsao.set_index('data_base', inplace=True)

print(dados_previsao)


# In[62]:


dados


# In[ ]:




