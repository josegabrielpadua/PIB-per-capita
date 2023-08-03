
## Bibliotecas e Funções Utilizadas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error

# Lendo os dados

df_gdp_capita = pd.read_csv('C:\Workspace\PIB-per-capita\dataset\gdppercapita_us_inflation_adjusted.csv')

df_gdp_capita.head(15)

df_gdp_capita.shape

# Fazendo uma busca e filtrando os dados para que sejam da argentina

df_argentina = df_gdp_capita.loc[df_gdp_capita['country'] == 'Argentina']

df_argentina.head()

# Deixando os dados prontos

# Verificando valor nulo

df_argentina.isnull().sum()

df_argentina = df_argentina.drop(columns='country')

df_gdp_capita.columns
df_gdp_capita_anos = df_gdp_capita.drop(columns='country')
df_gdp_capita_anos.columns

d = {'anos': df_gdp_capita_anos.columns, 'pib_per_capita': df_argentina.iloc[0].tolist()}
argentina = pd.DataFrame(data=d)
argentina.pib_per_capita.values

argentina.head(20)

argentina.dtypes

def convert_to_float(value):
    if 'k' in value:
        return float(value[:-1]) * 1000
    else:
        return float(value)

# Aplicar a função ao array usando list comprehension
argentina_pib_per_capita = np.array([convert_to_float(val) for val in argentina.pib_per_capita.values])

argentina_pib_per_capita.dtype

argentina['anos'] = argentina['anos'].astype(float)

# Funções de gráficos de comparação, gráficos de previsão

def mostrar_grafico(X, Y):

    fig = go.Figure(
        go.Scatter(x=X, y=Y)
    )

    fig.update_layout(height=600, width=800, title='Gráfico PIB per Capita', xaxis_title='Anos', yaxis_title='PIB per capita')
    fig.show()

    #fig.write_image(f"grafico-pib-percapita.png")

def grafico_comparar_markers(valores_reais, valores_previstos, anos):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anos, y=valores_reais, mode='markers', name='Valores Reais de Teste'))
    fig.add_trace(go.Scatter(x=anos, y=valores_previstos, mode='markers', name='Valores previstos pelo machine learning'))

    fig.update_layout(height=600, width=800, title='Gráfico de comparação',
                      xaxis_title='Ano', yaxis_title='Valores')
    fig.show()

    #fig.write_image("grafico-comparacao-markers.png")

def grafico_previsao_linha(valores_reais, valores_previstos, anos_x, anos_y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anos_x, y=valores_reais, mode='lines', name='Valores Originais'))
    fig.add_trace(go.Scatter(x=anos_y, y=valores_previstos, mode='markers', name='Valor Previsto', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=[anos_x[-1], anos_y[0]], y=[valores_reais[-1], valores_previstos[0]], mode='lines', showlegend=False, line=dict(color='green')))

    fig.update_layout(height=600, width=800, title='Gráfico de Previsão',
                      xaxis_title='Ano', yaxis_title='Valores')
    fig.show()

    #fig.write_image(f"grafico-previsao-linha.png")


def grafico_comparacao_linha(valores_reais, valores_previstos, anos_x, anos_y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anos_x, y=valores_reais, mode='lines', name='Valores Reais'))
    fig.add_trace(go.Scatter(x=anos_y, y=valores_previstos, mode='lines', name='Valores previstos pelo machine learning'))

    fig.update_layout(height=600, width=800, title='Gráfico de Comparação',
                      xaxis_title='Ano', yaxis_title='Valores')
    fig.show()

    #fig.write_image(f"grafico-comparacao-linha.png")

# Gráfico de PIB per Capita da Argentina.

mostrar_grafico(df_gdp_capita_anos.columns, argentina_pib_per_capita)


# Testando qual o melhor modelo para essa especifíca situação.


#Treinando o modelo

x=argentina['anos'].values.reshape(-1, 1)
y=argentina_pib_per_capita

SEED = 42
np.random.randint(SEED)

linear_model = LinearRegression(fit_intercept=True)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.20, random_state=SEED)

linear_model.fit(treino_x, treino_y)
previsoes = linear_model.predict(teste_x)

# Calcula o coeficiente de determinação R2
r2 = r2_score(teste_y, previsoes)

print('\n====================================================================\n')
print('Linear Regression Argentina')
print("R-squared: %.2f" % r2)

# Calcula o erro médio quadrático (MSE)
mse = mean_squared_error(teste_y, previsoes)
print("Mean Squared Error (MSE): %.2f" % mse)
print('\n====================================================================\n')

# Grau do polinômio
grau_polinomio = 2

x=argentina['anos'].values.reshape(-1, 1)
y=argentina_pib_per_capita

poly = PolynomialFeatures(degree=grau_polinomio)
x = poly.fit_transform(x)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.20, random_state=SEED)

modelo_polinomial = LinearRegression()
modelo_polinomial.fit(treino_x, treino_y)

previsoes_polinomiais = modelo_polinomial.predict(teste_x)

r2_polinomial = r2_score(teste_y, previsoes_polinomiais)
mse_polinomial = mean_squared_error(teste_y, previsoes_polinomiais)

print('\n====================================================================\n')
print('Regressão Polinomial Argentina')
print("Regressão Polinomial")
print("R-squared: %.2f" % r2_polinomial)
print("Mean Squared Error (MSE): %.2f" % mse_polinomial)
print('\n====================================================================\n')

x=argentina['anos'].values.reshape(-1, 1)
y=argentina_pib_per_capita

modelo_svm = SVR(kernel='linear')

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.20, random_state=SEED)

modelo_svm.fit(treino_x, treino_y)

previsoes_svm = modelo_svm.predict(teste_x)

r2_svm = r2_score(teste_y, previsoes_svm)
mse_svm = mean_squared_error(teste_y, previsoes_svm)

print('\n====================================================================\n')
print('SVM Argentina')
print("Máquinas de Vetores de Suporte (SVM)")
print("R-squared: %.2f" % r2_svm)
print("Mean Squared Error (MSE): %.2f" % mse_svm)
print('\n====================================================================\n')

x=argentina['anos'].values.reshape(-1, 1)
y=argentina_pib_per_capita

modelo_arvore = DecisionTreeRegressor(random_state=SEED, max_depth=50)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.10, random_state=SEED)

modelo_arvore.fit(treino_x, treino_y)

previsoes_arvore = modelo_arvore.predict(teste_x)

r2_arvore = r2_score(teste_y, previsoes_arvore)
mse_arvore = mean_squared_error(teste_y, previsoes_arvore)


print('\n====================================================================\n')
print('Árvore de Decisão Argentina')
print("Árvores de Decisão")
print("R-squared: %.2f" % r2_arvore)
print("Mean Squared Error (MSE): %.2f" % mse_arvore)
print('\n====================================================================\n')

x=argentina['anos'].values.reshape(-1, 1)
y=argentina_pib_per_capita

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.20, random_state=SEED)

teste_x_rf = teste_x

# Criação do modelo de Random Forest
modelo_rf = RandomForestRegressor(max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100, random_state=SEED)

modelo_rf.fit(treino_x, treino_y)

previsoes_rf = modelo_rf.predict(teste_x)

r2_rf = r2_score(teste_y, previsoes_rf)
mse_rf = mean_squared_error(teste_y, previsoes_rf)

# Aqui só estou garantindo que estou realmente pegando as variáveis corretas.

teste_y_rf = teste_y

print('\n====================================================================\n')
print("Random Forest Regressor Argentina")
print("R-squared: %.2f" % r2_rf)
print("Mean Squared Error (MSE): %.2f" % mse_rf)
print('\n====================================================================\n')

#Melhor modelo acaba sendo o RandomForestRegressor

previsoes_rf

teste_y_rf

teste_x_rf.sort()

# Gráfico de comparação de Valores Reais x Valores Previstos pelo Machine Learning do PIB per capita da Argentina.

grafico_comparar_markers(anos=teste_x_rf.ravel(), valores_reais=teste_y_rf, valores_previstos=previsoes_rf)


valores_prever_tudo = np.array(df_gdp_capita_anos.loc[:, '1960':].columns)
valores_prever_tudo = valores_prever_tudo.reshape(-1, 1)

# Aqui faço uma comparação de toda a linha do tempo, para visualizar isso de uma maneira geral

# Gráfico de Comparação de toda a linha do Tempo

# Modelo Decision Tree Regressor

grafico_comparacao_linha(anos_x=df_gdp_capita_anos.loc[:, '1960':].columns, anos_y=df_gdp_capita_anos.loc[:, '1960':].columns,
                 valores_reais=argentina_pib_per_capita, valores_previstos=modelo_arvore.predict(valores_prever_tudo))


# Filtrando para a região do Brasil

brasil = df_gdp_capita.loc[df_gdp_capita['country'] == 'Brazil']
brasil = brasil.drop(columns='country')
brasil.isnull().sum()

d = {'anos': df_gdp_capita_anos.columns, 'pib_per_capita': brasil.iloc[0].tolist()}
brasil = pd.DataFrame(data=d)
brasil.pib_per_capita.values
brasil_pib_per_capita = np.array([convert_to_float(val) for val in brasil.pib_per_capita.values])

# Gráfico de PIB per Capita do Brasil

mostrar_grafico(df_gdp_capita_anos.columns, brasil_pib_per_capita)


# Treinando o modelo para a região do Brasil

modelo_rf = RandomForestRegressor(random_state=SEED)

x=brasil['anos'].values.reshape(-1, 1)
y=brasil_pib_per_capita

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.30, random_state=SEED)

teste_x_rf = teste_x

modelo_rf.fit(treino_x, treino_y)

previsoes_rf = modelo_rf.predict(teste_x)

r2_rf = r2_score(teste_y, previsoes_rf)
mse_rf = mean_squared_error(teste_y, previsoes_rf)

# Aqui só estou garantindo que estou realmente pegando as variáveis corretas.

teste_y_rf = teste_y

print('\n====================================================================\n')
print("Random Forest Regressor Brasil")
print("R-squared: %.2f" % r2_rf)
print("Mean Squared Error (MSE): %.2f" % mse_rf)
print('\n====================================================================\n')

x=brasil['anos'].values.reshape(-1, 1)
y=brasil_pib_per_capita

modelo_arvore = DecisionTreeRegressor(random_state=SEED, max_depth=50)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.30, random_state=SEED)

modelo_arvore.fit(treino_x, treino_y)

previsoes_arvore = modelo_arvore.predict(teste_x)

r2_arvore = r2_score(teste_y, previsoes_arvore)
mse_arvore = mean_squared_error(teste_y, previsoes_arvore)

print('\n====================================================================\n')
print("Árvores de Decisão Brasil")
print("R-squared: %.2f" % r2_arvore)
print("Mean Squared Error (MSE): %.2f" % mse_arvore)
print('\n====================================================================\n')

# Gráfico de comparação de Valores Reais x Valores Previstos pelo Machine Learning do PIB do Brasil.

grafico_comparacao_linha(anos_x=df_gdp_capita_anos.loc[:, '1960':].columns, anos_y=df_gdp_capita_anos.loc[:, '1960':].columns,
                 valores_reais=brasil_pib_per_capita, valores_previstos=modelo_arvore.predict(valores_prever_tudo))


#Modelo Random Forest Regressor

grafico_comparacao_linha(anos_x=df_gdp_capita_anos.loc[:, '1960':].columns, anos_y=df_gdp_capita_anos.loc[:, '1960':].columns,
                 valores_reais=brasil_pib_per_capita, valores_previstos=modelo_rf.predict(valores_prever_tudo))


valores_prever = np.array([2022.]).reshape(-1, 1)

# Gráfico de previsão do PIB per capita do Brasil para o ano de 2022

grafico_previsao_linha(anos_x=df_gdp_capita_anos.loc[:, '2005':].columns, anos_y=valores_prever.ravel(),
                 valores_reais=brasil_pib_per_capita[45:], valores_previstos=modelo_arvore.predict(valores_prever))
