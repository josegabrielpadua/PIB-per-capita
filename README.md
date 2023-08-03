# Análise do PIB per Capita para Argentina e Brasil

### Introdução

Este projeto tem como objetivo analisar os dados do Produto Interno Bruto (PIB) per capita da Argentina e do Brasil usando vários modelos de Machine Learning. 
O conjunto de dados utilizado contém valores históricos do PIB per capita para ambos os países de 1960 a 2021.

### Bibliotecas e Funções Utilizadas

O projeto utiliza as seguintes bibliotecas e funções:

* Pandas: Para manipulação e análise de dados
* NumPy: Para operações numéricas
* Matplotlib e Plotly: Para visualização de dados
* scikit-learn: Para modelos de Machine Learning

### Carregamento e Pré-processamento dos Dados

1. Os dados são carregados a partir do arquivo CSV fornecido, que contém o PIB per capita para vários países, incluindo Argentina e Brasil.
2. Os dados são filtrados para obter apenas os valores do PIB per capita da Argentina e do Brasil.
3. Valores nulos, se houver, são verificados no conjunto de dados da Argentina e removidos, se encontrados.
4. Os dados são convertidos para um formato adequado para análise. O símbolo "k" (milhares) é removido dos valores do PIB, e os dados são convertidos para o tipo float.

### Visualização dos Dados

* O projeto visualiza o PIB per capita da Argentina ao longo dos anos usando um gráfico interativo do Plotly.

### Treinamento e Avaliação dos Modelos

1. O projeto treina e avalia diferentes modelos de Machine Learning para prever o PIB per capita da Argentina e do Brasil.
2. Os modelos utilizados são Linear Regression, Polinomial Regression, Support Vector Machines (SVM), Decision Tree e Random Forest.
3. Para cada modelo, os dados são divididos em conjuntos de treinamento e teste, e o modelo é treinado nos dados de treinamento e avaliado nos dados de teste.
4. As métricas de avaliação utilizadas são R-squared e Mean Squared Error (MSE).

### Seleção do Modelo

1. O Random Forest Regressor é selecionado como o melhor modelo tanto para a Argentina quanto para o Brasil, com base em seu alto valor de R-squared e baixo MSE.
2. O projeto fornece visualizações comparando os valores reais do PIB per capita com os valores previstos para ambos os países usando o melhor modelo.

### Previsões Futuras

1. O projeto também faz uma previsão futura para o PIB per capita do Brasil para o ano de 2022, utilizando o melhor modelo.

Este projeto demonstra a aplicação de vários modelos de Machine Learning para analisar e prever o PIB per capita da Argentina e do Brasil.
O Random Forest Regressor foi considerado o melhor modelo para ambos os países, fornecendo previsões precisas.
A visualização dos valores reais e previstos do PIB ajuda a entender o desempenho dos modelos ao longo do tempo.

### Autor: José Gabriel dos Santos Pádua
