# churn analyser

O projeto **churn analyser** tem como objetivo a análise e predição de portabilidades no setor de telecomunicações (churn), por meio de diferentes abordagens de regressão e modelagem estatística. Utiliza um conjunto de dados contendo métricas de satisfação e históricos de portabilidade, aplicando diversos algoritmos com e sem defasagens temporais.

---

## Experimentos

A aplicação executa automaticamente os seguintes experimentos a partir de uma base de dados central:

### 1. **Modelos de Regressão com Lag de 1 mês**

- Aplica regressão linear, árvore de decisão, random forest, XGBoost e ridge.
- Utiliza variáveis de satisfação com defasagem (lag=1).
- Codifica variáveis categóricas via OneHotEncoder.

### 2. **Modelos de Regressão sem Lag**

- Utiliza os mesmos modelos acima.
- Utiliza os valores brutos de satisfação (sem defasagem).
- Permite comparar impacto do lag no desempenho preditivo.

### 3. **Modelo SARIMAX por Estado e Operadora**

- Aplica modelo estatístico SARIMAX por grupo de operadora e UF.
- Utiliza o índice ISG como variável exógena.
- Gera gráficos individuais e métricas (R², MAE, RMSE).

### 4. **SARIMAX com Variação de Lag e Ordens**

- Aplica múltiplas configurações de `order` e `lag` para SARIMAX.
- Avalia impacto da defasagem da variável ISG na performance do modelo.
- Resultados são salvos em CSV e gráficos por grupo.

### 5. **Modelos Regressivos com Remoção de Outliers**

- Aplica regressão por operadora e estado com remoção de outliers via Isolation Forest.
- Compara modelos (RandomForest, DecisionTree, LinearRegression, XGBoost).
- ISG é utilizado como única variável preditiva.

### 6. **Modelos Regressivos com Otimização de Hiperparâmetros**

- Utiliza todas as variáveis de satisfação (exceto ISG).
- Aplica GridSearchCV para RandomForest e XGBoost.
- Também remove outliers antes do treino.
- Avaliação é feita com conjunto de teste separado (train_test_split).

---

## Preparação do Dataset

Para gerar o arquivo `.csv` utilizado pelos experimentos, utilize o notebook:

[`preparacao_dataset.ipynb`](./preparacao_dataset.ipynb)

Esse notebook realiza a limpeza, transformação e exportação dos dados necessários para a execução dos modelos.

---

## Dependências

- Python 3.8+
- pandas
- matplotlib
- scikit-learn
- xgboost
- statsmodels
- seaborn

Você pode instalar todas com:

`pip install -r requirements.txt`
