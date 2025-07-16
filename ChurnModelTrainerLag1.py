import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


"""
Classe responsável por treinar e avaliar modelos de regressão para predição de churn (portabilidade),
utilizando dados históricos com variáveis de satisfação defasadas (lag = 1). Executa o pipeline 
de pré-processamento, codificação de variáveis categóricas, particionamento dos dados, treino e
avaliação com métricas de desempenho (R², MAE, RMSE).

Modelos avaliados:
- Regressão Linear
- Regressão Ridge
- Árvore de Decisão
- Random Forest
- XGBoost Regressor

"""


class ChurnModelTrainerLag1:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.satisfacao_cols = ["ISG", "QIC", "QF", "QCR", "QAT", "QAD", "QAP"]
        self.modelos = {
            "Regressão Linear": LinearRegression(),
            "Árvore de Decisão": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0),
            "Regressão Ridge": Ridge()
        }
        self.resultados = []

    def _preprocessar(self):
        # Conversão e ordenação
        self.df["AM_EFETIVACAO"] = pd.to_datetime(self.df["AM_EFETIVACAO"])
        self.df = self.df.sort_values(by=["NO_PRESTADORA_DOADORA", "SG_UF", "AM_EFETIVACAO"])

        # Lags
        for col in self.satisfacao_cols:
            self.df[f"{col}_lag1"] = self.df.groupby(["NO_PRESTADORA_DOADORA", "SG_UF"])[col].shift(1)

        # Remove nulos
        self.df = self.df.dropna(subset=[f"{col}_lag1" for col in self.satisfacao_cols])

        # Data como número
        self.df["AM_EFETIVACAO"] = self.df["AM_EFETIVACAO"].dt.strftime("%Y%m").astype(int)

    def _dividir_dados(self):
        X = self.df[
            ["AM_EFETIVACAO", "SG_UF", "NO_PRESTADORA_DOADORA"] +
            [f"{col}_lag1" for col in self.satisfacao_cols]
        ]
        y = self.df["QT_PORTABILIDADE_EFETIVADA"]

        categorical_features = ["SG_UF", "NO_PRESTADORA_DOADORA"]
        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
            remainder="passthrough"
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, preprocessor

    def _treinar_e_avaliar(self, X_train, X_test, y_train, y_test, preprocessor):
        for nome, modelo in self.modelos.items():
            print(f"Treinando modelo: {nome}")
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", modelo)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            self.resultados.append({
                "Modelo": nome,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse
            })

    def run(self):
        self._preprocessar()
        X_train, X_test, y_train, y_test, preprocessor = self._dividir_dados()
        self._treinar_e_avaliar(X_train, X_test, y_train, y_test, preprocessor)

        df_resultados = pd.DataFrame(self.resultados)
        print("\nResultados Finais com Lag de 1 Mês:")
        print(df_resultados.sort_values(by="R2", ascending=False))
        return df_resultados
