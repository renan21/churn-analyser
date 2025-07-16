import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

class ModeloComGridSearchEOutliers:
    """
    Aplica modelos de regressão por estado e operadora usando variáveis de satisfação (exceto ISG),
    com remoção de outliers via Isolation Forest e otimização de hiperparâmetros (GridSearchCV)
    para RandomForest e XGBoost. Gera gráficos e salva métricas de desempenho.
    """

    def __init__(self, df: pd.DataFrame, pasta_resultados: str = "graficos_previsoes_melhorados"):
        self.df = df.copy()
        self.variaveis = ["QIC", "QF", "QCR", "QAT", "QAD", "QAP"]
        self.output_dir = pasta_resultados
        self.resultados = []

    def executar(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.df["AM_EFETIVACAO"] = pd.to_datetime(self.df["AM_EFETIVACAO"])
        self.df = self.df[["AM_EFETIVACAO", "SG_UF", "NO_PRESTADORA_DOADORA", "QT_PORTABILIDADE_EFETIVADA"] + self.variaveis]

        for estado in self.df["SG_UF"].unique():
            for operadora in self.df["NO_PRESTADORA_DOADORA"].unique():
                df_filtro = self.df[(self.df["SG_UF"] == estado) & (self.df["NO_PRESTADORA_DOADORA"] == operadora)].copy()
                if len(df_filtro) < 12:
                    continue

                # Remoção de outliers
                iso = IsolationForest(contamination=0.1, random_state=42)
                df_filtro["outlier"] = iso.fit_predict(df_filtro[["QT_PORTABILIDADE_EFETIVADA"] + self.variaveis])
                df_filtro = df_filtro[df_filtro["outlier"] == 1]

                X = df_filtro[self.variaveis]
                y = df_filtro["QT_PORTABILIDADE_EFETIVADA"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                modelos = {
                    "RandomForest": GridSearchCV(RandomForestRegressor(), {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 5, 10]
                    }, cv=3),

                    "XGBoost": GridSearchCV(XGBRegressor(), {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 5, 7]
                    }, cv=3),

                    "DecisionTree": DecisionTreeRegressor(),
                    "LinearRegression": LinearRegression()
                }

                for nome_modelo, modelo in modelos.items():
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = mean_squared_error(y_test, y_pred) ** 0.5

                    self.resultados.append({
                        "Estado": estado,
                        "Operadora": operadora,
                        "Modelo": nome_modelo,
                        "R2": r2,
                        "MAE": mae,
                        "RMSE": rmse
                    })

                    self._plot(df_filtro, y_test, y_pred, nome_modelo, estado, operadora)

        self._salvar_resultados()

    def _plot(self, df_filtro, y_test, y_pred, nome_modelo, estado, operadora):
        plt.figure(figsize=(10, 6))
        plt.plot(df_filtro.loc[y_test.index, "AM_EFETIVACAO"], y_test, label="Real", marker="o")
        plt.plot(df_filtro.loc[y_test.index, "AM_EFETIVACAO"], y_pred, label="Predito", marker="x")
        plt.title(f"{estado} - {operadora} - {nome_modelo}")
        plt.xlabel("Data")
        plt.ylabel("Portabilidades")
        plt.legend(loc="upper left")
        plt.tight_layout()
        nome_arquivo = f"{estado}_{operadora}_{nome_modelo}.png".replace("/", "_")
        plt.savefig(os.path.join(self.output_dir, nome_arquivo))
        plt.close()

    def _salvar_resultados(self):
        df_resultados = pd.DataFrame(self.resultados)
        df_resultados.to_csv("resultados_modelos_melhorados.csv", index=False)

        top_melhores = df_resultados.sort_values(by="R2", ascending=False).head(5)
        top_piores = df_resultados.sort_values(by="R2", ascending=True).head(5)

        top_melhores.to_csv("top_5_melhores_melhorados.csv", index=False)
        top_piores.to_csv("top_5_piores_melhorados.csv", index=False)
        print("✅ Resultados salvos: resultados_modelos_melhorados.csv")
