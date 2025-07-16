import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class ModeloPorGrupoComOutlierRemoval:
    """
    Aplica diferentes modelos de regressão (Random Forest, Árvore de Decisão, Regressão Linear, XGBoost)
    para prever a quantidade de portabilidades por operadora e estado. Remove outliers usando Isolation Forest
    e avalia desempenho por grupo. Gera gráficos e salva métricas (R², MAE, RMSE).
    """

    def __init__(self, df: pd.DataFrame, pasta_resultados: str = "graficos_previsoes"):
        self.df = df.copy()
        self.output_dir = pasta_resultados
        self.modelos = {
            "RandomForest": RandomForestRegressor(),
            "DecisionTree": DecisionTreeRegressor(),
            "LinearRegression": LinearRegression(),
            "XGBoost": XGBRegressor()
        }
        self.resultados = []

    def executar(self):
        os.makedirs(self.output_dir, exist_ok=True)

        self.df["AM_EFETIVACAO"] = pd.to_datetime(self.df["AM_EFETIVACAO"])
        self.df = self.df[["AM_EFETIVACAO", "SG_UF", "NO_PRESTADORA_DOADORA", "QT_PORTABILIDADE_EFETIVADA", "ISG"]]

        scaler = StandardScaler()
        self.df["ISG_NORM"] = scaler.fit_transform(self.df[["ISG"]])

        for estado in self.df["SG_UF"].unique():
            for operadora in self.df["NO_PRESTADORA_DOADORA"].unique():
                df_filtro = self.df[(self.df["SG_UF"] == estado) & (self.df["NO_PRESTADORA_DOADORA"] == operadora)].copy()
                if len(df_filtro) < 12:
                    continue

                iso = IsolationForest(contamination=0.1, random_state=42)
                df_filtro["outlier"] = iso.fit_predict(df_filtro[["QT_PORTABILIDADE_EFETIVADA", "ISG"]])
                df_filtro = df_filtro[df_filtro["outlier"] == 1]

                X = df_filtro[["ISG"]]
                y = df_filtro["QT_PORTABILIDADE_EFETIVADA"]

                for nome_modelo, modelo in self.modelos.items():
                    modelo.fit(X, y)
                    y_pred = modelo.predict(X)

                    r2 = r2_score(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    rmse = mean_squared_error(y, y_pred) ** 0.5

                    self.resultados.append({
                        "Estado": estado,
                        "Operadora": operadora,
                        "Modelo": nome_modelo,
                        "R2": r2,
                        "MAE": mae,
                        "RMSE": rmse
                    })

                    self._gerar_grafico(df_filtro, y, y_pred, estado, operadora, nome_modelo)

        self._salvar_resultados()

    def _gerar_grafico(self, df_filtro, y, y_pred, estado, operadora, nome_modelo):
        plt.figure(figsize=(10, 6))
        plt.plot(df_filtro["AM_EFETIVACAO"], y, label="Real", marker="o")
        plt.plot(df_filtro["AM_EFETIVACAO"], y_pred, label="Predito", marker="x")
        plt.twinx()
        plt.plot(df_filtro["AM_EFETIVACAO"], df_filtro["ISG_NORM"], label="ISG (normalizado)", color="green", linestyle="--")
        plt.title(f"{estado} - {operadora} - {nome_modelo}")
        plt.xlabel("Data")
        plt.legend(loc="upper left")
        plt.tight_layout()
        nome_arquivo = f"{estado}_{operadora}_{nome_modelo}.png".replace("/", "_")
        plt.savefig(os.path.join(self.output_dir, nome_arquivo))
        plt.close()

    def _salvar_resultados(self):
        df_resultados = pd.DataFrame(self.resultados)
        df_resultados.to_csv("resultados_modelos.csv", index=False)
        print("✅ Resultados gerais salvos em: resultados_modelos.csv")

        top_melhores = df_resultados.sort_values(by="R2", ascending=False).head(5)
        top_piores = df_resultados.sort_values(by="R2", ascending=True).head(5)

        top_melhores.to_csv("top_5_melhores.csv", index=False)
        top_piores.to_csv("top_5_piores.csv", index=False)
        print("📊 Top 5 melhores e piores salvos.")
