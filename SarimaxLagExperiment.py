import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

class SarimaxLagExperiment:
    """
    Classe responsável por conduzir experimentos SARIMAX variando diferentes valores de lag e configurações
    de ordem (p, d, q) para cada par de operadora e estado. Avalia o impacto da defasagem da variável ISG
    na previsão da quantidade de portabilidades.

    Gera gráficos de previsão e salva métricas como R², MAE e RMSE.
    """

    def __init__(self, df: pd.DataFrame, operadora: str, estado: str, lags: list, sarimax_orders: list, pasta_saida: str):
        self.df = df.copy()
        self.operadora = operadora
        self.estado = estado
        self.lags = lags
        self.orders = sarimax_orders
        self.pasta_saida = pasta_saida
        self.metricas = []

    def preparar_dados(self, lag: int):
        """
        Aplica lag à variável ISG, normaliza os dados e separa X e y.
        """
        self.df['AM_EFETIVACAO'] = pd.to_datetime(self.df['AM_EFETIVACAO'])
        grupo = self.df[
            (self.df['NO_PRESTADORA_DOADORA'] == self.operadora) &
            (self.df['SG_UF'] == self.estado)
        ].copy()
        grupo = grupo.sort_values('AM_EFETIVACAO').set_index('AM_EFETIVACAO')
        grupo['ISG_LAG'] = grupo['ISG'].shift(lag)
        grupo = grupo.dropna()

        y = grupo['QT_PORTABILIDADE_EFETIVADA']
        X = grupo[['ISG_LAG']]

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

        return y, X_scaled, grupo

    def executar_experimentos(self):
        """
        Executa os experimentos para todos os pares (lag, ordem SARIMAX) definidos.
        Salva gráficos e CSV com métricas preditivas.
        """
        os.makedirs(self.pasta_saida, exist_ok=True)

        for lag in self.lags:
            y, X_scaled, grupo = self.preparar_dados(lag)

            for order in self.orders:
                try:
                    model = SARIMAX(endog=y, exog=X_scaled, order=order, seasonal_order=(0, 1, 1, 12))
                    results = model.fit(disp=False)
                    forecast = results.get_prediction(start=0, end=len(y)-1, exog=X_scaled)
                    forecast_mean = forecast.predicted_mean

                    r2 = r2_score(y, forecast_mean)
                    mae = mean_absolute_error(y, forecast_mean)
                    rmse = mean_squared_error(y, forecast_mean) ** 0.5

                    self.metricas.append({
                        'Operadora': self.operadora,
                        'Estado': self.estado,
                        'Lag': lag,
                        'Order': str(order),
                        'R2': r2,
                        'MAE': mae,
                        'RMSE': rmse
                    })

                    self._plot_result(grupo, y, forecast_mean, lag, order)

                except Exception as e:
                    print(f"Erro com lag {lag} e order {order}: {e}")

        metricas_df = pd.DataFrame(self.metricas)
        metricas_df.to_csv(os.path.join(self.pasta_saida, 'metricas_experimentos.csv'), index=False)
        print(f"✅ Resultados salvos em: {os.path.join(self.pasta_saida, 'metricas_experimentos.csv')}")

    def _plot_result(self, grupo, y_real, y_pred, lag, order):
        """
        Plota e salva o gráfico da predição com ISG defasado.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_real.index, y_real, label='Reais')
        plt.plot(y_pred.index, y_pred, label='Preditos', linestyle='--')
        plt.plot(grupo.index, grupo['ISG_LAG'], label=f'ISG (lag {lag})', linestyle=':', color='green')
        plt.title(f'{self.operadora}-{self.estado} | Lag={lag} | Order={order}')
        plt.xlabel('Data')
        plt.ylabel('Portabilidades')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        nome_arquivo = f'{self.operadora}_{self.estado}_lag{lag}_order{order}.png'.replace("/", "_")
        plt.savefig(os.path.join(self.pasta_saida, nome_arquivo))
        plt.close()

    @classmethod
    def executar_para_todos_os_grupos(cls, df: pd.DataFrame, lags: list, orders: list, pasta_base: str):
        """
        Executa o experimento completo para todos os grupos de operadora + estado.
        """
        for (operadora, estado), _ in df.groupby(['NO_PRESTADORA_DOADORA', 'SG_UF']):
            pasta_saida = os.path.join(pasta_base, f"{operadora}_{estado}".replace("/", "_"))
            print(f"Executando experimentos para {operadora} - {estado}")
            experimento = cls(df, operadora, estado, lags, orders, pasta_saida)
            experimento.executar_experimentos()
