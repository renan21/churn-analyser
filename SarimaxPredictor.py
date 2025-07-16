import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class SarimaxPredictor:
    """
    Aplica o modelo SARIMAX com variável exógena (ISG) para previsão de churn por operadora e estado.
    Gera previsões, gráficos e métricas de avaliação (R², MAE, RMSE) para cada grupo analisado.
    """

    def __init__(self, grupo: pd.DataFrame, operadora: str, estado: str, pasta_saida: str):
        self.grupo = grupo.sort_values('AM_EFETIVACAO')
        self.operadora = operadora
        self.estado = estado
        self.pasta_saida = pasta_saida
        self.metricas = {}

    def preprocessar(self):
        """
        Prepara os dados temporais e normaliza a variável exógena ISG.
        Retorna False se os dados forem insuficientes ou o ISG não variar.
        """
        self.grupo['AM_EFETIVACAO'] = pd.to_datetime(self.grupo['AM_EFETIVACAO'])
        self.grupo = self.grupo.set_index('AM_EFETIVACAO')
        self.y = self.grupo['QT_PORTABILIDADE_EFETIVADA']
        self.X = self.grupo[['ISG']]

        if self.X['ISG'].std() == 0 or len(self.y) < 12:
            print(f"[{self.operadora} - {self.estado}] Dados insuficientes ou ISG constante.")
            return False

        scaler = StandardScaler()
        self.X_scaled = pd.DataFrame(scaler.fit_transform(self.X), index=self.X.index, columns=self.X.columns)
        return True

    def treinar_e_prever(self):
        """
        Treina o modelo SARIMAX e gera previsões dentro da série histórica.
        Retorna dicionário de métricas (R², MAE, RMSE) ou None em caso de falha.
        """
        try:
            model = SARIMAX(endog=self.y, exog=self.X_scaled, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
            results = model.fit(disp=False)

            forecast = results.get_prediction(start=0, end=len(self.y)-1, exog=self.X_scaled)
            forecast_mean = forecast.predicted_mean
            conf_int = forecast.conf_int()

            self.metricas = {
                'Operadora': self.operadora,
                'Estado': self.estado,
                'R2': r2_score(self.y, forecast_mean),
                'MAE': mean_absolute_error(self.y, forecast_mean),
                'RMSE': mean_squared_error(self.y, forecast_mean) ** 0.5
            }

            self.plotar_resultado(forecast_mean, conf_int)
            return self.metricas

        except Exception as e:
            print(f"[{self.operadora} - {self.estado}] Falha no modelo: {e}")
            return None

    def plotar_resultado(self, forecast_mean, conf_int):
        """
        Plota e salva o gráfico com valores reais, preditos e ISG ao longo do tempo.
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_title(f'Previsão: {self.operadora} - {self.estado}')
        ax1.set_xlabel('Data')
        ax1.set_ylabel('Portabilidades Efetivadas')
        ax1.plot(self.y.index, self.y, label='Valores Reais')
        ax1.plot(forecast_mean.index, forecast_mean, label='Valores Preditos', linestyle='--')
        ax1.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.2)

        ax2 = ax1.twinx()
        ax2.set_ylabel('ISG (Satisfação)')
        ax2.plot(self.X.index, self.X['ISG'], label='ISG (Satisfação)', linestyle=':', color='green')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.grid(True)
        plt.tight_layout()

        nome_arquivo = f"{self.operadora}_{self.estado}.png".replace("/", "_")
        caminho_completo = os.path.join(self.pasta_saida, nome_arquivo)
        os.makedirs(self.pasta_saida, exist_ok=True)
        plt.savefig(caminho_completo)
        plt.close()

    @classmethod
    def executar_previsao_por_estado_operadora(cls, df: pd.DataFrame, pasta_saida: str):
        """
        Executa o SARIMAX para cada par (operadora, estado) do DataFrame.
        Salva gráficos e uma planilha com as métricas de desempenho agregadas.
        """
        metricas_gerais = []
        for (operadora, estado), grupo in df.groupby(['NO_PRESTADORA_DOADORA', 'SG_UF']):
            print(f"Analisando {operadora} - {estado}")
            predictor = cls(grupo, operadora, estado, pasta_saida)
            if predictor.preprocessar():
                resultado = predictor.treinar_e_prever()
                if resultado:
                    metricas_gerais.append(resultado)

        metricas_df = pd.DataFrame(metricas_gerais)
        metricas_df = metricas_df.sort_values(by='R2', ascending=False)
        os.makedirs(pasta_saida, exist_ok=True)
        caminho_metricas = os.path.join(pasta_saida, 'metricas_modelo.csv')
        metricas_df.to_csv(caminho_metricas, index=False)
        print(f"\n✅ Tabela de métricas salva em: {caminho_metricas}")
