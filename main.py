from ChurnModelTrainerLag1 import ChurnModelTrainerLag1
from ChurnModelNoLagTrainer import ChurnModelNoLagTrainer
from SarimaxPredictor import SarimaxPredictor
from SarimaxLagExperiment import SarimaxLagExperiment
from ModeloPorGrupoComOutlierRemoval import ModeloPorGrupoComOutlierRemoval
from ModeloComGridSearchEOutliers import ModeloComGridSearchEOutliers
import pandas as pd

def main():
    # Caminhos dos dados e saída
    caminho_csv = "D:/Renan/Documents/Mestrado/Inteligencia artificial/Artigo/churn_analyser/data/base_churn_treinamento_final_2.csv"
    pasta_saida_sarimax = "D:/Renan/Documents/Mestrado/Inteligencia artificial/Artigo/churn_analyser/resultados_sarimax"
    pasta_saida_lag = "D:/Renan/Documents/Mestrado/Inteligencia artificial/Artigo/churn_analyser/resultados_lag_sarimax"

    # Carrega o dataset base
    df = pd.read_csv(caminho_csv)

    # Experimento 1: Modelos com LAG de 1 mês
    print("\n--- Execução com Lag de 1 Mês ---")
    trainer_lag = ChurnModelTrainerLag1(df)
    resultados_lag = trainer_lag.run()

    # Experimento 2: Modelos SEM LAG
    print("\n--- Execução Sem Lag (dados brutos) ---")
    trainer_sem_lag = ChurnModelNoLagTrainer(df)
    resultados_sem_lag = trainer_sem_lag.run()

    # Experimento 3: Modelo SARIMAX por estado e operadora
    print("\n--- Execução com SARIMAX por Estado e Operadora ---")
    sarimaxPredictor = SarimaxPredictor()
    sarimaxPredictor.executar_previsao_por_estado_operadora(df, pasta_saida_sarimax)

    # Experimento 4: SARIMAX com múltiplos lags e ordens
    print("\n--- Execução SARIMAX com Lags e Múltiplas Ordens ---")
    lags = [0, 1, 2, 3]
    orders = [(1, 1, 1), (2, 1, 1), (1, 0, 1)]
    sarimaxLagExperiment = SarimaxLagExperiment()
    sarimaxLagExperiment.executar_para_todos_os_grupos(df, lags, orders, pasta_saida_lag)

    # Experimento 5: Modelos com remoção de outliers    print("\n--- Execução com Modelos Regressivos e Outlier Removal ---")
    experimento_outliers = ModeloPorGrupoComOutlierRemoval(df)
    experimento_outliers.executar()

    # Experimento 6: Modelos com GridSearch e remoção de outliers
    print("\n--- Execução com Modelos Otimizados (GridSearch + Outlier Removal) ---")
    experimento_otimizado = ModeloComGridSearchEOutliers(df)
    experimento_otimizado.executar()

if __name__ == "__main__":
    main()
