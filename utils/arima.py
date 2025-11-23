import time
import numpy as np
import warnings
import os
import pandas as pd

warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA

# ========================================================================================
# IMPORTAÇÕES DO PROJETO
# ========================================================================================
import utils.utils as util
import utils.plotUtils as plot
from utils.logger import Logger


class ArimaRunner:
    """
    Runner para modelo ARIMA univariado (chuva) com busca de hiperparâmetros via AIC.

    Usa:
      - timeseries: DataFrame original (desescalado) com índice de datas
      - scaler: scaler usado no pré-processamento (para desescalar previsão)
      - ts_scaled: DataFrame escalado (com coluna 'chuva')
      - n_test: tamanho do conjunto de teste
      - titulo: sufixo para logs e gráficos
    """

    def __init__(self, timeseries, scaler, ts_scaled, ts_scaled_df, n_test, lookback, index, titulo):
        self.timeseries = timeseries
        self.scaler = scaler
        self.ts_scaled = ts_scaled
        self.n_test = n_test
        self.lookback = lookback
        self.index = index
        self.titulo = titulo
        self.ts_scaled_df = ts_scaled_df

        # Logger da classe
        self.logger = Logger.configurar_logger(
            nome_arquivo=f"arima{titulo}.log",
            nome_classe=f"ARIMA_{titulo}"
        )

        # Será preenchido depois
        self.best_order = None
        self.best_aic = None
        self.best_model = None

    # ---------------------------------------------------------------------
    # MÉTODOS AUXILIARES
    # ---------------------------------------------------------------------

    def _train_arima(self, endog, order):
        """
        Treina um ARIMA simples (sem exógenas) com a ordem passada.
        """
        model = ARIMA(
            endog=endog,
            order=order,
        )
        res = model.fit()
        return res

    def _grid_search_aic(self, endog_train, orders):
        """
        Faz busca manual por menor AIC numa grade de (p,d,q).

        Retorna:
          best_res: resultado do melhor ajuste
          best_order: tupla (p,d,q)
          best_aic: valor de AIC
        """
        best_res = None
        best_order = None
        best_aic = np.inf

        for order in orders:
            res = self._train_arima(endog_train, order)
            aic = res.aic
            if aic < best_aic:
                best_aic = aic
                best_res = res
                best_order = order

        return best_res, best_order, best_aic

    # ---------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # ---------------------------------------------------------------------

    def run(self):
        """
        Executa todo o pipeline:
          - split treino/teste
          - grid search em (p,d,q) via AIC
          - refit no treino
          - previsão no teste
          - desescalonamento e cálculo de métricas
          - geração de gráfico

        Retorna:
          best_model, y_pred_mm, testY_mm
        """
        t0_total = time.time()

        self.logger.info("=" * 90)
        self.logger.info(f"Iniciando script ARIMA/ARIMAX {self.titulo} com 1 feature (chuva).")
        self.logger.info("=" * 90)

        # Série endógena (chuva escalonada)
        endog = self.ts_scaled_df['chuva'].astype('float64')

        # Split treino/teste
        endog_train, endog_test = endog.iloc[:-self.n_test], endog.iloc[-self.n_test:]

        self.logger.info(f"Tamanho treino: {len(endog_train)} | teste: {len(endog_test)}")

        # ================================================================
        # FASE 4 - BUSCA MANUAL DE HIPERPARÂMETROS (SEM AUTO_ARIMA)
        # ================================================================
        t3 = time.time()
        self.logger.info("[FASE 4] Iniciando busca manual por hiperparâmetros (AIC).")

        # Aqui é onde você controla o "tamanho da janela" via p
        p_values = [self.lookback] 
        d_values = [0, 1]
        q_values = [0, 1, 2]

        orders = [(p, d, q) for p in p_values for d in d_values for q in q_values]

        best_res, best_order, best_aic = self._grid_search_aic(endog_train, orders)

        if best_res is None:
            self.logger.info("Nenhum modelo pôde ser ajustado com a grade fornecida.")
            raise RuntimeError("Falha na busca de hiperparâmetros.")

        self.best_order = best_order
        self.best_aic = best_aic

        self.logger.info(f"Melhor order encontrado: {best_order}")
        self.logger.info(f"Melhor AIC (treino): {best_aic:.2f}")
        self.logger.info(f"Tempo da Fase 4: {time.time() - t3:.2f}s")

        # ================================================================
        # FASE 5 - REFIT NO CONJUNTO COMPLETO DE TREINO
        # ================================================================
        t4 = time.time()
        self.logger.info("[FASE 5] Reajustando melhor modelo no conjunto de treino.")

        self.best_model = ARIMA(
            endog=endog_train,
            order=self.best_order,
        ).fit()

        self.logger.info("Melhor modelo ajustado no treino.")
        self.logger.info(f"Tempo da Fase 5: {time.time() - t4:.2f}s")

        # ================================================================
        # FASE 6 - PREVISÃO E AVALIAÇÃO NO TESTE
        # ================================================================
        t5 = time.time()
        self.logger.info("[FASE 6] Gerando previsões no conjunto de teste e avaliando métricas.")
        # ----------------------------
        # PREVISÃO DENTRO DO TREINO
        # ----------------------------
        # Pode usar fittedvalues (in-sample) ou predict com o índice do treino
        y_pred_train = self.best_model.predict(
            start=endog_train.index[0],
            end=endog_train.index[-1],
        )
        # ou simplesmente:
        # y_pred_train = self.best_model.fittedvalues

        # desescalar previsão de treino
        y_pred_train_mm, trainY_mm = util.desescalar_pred_generico(
            y_pred_train,
            scaler=self.scaler,
            ts_scaled=self.ts_scaled_df,
            timeseries=self.timeseries,
            target='chuva',
            start=0,  # treino começa no início da série
            index=endog_train.index
        )

        rmse_tr, mse_tr, mae_tr, csi_tr = util.calcular_erros(
            logger=self.logger,
            dadoPrevisao=y_pred_train_mm,
            dadoReal=trainY_mm
        )

        self.logger.info(f"[TREINO] RMSE={rmse_tr:.4f} | MSE={mse_tr:.4f} | MAE={mae_tr:.4f} | CSI={csi_tr:.4f}")


        # Previsões no período de teste
        y_pred = self.best_model.predict(
            start=endog_test.index[0],
            end=endog_test.index[-1],
        )

        # para desescalar corretamente
        train_size = len(self.timeseries) - len(y_pred)

        y_pred_mm, testY_mm = util.desescalar_pred_generico(
            y_pred,
            scaler=self.scaler,
            ts_scaled=self.ts_scaled_df,
            timeseries=self.timeseries,
            target='chuva',
            start=train_size,
            index=endog_test.index
        )

        rmse, mse , mae, csi = util.calcular_erros(logger=self.logger, dadoPrevisao=y_pred_mm, dadoReal=testY_mm)

        self.logger.info(f"Tempo total da Fase 6: {time.time() - t5:.2f}s")

        # ================================================================
        # FASE 7 - VISUALIZAÇÃO
        # ================================================================
        self.logger.info("[FASE 7] Salvando gráfico de previsão vs. observado.")
        plot.gerar_plot_dois_eixo(
            eixo_x=trainY_mm,
            eixo_y=y_pred_train_mm,
            titulo=f"TRAIN [{self.index}] - arima{self.titulo}_result",
            xlabel="Amostra",
            ylabel="Chuva",
            legenda=['Real', 'Previsto']
        )
        plot.gerar_plot_dois_eixo(
            eixo_x=testY_mm,
            eixo_y=y_pred_mm,
            titulo=f"TEST [{self.index}] - arima{self.titulo}_result",
            xlabel="Amostra",
            ylabel="Chuva",
            legenda=['Real', 'Previsto']
        )
        self.logger.info(f"Gráfico salvo como 'arima{self.titulo}_result.png'.")

        # ================================================================
        # FINALIZAÇÃO
        # ================================================================
        self.logger.info("=" * 90)
        self.logger.info(f"Execução ARIMA {self.titulo} finalizada com sucesso.")
        self.logger.info(f"Tempo total de execução: {time.time() - t0_total:.2f}s")
        self.logger.info("=" * 90)
        t_total = time.time() - t0_total
        return {
            "lookback": 30,
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "csi": csi,
            "tempoTreinamento":t_total,
            }


def rodarARIMA(timeseries, scaler, ts_scaled, ts_scaled_df, n_test, lookback , index, titulo):
    runner = ArimaRunner(
        timeseries=timeseries,
        scaler=scaler,
        ts_scaled=ts_scaled,
        ts_scaled_df=ts_scaled_df,
        n_test=n_test,
        lookback=lookback,
        index = index,
        titulo=titulo,
    )
    resultados = []
    resultados.append(runner.run())
    df_resultados = pd.DataFrame(resultados)
    caminho = f"pictures/resultados_arima_{titulo}.csv"
    df_resultados.to_csv(caminho, 
                            mode="a",                     
                            header=not os.path.exists(caminho),
                            index=False)
