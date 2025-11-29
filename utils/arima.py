import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from utils.ModeloBase import ModeloBase

# ========================================================================================
# IMPORTAÇÕES DO PROJETO
# ========================================================================================
import utils.utils as util
import utils.plotUtils as plot

class Arima(ModeloBase):
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

    def run(self, index):
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
        self.logger.info(f"Iniciando script ARIMA {self.base_dados}.")
        self.logger.info("=" * 90)

        # Série endógena (chuva escalonada)
        endog = self.timeseries['chuva'].astype('float64')

        # Split treino/teste
        endog_train, endog_test = endog.iloc[:-self.num_test], endog.iloc[-self.num_test:]

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
        tempoTreinamento = time.time() - t3/60

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

        # desescalar previsão de treino
        # y_pred_train_mm, trainY_mm = util.desescalar_pred_generico(
        #     y_pred_train,
        #     self.colunas_normalizar,
        #     scaler=self.scaler,
        #     ts_scaled=self.ts_scaled_df,
        #     timeseries=self.timeseries,
        #     target='chuva',
        #     start=0,  # treino começa no início da série
        #     index=endog_train.index
        # )
        y_pred_train_mm, trainY_mm = y_pred_train, endog_train

        rmseTrain, mseTrain, maeTrain, csiTrain = util.calcular_erros(
            logger=self.logger,
            dadoPrevisao=y_pred_train_mm,
            dadoReal=trainY_mm
        )

        self.logger.info(f"[TREINO] RMSE={rmseTrain:.4f} | MSE={mseTrain:.4f} | MAE={maeTrain:.4f} | CSI={csiTrain:.4f}")


        # Previsões no período de teste
        y_pred = self.best_model.predict(
            start=endog_test.index[0],
            end=endog_test.index[-1],
        )

        # para desescalar corretamente
        # train_size = len(self.timeseries) - len(y_pred)

        # y_pred_mm, testY_mm = util.desescalar_pred_generico(
        #     y_pred,
        #     self.colunas_normalizar,
        #     scaler=self.scaler,
        #     ts_scaled=self.ts_scaled_df,
        #     timeseries=self.timeseries,
        #     target='chuva',
        #     start=train_size,
        #     index=endog_test.index
        # )
        y_pred_mm, testY_mm = y_pred, endog_test

        rmse, mse , mae, csi = util.calcular_erros(logger=self.logger, dadoPrevisao=y_pred_mm, dadoReal=testY_mm)

        self.logger.info(f"Tempo total da Fase 6: {time.time() - t5:.2f}s")

        # ================================================================
        # FASE 7 - VISUALIZAÇÃO
        # ================================================================
        self.logger.info("[FASE 7] Salvando gráfico de previsão vs. observado.")
        plot.gerar_plot_dois_eixo(
            eixo_x=trainY_mm,
            eixo_y=y_pred_train_mm,
            titulo=f"TRAIN [{index}] - arima{self.base_dados}",
            xlabel="Amostra",
            ylabel="Chuva",
            legenda=['Real', 'Previsto'],
            dataset=self.base_dados,
            index=index
        )
        plot.gerar_plot_dois_eixo(
            eixo_x=testY_mm,
            eixo_y=y_pred_mm,
            titulo=f"TEST [{index}] - arima{self.base_dados}",
            xlabel="Amostra",
            ylabel="Chuva",
            legenda=['Real', 'Previsto'],
            dataset=self.base_dados,
            index=index
        )
        self.logger.info(f"Gráfico salvo como 'arima{self.base_dados}_result.png'.")

        # ================================================================
        # FINALIZAÇÃO
        # ================================================================
        self.logger.info("=" * 90)
        self.logger.info(f"Execução ARIMA {self.base_dados} finalizada com sucesso.")
        self.logger.info(f"Tempo total de execução: {time.time() - t0_total:.2f}s")
        self.logger.info("=" * 90)
        return {
            "lookback": 30,
            "rmseTrain": rmseTrain,
            "mseTrain": mseTrain,
            "maeTrain": maeTrain,
            "csiTrain": csiTrain,
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "csi": csi,
            "tempoTreinamento":tempoTreinamento,
            "y_pred": y_pred_mm,
            }


# def rodarARIMA(timeseries,colunas_normalizar, scaler, ts_scaled_df, n_test, lookback , index, titulo):
#     runner = ArimaRunner(
#         timeseries=timeseries,
#         colunas_normalizar=colunas_normalizar,
#         scaler=scaler,
#         ts_scaled_df=ts_scaled_df,
#         n_test=n_test,
#         lookback=lookback,
#         index = index,
#         titulo=titulo,
#     )
#     nome = f'arima_{titulo}'
#     tracker = EmissionsTracker(
#         project_name=f"{nome}",
#         output_dir= f"../results/{titulo}/code_carbon/",
#         output_file=f"emissions_{titulo}.csv",
#         log_level="error"
#     )
#     tracker.start()
#     resultado = runner.run()
#     tracker.stop()
  
#     return resultado
