import time
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import warnings
warnings.filterwarnings("ignore")

# ========================================================================================
# IMPORTAÇÕES DO PROJETO
# ========================================================================================
import utils.utils as util
from utils.logger import Logger
import utils.plotUtils as plot


class RandomForestRunner:
    """
    Runner para Random Forest com janela temporal (univariada) e GridSearchCV.

    Parâmetros:
      - timeseries: DataFrame com pelo menos a coluna 'chuva'
      - n_test: quantidade de amostras de teste (em janelas)
      - titulo: sufixo para logs e figuras
      - window_size: tamanho da janela temporal (número de lags)
    """

    def __init__(self, timeseries, n_test, titulo, window_size=30):
        self.timeseries = timeseries
        self.n_test = n_test
        self.titulo = titulo
        self.window_size = window_size

        # Logger da classe
        self.logger = Logger.configurar_logger(
            nome_arquivo=f"random_forest_{titulo}.log",
            nome_classe=f"RANDOM_FOREST_{titulo}"
        )

        # Guardar melhor modelo
        self.best_model = None
        self.best_params_ = None
        self.best_score_ = None

    # ---------------------------------------------------------------------
    # MÉTODOS AUXILIARES
    # ---------------------------------------------------------------------

    def _criar_janelas_univariadas(self, series, window_size):
        """
        Cria janelas [t-W, ..., t-1] -> alvo em t, usando só a série 'series'.

        Retorna:
            X_window: [n_amostras, window_size]
            y_window: [n_amostras]
        """
        vals = series.values  # [T]
        X_list, y_list = [], []

        for i in range(len(vals) - window_size):
            X_list.append(vals[i: i + window_size])   # janela de W dias
            y_list.append(vals[i + window_size])      # valor no dia seguinte

        return np.array(X_list), np.array(y_list)

    def _configurar_param_grid(self):
        """
        Define o grid de hiperparâmetros para o RandomForest.
        """
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [None, 30, 60],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5],
        }
        return param_grid

    # ---------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # ---------------------------------------------------------------------

    def run(self):
        """
        Executa todo o pipeline:
          - cria janelas temporais univariadas usando 'chuva'
          - split treino/teste (mantendo ordem)
          - GridSearchCV com TimeSeriesSplit
          - treino final com best_estimator_
          - previsão no teste
          - cálculo de métricas
          - gráfico de real vs previsto

        Retorna:
          best_model, y_pred_mm, testY_mm
        """
        t0_total = time.time()
        self.logger.info("=" * 100)
        self.logger.info(f"Iniciando Random Forest {self.titulo} com GridSearchCV.")
        self.logger.info("=" * 100)

        # ================================================================
        # FASE 3 — SPLIT E PREPARAÇÃO
        # ================================================================
        inicio = time.time()
        self.logger.info("[FASE 3] Split treino/teste e preparação.")

        WINDOW_SIZE = self.window_size

        # Série alvo
        y = self.timeseries['chuva'].astype(float)

        # Cria janelas univariadas a partir da série completa
        X_window, y_window = self._criar_janelas_univariadas(self.timeseries['chuva'], WINDOW_SIZE)
        # (n_samples, window_size)
        X_window = X_window.squeeze(-1) if X_window.ndim == 3 else X_window
        y_window = y_window.squeeze(-1) if y_window.ndim == 2 else y_window

        # Ajusta n_test para não passar do tamanho da série de janelas
        n_test = self.n_test
        if n_test is None:
            n_test = 30  # default
        n_test = min(n_test, len(y_window) // 2)

        X_train_s, X_test_s = util.split_last_n(X_window, n_test=n_test)
        y_train_s, y_test_s = util.split_last_n(y_window, n_test=n_test)

        # datas para gráficos (últimos n_test pontos)
        dates = self.timeseries.index
        test_dates = dates[-n_test:]

        self.logger.info(
            f"[FASE 3] Treino: X={X_train_s.shape}, y={len(y_train_s)} | "
            f"Teste: X={X_test_s.shape}, y={len(y_test_s)}"
        )
        self.logger.info(f"[FASE 3] Tempo: {time.time() - inicio:.2f}s")

        # ================================================================
        # FASE 4 — GRIDSEARCHCV (TimeSeriesSplit)
        # ================================================================
        inicio = time.time()
        self.logger.info("[FASE 4] Iniciando GridSearchCV (TimeSeriesSplit).")

        param_grid = self._configurar_param_grid()
        tscv = TimeSeriesSplit(n_splits=2)
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)

        grid = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=2
        )

        t_grid = time.time()
        grid.fit(X_train_s, y_train_s)
        grid_time = time.time() - t_grid

        self.best_model = grid.best_estimator_
        self.best_params_ = grid.best_params_
        self.best_score_ = grid.best_score_

        self.logger.info(f"[FASE 4] GridSearchCV concluído em {grid_time:.2f}s")
        self.logger.info(f"[FASE 4] Melhores parâmetros: {grid.best_params_}")
        self.logger.info(f"[FASE 4] Melhor score (neg_MSE): {grid.best_score_:.6f}")
        self.logger.info(f"[FASE 4] Tempo: {time.time() - inicio:.2f}s")

        # ================================================================
        # FASE 5 — TREINO FINAL (refit)
        # ================================================================
        inicio = time.time()
        self.logger.info("[FASE 5] Treinando best_estimator_ no conjunto de treino.")
        self.best_model.fit(X_train_s, y_train_s)
        self.logger.info(f"[FASE 5] Tempo: {time.time() - inicio:.2f}s")

        # ================================================================
        # FASE 6 — PREVISÃO E MÉTRICAS
        # ================================================================
        inicio = time.time()
        self.logger.info("[FASE 6] Previsão e cálculo de métricas.")

        pred = self.best_model.predict(X_test_s).reshape(-1, 1)

        print('y_pred raw min/max:', float(pred.min()), float(pred.max()))
        print('y_TRUE raw min/max:', float(y_test_s.min()), float(y_test_s.max()))

        y_pred_mm = pred
        testY_mm = y_test_s

        print('y_pred mm min/max:', float(y_pred_mm.min()), float(y_pred_mm.max()))
        print('y_TRUE mm min/max:', float(testY_mm.min()), float(testY_mm.max()))

        rmse, mse , mae, csi = util.calcular_erros(logger=self.logger, dadoPrevisao=y_pred_mm, dadoReal=testY_mm)
        tempoFinal = time.time() - inicio
        # ================================================================
        # FASE 7 — GRÁFICOS
        # ================================================================
        inicio = time.time()
        self.logger.info("[FASE 7] Gerando gráficos.")

        plot.gerar_plot_dois_eixo(
            eixo_x=testY_mm,
            eixo_y=y_pred_mm,
            titulo=f"lstmRandomForest_{self.titulo}_result",
            xlabel="Amostra",
            ylabel="Chuva",
            legenda=['Real', 'Previsto']
        )

        self.logger.info(f"Gráfico salvo como 'random_forest_{self.titulo}_result.png'.")
        self.logger.info(f"[FASE 7] Tempo: {time.time() - inicio:.2f}s")

        # ================================================================
        # RESUMO FINAL
        # ================================================================
        self.logger.info("=" * 100)
        self.logger.info("Resumo Final")
        self.logger.info(f"Melhores parâmetros: {self.best_params_}")
        self.logger.info(f"Tempo total de execução: {time.time() - t0_total:.2f}s")
        self.logger.info("=" * 100)

        return {
            "lookback": WINDOW_SIZE,
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "csi": csi,
            "tempoTreinamento":tempoFinal,
            }



def rodarRandomForest(timeseries, n_test, titulo, window_size=30):
    runner = RandomForestRunner(
        timeseries=timeseries,
        n_test=n_test,
        titulo=titulo,
        window_size=window_size,  
    )
    return runner.run()
