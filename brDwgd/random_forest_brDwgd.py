import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# ========================================================================================
# IMPORTAÇÕES DO PROJETO
# ========================================================================================
sys.path.append(os.path.abspath(".."))
import utils.utils as util
import utils.utilDataset as utilDataset
from utils.logger import Logger
import access_br_dwgd as access_br_dwgd
import utils.plotUtils as plot

# ========================================================================================
# LOGGER
# ========================================================================================
logger = Logger.configurar_logger(
    nome_arquivo="random_forest_brDwgd.log",
    nome_classe="RANDOM_FOREST_BR_DWGD"
)

logger.info("=" * 100)
logger.info("Iniciando Random Forest BR_DWGD com GridSearchCV.")
logger.info("=" * 100)

# ========================================================================================
# FASE 1 — CARREGAMENTO
# ========================================================================================
t0_total = time.time()
inicio = time.time()
logger.info("[FASE 1] Carregando dados de access_br_dwgd.recuperar_dados_br_dwgd_com_area()")
timeseries = access_br_dwgd.recuperar_dados_br_dwgd_com_area()
logger.info(f"[FASE 1] Registros: {len(timeseries)} | Período: {timeseries.index.min()} a {timeseries.index.max()}")
logger.info(f"[FASE 1] Tempo: {time.time() - inicio:.2f}s")

# ========================================================================================
# FASE 2 — FEATURES 
# ========================================================================================
inicio2 = time.time()
logger.info("[FASE 2] Criando features temporais e estatísticas...")

#timeseries, colunas_normalizar = utilDataset.criar_data_frame_chuva(df=timeseries, tmax_col='Tmax', tmin_col='Tmin', W=30,wet_thr=1.0)

logger.info(f"Engenharia de features concluída. Total de colunas: {timeseries.shape[1]}")
logger.info(f"Colunas criadas: {list(timeseries.columns)}")
logger.info(f"Tempo total da Fase 2: {time.time() - inicio:.2f} segundos.")


# ========================================================================================
# FASE 3 — SPLIT E PREPARAÇÃO
# ========================================================================================
inicio = time.time()
logger.info("[FASE 3] Split treino/teste e preparação.")


WINDOW_SIZE = 30  # por exemplo, igual ao W que vc já usa

def criar_janelas_multivariadas(X_df, y_series, window_size):
    """
    Transforma uma série multivariada em um problema supervisionado:
    - X_window: [n_amostras, window_size * n_features]
    - y_window: [n_amostras], alvo = chuva no tempo (t + 1) depois da janela
    """
    X_vals = X_df.values          # [T, n_features]
    y_vals = y_series.values      # [T]
    
    n_total = len(X_df)
    n_feats = X_df.shape[1]
    n_samples = n_total - window_size

    X_window = np.zeros((n_samples, window_size * n_feats), dtype=float)
    y_window = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        # janela de [i ... i+window_size-1]
        janela = X_vals[i : i + window_size]          # [window_size, n_feats]
        X_window[i, :] = janela.reshape(-1)           # flatten -> [window_size * n_feats]
        # alvo = chuva no dia imediatamente após a janela
        y_window[i] = y_vals[i + window_size]         # t = i+window_size

    return X_window, y_window

def criar_janelas_univariadas(series, window_size):
    """
    Cria janelas [t-W, ..., t-1] -> alvo em t, usando só a série 'series'.
    Retorna:
        X_window: [n_amostras, window_size]
        y_window: [n_amostras]
    """
    vals = series.values  # [T]
    X_list, y_list = [], []

    for i in range(len(vals) - window_size):
        X_list.append(vals[i : i + window_size])   # janela de W dias
        y_list.append(vals[i + window_size])       # valor no dia seguinte

    return np.array(X_list), np.array(y_list)

# aplica a transformação
X_window, y_window = criar_janelas_univariadas(timeseries, WINDOW_SIZE)
X_window = X_window.squeeze(-1)
y_window = y_window.squeeze(-1) 
n_test = 30
X_train_s, X_test_s = util.split_last_n(X_window, n_test=n_test)
y_train_s, y_test_s = util.split_last_n(y_window, n_test=n_test)

# datas para gráficos
dates = timeseries.index
test_dates = dates[-n_test:]

logger.info(f"[FASE 3] Treino: X={X_train_s.shape}, y={len(y_train_s)} | Teste: X={X_test_s.shape}, y={len(y_test_s)}")
logger.info(f"[FASE 3] Tempo: {time.time() - inicio:.2f}s")

# ========================================================================================
# FASE 4 — GRIDSEARCHCV (TimeSeriesSplit)
# ========================================================================================
inicio = time.time()
logger.info("[FASE 4] Iniciando GridSearchCV (TimeSeriesSplit).")

param_grid= {
    "n_estimators": [200, 400],
    "max_depth": [None, 30, 60],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.5],
}

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

# rf = RandomForestRegressor(
#     n_estimators=400,
#     max_depth=None,
#     max_features="sqrt",   # evita árvores muito caras
#     min_samples_split=2,
#     min_samples_leaf=1,
#     bootstrap=True,
#     max_samples=0.9,       # subsampling acelera e reduz overfit
#     n_jobs=-1,
#     random_state=42
# )

t_grid = time.time()
grid.fit(X_train_s, y_train_s)
grid_time = time.time() - t_grid

logger.info(f"[FASE 4] GridSearchCV concluído em {grid_time:.2f}s")
logger.info(f"[FASE 4] Melhores parâmetros: {grid.best_params_}")
logger.info(f"[FASE 4] Melhor score (neg_MSE): {grid.best_score_:.6f}")
best_model = grid.best_estimator_
#best_model =rf
logger.info(f"[FASE 4] Tempo: {time.time() - inicio:.2f}s")

# ========================================================================================
# FASE 5 — TREINO FINAL (refit)
# ========================================================================================
inicio = time.time()
logger.info("[FASE 5] Treinando best_estimator_ no conjunto de treino.")
best_model.fit(X_train_s, y_train_s)
logger.info(f"[FASE 5] Tempo: {time.time() - inicio:.2f}s")

# ========================================================================================
# FASE 6 — PREVISÃO E MÉTRICAS (invertendo escala e log1p)
# ========================================================================================
inicio = time.time()
logger.info("[FASE 6] Previsão e cálculo de métricas.")

pred = best_model.predict(X_test_s).reshape(-1, 1)

print('y_pred raw min/max:', float(pred.min()), float(pred.max()))
print('y_TRUE raw min/max:', float(y_test_s.min()), float(y_test_s.max()))

y_pred_mm = pred
testY_mm = y_test_s

print('y_pred mm min/max:', float(y_pred_mm.min()), float(y_pred_mm.max()))
print('y_TRUE mm min/max:', float(testY_mm.min()), float(testY_mm.max()))

util.calcular_erros(logger=logger,
                     dadoReal=testY_mm,
                     dadoPrevisao=y_pred_mm
                    )

# ========================================================================================
# FASE 7 — GRÁFICOS (previsão e importâncias)
# ========================================================================================
inicio = time.time()
logger.info("[FASE 7] Gerando gráficos.")

plot.gerar_plot_dois_eixo(eixo_x=testY_mm, eixo_y=y_pred_mm, titulo="lstmRandomForest_br_dwgd_result", xlabel="Amostra", ylabel="Chuva", legenda=['Real', 'Previsto'])

logger.info("Gráfico salvo como 'random_forest_brDwgd_result.png'.")

# Importância das Features
# feat_names = X.columns.tolist()
# importances = best_model.feature_importances_
# imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)

# plt.figure(figsize=(12, 8))
# plt.barh(imp_df["feature"], imp_df["importance"])
# plt.gca().invert_yaxis()
# plt.title("Importância das Features - Random Forest")
# plt.tight_layout()
# plt.savefig("pictures/random_forest_brDwgd_feature_importance.png")
# plt.close()
# logger.info("Gráfico salvo como 'random_forest_brDwgd_feature_importance.png'.")
logger.info(f"[FASE 7] Tempo: {time.time() - inicio:.2f}s")

# ========================================================================================
# FASE 8 — SALVAR MODELO E MÉTRICAS
# ========================================================================================
#inicio = time.time()
#logger.info("[FASE 8] Salvando modelo e métricas.")
#os.makedirs("models", exist_ok=True)
#os.makedirs("reports", exist_ok=True)

#model_path = os.path.join("models", "random_forest_brDwgd_best.pkl")
#with open(model_path, "wb") as f:
#    pickle.dump(best_model, f)
#logger.info(f"Modelo salvo em '{model_path}'.")

#metrics_path = os.path.join("reports", "random_forest_brDwgd_metrics.csv")
#pd.DataFrame([{"rmse": rmse, "mse": mse, "mae": mae}]).to_csv(metrics_path, index=False)
#logger.info(f"Métricas salvas em '{metrics_path}'.")
#logger.info(f"[FASE 8] Tempo: {time.time() - inicio:.2f}s")

# ========================================================================================
# RESUMO FINAL
# ========================================================================================
logger.info("=" * 100)
logger.info("Resumo Final")
logger.info(f"Melhores parâmetros: {grid.best_params_}")
logger.info(f"Tempo total de execução: {time.time() - t0_total:.2f}s")
logger.info("=" * 100)