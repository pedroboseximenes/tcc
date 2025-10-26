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
from utils.logger import Logger
import access_br_dwgd as access_br_dwgd

# ========================================================================================
# LOGGER
# ========================================================================================
logger = Logger.configurar_logger(
    nome_arquivo="random_forest_brDwgd.log",
    nome_classe="RANDOM_FOREST_BR_DWGD"
)

logger.info("=" * 100)
logger.info("Iniciando Random Forest BR_DWGD com 18 features e GridSearchCV.")
logger.info("=" * 100)

# ========================================================================================
# FASE 1 — CARREGAMENTO
# ========================================================================================
t0_total = time.time()
inicio = time.time()
logger.info("[FASE 1] Carregando dados de access_br_dwgd.recuperar_dados_br_dwgd_com_area()")
series = access_br_dwgd.recuperar_dados_br_dwgd_com_area()
logger.info(f"[FASE 1] Registros: {len(series)} | Período: {series.index.min()} a {series.index.max()}")
logger.info(f"[FASE 1] Tempo: {time.time() - inicio:.2f}s")

# ========================================================================================
# FASE 2 — FEATURES (18)
# ========================================================================================
inicio = time.time()
logger.info("[FASE 2] Construindo 18 features.")
timeseries = util.criar_data_frame_chuva(series)
logger.info(f"[FASE 2] Colunas: {list(timeseries.columns)}")
logger.info(f"[FASE 2] Shape: {timeseries.shape}")
logger.info(f"[FASE 2] Tempo: {time.time() - inicio:.2f}s")

# ========================================================================================
# FASE 3 — SPLIT E PREPARAÇÃO
# ========================================================================================
inicio = time.time()
logger.info("[FASE 3] Split treino/teste e preparação.")

# Target = chuva (log1p). Inputimeseries = demais 17 colunas
y = timeseries['chuva'].astype(float)
X = timeseries.drop(columns=['chuva']).astype(float)

# Split temporal 75/25 (ajuste se quiser)
split_idx = int(len(timeseries) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# datas para gráficos
dates = timeseries.index
test_dates = dates[split_idx:]

# Normalização apenas das colunas 'chuva*' para inverter ao fim (mesma técnica dos seus scriptimeseries)
chuva_cols = [c for c in timeseries.columns if 'chuva' in c]
scaler_chuva = MinMaxScaler()
timeseries_scaled = timeseries.copy()
timeseries_scaled[chuva_cols] = scaler_chuva.fit_transform(timeseries[chuva_cols])

# Reconstituir conjuntos escalados
y_train_s = timeseries_scaled['chuva'].iloc[:split_idx].values
y_test_s  = timeseries_scaled['chuva'].iloc[split_idx:].values
X_train_s = timeseries_scaled.drop(columns=['chuva']).iloc[:split_idx].values
X_test_s  = timeseries_scaled.drop(columns=['chuva']).iloc[split_idx:].values

logger.info(f"[FASE 3] Treino: X={X_train_s.shape}, y={len(y_train_s)} | Teste: X={X_test_s.shape}, y={len(y_test_s)}")
logger.info(f"[FASE 3] Tempo: {time.time() - inicio:.2f}s")

# ========================================================================================
# FASE 4 — GRIDSEARCHCV (TimeSeriesSplit)
# ========================================================================================
inicio = time.time()
logger.info("[FASE 4] Iniciando GridSearchCV (TimeSeriesSplit).")

param_grid = {
    "n_estimators": [500, 1000],
    "max_depth": [None, 20, 50, 100],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2", None]
}

tscv = TimeSeriesSplit(n_splits=3)
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

logger.info(f"[FASE 4] GridSearchCV concluído em {grid_time:.2f}s")
logger.info(f"[FASE 4] Melhores parâmetros: {grid.best_params_}")
logger.info(f"[FASE 4] Melhor score (neg_MSE): {grid.best_score_:.6f}")
best_model = grid.best_estimator_
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

y_pred_s = best_model.predict(X_test_s).reshape(-1, 1)

# Inversão do MinMaxScaler para voltar ao espaço log1p
n_features_chuva = scaler_chuva.n_features_in_
pred_dummy = np.zeros((len(y_pred_s), n_features_chuva))
pred_dummy[:, 0] = y_pred_s.flatten()

ytest_dummy = np.zeros((len(y_test_s), n_features_chuva))
ytest_dummy[:, 0] = y_test_s.flatten()

pred_log = scaler_chuva.inverse_transform(pred_dummy)[:, 0]
ytest_log = scaler_chuva.inverse_transform(ytest_dummy)[:, 0]

# Volta para escala original (mm)
y_pred = np.expm1(pred_log)
y_true = np.expm1(ytest_log)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

logger.info("--- Métricas no conjunto de teste ---")
logger.info(f"RMSE: {rmse:.4f}")
logger.info(f"MSE : {mse:.4f}")
logger.info(f"MAE : {mae:.4f}")
logger.info("-------------------------------------")
logger.info(f"[FASE 6] Tempo: {time.time() - inicio:.2f}s")

# ========================================================================================
# FASE 7 — GRÁFICOS (previsão e importâncias)
# ========================================================================================
inicio = time.time()
logger.info("[FASE 7] Gerando gráficos.")

# Previsão vs Real
plt.figure(figsize=(15, 7))
plt.plot(test_dates, y_true, label="Real")
plt.plot(test_dates, y_pred, label="Previsto", alpha=0.8)
plt.legend()
plt.title("Previsão de Chuva - Random Forest (18 features)")
plt.xlabel("Data")
plt.ylabel("Chuva (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("random_forest_brDwgd_result.png")
plt.close()
logger.info("Gráfico salvo como 'random_forest_brDwgd_result.png'.")

# Importância das Features
feat_names = X.columns.tolist()
importances = best_model.feature_importances_
imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.gca().invert_yaxis()
plt.title("Importância das Features - Random Forest")
plt.tight_layout()
plt.savefig("pictures/random_forest_brDwgd_feature_importance.png")
plt.close()
logger.info("Gráfico salvo como 'random_forest_brDwgd_feature_importance.png'.")
logger.info(f"[FASE 7] Tempo: {time.time() - inicio:.2f}s")

# ========================================================================================
# FASE 8 — SALVAR MODELO E MÉTRICAS
# ========================================================================================
inicio = time.time()
logger.info("[FASE 8] Salvando modelo e métricas.")
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

model_path = os.path.join("models", "random_forest_brDwgd_best.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
logger.info(f"Modelo salvo em '{model_path}'.")

metrics_path = os.path.join("reports", "random_forest_brDwgd_metrics.csv")
pd.DataFrame([{"rmse": rmse, "mse": mse, "mae": mae}]).to_csv(metrics_path, index=False)
logger.info(f"Métricas salvas em '{metrics_path}'.")
logger.info(f"[FASE 8] Tempo: {time.time() - inicio:.2f}s")

# ========================================================================================
# RESUMO FINAL
# ========================================================================================
logger.info("=" * 100)
logger.info("Resumo Final")
logger.info(f"Melhores parâmetros: {grid.best_params_}")
logger.info(f"RMSE (teste): {rmse:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f}")
logger.info(f"Tempo total de execução: {time.time() - t0_total:.2f}s")
logger.info("=" * 100)