import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys, os, tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import access_br_dwgd as access_br_dwgd

# ========================================================================================
# CONFIGURAÇÃO DE LOGGER
# ========================================================================================
sys.path.append(os.path.abspath(".."))
from utils.logger import Logger
import utils.utils as util
logger = Logger.configurar_logger(nome_arquivo="lstmBrDwgd.log", nome_classe="LSTM_BR_DWGD_KERAS")

logger.info("=" * 80)
logger.info("Iniciando execução do script LSTM com suporte a GPU e logs detalhados.")
logger.info("=" * 80)

# ========================================================================================
# VERIFICAÇÃO DE GPU
# ========================================================================================
gpu_disponiveis = tf.config.list_physical_devices('GPU')
if gpu_disponiveis:
    try:
        tf.config.experimental.set_memory_growth(gpu_disponiveis[0], True)
        logger.info("Configuração de crescimento de memória da GPU ativada com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao configurar memória da GPU: {e}")
else:
    logger.warning("Nenhuma GPU detectada. O treinamento ocorrerá na CPU.")

# ========================================================================================
# FASE 1 - CARREGAMENTO DOS DADOS
# ========================================================================================
inicio = time.time()
logger.info("[FASE 1] Iniciando carregamento e pré-processamento dos dados...")
timeseries = access_br_dwgd.recuperar_dados_br_dwgd_com_area()
logger.info(f"Dados carregados com sucesso. Total de {len(timeseries)} registros recebidos.")
logger.info(f"Primeiras linhas:\n{timeseries.head()}")
logger.info(f"Colunas iniciais: {list(timeseries.columns)}")
logger.info(f"Índice temporal: {timeseries.index.min()} -> {timeseries.index.max()}")
logger.info(f"Valores ausentes: {timeseries.isna().sum().sum()}")
timeseries['chuva'] = np.log1p(timeseries['chuva'])
logger.info("Transformação log1p aplicada na variável 'chuva'.")
logger.info(f"Tempo total da Fase 1: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 2 - ENGENHARIA DE FEATURES
# ========================================================================================
inicio = time.time()
logger.info("[FASE 2] Iniciando engenharia de features...")

# Criação de features temporais
timeseries['dia_seno'] = np.sin(2 * np.pi * timeseries.index.dayofyear / 365)
timeseries['dia_cosseno'] = np.cos(2 * np.pi * timeseries.index.dayofyear / 365)
timeseries['mes_seno'] = np.sin(2 * np.pi * timeseries.index.month / 12)
timeseries['mes_cosseno'] = np.cos(2 * np.pi * timeseries.index.month / 12)
timeseries['ano'] = timeseries.index.year - timeseries.index.year.min()

# Médias móveis
for w in [3, 7, 14, 30]:
    timeseries[f'chuva_ma{w}'] = timeseries['chuva'].shift(1).rolling(window=w, min_periods=1).mean().fillna(0)

# Estatísticas móveis
timeseries['chuva_std7'] = timeseries['chuva'].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
timeseries['chuva_max7'] = timeseries['chuva'].shift(1).rolling(window=7, min_periods=1).max().fillna(0)
timeseries['chuva_min7'] = timeseries['chuva'].shift(1).rolling(window=7, min_periods=1).min().fillna(0)

# Lags
for lag in [1, 3, 7]:
    timeseries[f'chuva_lag{lag}'] = timeseries['chuva'].shift(lag).fillna(0)

# Flags
timeseries['choveu_ontem'] = (timeseries['chuva_lag1'] > 0).astype(int)
timeseries['choveu_semana'] = (timeseries['chuva_ma7'] > 0).astype(int)

logger.info(f"Engenharia de features concluída. Total de colunas: {timeseries.shape[1]}")
logger.info(f"Colunas criadas: {list(timeseries.columns)}")
logger.info(f"Tempo total da Fase 2: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 3 - NORMALIZAÇÃO E PREPARAÇÃO DOS DADOS
# ========================================================================================
inicio = time.time()
logger.info("[FASE 3] Normalizando e criando sequências de treino/teste...")

features_dinamicas = [col for col in timeseries.columns if 'chuva' in col]
scaler_chuva = MinMaxScaler()
timeseries_scaled = timeseries.copy()
timeseries_scaled[features_dinamicas] = scaler_chuva.fit_transform(timeseries[features_dinamicas])
logger.info(f"Normalização aplicada às colunas: {features_dinamicas}")

lookback = 60
X, y = util.create_sequence(timeseries_scaled.values, lookback)
train_size = int(len(X) * 0.75)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

logger.info(f"Sequências criadas - Lookback: {lookback}")
logger.info(f"Tamanho treino: {len(X_train)} | teste: {len(X_test)}")
logger.info(f"Shape entrada: {X_train.shape} | Saída: {y_train.shape}")
logger.info(f"Tempo total da Fase 3: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 4 - CRIAÇÃO E TREINAMENTO DO MODELO
# ========================================================================================
inicio = time.time()
logger.info("[FASE 4] Criando e treinando o modelo LSTM...")

num_features = timeseries.shape[1]
model = util.build_deep_lstm()
model.compile(optimizer='adam', loss='mean_squared_error')

n_epochs = 1000
batch_size = 32
logger.info(f"Parâmetros de treinamento -> Épocas: {n_epochs}, Batch: {batch_size}, Features: {num_features}")
logger.info(f"Resumo do modelo:\n{model.summary(print_fn=lambda x: logger.info(x))}")

start_train = time.time()
hist = model.fit(X_train, y_train,
                 validation_data=(X_test, y_test),
                 epochs=n_epochs,
                 batch_size=batch_size,
                 verbose=2)
logger.info(f"Treinamento concluído em {(time.time() - start_train) / 60:.2f} minutos.")
logger.info(f"Tempo total da Fase 4: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 5 - AVALIAÇÃO E MÉTRICAS
# ========================================================================================
inicio = time.time()
logger.info("[FASE 5] Avaliando modelo e gerando previsões...")

pred = model.predict(X_test)
logger.info(f"Previsões geradas: {pred.shape}")

# Desnormalizar
n_features_chuva = scaler_chuva.n_features_in_
pred_dummy = np.zeros((len(pred), n_features_chuva))
pred_dummy[:, 0] = pred.flatten()
pred_log = scaler_chuva.inverse_transform(pred_dummy)[:, 0]
pred_final = np.expm1(pred_log)

y_test_dummy = np.zeros((len(y_test), n_features_chuva))
y_test_dummy[:, 0] = y_test.flatten()
y_test_log = scaler_chuva.inverse_transform(y_test_dummy)[:, 0]
y_test_final = np.expm1(y_test_log)


rmse = np.sqrt(mean_squared_error(y_test_final, pred_final))
mae = mean_absolute_error(y_test_final, pred_final)
mse = mean_squared_error(y_test_final, pred_final)

logger.info(f"Métricas de avaliação:")
logger.info(f"RMSE: {rmse:.4f}")
logger.info(f"MSE:  {mse:.4f}")
logger.info(f"MAE:  {mae:.4f}")
logger.info(f"Tempo total da Fase 5: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 6 - VISUALIZAÇÃO FINAL
# ========================================================================================
inicio = time.time()
logger.info("[FASE 6] Gerando gráfico de previsão final...")

plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Real (Normalizado)")
plt.plot(pred, label="Previsto (Normalizado)")
plt.legend()
plt.title("Previsão de Chuva - LSTM com GPU")
plt.xlabel("Amostra")
plt.ylabel("Chuva (Normalizada)")
plt.tight_layout()
plt.savefig("pictures/lstm_gpu_br_dwgd_result.png")
plt.close()
logger.info("Gráfico salvo como 'lstm_gpu_br_dwgd_result.png'.")
logger.info(f"Tempo total da Fase 6: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FINALIZAÇÃO
# ========================================================================================
logger.info("=" * 80)
logger.info("Execução finalizada com sucesso.")
logger.info(f"Ambiente de execução: {'GPU' if gpu_disponiveis else 'CPU'}")
logger.info("=" * 80)
