import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys, os, tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
#timeseries['chuva'] = np.log1p(timeseries['chuva'])
logger.info("Transformação log1p aplicada na variável 'chuva'.")
logger.info(f"Tempo total da Fase 1: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 2 - ENGENHARIA DE FEATURES
# ========================================================================================
inicio = time.time()
logger.info("[FASE 2] Iniciando engenharia de features...")

#timeseries = util.criar_data_frame_chuva(timeseries)

logger.info(f"Engenharia de features concluída. Total de colunas: {timeseries.shape}")
logger.info(f"Colunas criadas: {list(timeseries.columns)}")
logger.info(f"Tempo total da Fase 2: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 3 - NORMALIZAÇÃO E PREPARAÇÃO DOS DADOS
# ========================================================================================
inicio = time.time()
logger.info("[FASE 3] Normalizando e criando sequências de treino/teste...")

features_dinamicas = [c for c in timeseries.columns if ('chuva' in c) or ('mediana' in c) or ('iqr_7' in c) or ('Tmax' in c) or ('Tmin' in c)]
scaler_chuva = MinMaxScaler(feature_range=(0, 1))
timeseries[features_dinamicas] = scaler_chuva.fit_transform(timeseries[features_dinamicas])
logger.info(f"Normalização aplicada às colunas: {features_dinamicas}")

train_size = int(len(timeseries) * 0.7)
test_size = len(timeseries) - train_size
train, test = timeseries.iloc[:train_size], timeseries.iloc[train_size:]

lookback = 60
trainX, trainY = util.create_sequence(train.values, lookback)
testX, testY = util.create_sequence(test.values, lookback)



logger.info(f"Sequências criadas - Lookback: {lookback}")
logger.info(f"Tamanho treino: {len(trainX)} | teste: {len(testX)}")
logger.info(f"Shape entrada: {trainX.shape} | Saída: {testY.shape}")
logger.info(f"Tempo total da Fase 3: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 4 - CRIAÇÃO E TREINAMENTO DO MODELO
# ========================================================================================
inicio = time.time()
logger.info("[FASE 4] Criando e treinando o modelo LSTM...")

num_features = timeseries.shape[1]
model = util.build_deep_lstm(lookback=lookback, num_features=num_features)

n_epochs = 10
batch_size = 64
logger.info(f"Parâmetros de treinamento -> Épocas: {n_epochs}, Batch: {batch_size}, Features: {num_features}")
logger.info(f"Resumo do modelo:\n{model.summary(print_fn=lambda x: logger.info(x))}")

start_train = time.time()
# hist = model.fit(X_train, y_train,
#                  validation_data=(X_test, y_test),
#                  epochs=n_epochs,
#                  batch_size=batch_size,
#                  verbose=2)
es = EarlyStopping(monitor='val_loss', patience=30, mode='min',
                   restore_best_weights=True, verbose=1)

rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                        patience=10, min_lr=1e-5, verbose=1)

hist = model.fit(
    trainX, trainY,
    epochs=n_epochs,
    batch_size=batch_size,
    shuffle=False,          
    callbacks=[es, rlr],
    verbose=1
)
logger.info(f"Treinamento concluído em {(time.time() - start_train) / 60:.2f} minutos.")
logger.info(f"Tempo total da Fase 4: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 5 - AVALIAÇÃO E MÉTRICAS
# ========================================================================================
inicio6 = time.time()
logger.info("[FASE 5] Avaliando modelo LSTMKeras no conjunto de teste.")
pred_scaled = model.predict(testX).reshape(-1)
y_test_scaled = testY

alvo = "chuva"  # ou o nome exato da coluna alvo
idx_alvo = features_dinamicas.index(alvo)

pred_scaled_1d = pred_scaled.ravel()
y_test_scaled_1d = np.array(y_test_scaled).ravel()

pred_chuva = (pred_scaled_1d - scaler_chuva.min_[idx_alvo]) / scaler_chuva.scale_[idx_alvo]
y_test_chuva = (y_test_scaled_1d - scaler_chuva.min_[idx_alvo]) / scaler_chuva.scale_[idx_alvo]


mse  = mean_squared_error(y_test_chuva, pred_chuva)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_chuva, pred_chuva)
r2   = r2_score(y_test_chuva, pred_chuva)
logger.info(f"RMSE: {rmse:.4f}")
logger.info(f"MSE : {mse:.4f}")
logger.info(f"MAE : {mae:.4f}")
logger.info(f"R2 : {r2:.4f}")
logger.info(f"Tempo total da Fase 5: {time.time() - inicio6:.2f}s")

# ========================================================================================
# FASE 6 - AVALIAÇÃO FINAL COM O OS DADOS P TREINAR
# ========================================================================================
inicio6 = time.time()
logger.info("[FASE 6] Avaliando modelo LSTMKeras no conjunto TRAIN.")
pred_train= model.predict(trainX, batch_size=batch_size, verbose=0)
y_test_train= trainY

pred_scaled_train_1d = pred_train.ravel()
y_test_scaled_train_1d = np.array(y_test_train).ravel()

pred_chuva_train = (pred_scaled_train_1d - scaler_chuva.min_[idx_alvo]) / scaler_chuva.scale_[idx_alvo]
y_test_chuva_train = (y_test_scaled_train_1d - scaler_chuva.min_[idx_alvo]) / scaler_chuva.scale_[idx_alvo]

mse  = mean_squared_error(y_test_chuva_train, pred_chuva_train)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_chuva_train, pred_chuva_train)
r2   = r2_score(y_test_chuva_train,pred_chuva_train)

logger.info(f"RMSE: {rmse:.4f}")
logger.info(f"MSE : {mse:.4f}")
logger.info(f"MAE : {mae:.4f}")
logger.info(f"R2 : {r2:.4f}")
logger.info(f"Tempo total da Fase 6: {time.time() - inicio6:.2f}s")

# ========================================================================================
# FASE 7 - VISUALIZAÇÃO FINAL
# ========================================================================================
inicio = time.time()
logger.info("[FASE 7] Gerando gráfico de previsão final...")

plt.figure(figsize=(12, 6))
plt.plot(y_test_chuva_train)
plt.plot(pred_chuva_train, label="Previsto (Normalizado)")
plt.legend()
plt.title("Previsão de Chuva - LSTM com GPU")
plt.xlabel("Amostra")
plt.ylabel("Chuva")
plt.tight_layout()
plt.savefig("lstm_gpu_br_dwgd_result.png")
plt.close()
logger.info("Gráfico salvo como 'lstm_gpu_br_dwgd_result.png'.")

# logger.info("[FASE 7] Gerando gráfico de previsão final...")

plt.figure(figsize=(12, 6))
plt.plot(y_test_chuva, label="Real (Normalizado)")
plt.plot(pred_chuva, label="Previsto (Normalizado)")
plt.legend()
plt.title("Previsão de Chuva - LSTM com GPU")
plt.xlabel("Amostra")
plt.ylabel("Chuva (Normalizada)")
plt.tight_layout()
plt.savefig("lstm_gpu_br_dwgd_resultTeste.png")
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
