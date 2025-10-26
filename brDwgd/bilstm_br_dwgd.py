import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasRegressor

# ========================================================================================
# CONFIGURAÇÃO DE AMBIENTE E GPU
# ========================================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs disponíveis: {gpus}")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Configuração de crescimento de memória da GPU ativada.")
    except Exception as e:
        print(f"Erro ao configurar GPU: {e}")
else:
    print("Nenhuma GPU disponível. Rodando no CPU.")

# ========================================================================================
# IMPORTAÇÕES DO PROJETO
# ========================================================================================
sys.path.append(os.path.abspath(".."))
from utils.logger import Logger
import access_br_dwgd as access_br_dwgd
import utils.utils as util

# ========================================================================================
# CONFIGURAÇÃO DO LOGGER
# ========================================================================================
logger = Logger.configurar_logger(
    nome_arquivo="biLstmBrDwgd.log",
    nome_classe="BiLstm BR DWGD"
)
logger.info("=" * 90)
logger.info("Iniciando script BiLSTM BR DWGD (TensorFlow/Keras).")
logger.info("=" * 90)

# ========================================================================================
# CALLBACK CUSTOMIZADO PARA LOGAR TREINAMENTO
# ========================================================================================
class LoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = (
            f"Época {epoch + 1:04d} - "
            f"Loss: {logs.get('loss', -1):.4f} - MAE: {logs.get('mae', -1):.4f} - "
            f"Val_Loss: {logs.get('val_loss', -1):.4f} - Val_MAE: {logs.get('val_mae', -1):.4f}"
        )
        logger.info(msg)

# ========================================================================================
# FASE 1 - CARREGAMENTO DOS DADOS
# ========================================================================================
inicio = time.time()
logger.info("[FASE 1] Carregando e pré-processando dados.")
timeseries = access_br_dwgd.recuperar_dados_br_dwgd_com_area()
logger.info(f"Dados carregados com sucesso: {len(timeseries)} registros.")
logger.info(f"Período: {timeseries.index.min()} até {timeseries.index.max()}")

# ========================================================================================
# FASE 2 - ENGENHARIA DE FEATURES
# ========================================================================================
inicio2 = time.time()
logger.info("[FASE 2] Criando features temporais e estatísticas.")

timeseries = util.criar_data_frame_chuva(timeseries)

logger.info(f"Engenharia de features concluída. Total de colunas: {timeseries.shape[1]}")
logger.info(f"Colunas criadas: {list(timeseries.columns)}")
logger.info(f"Tempo total da Fase 2: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 3 - NORMALIZAÇÃO E PREPARAÇÃO DE DADOS
# ========================================================================================
inicio3 = time.time()
logger.info("[FASE 3] Normalizando dados e criando sequências.")

features_dinamicas = [c for c in timeseries.columns if ('chuva' in c) or ('mediana' in c) or ('iqr_7' in c)]
scaler_chuva = MinMaxScaler()
timeseries[features_dinamicas] = scaler_chuva.fit_transform(timeseries[features_dinamicas])

lookback = 240
X, y = util.create_sequence(timeseries, lookback)
dates_aligned = timeseries.index[lookback:]

train_size = int(len(X) * 0.70)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
train_date, test_date = dates_aligned[:train_size], dates_aligned[train_size:]

logger.info(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
logger.info(f"Treino: {y_train.shape}, Teste: {y_test.shape}")

logger.info(f"Tempo total da Fase 3: {time.time() - inicio3:.2f}s")

# ========================================================================================
# FASE 4 - GRIDSEARCHCV (OTIMIZAÇÃO DE HIPERPARÂMETROS)
# ========================================================================================
#inicio4 = time.time()
#logger.info("[FASE 4] Iniciando GridSearchCV para otimização de hiperparâmetros.")

#param_grid = {
#    'model__units_camada1': [32, 64],
#    'model__units_camada2': [16, 32],
#    'model__dropout_rate': [0.2, 0.3],
#    'model__activation': ['relu', 'tanh'],
#    'model__optimizer': ['adam', 'rmsprop'],
#    'batch_size': [32, 64],
#    'epochs': [50]
#}

#model_wrapper = KerasRegressor(
#    model=util.criar_modelo_avancado,
#    verbose=0,
#    n_features=timeseries.shape[1],
#    lookback=lookback
#)

#tscv = TimeSeriesSplit(n_splits=3)
#grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2)

#start_grid = time.time()
#grid_result = grid.fit(X_train, y_train)
#end_grid = time.time()

#logger.info(f"GridSearchCV finalizado em {end_grid - start_grid:.2f}s")
#logger.info(f"Melhores parâmetros encontrados: {grid_result.best_params_}")
#logger.info(f"Melhor score de validação cruzada (negativo da MSE): {grid_result.best_score_:.4f}")
#best_model = grid_result.best_estimator_

# ========================================================================================
# FASE 5 - TREINAMENTO FINAL DO MELHOR MODELO
# ========================================================================================
inicio5 = time.time()
logger.info("[FASE 5] Treinando modelo BiLSTM com os melhores hiperparâmetros encontrados.")

epochs = 1000
batch_size = 240

logging_callback = LoggingCallback()
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=25, min_lr=1e-7, verbose=1)

model = util.criar_modelo_avancado(
    lookback=lookback,
    n_features=timeseries.shape[1],
    num_neuronios_1_camada=128,
    num_neuronios_2_camada=64,
    dropout_rate=0.2
)
model.summary(print_fn=logger.info)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr, logging_callback]
)

logger.info(f"Treinamento final concluído em {time.time() - inicio5:.2f}s")

# ========================================================================================
# FASE 6 - AVALIAÇÃO FINAL COM O OS DADOS P TREINAR
# ========================================================================================
inicio6 = time.time()
logger.info("[FASE 6] Avaliando modelo BiLSTM no conjunto de teste.")
pred_scaled = model.predict(X_train, batch_size=batch_size, verbose=0).reshape(-1)
y_test_scaled = y_train.reshape(-1)

idx_chuva = features_dinamicas.index('chuva')
ncols = len(features_dinamicas)

dummy_pred = np.zeros((len(pred_scaled), ncols))
dummy_true = np.zeros((len(y_test_scaled), ncols))
dummy_pred[:, idx_chuva] = pred_scaled
dummy_true[:, idx_chuva] = y_test_scaled

pred = scaler_chuva.inverse_transform(dummy_pred)[:, idx_chuva]
y_true = scaler_chuva.inverse_transform(dummy_true)[:, idx_chuva]
mse  = mean_squared_error(y_true, pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, pred)
r2   = r2_score(y_true, pred)
logger.info(f"RMSE: {rmse:.4f}")
logger.info(f"MSE : {mse:.4f}")
logger.info(f"MAE : {mae:.4f}")
logger.info(f"R2 : {r2:.4f}")
logger.info(f"Tempo total da Fase 6: {time.time() - inicio6:.2f}s")

# ========================================================================================
# FASE 6 - AVALIAÇÃO FINAL TESTE
# ========================================================================================
inicio6 = time.time()
logger.info("[FASE 6] Avaliando modelo BiLSTM no conjunto de teste.")
pred_scaled = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)
y_test_scaled = y_test.reshape(-1)

idx_chuva = features_dinamicas.index('chuva')
ncols = len(features_dinamicas)

dummy_pred = np.zeros((len(pred_scaled), ncols))
dummy_true = np.zeros((len(y_test_scaled), ncols))
dummy_pred[:, idx_chuva] = pred_scaled
dummy_true[:, idx_chuva] = y_test_scaled

pred = scaler_chuva.inverse_transform(dummy_pred)[:, idx_chuva]
y_true = scaler_chuva.inverse_transform(dummy_true)[:, idx_chuva]
mse  = mean_squared_error(y_true, pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, pred)
r2   = r2_score(y_true, pred)
logger.info(f"RMSE: {rmse:.4f}")
logger.info(f"MSE : {mse:.4f}")
logger.info(f"MAE : {mae:.4f}")
logger.info(f"R2 : {r2:.4f}")
logger.info(f"Tempo total da Fase 6: {time.time() - inicio6:.2f}s")

# ========================================================================================
# FASE 7 - VISUALIZAÇÃO FINAL TESTE
# ========================================================================================
logger.info("[FASE 7] Gerando gráfico de previsão final.")
plt.figure(figsize=(15, 7))
plt.plot(test_date, y_true, label="Real")
plt.plot(test_date , pred, label="Previsto", alpha=0.7)
plt.legend()
plt.title("Previsão de Chuva com BiLSTM (TensorFlow/Keras)")
plt.xlabel("Data")
plt.ylabel("Chuva (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("previsao_final_bilstmBrDwgd.png")
plt.close()
logger.info("Gráfico salvo como 'previsao_final_bilstmBrDwgd.png'.")

# ========================================================================================
# FASE 7 - VISUALIZAÇÃO FINAL TRAIN
# ========================================================================================
logger.info("[FASE 7] Gerando gráfico de previsão final.")
plt.figure(figsize=(15, 7))
plt.plot(train_date, y_true, label="Real")
plt.plot(train_date , pred, label="Previsto", alpha=0.7)
plt.legend()
plt.title("Previsão de Chuva com BiLSTM (TensorFlow/Keras)")
plt.xlabel("Data")
plt.ylabel("Chuva (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("previsao_final_bilstmBrDwgd.png")
plt.close()
logger.info("Gráfico salvo como 'previsao_final_bilstmBrDwgd.png'.")


# ========================================================================================
# FINALIZAÇÃO
# ========================================================================================
logger.info("=" * 90)
logger.info("Execução BiLSTM BR DWGD finalizada com sucesso.")
logger.info("=" * 90)
