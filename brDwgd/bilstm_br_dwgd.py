import time
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Configuração do Ambiente e Logger ---
sys.path.append(os.path.abspath(".."))
try:
    import utils.lstm as lstm
    from utils.logger import Logger
    import access_br_dwgd as access_br_dwgd
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Certifique-se de que os arquivos necessários estão nos diretórios corretos.")
    sys.exit(1)

# Configura o logger
logger = Logger.configurar_logger(nome_arquivo="biLstmBRDWGD.log", nome_classe="BiLstm BR_DWGD")
logger.info("Iniciando script de previsão com BiLSTM (TensorFlow/Keras).")


# --- Callback Customizado para Logging por Época ---
class LoggingCallback(tf.keras.callbacks.Callback):
    """Callback para logar informações ao final de cada época."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Época {epoch+1:04d} concluída - "
        msg += f"Loss: {logs.get('loss', -1):.4f} - MAE: {logs.get('mae', -1):.4f} - "
        msg += f"Val_Loss: {logs.get('val_loss', -1):.4f} - Val_MAE: {logs.get('val_mae', -1):.4f}"
        logger.info(msg)


# --- 1. Carregamento e Pré-processamento dos Dados ---
logger.info("Iniciando carregamento e pré-processamento dos dados.")
timeseries = access_br_dwgd.acessar_dados_br_dwgd()
timeseries['chuva'] = np.log1p(timeseries['chuva'])
logger.info(f"Dados carregados com sucesso. Total de {len(timeseries)} registros.")
datas = timeseries.index

# --- 2. Normalização dos Dados ---
logger.info("Normalizando a série temporal de chuva.")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(timeseries[['chuva']])

# --- 3. Preparação das Sequências para o Modelo ---
lookback = 14
logger.info(f"Preparando sequências com um lookback de {lookback} dias.")
X, y = lstm.create_sequence(scaled_data, lookback)
dates_aligned = datas[lookback:]

train_size = int(len(X) * 0.70)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
train_date, test_date = dates_aligned[:train_size], dates_aligned[train_size:]

logger.info(f"Sequências criadas. Treino: {len(X_train)} amostras, Teste: {len(X_test)} amostras.")
logger.info(f"Shape dos dados de treino (X_train): {X_train.shape}")

# --- 4. Treinamento do Modelo BiLSTM ---
epochs = 1000
batch_size = 32

# Callbacks
logging_callback = LoggingCallback()
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=25, min_lr=1e-6, verbose=1)

logger.info("Compilando o modelo BiLSTM.")
model = lstm.criar_modelo_br_dwgd(lookback=lookback)
model.summary(print_fn=logger.info)

logger.info(f"Iniciando treinamento por até {epochs} épocas com batch_size: {batch_size}.")
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr, logging_callback]
)

training_duration = time.time() - start_time
logger.info(f"Treinamento concluído em {training_duration:.2f} segundos.")

# Log dos melhores resultados
best_epoch = early_stopping.stopped_epoch - early_stopping.patience if early_stopping.stopped_epoch > 0 else len(history.history['loss'])
best_val_loss = min(history.history['val_loss'])
logger.info(f"Melhor resultado obtido na época ~{best_epoch}: Val_Loss: {best_val_loss:.4f}")

# --- 5. Avaliação e Previsão ---
logger.info("Realizando previsões no conjunto de teste.")
pred_scaled = model.predict(X_test)

# Inversão da transformação para a escala original
pred_log = scaler.inverse_transform(pred_scaled)
predictions = np.expm1(pred_log)

y_test_log = scaler.inverse_transform(y_test)
y_test_final = np.expm1(y_test_log)

# Cálculo das métricas
mse = mean_squared_error(y_test_final, predictions)
mae = mean_absolute_error(y_test_final, predictions)
r2 = r2_score(y_test_final, predictions)

logger.info("--- Métricas de Avaliação no Conjunto de Teste ---")
logger.info(f"MSE (Erro Quadrático Médio): {mse:.4f}")
logger.info(f"MAE (Erro Absoluto Médio): {mae:.4f}")
logger.info(f"R² (Coeficiente de Determinação): {r2:.4f}")
logger.info("-------------------------------------------------")

# --- 6. Visualização Final ---
logger.info("Gerando gráfico de previsão final.")
plt.figure(figsize=(15, 7))
plt.plot(test_date, y_test_final, label='Valor Real', color='blue')
plt.plot(test_date, predictions, label='Previsão BiLSTM', color='red', alpha=0.7)
plt.title('Previsão de Chuva com BiLSTM vs. Dados Reais')
plt.xlabel('Data')
plt.ylabel('Chuva (mm)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('previsao_final_bilstm_brdwgd.png')
plt.close()

logger.info("Script BiLSTM (TensorFlow/Keras) finalizado com sucesso.")