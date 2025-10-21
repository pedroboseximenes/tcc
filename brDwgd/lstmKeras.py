import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
import access_br_dwgd as access_br_dwgd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys, os
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs disponíveis: {gpus}")
    tf.config.experimental.set_memory_growth(gpus[0], True)  # evita usar toda a memória GPU
else:
    print("Nenhuma GPU disponível. Rodando no CPU.")
sys.path.append(os.path.abspath(".."))

from utils.logger import Logger
logger = Logger.configurar_logger(nome_arquivo="lstmBrDwgd.log", nome_classe="Lstm BrDwgd")
logger.info("Iniciando script de previsão com LSTM BrDwgd.")

def create_sequence(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :])
        y.append(data[i+lookback, 0])
    return np.array(X), np.array(y)

# --- 1. Carregamento e Pré-processamento dos Dados ---
logger.info("Iniciando carregamento e pré-processamento dos dados.")
timeseries = access_br_dwgd.recuperar_dados_br_dwgd_com_area()
# Aplica transformação logarítmica para estabilizar a variância
timeseries['chuva'] = np.log1p(timeseries['chuva'])
logger.info(f"Dados carregados com sucesso. Total de {len(timeseries)} registros.")
datas = timeseries.index 
num_features = 18
# --- 2. Engenharia de Features ---
logger.info("Iniciando engenharia de features.")
# Features Temporais
timeseries['dia_seno'] = np.sin(2 * np.pi * timeseries.index.dayofyear / 365)
timeseries['dia_cosseno'] = np.cos(2 * np.pi * timeseries.index.dayofyear / 365)
timeseries['mes_seno'] = np.sin(2 * np.pi * timeseries.index.month / 12)
timeseries['mes_cosseno'] = np.cos(2 * np.pi * timeseries.index.month / 12)
timeseries['ano'] = timeseries.index.year - timeseries.index.year.min()

# Features de Médias Móveis
timeseries['chuva_ma3'] = timeseries['chuva'].shift(1).rolling(window=3, min_periods=1).mean().fillna(0)
timeseries['chuva_ma7'] = timeseries['chuva'].shift(1).rolling(window=7, min_periods=1).mean().fillna(0)
timeseries['chuva_ma14'] = timeseries['chuva'].shift(1).rolling(window=14, min_periods=1).mean().fillna(0)
timeseries['chuva_ma30'] = timeseries['chuva'].shift(1).rolling(window=30, min_periods=1).mean().fillna(0)

# Features de Estatísticas Móveis
timeseries['chuva_std7'] = timeseries['chuva'].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
timeseries['chuva_max7'] = timeseries['chuva'].shift(1).rolling(window=7, min_periods=1).max().fillna(0)
timeseries['chuva_min7'] = timeseries['chuva'].shift(1).rolling(window=7, min_periods=1).min().fillna(0)

# Features de Lags
timeseries['chuva_lag1'] = timeseries['chuva'].shift(1).fillna(0)
timeseries['chuva_lag3'] = timeseries['chuva'].shift(3).fillna(0)
timeseries['chuva_lag7'] = timeseries['chuva'].shift(7).fillna(0)

# Flags Binários
timeseries['choveu_ontem'] = (timeseries['chuva_lag1'] > 0).astype(int)
timeseries['choveu_semana'] = (timeseries['chuva_ma7'] > 0).astype(int)

logger.info(f"Engenharia de features concluída. Total de features: {timeseries.shape[1]}")

# Normalização
features_dinamicas = [col for col in timeseries.columns if 'chuva' in col]
scaler_chuva = MinMaxScaler()
timeseries_scaled = timeseries.copy()
timeseries_scaled[features_dinamicas] = scaler_chuva.fit_transform(timeseries[features_dinamicas])
logger.info("Features normalizadas com sucesso.")

# --- 3. Visualização e Análise de Correlação ---
logger.info("Gerando gráficos de análise exploratória.")
plt.figure(figsize=(10, 5))
plt.plot(timeseries.index, timeseries['chuva'])
plt.title('Chuva Diária na Estação (Log-Normalizada)')
plt.xlabel('Data')
plt.ylabel('Chuva (log1p)')
plt.savefig('rf_chuva_diaria.png')
plt.close()

lookback = 60
logger.info(f"Preparando sequências com um lookback de {lookback} dias.")
X, y = create_sequence(timeseries.values, lookback=lookback)
train_size = int(len(timeseries) * 0.75)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


dates_aligned = datas[lookback:]
train_date, test_date = dates_aligned[:train_size] , dates_aligned[train_size:]
logger.info(f"Sequências criadas. Treino: {len(X_train)} amostras, Teste: {len(X_test)} amostras.")


n_epochs = 2000
batch_size = 32
logger.info(f"Fase 4: Iniciando treinamento do modelo por {n_epochs} épocas com batch_size de {batch_size}.")
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(lookback, num_features)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dense(units=1))  
model.compile(optimizer='adam', loss='mean_squared_error')
hist = model.fit(X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=n_epochs,
                 batch_size = batch_size,
                 verbose=2)
logger.info("Treinamento do modelo finalizado.")

# --- 5. Avaliação do Modelo ---
logger.info("Fase 5: Realizando previsões no conjunto de teste e avaliando a performance.")
pred = model.predict(X_test)
logger.info(f"Previsões geradas com sucesso. Shape do resultado: {pred.shape}")

# Desnormalizar os dados para avaliação na escala original (opcional, mas recomendado)

pred_rescaled = scaler_chuva.inverse_transform(pred)
y_test_rescaled = scaler_chuva.inverse_transform(y_test.reshape(-1, 1))
logger.info("Resultados (previstos e reais) desnormalizados para avaliação.")
rmse = np.sqrt(mean_squared_error(y_test_rescaled, pred_rescaled))
mse = mean_squared_error(y_test_rescaled, pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
logger.info(f"Avaliação na escala original - RMSE: {rmse:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")


# --- 6. Visualização dos Resultados ---
logger.info("Fase 6: Gerando gráfico de previsão final.")
plt.figure(figsize=(12, 6))
plt.plot(test_date, y_test, label="Real (Normalizado)")
plt.plot(test_date, pred, label="Previsto (Normalizado)")
plt.legend()
plt.title("Previsão de Chuva com LSTM")
plt.xlabel("Data")
plt.ylabel("Chuva (Normalizada)")
plt.xticks(rotation=45)
plt.tight_layout()
logger.info("Exibindo gráfico com os resultados reais vs. previstos.")
plt.show()