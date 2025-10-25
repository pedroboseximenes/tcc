import time
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Activation, BatchNormalization
from scikeras.wrappers import KerasRegressor
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs disponíveis: {gpus}")
    tf.config.experimental.set_memory_growth(gpus[0], True)  # evita usar toda a memória GPU
else:
    print("Nenhuma GPU disponível. Rodando no CPU.")
# --- Configuração do Ambiente e Logger ---
# Adiciona o diretório pai ao path para encontrar os módulos 'utils' e 'access_merge'
sys.path.append(os.path.abspath(".."))

from utils.logger import Logger
import utils.utils as util
import access_merge as access_merge


# Configura o logger
logger = Logger.configurar_logger(nome_arquivo="biLstmMERGE.log", nome_classe="BiLstm MERGE")
logger.info("Iniciando script de previsão com BiLSTM Merge (TensorFlow/Keras).")
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
timeseries = access_merge.acessar_dados_merge(logger=logger)
timeseries['chuva'] = np.log1p(timeseries['chuva'])

# --- 2. Engenharia de Features ---
logger.info("Iniciando engenharia de features.")
num_features_inicial = timeseries.shape[1]

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

num_features_final = timeseries.shape[1]
logger.info(f"Engenharia de features concluída. Total de features: {num_features_final}")

# Normalização
features_dinamicas = [col for col in timeseries.columns if 'chuva' in col]
scaler_chuva = MinMaxScaler()
timeseries[features_dinamicas] = scaler_chuva.fit_transform(timeseries[features_dinamicas])
logger.info("Features normalizadas com sucesso.")
logger.info(f"Dados carregados com sucesso. Total de {len(timeseries)} registros.")
datas = timeseries.index

# --- 3. Visualização e Análise de Correlação ---
logger.info("Gerando gráficos de análise exploratória.")
plt.figure(figsize=(10, 5))
plt.plot(timeseries['chuva'])
plt.title('Chuva Diária na Estação (Log-Normalizada)')
plt.xlabel('Data')
plt.ylabel('Chuva (log1p)')
plt.savefig('chuva_diaria.png')
plt.close() # Fecha a figura para não exibir em modo script

plt.figure(figsize=(18, 14))
corr = timeseries.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlação entre as Variáveis")
plt.savefig('correlacao_variaveis.png')
plt.close()

# --- 4. Preparação das Sequências para o Modelo ---
lookback = 14
logger.info(f"Preparando sequências com um lookback de {lookback} dias.")
X, y = util.create_sequence(timeseries.values, lookback)
dates_aligned = datas[lookback:]

train_size = int(len(X) * 0.70)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
train_date, test_date = dates_aligned[:train_size], dates_aligned[train_size:]

logger.info(f"Sequências criadas. Treino: {len(X_train)} amostras, Teste: {len(X_test)} amostras.")
logger.info(f"Shape dos dados de treino (X_train): {X_train.shape}")


# --- 5. Otimização de Hiperparâmetros (GridSearchCV - Opcional) ---
# Esta seção está comentada por ser computacionalmente intensiva.
# Descomente para executar a busca de hiperparâmetros.

logger.info("Iniciando GridSearchCV para otimização de hiperparâmetros.")
param_grid = {
    'model__units_camada1': [32, 64],
    'model__units_camada2': [16, 32],
    'model__dropout_rate': [0.2, 0.3],
    'model__activation': ['relu', 'tanh'],
    'model__optimizer': ['adam', 'rmsprop'],
    'batch_size': [32, 64],
    'epochs': [50] 
}
model_wrapper = KerasRegressor(model=util.criar_modelo_avancado, verbose=0, n_features=num_features_final, lookback=lookback)
tscv = TimeSeriesSplit(n_splits=3)
grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2)

start_grid = time.time()
grid_result = grid.fit(X_train, y_train)
end_grid = time.time()

logger.info(f"GridSearchCV finalizado em {end_grid - start_grid:.2f} segundos.")
logger.info(f"Melhores parâmetros encontrados: {grid_result.best_params_}")
logger.info(f"Melhor score de validação cruzada (negativo da MSE): {grid_result.best_score_:.4f}")
best_model = grid_result.best_estimator_


# --- 6. Treinamento do Modelo BiLSTM ---
epochs = 1000
batch_size = 32

# Callbacks
logging_callback = LoggingCallback()
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=25, min_lr=1e-6, verbose=1)

logger.info("Compilando o modelo BiLSTM com arquitetura avançada.")
model = util.criar_modelo_avancado(
    lookback=lookback,
    n_features=num_features_final,
    units_camada1=128,  # Exemplo de valor, ajuste se usar GridSearchCV
    units_camada2=64,   # Exemplo de valor
    dropout_rate=0.2    # Exemplo de valor
)
model.summary(print_fn=logger.info) # Loga o sumário do modelo

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

# --- 7. Avaliação e Previsão ---
logger.info("Realizando previsões no conjunto de teste.")
pred = model.predict(X_test)

# Inversão da transformação para a escala original
n_features_chuva = scaler_chuva.n_features_in_
pred_dummy = np.zeros((len(pred), n_features_chuva))
pred_dummy[:, 0] = pred.flatten()
pred_log = scaler_chuva.inverse_transform(pred_dummy)[:, 0]
pred_final = np.expm1(pred_log)

y_test_dummy = np.zeros((len(y_test), n_features_chuva))
y_test_dummy[:, 0] = y_test.flatten()
y_test_log = scaler_chuva.inverse_transform(y_test_dummy)[:, 0]
y_test_final = np.expm1(y_test_log)

# Cálculo das métricas
rmse = np.sqrt(mean_squared_error(y_test_final, pred_final))
mse = mean_squared_error(y_test_final, pred_final)
mae = mean_absolute_error(y_test_final, pred_final)

logger.info("--- Métricas de Avaliação no Conjunto de Teste ---")
logger.info(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")
logger.info(f"MSE (Erro Quadrático Médio): {mse:.4f}")
logger.info(f"MAE (Erro Absoluto Médio): {mae:.4f}")
logger.info("-------------------------------------------------")

# --- 8. Visualização Final ---
logger.info("Gerando gráfico de previsão final.")
plt.figure(figsize=(15, 7))
plt.plot(test_date, y_test_final, label="Real")
plt.plot(test_date, pred_final, label="Previsto", alpha=0.7)
plt.legend()
plt.title("Previsão de Chuva com BiLSTM vs. Dados Reais")
plt.xlabel("Data")
plt.ylabel("Chuva (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('previsao_final_bilstm.png')
plt.close()

logger.info("Script BiLSTM Merge (TensorFlow/Keras) finalizado com sucesso.")