import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys, os, tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
from keras.losses import MeanSquaredError
from keras.activations import mish
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import access_br_dwgd as access_br_dwgd
tf.random.set_seed(7)
# ========================================================================================
# CONFIGURAÇÃO DE LOGGER
# ========================================================================================
sys.path.append(os.path.abspath(".."))
from utils.logger import Logger
import utils.utils as util
import utils.plotUtils as plot
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
th_mm = 0.1
timeseries["chuva_log1p"] = np.log1p(timeseries["chuva"].astype(np.float32))
timeseries["chuva_lag1"]  = timeseries["chuva_log1p"].shift(1).fillna(0)
timeseries["chuva_lag2"]  = timeseries["chuva_log1p"].shift(2).fillna(0)
timeseries["chuva_ma7"]   = timeseries["chuva_log1p"].rolling(7, min_periods=1).mean()
timeseries['choveu_ontem'] = (timeseries['chuva'].shift(1) > th_mm).astype(int)
timeseries['dia_seno']    = np.sin(2 * np.pi * timeseries.index.dayofyear / 365)
timeseries['dia_cosseno'] = np.cos(2 * np.pi * timeseries.index.dayofyear / 365)

timeseries = timeseries.dropna(subset=["chuva_lag1"]) 

logger.info(f"Engenharia de features concluída. Total de colunas: {timeseries.shape}")
logger.info(f"Colunas criadas: {list(timeseries.columns)}")
logger.info(f"Tempo total da Fase 2: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 3 - NORMALIZAÇÃO E PREPARAÇÃO DOS DADOS
# ========================================================================================
inicio = time.time()
logger.info("[FASE 3] Normalizando e criando sequências de treino/teste...")
feat_colnum = ["chuva","Tmin","Tmax", "chuva_lag1", "chuva_log1p" , "chuva_lag2", "chuva_ma7"] 
feat_bin = ["choveu_ontem"]             
feat_cyc = ["dia_seno","dia_cosseno"]   
todas = feat_colnum + feat_bin + feat_cyc


train_size = int(len(timeseries) * 0.80)
valid_size = int(len(timeseries) * 0.9)
print(timeseries.iloc[valid_size:])
scaler = MinMaxScaler().fit(timeseries.iloc[:train_size][feat_colnum])

df_scaled = timeseries.copy()
df_scaled.loc[:, feat_colnum] = scaler.transform(df_scaled[feat_colnum]).astype(np.float32)
lookback = 14

#treino, test = X_all[:train_size], X_all[train_size:]
X,y = util.create_sequence(df_scaled.values, lookback)


trainX, trainY = X[:train_size], y[:train_size]
X_val1, y_val1 = X[train_size:valid_size], y[train_size:valid_size]
testX, testY = X[valid_size:], y[valid_size:]
y_index = timeseries.index[lookback:]
test_date  = y_index[train_size:]
# reshape input to be [samples, time steps, features]
#trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

logger.info(f"Sequências criadas - Lookback: {lookback}")
logger.info(f"Shape X treino: {trainX.shape} | teste: {testX.shape}")
logger.info(f"Shape Y treino: {trainY.shape} | teste: {testY.shape}")
logger.info(f"Tempo total da Fase 3: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 4 - CRIAÇÃO E TREINAMENTO DO MODELO
# ========================================================================================
inicio = time.time()
logger.info("[FASE 4] Criando e treinando o modelo LSTM...")

num_features = timeseries.shape[1]
#model = util.build_deep_lstm(lookback=lookback, num_features=num_features)

n_epochs = 5
batch_size = 32
model = Sequential([
    LSTM(128, activation=mish, input_shape=(lookback, num_features),return_sequences=True),
    Dropout(0.1),
    Dense(128),
    LSTM(64,activation=mish , return_sequences=False),
    Dropout(0.3),
    Dense(1, 'linear')
])
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01), metrics=[RootMeanSquaredError()])

# ---- 8) Callbacks e treino (sem embaralhar!)
cbs = [
    EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-10, monitor="val_loss")
]
start_train = time.time()
sw = 1 + 5*(trainY > 0) 
hist = model.fit(
    trainX, trainY,
    validation_data=(X_val1, y_val1),
    sample_weight=sw,
    epochs=n_epochs, batch_size=batch_size,
    shuffle=False,
    #callbacks=cbs,
    verbose=1
)
logger.info(f"Parâmetros de treinamento -> Épocas: {n_epochs}, Batch: {batch_size}, Features: {num_features}")
logger.info(f"Resumo do modelo:\n{model.summary(print_fn=lambda x: logger.info(x))}")

logger.info(f"Treinamento concluído em {(time.time() - start_train) / 60:.2f} minutos.")
logger.info(f"Tempo total da Fase 4: {time.time() - inicio:.2f} segundos.")
plot.gerar_plot(eixo_x=hist.history['loss'], eixo_y=hist.history['val_loss'], titulo="lstmLoss", xlabel="Epoca", ylabel="Loss", legenda=['Train', 'Validation'])
# ========================================================================================
# FASE 5 - AVALIAÇÃO E MÉTRICAS
# ========================================================================================
inicio6 = time.time()
logger.info("[FASE 5] Avaliando modelo LSTMKeras no conjunto de teste.")
pred_te_s = model.predict(testX)
print(pred_te_s)
util.calcular_erros(logger=logger,
                     dadoReal=testY,
                     dadoPrevisao=pred_te_s
                    )
logger.info(f"Tempo total da Fase 5: {time.time() - inicio6:.2f}s")

# test_y_copies = np.repeat(testY.reshape(-1, 1), testX.shape[-1], axis=-1)
# true_temp = scaler.inverse_transform(test_y_copies)[:,1]

# prediction = model.predict(testX)
# prediction_copies = np.repeat(prediction, num_features, axis=-1)
# predicted_temp = scaler.inverse_transform(prediction_copies)[:,1]

# ========================================================================================
# FASE 6 - AVALIAÇÃO FINAL COM O OS DADOS P TREINAR
# ========================================================================================
inicio6 = time.time()
logger.info("[FASE 6] Avaliando modelo LSTMKeras no conjunto TRAIN.")
pred_train= model.predict(trainX)

util.calcular_erros(logger=logger,
                     dadoReal=trainY,
                     dadoPrevisao=pred_train
                    )
logger.info(f"Tempo total da Fase 6: {time.time() - inicio6:.2f}s")

# ========================================================================================
# FASE 7 - VISUALIZAÇÃO FINAL
# ========================================================================================
inicio = time.time()
logger.info("[FASE 7] Gerando gráficos...")
plot.gerar_plot(eixo_x=testY, eixo_y=pred_te_s, titulo="lstm_gpu_br_dwgd_result", xlabel="Amostra", ylabel="Chuva", legenda=['Real', 'Previsto'])

logger.info("Gráfico salvo como 'lstm_gpu_br_dwgd_result.png'.")

plot.gerar_plot(eixo_x=trainY, eixo_y=pred_train, titulo="lstm_gpu_br_dwgd_resultTrain", xlabel="Amostra", ylabel="Chuva", legenda=['Real', 'Previsto'])

logger.info("Gráfico salvo como 'lstm_gpu_br_dwgd_resultTrain.png'.")
# ========================================================================================
# FINALIZAÇÃO
# ========================================================================================
logger.info("=" * 80)
logger.info("Execução finalizada com sucesso.")
logger.info(f"Ambiente de execução: {'GPU' if gpu_disponiveis else 'CPU'}")
logger.info("=" * 80)
