import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout, Activation, BatchNormalization, LayerNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.activations import mish as tf_mish
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_sequence(data, lookback, target_col=0):
    dataX, dataY = [], []
    for i in range(len(data)-lookback-1):
        a = data[i:(i+lookback), :]
        dataX.append(a)
        dataY.append(data[i + lookback, 0])
    return np.array(dataX), np.array(dataY)

def calcular_erros(logger, dadoReal, dadoPrevisao):
    mse  = mean_squared_error(dadoReal, dadoPrevisao)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(dadoReal, dadoPrevisao)
    r2   = r2_score(dadoReal, dadoPrevisao)
    
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MSE : {mse:.4f}")
    logger.info(f"MAE : {mae:.4f}")
    logger.info(f"R2 : {r2:.4f}")

def criar_data_frame_chuva(timeseries):
    #timeseries['chuva'] = np.log1p(timeseries['chuva'])
    th_mm = 0.1     

    # Série em mm (original) a partir de log1p(mm)
    #chuva_mm = np.expm1(timeseries['chuva'])
    chuva_mm = timeseries['chuva']

    # 1) Sazonalidade (cíclico)
    timeseries['dia_seno']    = np.sin(2 * np.pi * timeseries.index.dayofyear / 365)
    timeseries['dia_cosseno'] = np.cos(2 * np.pi * timeseries.index.dayofyear / 365)

    # 2) Médias móveis (em mm, sem vazamento)
    for w in [3, 14, 30]:
        timeseries[f'chuva_ma{w}'] = chuva_mm.shift(1).rolling(window=w, min_periods=1).mean().fillna(0)

    # 3) Acumulações e ocorrência recente (em mm → opcional log1p)
    for w in [7, 30]:
        acc = chuva_mm.shift(1).rolling(w, min_periods=1).sum()
        timeseries[f'chuva_acum{w}'] = np.log1p(acc).fillna(0)  # compacta cauda pesada

    mask_past = chuva_mm.shift(1) > th_mm
    for w in [7, 30]:
        timeseries[f'dias_chuva_{w}'] = mask_past.rolling(window=w, min_periods=1).sum().fillna(0)

    # 4) Tendência e estatísticas robustas (em mm)
    #timeseries['slope_7'] = (
    #    chuva_mm.shift(1)
    #            .rolling(7, min_periods=7)
    #            .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    #).fillna(0)

    med7 = chuva_mm.shift(1).rolling(7, min_periods=1).median()
    q75  = chuva_mm.shift(1).rolling(7, min_periods=1).quantile(0.75)
    q25  = chuva_mm.shift(1).rolling(7, min_periods=1).quantile(0.25)
    timeseries['mediana_7'] = med7.fillna(0)
    timeseries['iqr_7']     = (q75 - q25).fillna(0)

    # 5) Lags do alvo (na escala que você treina: log1p)
    for lag in [1, 3, 7]:
        timeseries[f'chuva_lag{lag}'] = timeseries['chuva'].shift(lag).fillna(0)

    # Flags
    timeseries['choveu_ontem'] = (chuva_mm.shift(1) > th_mm).astype(int)

    # 6) Streaks (em mm)
    mask0 = chuva_mm.shift(1) > th_mm
    grp   = (mask0 != mask0.shift()).cumsum()
    timeseries['cwd'] = mask0.groupby(grp).cumsum().fillna(0)        # dias chuvosos seguidos
    timeseries['cdd'] = (~mask0).groupby(grp).cumsum().fillna(0)     # dias secos seguidos

    # Dias desde a última chuva (até t-1)
    idx = np.arange(len(timeseries))
    last_idx = pd.Series(np.where(mask0, idx, np.nan), index=timeseries.index).ffill()
    timeseries['dias_desde_ultima'] = (idx - last_idx).fillna(0).astype(int)

    # 7) Evento forte nos últimos 3 dias (p95 em mm)
    # Ideal: calcule p95 usando APENAS o conjunto de treino e reutilize no teste.
    p95_mm = chuva_mm.quantile(0.95)
    timeseries['evento_forte_3d'] = (chuva_mm.shift(1).rolling(3).max() >= p95_mm).astype(int)
    return timeseries

def criar_modelo_avancado(
    lookback=60, n_features=18,
    num_neuronios_1_camada = 32,
    num_neuronios_2_camada = 16,
    l2_kernel=1e-3, l2_bias=1e-3,
    dropout_rate=0.2, lr=1e-3, clipnorm=1.0
):
    """
    Cria modelo BiLSTM com configurações avançadas:
    - Mish activation
    - L2 regularization (kernel e bias)
    - Dropout
    - BatchNormalization
    """
  
    model = Sequential(name='BiLSTM_128_64')

    # LSTM 1 (retorna sequência)
    model.add(Bidirectional(LSTM(
        num_neuronios_1_camada, return_sequences=True,
        activation='tanh', recurrent_activation='sigmoid',
        kernel_regularizer=l2(l2_kernel),
        recurrent_regularizer=l2(l2_kernel),
        bias_regularizer=l2(l2_bias),
        dropout=dropout_rate
    ), input_shape=(lookback, n_features)))
    model.add(BatchNormalization())
    model.add(Activation(tf_mish))
    model.add(Dropout(dropout_rate))

    # LSTM 2 (última)
    model.add(Bidirectional(LSTM(
        num_neuronios_2_camada, return_sequences=False,
        activation='tanh', recurrent_activation='sigmoid',
        kernel_regularizer=l2(l2_kernel),
        recurrent_regularizer=l2(l2_kernel),
        bias_regularizer=l2(l2_bias),
        dropout=dropout_rate
    )))
    model.add(BatchNormalization())
    model.add(Activation(tf_mish))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1, kernel_regularizer=l2(l2_kernel), bias_regularizer=l2(l2_bias)))

    opt = Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

def build_deep_lstm(lookback, num_features,
                    units=(32, 16),
                    dropout=0.1,
                    rec_dropout=0.1,
                    l2_reg=1e-4):
    model = Sequential(name="DeepLSTM")

    # LSTM 1
    model.add(LSTM(units=units[0],
                   return_sequences=True,
                   input_shape=(lookback, num_features),
                   ))
    #model.add(Dropout(dropout))

    # LSTM 2
    #model.add(LSTM(units=units[1],
                   #return_sequences=True,
                   #recurrent_dropout=rec_dropout,
                   #kernel_regularizer=l2(l2_reg)))
    #model.add(LayerNormalization())
    #model.add(Dropout(dropout))

    # LSTM 4 (última sem return_sequences)
    model.add(LSTM(units=units[1],
                   return_sequences=False,))
    model.add(Dropout(dropout))

    # Saída
    model.add(Dense(1))

    # otimizador com clip de gradiente
    opt = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=opt, loss="mse")
    return model