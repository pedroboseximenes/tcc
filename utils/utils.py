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
from statsmodels.tsa.stattools import adfuller

def create_sequence(data, lookback):
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

def desescalar_e_delogar_pred(y_pred, scaler):
    arr = y_pred.reshape(-1, 1)
    y_log = scaler.inverse_transform(arr)
    y = np.expm1(y_log)
   
    return y

def criar_modelo_bilstm(
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

def check_stationarity(series):
    result = adfuller(series.dropna())  # Drop NaN values if any
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")