import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Activation, BatchNormalization, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.activations import mish as tf_mish
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

def create_sequence(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :])
        y.append(data[i+lookback, 0])
    return np.array(X), np.array(y)



def criar_modelo_avancado(
    lookback=60, n_features=18,
    num_neuronios_1_camada = 128,
    num_neuronios_2_camada = 64,
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
                    units=(128, 96, 64, 32),
                    dropout=0.2,
                    rec_dropout=0.1,
                    l2_reg=1e-4):
    model = Sequential(name="DeepLSTM")

    # LSTM 1
    model.add(LSTM(units=units[0],
                   return_sequences=True,
                   input_shape=(lookback, num_features),
                   recurrent_dropout=rec_dropout,
                   kernel_regularizer=l2(l2_reg)))
    model.add(LayerNormalization())
    model.add(Dropout(dropout))

    # LSTM 2
    model.add(LSTM(units=units[1],
                   return_sequences=True,
                   recurrent_dropout=rec_dropout,
                   kernel_regularizer=l2(l2_reg)))
    model.add(LayerNormalization())
    model.add(Dropout(dropout))

    # LSTM 3
    model.add(LSTM(units=units[2],
                   return_sequences=True,
                   recurrent_dropout=rec_dropout,
                   kernel_regularizer=l2(l2_reg)))
    model.add(LayerNormalization())
    model.add(Dropout(dropout))

    # LSTM 4 (última sem return_sequences)
    model.add(LSTM(units=units[3],
                   return_sequences=False,
                   recurrent_dropout=rec_dropout,
                   kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout))

    # Saída
    model.add(Dense(1, name="output"))

    # otimizador com clip de gradiente
    opt = Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model