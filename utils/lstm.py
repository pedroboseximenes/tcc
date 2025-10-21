import torch
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Activation, BatchNormalization
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

def create_sequences_pytorch(input_data, lookback):
    X, y = [], []
    for i in range(len(input_data) - lookback):
        # A sequência de entrada (X) são os 'lookback' dias
        feature = input_data[i:i+lookback, :]
        # O alvo (y) é o valor do dia seguinte
        target = input_data[i+lookback, 0]
        X.append(feature)
        y.append(target)
    
    # Converte para tensores com o formato correto
    X_arr = np.array(X).reshape(-1, lookback, 1)
    y_arr = np.array(y).reshape(-1, 1)
    
    return torch.tensor(X_arr, dtype=torch.float32), torch.tensor(y_arr, dtype=torch.float32)


def criar_modelo_avancado(
    lookback=60,
    n_features=18,
    units_camada1=16,
    units_camada2=4,
    dropout_rate=0.2,
    activation='relu', optimizer='adam'
):
    """
    Cria modelo BiLSTM com configurações avançadas:
    - Mish activation
    - L2 regularization (kernel e bias)
    - Dropout
    - BatchNormalization
    """
  
    model = Sequential(name='BiLSTM_Melhorado')
    
    model.add(Bidirectional(
        LSTM(units_camada1, return_sequences=True,  activation=activation),
        input_shape=(lookback, n_features)
    ))
    model.add(Bidirectional(
        LSTM(units_camada2, return_sequences=False,  activation=activation)
    ))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    # Compilar modelo
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model