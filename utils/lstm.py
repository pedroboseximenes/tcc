import torch
import numpy as np

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