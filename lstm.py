import numpy as np
import torch.optim as optim
import torch.utils.data as data
from lstmModel import LstmModel 
import torch.nn as nn
import torch
from access_br_dwgd import recuperar_dados_br_dwgd
import matplotlib.pyplot as plt


def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)  