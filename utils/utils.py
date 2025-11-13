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

def calcular_erros(logger, dadoReal, dadoPrevisao, thr_mm=1.0):
    y_true = np.asarray(dadoReal).ravel()
    y_pred = np.asarray(dadoPrevisao).ravel()

    # métricas contínuas
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)

    # binarização para CSI
    obs_rain  = y_true >= thr_mm
    pred_rain = y_pred >= thr_mm

    TP = int(np.sum(pred_rain & obs_rain))
    FP = int(np.sum(pred_rain & ~obs_rain))
    FN = int(np.sum(~pred_rain & obs_rain))
    denom = TP + FP + FN
    csi = (TP / denom) if denom > 0 else np.nan

    # logs
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MSE : {mse:.4f}")
    logger.info(f"MAE : {mae:.4f}")
    logger.info(f"CSI (thr={thr_mm} mm): {csi:.4f}  [TP={TP}, FP={FP}, FN={FN}]")

def desescalar_e_delogar_pred(pred, scaler, timeseries, ts_scaled , train_size, lookback):
    idx_chuva = timeseries.columns.get_loc('chuva')
    pred_scaled = pred.squeeze(-1).cpu().numpy().reshape(-1)
    template = ts_scaled[train_size+lookback:train_size+lookback+len(pred_scaled), :].copy()
    # substitua somente a coluna 'chuva' pelo que o modelo previu (em escala)
    template[:, idx_chuva] = pred_scaled
    # volte ao espaço original
    template_mm = scaler.inverse_transform(template)
    y_pred_mm = template_mm[:, idx_chuva]

    testY_mm = timeseries.iloc[train_size+lookback:train_size+lookback+len(pred_scaled)]['chuva'].to_numpy()
    return y_pred_mm, testY_mm

def split_last_n(dados, n_test=100):
    """
    timeseries: pd.Series, pd.DataFrame ou np.ndarray (1D/2D)
    n_test: quantidade de registros finais para teste
    """
    train = dados[:-n_test]
    test  = dados[-n_test:]
    return train, test