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

# def desescalar_e_delogar_pred(pred, scaler, timeseries, ts_scaled , train_size, lookback):
#     idx_chuva = timeseries.columns.get_loc('chuva')
#     pred_scaled = pred
#     template = ts_scaled[train_size+lookback:train_size+lookback+len(pred_scaled), :].copy()
#     # substitua somente a coluna 'chuva' pelo que o modelo previu (em escala)
#     template[:, idx_chuva] = pred_scaled
#     # volte ao espaço original
#     template_mm = scaler.inverse_transform(template)
#     y_pred_mm = template_mm[:, idx_chuva]

#     testY_mm = timeseries.iloc[train_size+lookback:train_size+lookback+len(pred_scaled)]['chuva'].to_numpy()
#     return y_pred_mm, testY_mm
def _to_series_1d(y_pred, index=None, name="pred"):
    """Converte torch/np/Series para Series 1D com um índice."""
    try:
        import torch
        is_torch = isinstance(y_pred, torch.Tensor)
    except Exception:
        is_torch = False

    if is_torch:
        y_pred = y_pred.detach().cpu().numpy()
    elif isinstance(y_pred, pd.Series):
        s = y_pred.copy()
        s.name = name
        return s

    # agora y_pred é np.ndarray ou lista
    y_pred = np.asarray(y_pred)
    if y_pred.ndim > 1:
        if y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        else:
            raise ValueError(f"y_pred tem shape {y_pred.shape}; esperava 1D ou (n,1).")

    if index is None:
         index = pd.RangeIndex(start=0, stop=len(y_pred))
    return pd.Series(y_pred, index=index, name=name)

def desescalar_pred_generico(
    y_pred,
    *,
    scaler,
    ts_scaled: pd.DataFrame,
    timeseries: pd.DataFrame,
    target: str = "chuva",
    index=None,
    start=None,
    lookback: int = 0,
):
    """
    Desescala previsões feitas em espaço MinMax (ou outro scaler) para o espaço original.

    Parâmetros
    ----------
    y_pred : pd.Series | np.ndarray | torch.Tensor
        Predições NO ESPAÇO ESCALADO para a coluna `target`.
        Pode vir do ARIMA, RandomForest, PyTorch etc.
    scaler : sklearn Scaler já .fit(...)
        O mesmo usado para criar `ts_scaled`.
    ts_scaled : pd.DataFrame
        DataFrame já escalado (mesmos index/columns de `timeseries`).
    timeseries : pd.DataFrame
        DataFrame original (não escalado), para coletar o y verdadeiro no mesmo índice.
    target : str
        Nome da coluna alvo.
    index : pandas.Index | array-like | None
        Índice (datas/posições) das predições. Use isto quando seu modelo preservar índice
        (ex.: ARIMA → use `endog_test.index`).
    start : int | None
        Posição inicial (base 0) no `ts_scaled` para prever quando NÃO houver `index`.
        Útil para modelos que retornam apenas array (ex.: RandomForest, Torch).
        Ex.: se o teste são as últimas N linhas, `start = len(timeseries) - N`.
    lookback : int
        Offset adicional típico de LSTM (janelas). Some ao `start` quando usar posição.

    Retorna
    -------
    y_pred_mm : pd.Series
        Previsões no espaço original (mesmo índice passado/derivado).
    y_true_mm : pd.Series
        Valores verdadeiros no espaço original, no mesmo índice.
    """
    # 1) Determinar o índice das predições

    start_pos = int(start) + int(lookback)
    stop_pos = start_pos + len(y_pred)
    idx = ts_scaled.index[start_pos:stop_pos]
    if len(idx) != len(y_pred):
        raise ValueError(
            f"Comprimento do índice ({len(idx)}) difere de y_pred ({len(y_pred)}). "
            f"Cheque `start` e `lookback`."
        )

    # 2) Converter y_pred para Series 1D com o índice determinado
    y_pred_series = _to_series_1d(y_pred, index=idx, name=target).astype(float)

    # 3) Montar template e substituir SOMENTE a coluna target
    if target not in ts_scaled.columns:
        raise KeyError(f"Coluna alvo '{target}' não existe em ts_scaled.columns.")
    target_pos = ts_scaled.columns.get_loc(target)

    template = ts_scaled.loc[idx].copy()
    template.iloc[:, target_pos] = y_pred_series.values

    # 4) Inverter o scaling
    inv = scaler.inverse_transform(template.values)
    y_pred_mm = pd.Series(inv[:, target_pos], index=idx, name=target)

    # 5) Verdade-terreno no espaço original, mesmo índice
    y_true_mm = timeseries.loc[idx, target].astype(float)

    return y_pred_mm, y_true_mm

def split_last_n(dados, n_test=100):
    """
    timeseries: pd.Series, pd.DataFrame ou np.ndarray (1D/2D)
    n_test: quantidade de registros finais para teste
    """
    train = dados[:-n_test]
    test  = dados[-n_test:]
    return train, test