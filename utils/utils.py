import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
def predict_in_batches(model, X, device, batch_size=32):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].to(device)
            out = model(batch)
            out = torch.clamp(out, min=0.0)

            preds.append(out.cpu())
    return torch.cat(preds, dim=0)

def criar_experimentos(lookback):
    experimentos = [
        {"lookback": lookback, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
        {"lookback": lookback, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
        # {"lookback": lookback, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
        # {"lookback": lookback, "hidden_dim": 256,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
    ]
    return experimentos

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

    TP = int(np.sum(pred_rain & obs_rain))  # previu chuva e choveu
    FP = int(np.sum(pred_rain & ~obs_rain)) # previu chuva e NÃO choveu
    FN = int(np.sum(~pred_rain & obs_rain)) # NÃO previu chuva e choveu
    denom = TP + FP + FN
    csi = (TP / denom) if denom > 0 else np.nan

    # logs
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MSE : {mse:.4f}")
    logger.info(f"MAE : {mae:.4f}")
    logger.info(f"CSI (thr={thr_mm} mm): {csi:.4f}  [TP={TP}, FP={FP}, FN={FN}]")
    return rmse, mse , mae, csi


def _to_series_1d(y_pred, index=None, name="pred"):
    """Converte torch/np/Series para Series 1D com um índice."""
    try:
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
    colunas_normalizar,
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
    y_pred_series = _to_series_1d(y_pred, index=idx, name=target).astype(float)

    # 4) construir template apenas com cols_to_scale (no índice das preds)
    template_scaled = ts_scaled.loc[idx, colunas_normalizar].copy()

    # substituir a coluna alvo no template pelo y_pred escalado (já no espaço do scaler)
    template_scaled[target] = y_pred_series.values

    # 5) inverse_transform apenas desse template
    # scaler.inverse_transform espera np.ndarray com a mesma ordem de features usadas no fit
    inv_vals = scaler.inverse_transform(template_scaled.values)  # shape (n_samples, n_cols_to_scale)

    # achar posição da coluna target dentro de cols_to_scale
    target_pos = colunas_normalizar.index(target)
    y_pred_mm = pd.Series(inv_vals[:, target_pos], index=idx, name=target)

    # 6) extrair y_true no espaço original (mesmo índice)
    # se índice não existir em timeseries, gerará KeyError
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

def get_metricas(resultado):
    return  (
        resultado['rmseTrain'],
        resultado['mseTrain'],
        resultado['maeTrain'],
        resultado['csiTrain'],
        resultado['mse'], 
        resultado['rmse'], 
        resultado['mae'], 
        resultado['csi'], 
        resultado['tempoTreinamento'],
        resultado['y_pred']
    )
def registrar_resultado(modelo,configuracao, resultado, index, isRedeNeural):
    metricas = get_metricas(resultado)
    rmseTrain, mseTrain, maeTrain, csiTrain, mse, rmse, mae, csi, tempo , y_pred_mm= metricas

    if(isRedeNeural):
        configuracao = (
            f"LB={resultado['lookback']}_HD={resultado['hidden_dim']}_"
            f"LD={resultado['layer_dim']}_LR={resultado['learning_rate']}_"
            f"DR={resultado['drop_rate']}"
        )
    return {
        'Modelo': modelo,
        'Configuracao': configuracao, 
        'index': index, 
        'MSE_TRAIN': mseTrain,
        'RMSE_TRAIN': rmseTrain,
        'MAE_TRAIN': maeTrain,
        'CSI_TRAIN': csiTrain,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'CSI': csi,
        'Tempo_treinamento': tempo,
        'y_pred': y_pred_mm
    }

def pegar_melhor_curva(df, nome_modelo, todos_resultados):
    """
    1. Acha a melhor configuração (baseada na média do RMSE).
    2. Busca nos resultados brutos a execução dessa config que teve o menor RMSE individual.
    """
    df_modelo = df[df['Modelo'] == nome_modelo]
    
    # Pega a config vencedora (menor RMSE médio)
    melhor_row = df_modelo.sort_values(by='MAE').iloc[0]
    melhor_config = melhor_row['Configuracao']
    
    # Busca nos dados brutos a melhor execução dessa config
    candidatos = [
        r for r in todos_resultados 
        if r['Modelo'] == nome_modelo and r['Configuracao'] == melhor_config
    ]
    
    # Escolhe a execução com menor RMSE para o gráfic
    melhorCanditado = min(candidatos, key=lambda x: x['MAE'])
    
    return melhorCanditado['y_pred']