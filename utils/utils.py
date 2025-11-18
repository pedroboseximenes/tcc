import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller
import torch
import torch.utils.data as data
from utils.lstmModel import LstmModel
from utils.biLstmModel import BiLstmModel
import torch.nn.functional as F
import time
import utils.plotUtils as plot
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

def rodar_experimento_lstm(
    timeseries,
    scaler,
    ts_scaled,
    ts_scaled_df,
    device,
    lookback,
    hidden_dim,
    layer_dim,
    learning_rate,
    drop_rate,
    logger,
    dataset,
    n_test=500,
    n_epochs=300,
    batch_size=32
):
    logger.info("="*70)
    logger.info(
        f"Iniciando experimento: lookback={lookback}, "
        f"hidden_dim={hidden_dim}, layers={layer_dim}, "
        f"lr={learning_rate}, drop={drop_rate}"
    )

    # ---------- FASE 3: criar sequências para esse lookback ----------
    X, y = create_sequence(ts_scaled, lookback)
    X_train, X_test = split_last_n(X, n_test=n_test)
    y_train, y_test = split_last_n(y, n_test=n_test)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1).to(device)

    logger.info(f"Shape treino: {X_train.shape}, teste: {X_test.shape}")
    # ---------- FASE 4: modelo ----------
    model = LstmModel(
        input_dim=X_train.shape[2],
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        output_dim=1,
        drop_rate=drop_rate
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = data.DataLoader(
        data.TensorDataset(X_train, y_train),
        shuffle=True,
        batch_size=batch_size
    )

    logger.info(f"Treinando por {n_epochs} épocas...")
    inicio = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)

            mse = F.mse_loss(outputs, y_batch, reduction='mean')
            mae = F.l1_loss(outputs, y_batch, reduction='mean')

            # pesos opcionais: alpha*MSE + beta*MAE
            alpha, beta = 1.0, 1.0
            loss = alpha*mse + beta*mae


            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 50 == 0 or epoch == 1:
            logger.info(
                f"[EXP] Época {epoch}/{n_epochs} - Loss: {epoch_loss / len(train_loader):.6f}"
            )
    tempoFinal = (time.time() - inicio)/60
    logger.info(f"Treinamento concluído em {tempoFinal:.2f} minutos")

    # ---------- FASE 5: avaliação ----------
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = torch.clamp(y_pred, min=0.0)

    # desescalar chuva
    train_size = len(timeseries) - len(y_pred) - lookback

    y_pred_mm, y_true_mm = desescalar_pred_generico(
        y_pred,
        scaler=scaler,
        ts_scaled=ts_scaled_df,
        timeseries=timeseries,
        target='chuva',
        start=train_size,
        lookback=lookback
    )
    rmse, mse , mae, csi = calcular_erros(logger=logger, dadoPrevisao=y_pred_mm, dadoReal=y_true_mm)
    logger.info(f"y_pred mm min/max: {float(y_pred_mm.min())}, {float(y_pred_mm.max())}")
    logger.info(f"y_TRUE mm min/max: {float(y_true_mm.min())}, {float(y_true_mm.max())}")

    logger.info(" Gerando gráficos...")
    plot.gerar_plot_dois_eixo(eixo_x=y_true_mm, eixo_y=y_pred_mm, titulo=f"lstm{dataset}_lookback={lookback}_neuronios={hidden_dim}_lr={learning_rate}_droprate={drop_rate}", xlabel="Amostra", ylabel="Chuva", legenda=['Real', 'Previsto'])
    logger.info(" Gráficos gerados...")
    logger.info("=" * 90)
    logger.info("Execução finalizada com sucesso.")
    logger.info(f"Dispositivo utilizado: {device}")
    logger.info("=" * 90)

    return {
    "lookback": lookback,
    "hidden_dim": hidden_dim,
    "layer_dim": layer_dim,
    "learning_rate": learning_rate,
    "drop_rate": drop_rate,
    "rmse": rmse,
    "mse": mse,
    "mae": mae,
    "csi": csi,
    "tempoTreinamento":tempoFinal,
    }


def rodar_experimento_bilstm(
    timeseries,
    scaler,
    ts_scaled,
    ts_scaled_df,
    device,
    lookback,
    hidden_dim,
    layer_dim,
    learning_rate,
    drop_rate,
    logger,
    dataset,
    n_test=500,
    n_epochs=300,
    batch_size=32
):
    logger.info("="*70)
    logger.info(
        f"Iniciando experimento: lookback={lookback}, "
        f"hidden_dim={hidden_dim}, layers={layer_dim}, "
        f"lr={learning_rate}, drop={drop_rate}"
    )

    # ---------- FASE 3: criar sequências para esse lookback ----------
    X, y = create_sequence(ts_scaled, lookback)
    X_train, X_test = split_last_n(X, n_test=n_test)
    y_train, y_test = split_last_n(y, n_test=n_test)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1).to(device)

    logger.info(f"Shape treino: {X_train.shape}, teste: {X_test.shape}")
    # ---------- FASE 4: modelo ----------
    model = BiLstmModel(
        input_dim=X_train.shape[2],
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        output_dim=1,
        drop_rate=drop_rate
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = data.DataLoader(
        data.TensorDataset(X_train, y_train),
        shuffle=True,
        batch_size=batch_size
    )

    logger.info(f"Treinando por {n_epochs} épocas...")
    inicio = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)

            mse = F.mse_loss(outputs, y_batch, reduction='mean')
            mae = F.l1_loss(outputs, y_batch, reduction='mean')

            # pesos opcionais: alpha*MSE + beta*MAE
            alpha, beta = 1.0, 1.0
            loss = alpha*mse + beta*mae

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 50 == 0 or epoch == 1:
            logger.info(
                f"[EXP] Época {epoch}/{n_epochs} - Loss: {epoch_loss / len(train_loader):.6f}"
            )
    tempoFinal = (time.time() - inicio)/60
    logger.info(f"Treinamento concluído em {tempoFinal:.2f} minutos")

    # ---------- FASE 5: avaliação ----------
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = torch.clamp(y_pred, min=0.0)

    # desescalar chuva
    train_size = len(timeseries) - len(y_pred) - lookback

    y_pred_mm, y_true_mm = desescalar_pred_generico(
        y_pred,
        scaler=scaler,
        ts_scaled=ts_scaled_df,
        timeseries=timeseries,
        target='chuva',
        start=train_size,
        lookback=lookback
    )
    rmse, mse , mae, csi = calcular_erros(logger=logger, dadoPrevisao=y_pred_mm, dadoReal=y_true_mm)
    logger.info(f"y_pred mm min/max: {float(y_pred_mm.min())}, {float(y_pred_mm.max())}")
    logger.info(f"y_TRUE mm min/max: {float(y_true_mm.min())}, {float(y_true_mm.max())}")

    logger.info(" Gerando gráficos...")
    plot.gerar_plot_dois_eixo(eixo_x=y_true_mm, eixo_y=y_pred_mm, titulo=f"Bilstm{dataset}_lookback={lookback}_neuronios={hidden_dim}_lr={learning_rate}_droprate={drop_rate}", xlabel="Amostra", ylabel="Chuva", legenda=['Real', 'Previsto'])
    logger.info(" Gráficos gerados...")
    logger.info("=" * 90)
    logger.info("Execução finalizada com sucesso.")
    logger.info(f"Dispositivo utilizado: {device}")
    logger.info("=" * 90)

    return {
    "lookback": lookback,
    "hidden_dim": hidden_dim,
    "layer_dim": layer_dim,
    "learning_rate": learning_rate,
    "drop_rate": drop_rate,
    "rmse": rmse,
    "mse": mse,
    "mae": mae,
    "csi": csi,
    "tempoTreinamento":tempoFinal,
    }