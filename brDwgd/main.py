import access_br_dwgd as access_br_dwgd
import sys
import os
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time
import numpy as np


sys.path.append(os.path.abspath(".."))
from utils.logger import Logger
from utils.utils import criar_experimentos
from utils.arima import rodarARIMA
from utils.randomforest import rodarRandomForest
from utils.lstm import rodarLSTM
from utils.bilstm import rodarBILSTM


logger = Logger.configurar_logger(nome_arquivo="mainBRDWGD.log", nome_classe="MAIN_BR_DWGD")
# ========================================================================================
# CONFIGURAÇÃO DE GPU
# ========================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memória disponível: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
else:
    logger.warning("Nenhuma GPU disponível. Rodando no CPU.")

# ========================================================================================
# FASE 1 - CARREGAMENTO E PRÉ-PROCESSAMENTO
# ========================================================================================
inicio = time.time()
logger.info("[FASE 1] Carregando e pré-processando dados...")
timeseries = access_br_dwgd.recuperar_dados_br_dwgd_com_area()
logger.info(f"Dados carregados com {len(timeseries)} registros.")
logger.info(f"Período: {timeseries.index.min()} → {timeseries.index.max()}")
logger.info(f"Primeiras linhas:\n{timeseries.head()}")


# ========================================================================================
# FASE 2 - ENGENHARIA DE FEATURES
# ========================================================================================
inicio2 = time.time()
logger.info("[FASE 2] Criando features temporais e estatísticas...")

#timeseries, colunas_normalizar = utilDataset.criar_data_frame_chuva_br_dwgd(df=timeseries, tmax_col='Tmax', tmin_col='Tmin', W=30,wet_thr=1.0)
colunas_normalizar = ["chuva"]
logger.info(f"Engenharia de features concluída. Total de colunas: {timeseries.shape[1]}")
logger.info(f"Colunas criadas: {list(timeseries.columns)}")
logger.info(f"Tempo total da Fase 2: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 3 - NORMALIZAÇÃO E DIVISÃO DE DADOS
# ========================================================================================
inicio3 = time.time()
logger.info("[FASE 3] Normalizando e criando sequências...")
n_test = 30
lookback = 30
scaler = MinMaxScaler().fit(timeseries.iloc[:-n_test])
ts_scaled = scaler.transform(timeseries).astype(np.float32)

experimentos  = criar_experimentos(lookback)

ts_scaled_df = pd.DataFrame(
    ts_scaled,
    index=timeseries.index,
    columns=timeseries.columns
)
titulo = "BRDWGD"
for i in range(1):
    rodarARIMA(
        timeseries,
        scaler,
        ts_scaled,
        ts_scaled_df,
        n_test,
        lookback,
        i,
        titulo
    )
    rodarRandomForest(
        timeseries,
        n_test,
        i,
        titulo
    )
    rodarLSTM(
        timeseries,
        device,
        experimentos,
        scaler,
        ts_scaled,
        ts_scaled_df,
        n_test,
        i,
        titulo
    )
    rodarBILSTM(
        timeseries,
        device,
        experimentos,
        scaler,
        ts_scaled,
        ts_scaled_df,
        n_test,
        i,
        titulo
    )