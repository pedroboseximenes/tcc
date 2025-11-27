import access_merge as access_merge
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
import utils.utils as utils
import utils.utilDataset as utilDataset
import utils.plotUtils as plot

logger = Logger.configurar_logger(nome_arquivo="mainMERGE.log", nome_classe="MAIN_MERGE")
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
timeseries = access_merge.acessar_dados_merge_lat_long()
timeseries["chuva"] = timeseries["chuva"].apply(lambda x: 0 if x < 0.001 else x)
logger.info(f"Dados carregados com {len(timeseries)} registros.")
logger.info(f"Período: {timeseries.index.min()} → {timeseries.index.max()}")
logger.info(f"Primeiras linhas:\n{timeseries.head()}")


# ========================================================================================
# FASE 2 - ENGENHARIA DE FEATURES
# ========================================================================================
inicio2 = time.time()
logger.info("[FASE 2] Criando features temporais e estatísticas...")

#timeseries, colunas_normalizar = criar_data_frame_chuva_br_dwgd(df=timeseries, tmax_col='Tmax', tmin_col='Tmin', W=30,wet_thr=1.0)
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
ts_scaled = scaler.transform(timeseries[colunas_normalizar]).astype(np.float32)

experimentos = criar_experimentos(lookback)

ts_scaled_df = pd.DataFrame(
    ts_scaled,
    index=timeseries.index,
    columns=timeseries.columns
)
titulo = "MERGE"
resultados_acumulados = []

for i in range(10):
    resultado_arima = rodarARIMA(
        timeseries,
        colunas_normalizar,
        scaler,
        timeseries,
        n_test,
        lookback,
        i,
        titulo
    )
    result_arima = utils.registrar_resultado('ARIMA', "Padrão", resultado_arima, i, False)

    resultado_rf= rodarRandomForest(
        timeseries,
        n_test,
        i,
        titulo
    )
    result_rf = utils.registrar_resultado('RF',"Padrão", resultado_rf, i, False)

    for exp in experimentos:
        resultadolstm = rodarLSTM(
            timeseries,
            colunas_normalizar,
            device,
            scaler,
            timeseries,
            n_test,
            i,
            titulo,
            lookback      = exp['lookback'],
            hidden_dim    = exp["hidden_dim"],
            layer_dim     = exp["layer_dim"],
            learning_rate = exp["learning_rate"],
            drop_rate     = exp["drop_rate"],
        )
        result_lstm= utils.registrar_resultado('LSTM', "", resultadolstm, i, True)

        resultadobilstm = rodarBILSTM(
            timeseries,
            colunas_normalizar,
            device,
            scaler,
            timeseries,
            n_test,
            i,
            titulo,
            lookback      = exp['lookback'],
            hidden_dim    = exp["hidden_dim"],
            layer_dim     = exp["layer_dim"],
            learning_rate = exp["learning_rate"],
            drop_rate     = exp["drop_rate"],
        )
        result_bilstm = utils.registrar_resultado('BILSTM', "", resultadobilstm, i, True)
        config = result_bilstm['Configuracao']
        tituloIteracao = f'Exec{i}_config{config}_{titulo}'
        plot.gerar_grafico_modelos(timeseries.iloc[-n_test:], result_arima['y_pred'], result_rf['y_pred'], result_lstm['y_pred'], result_bilstm['y_pred'], tituloIteracao, titulo, i)

        result_lstm.pop('y_pred')
        result_bilstm.pop('y_pred')
        resultados_acumulados.append(result_lstm)
        resultados_acumulados.append(result_bilstm)

    
    result_arima.pop('y_pred')
    result_rf.pop('y_pred')
    resultados_acumulados.append(result_arima)
    resultados_acumulados.append(result_rf)

df_bruto = pd.DataFrame(resultados_acumulados)
utilDataset.criar_csv(logger, df_bruto, titulo)
