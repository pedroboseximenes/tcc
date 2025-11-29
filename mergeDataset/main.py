from codecarbon import EmissionsTracker
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
import utils.utils as utils
import utils.utilDataset as utilDataset
import utils.plotUtils as plot
from utils.ModeloBase import ModeloBase

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
num_test = 30
lookback = 30
scaler = MinMaxScaler().fit(timeseries.iloc[:-num_test])
ts_scaled = scaler.transform(timeseries[colunas_normalizar]).astype(np.float32)

ts_scaled_df = pd.DataFrame(
    ts_scaled,
    index=timeseries.index,
    columns=timeseries.columns
)
base_dados = "MERGE"
lista_modelos = utils.criar_modelos(timeseries, colunas_normalizar, scaler, ts_scaled_df, num_test, lookback, base_dados, device)

resultados_acumulados = []
for i in range(2):
    resultado_tmp = {}
    for index, modelo in enumerate(lista_modelos):        
        tracker = utils.configurar_track_carbon(modelo.nome_modelo, base_dados, i)
        tracker.start()
        resultado = modelo.run(i)
        tracker.stop()
        result = utils.registrar_resultado(modelo.nome_modelo, modelo.config_registrar_resultado, resultado, i)
        resultados_acumulados.append(result)
        #resultado_tmp[modelo.nome_modelo] = utils.limpar_predicao(resultado['y_pred'])
        resultado_tmp[modelo.nome_modelo] =resultado['y_pred']

        if(isinstance(modelo, utils.BILSTM)):
            config = result['Configuracao']
            tituloIteracao = f'Exec{i}_config{config}_{base_dados}'
            print( resultado_tmp['ARIMA'])
            print( resultado_tmp['RANDOM_FOREST'])
            print( resultado_tmp['LSTM'])
            print( resultado_tmp['BiLSTM'])

            plot.gerar_grafico_modelos(timeseries.iloc[-num_test:], resultado_tmp['ARIMA'], resultado_tmp['RANDOM_FOREST'], resultado_tmp['LSTM'], resultado_tmp['BiLSTM'], tituloIteracao, base_dados, i)
            resultado_tmp['LSTM'] = []
            resultado_tmp['BiLSTM'] = []

df_bruto = pd.DataFrame(resultados_acumulados)
utilDataset.criar_csv(logger, df_bruto, base_dados)
