import torch
import torch.utils.data as data
import numpy as np
import time
import os, sys
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import access_br_dwgd as access_br_dwgd
import pandas as pd

# ========================================================================================
# LOGGER CONFIG
# ========================================================================================
sys.path.append(os.path.abspath(".."))
from utils.logger import Logger
from utils.biLstmModel import BiLstmModel
import utils.utils as util
import utils.utilDataset as utilDataset
import utils.plotUtils as plot
logger = Logger.configurar_logger(nome_arquivo="BilstmBrDwgd_torch.log", nome_classe="BILSTM_BR_DWGD_TORCH")

logger.info("=" * 90)
logger.info("Iniciando script BILSTM (PyTorch) com suporte a GPU e logs detalhados.")
logger.info("=" * 90)

# ========================================================================================
# CONFIGURAÇÃO DE GPU
# ========================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Dispositivo em uso: {device}")
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

#timeseries, colunas_normalizar = utilDataset.criar_data_frame_chuva(df=timeseries, tmax_col='Tmax', tmin_col='Tmin', W=30,wet_thr=1.0)
colunas_normalizar = ["chuva", "Tmax", "Tmin"]
logger.info(f"Engenharia de features concluída. Total de colunas: {timeseries.shape[1]}")
logger.info(f"Colunas criadas: {list(timeseries.columns)}")
logger.info(f"Tempo total da Fase 2: {time.time() - inicio:.2f} segundos.")


# ========================================================================================
# FASE 3 - NORMALIZAÇÃO E DIVISÃO DE DADOS
# ========================================================================================
inicio3 = time.time()
logger.info("[FASE 3] Normalizando e criando sequências...")
n_test = 500
scaler = MinMaxScaler().fit(timeseries.iloc[:-n_test])
ts_scaled = scaler.transform(timeseries).astype(np.float32)

experimentos = [
    # lookback, hidden_dim, layer_dim, learning_rate, drop_rate
    #hiddem_dim 32, layer_dim 1
    {"lookback": 30, "hidden_dim": 32,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.3},
    {"lookback": 45, "hidden_dim": 32,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.3},
    {"lookback": 60, "hidden_dim": 32,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.3},
    {"lookback": 100, "hidden_dim": 32,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.3},

    # {"lookback": 30, "hidden_dim": 32,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 45, "hidden_dim": 32,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 60, "hidden_dim": 32,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 100, "hidden_dim": 32,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.5},

    # {"lookback": 30, "hidden_dim": 32,  "layer_dim": 1, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 45, "hidden_dim": 32,  "layer_dim": 1, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 60, "hidden_dim": 32,  "layer_dim": 1, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 100, "hidden_dim": 32,  "layer_dim": 1, "learning_rate": 3e-4, "drop_rate": 0.5},
    # #hiddem_dim 32, layer_dim 2
    # {"lookback": 30, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 45, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 60, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 100, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.3},

    # {"lookback": 30, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 45, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 60, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 100, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},

    # {"lookback": 30, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 45, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 60, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 100, "hidden_dim": 32,  "layer_dim": 2, "learning_rate": 3e-4, "drop_rate": 0.5},
    # #hiddem_dim 64, layer_dim 1
    # {"lookback": 30, "hidden_dim": 64,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 45, "hidden_dim": 64,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 60, "hidden_dim": 64,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 100, "hidden_dim": 64,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.3},

    # {"lookback": 30, "hidden_dim": 64,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 45, "hidden_dim": 64,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 60, "hidden_dim": 64,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 100, "hidden_dim": 64,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.5},

    # {"lookback": 30, "hidden_dim": 64,  "layer_dim": 1, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 45, "hidden_dim": 64,  "layer_dim": 1, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 60, "hidden_dim": 64,  "layer_dim": 1, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 100, "hidden_dim": 64,  "layer_dim": 1, "learning_rate": 3e-4, "drop_rate": 0.5},
    # #hiddem_dim 64, layer_dim 2
    # {"lookback": 30, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 45, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 60, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 100, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.3},

    # {"lookback": 30, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 45, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 60, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 100, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},

    # {"lookback": 30, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 45, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 60, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 100, "hidden_dim": 64,  "layer_dim": 2, "learning_rate": 3e-4, "drop_rate": 0.5},
    # #hiddem_dim 128, layer_dim 1
    # {"lookback": 30, "hidden_dim": 128,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 45, "hidden_dim": 128,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 60, "hidden_dim": 128,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 100, "hidden_dim": 128,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.3},

    # {"lookback": 30, "hidden_dim": 128,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 45, "hidden_dim": 128,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 60, "hidden_dim": 128,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 100, "hidden_dim": 128,  "layer_dim": 1, "learning_rate": 1e-3, "drop_rate": 0.5},

    # {"lookback": 30, "hidden_dim": 128,  "layer_dim": 1, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 45, "hidden_dim": 128,  "layer_dim": 1, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 60, "hidden_dim": 128,  "layer_dim": 1, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 100, "hidden_dim": 128,  "layer_dim": 1, "learning_rate": 3e-4, "drop_rate": 0.5},

    #hiddem_dim 128, layer_dim 2
    # {"lookback": 30, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 45, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 60, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.3},
    # {"lookback": 100, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.3},

    # {"lookback": 30, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 45, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 60, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},
    # {"lookback": 100, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 1e-3, "drop_rate": 0.5},

    # {"lookback": 30, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 45, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 60, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 3e-4, "drop_rate": 0.5},
    # {"lookback": 100, "hidden_dim": 128,  "layer_dim": 2, "learning_rate": 3e-4, "drop_rate": 0.5},

]

ts_scaled_df = pd.DataFrame(
    ts_scaled,
    index=timeseries.index,
    columns=timeseries.columns
)
resultados = []
inicio4 = time.time()
logger.info("[FASE 4] Experimentando com várias variações....")
for exp in experimentos:
    resultado = util.rodar_experimento_bilstm(
        timeseries,
        scaler,
        ts_scaled,
        ts_scaled_df,
        device,
        lookback      = exp['lookback'],
        hidden_dim    = exp["hidden_dim"],
        layer_dim     = exp["layer_dim"],
        learning_rate = exp["learning_rate"],
        drop_rate     = exp["drop_rate"],
        logger = logger,
        dataset= "BRDWGD",
        n_epochs      = 10,
        batch_size    = 32,
    )
    resultados.append(resultado)

logger.info("[FASE 4] Fim experimentos....")
melhor = min(resultados, key=lambda r: r["rmse"])
logger.info(f"*** MELHOR CONFIG: {melhor}")

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("pictures/resultados_bilstm_brdwgd.csv", index=False)