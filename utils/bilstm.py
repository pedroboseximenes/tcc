import os
import pandas as pd

# ========================================================================================
# LOGGER CONFIG
# ========================================================================================
from utils.logger import Logger
import utils.utils as util
def rodarBILSTM(timeseries, device, experimentos, scaler, ts_scaled_df, n_test,index , titulo):
    logger = Logger.configurar_logger(nome_arquivo=f"Bilstm{titulo}_torch.log", nome_classe=f"BILSTM_{titulo}_TORCH")

    logger.info("=" * 90)
    logger.info(f"Iniciando script BILSTM (PyTorch) {titulo} com suporte a GPU e logs detalhados.")
    logger.info("=" * 90)


    resultados = []
    logger.info("[FASE 4] Experimentando com várias variações....")
    for exp in experimentos:
        resultado = util.rodar_experimento_bilstm(
            timeseries,
            scaler,
            ts_scaled_df,
            device,
            lookback      = exp['lookback'],
            hidden_dim    = exp["hidden_dim"],
            layer_dim     = exp["layer_dim"],
            learning_rate = exp["learning_rate"],
            drop_rate     = exp["drop_rate"],
            logger = logger,
            dataset= titulo,
            index=index,
            n_epochs      = 500,
            n_test=n_test,
            batch_size    = 32,
        )
        resultados.append(resultado)

    logger.info(f"[FASE 4] Fim experimentos index {index}....")
    melhor = min(resultados, key=lambda r: r["rmse"])
    logger.info(f"*** MELHOR CONFIG: {melhor}")

    df_resultados = pd.DataFrame(resultados)
    caminho = f"pictures/resultados_bilstm_{titulo}.csv"
    df_resultados.to_csv(caminho, 
                            mode="a",                     
                            header=not os.path.exists(caminho),
                            index=False)