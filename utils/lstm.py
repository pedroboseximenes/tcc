import pandas as pd

# ========================================================================================
# LOGGER CONFIG
# ========================================================================================
from utils.logger import Logger
import utils.utils as util

def rodarLSTM(timeseries, device, experimentos, scaler, ts_scaled, ts_scaled_df, n_test, titulo):
    logger = Logger.configurar_logger(nome_arquivo=f"lstm{titulo}_torch.log", nome_classe=f"LSTM_{titulo}_TORCH")

    logger.info("=" * 90)
    logger.info(f"Iniciando script LSTM (PyTorch) {titulo} com suporte a GPU e logs detalhados.")
    logger.info("=" * 90)
    resultados = []
    logger.info("[FASE 4] Experimentando com várias variações....")
    for exp in experimentos:
        resultado = util.rodar_experimento_lstm(
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
            dataset= titulo,
            n_epochs      = 1500,
            n_test=n_test,
            batch_size    = 32,
        )
        resultados.append(resultado)

    logger.info("Fim experimentos....")
    melhor = min(resultados, key=lambda r: r["rmse"])
    logger.info(f"*** MELHOR CONFIG: {melhor}")

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(f"pictures/resultados_lstm_{titulo}.csv", index=False)