import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# statimeseriesmodels - ARIMAX/SARIMAX
from statimeseriesmodels.timeseriesa.statespace.sarimax import SARIMAX

# ========================================================================================
# IMPORTAÇÕES DO PROJETO
# ========================================================================================
sys.path.append(os.path.abspath(".."))
import utils.utils as util
from utils.logger import Logger
import access_br_dwgd as access_br_dwgd

# ========================================================================================
# CONFIGURAÇÃO DO LOGGER
# ========================================================================================
logger = Logger.configurar_logger(
    nome_arquivo="arimaBrDwgd.log",
    nome_classe="ARIMA_BR_DWGD"
)

logger.info("=" * 90)
logger.info("Iniciando script ARIMA/ARIMAX BR_DWGD com 18 features (sem auto_arima).")
logger.info("=" * 90)

# ========================================================================================
# FUNÇÕES AUXILIARES
# ========================================================================================

def train_sarimax(endog, exog, order, seasonal_order=None, enforce_stationarity=True, enforce_invertibility=True):
    model = SARIMAX(
        endog=endog,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order if seasonal_order is not None else (0,0,0,0),
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility
    )
    res = model.fit(disp=False)
    return res

def grid_search_aic(endog_train, exog_train, orders, seasonal_orders):
    """
    Faz busca manual por menor AIC em uma grade pequena de (p,d,q) e (P,D,Q,s).
    Retorna (best_res, best_order, best_seasonal, best_aic).
    """
    best_res = None
    best_order = None
    best_seasonal = None
    best_aic = np.inf

    for order in orders:
        for sorder in seasonal_orders:
            try:
                res = train_sarimax(endog_train, exog_train, order, sorder)
                aic = res.aic
                if aic < best_aic:
                    best_aic = aic
                    best_res = res
                    best_order = order
                    best_seasonal = sorder
            except Exception as e:
                # apenas registra e segue
                logger.info(f"Falha ao ajustar SARIMAX para order={order}, seasonal={sorder}: {e}")
                continue
    return best_res, best_order, best_seasonal, best_aic

# ========================================================================================
# FASE 1 - CARREGAMENTO
# ========================================================================================
t0 = time.time()
logger.info("[FASE 1] Carregando dados de access_br_dwgd.recuperar_dados_br_dwgd_com_area()")
timeseries = access_br_dwgd.recuperar_dados_br_dwgd_com_area()  # Series univariada: chuva diária da estação
logger.info(f"Registros carregados: {len(timeseries)} | Período: {timeseries.index.min()} a {timeseries.index.max()}")

# ========================================================================================
# FASE 2 - FEATURES (18)
# ========================================================================================
t1 = time.time()
logger.info("[FASE 2] Construindo exatamente 18 features conforme os scriptimeseries LSTM.")
timeseries = util.criar_data_frame_chuva(timeseries)
logger.info(f"Total de colunas após engenharia: {timeseries.shape[1]} (esperado: 18)")
logger.info(f"Tempo da Fase 2: {time.time() - t1:.2f}s")

# ========================================================================================
# FASE 3 - SPLIT E PADRONIZAÇÃO DAS EXÓGENAS
# ========================================================================================
t2 = time.time()
logger.info("[FASE 3] Separando alvo (endog) e exógenas (exog) e definindo partição treino/teste.")

# endog = chuva (log1p), exog = demais colunas (17 exógenas)
endog = timeseries['chuva'].astype(float)
exog = timeseries.drop(columns=['chuva']).astype(float)

# Padronização das exógenas é comum para ARIMAX (não do alvo)
scaler_exog = StandardScaler()
exog_scaled = pd.DataFrame(
    scaler_exog.fit_transform(exog),
    index=exog.index,
    columns=exog.columns
)

# split 70/30
split_idx = int(len(timeseries) * 0.80)
endog_train, endog_test = endog.iloc[:split_idx], endog.iloc[split_idx:]
exog_train, exog_test = exog_scaled.iloc[:split_idx], exog_scaled.iloc[split_idx:]

logger.info(f"Tamanho treino: {len(endog_train)} | teste: {len(endog_test)}")
logger.info(f"Tempo da Fase 3: {time.time() - t2:.2f}s")

# ========================================================================================
# FASE 4 - BUSCA MANUAL DE HIPERPARÂMETROS (SEM AUTO_ARIMA)
# ========================================================================================
t3 = time.time()
logger.info("[FASE 4] Iniciando busca manual por hiperparâmetros (AIC).")

# Grade pequena e segura (ajuste se quiser explorar mais)
# d e D pequenos para evitar over-differencing com log1p e médias/defasagens já estabilizando
orders = [(p,d,q) for p in range(0,3) for d in [0,1] for q in range(0,3)]
seasonal_orders = [(0,0,0,0), (0,1,0,7), (1,0,1,7)]  # sem sazonal, ou semanal simples

best_res, best_order, best_seasonal, best_aic = grid_search_aic(endog_train, exog_train, orders, seasonal_orders)

if best_res is None:
    logger.info("Nenhum modelo pôde ser ajustado com a grade fornecida.")
    raise RuntimeError("Falha na busca de hiperparâmetros.")

logger.info(f"Melhor order encontrado: {best_order}")
logger.info(f"Melhor seasonal_order encontrado: {best_seasonal}")
logger.info(f"Melhor AIC (treino): {best_aic:.2f}")
logger.info(f"Tempo da Fase 4: {time.time() - t3:.2f}s")

# ========================================================================================
# FASE 5 - REFIT NO CONJUNTO COMPLETO DE TREINO
# ========================================================================================
t4 = time.time()
logger.info("[FASE 5] Reajustando melhor modelo no conjunto de treino.")
best_model = SARIMAX(
    endog=endog_train,
    exog=exog_train,
    order=best_order,
    seasonal_order=best_seasonal if best_seasonal is not None else (0,0,0,0),
    enforce_stationarity=True,
    enforce_invertibility=True
).fit(disp=False)

logger.info("Melhor modelo ajustado no treino.")
logger.info(f"Tempo da Fase 5: {time.time() - t4:.2f}s")

# ========================================================================================
# FASE 6 - PREVISÃO E AVALIAÇÃO NO TESTE
# ========================================================================================
t5 = time.time()
logger.info("[FASE 6] Gerando previsões no conjunto de teste e avaliando métricas.")

# previsões one-shot no período de teste usando exógenas de teste
pred_log = best_model.predict(
    start=endog_test.index[0],
    end=endog_test.index[-1],
    exog=exog_test
)

# Converter de log1p para escala original
pred_final = np.expm1(pred_log.values)
y_test_final = np.expm1(endog_test.values)

rmse = np.sqrt(mean_squared_error(y_test_final, pred_final))
mse = mean_squared_error(y_test_final, pred_final)
mae = mean_absolute_error(y_test_final, pred_final)

logger.info("--- Métricas no conjunto de teste ---")
logger.info(f"RMSE: {rmse:.4f}")
logger.info(f"MSE : {mse:.4f}")
logger.info(f"MAE : {mae:.4f}")
logger.info("-------------------------------------")
logger.info(f"Tempo da Fase 6: {time.time() - t5:.2f}s")

# ========================================================================================
# FASE 7 - VISUALIZAÇÃO
# ========================================================================================
logger.info("[FASE 7] Salvando gráfico de previsão vs. observado.")
plt.figure(figsize=(15, 7))
plt.plot(endog_test.index, y_test_final, label="Real")
plt.plot(endog_test.index, pred_final, label="Previsto", alpha=0.8)
plt.legend()
plt.title("Previsão de Chuva - ARIMAX (18 features)")
plt.xlabel("Data")
plt.ylabel("Chuva (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("pictures/arimaBrDwgd_result.png")
plt.close()
logger.info("Gráfico salvo como 'arimaBrDwgd_result.png'.")

# ========================================================================================
# FINALIZAÇÃO
# ========================================================================================
logger.info("=" * 90)
logger.info("Execução ARIMAX BR_DWGD finalizada com sucesso.")
logger.info(f"Tempo total de execução: {time.time() - t0:.2f}s")
logger.info("=" * 90)
