import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# statimeseriesmodels - ARIMAX/SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# ========================================================================================
# IMPORTAÇÕES DO PROJETO
# ========================================================================================
sys.path.append(os.path.abspath(".."))
import utils.utils as util
import utils.plotUtils as plot
import utils.utilDataset as utilDataset
from utils.logger import Logger
import access_merge as access_merge

# ========================================================================================
# CONFIGURAÇÃO DO LOGGER
# ========================================================================================
logger = Logger.configurar_logger(
    nome_arquivo="arimaMerge.log",
    nome_classe="ARIMA_MERGE"
)

logger.info("=" * 90)
logger.info("Iniciando script ARIMA MERGE com 1 features.")
logger.info("=" * 90)

# ========================================================================================
# FUNÇÕES AUXILIARES
# ========================================================================================

def train_arima(endog, exog, order, enforce_stationarity=True, enforce_invertibility=True):
    model = ARIMA(
        endog=endog,
        #exog=exog,
        order=order,
        #enforce_stationarity=enforce_stationarity,
        #enforce_invertibility=enforce_invertibility
    )
    res = model.fit()
    return res

def grid_search_aic(endog_train, orders, exog_train=None):
    """
    Faz busca manual por menor AIC em uma grade pequena de (p,d,q) e (P,D,Q,s).
    Retorna (best_res, best_order, best_seasonal, best_aic).
    """
    best_res = None
    best_order = None
    best_aic = np.inf

    for order in orders:
        try:
            res = train_arima(endog_train, exog_train, order)
            aic = res.aic
            if aic < best_aic:
                best_aic = aic
                best_res = res
                best_order = order
        except Exception as e:
            # apenas registra e segue
            logger.info(f"Falha ao ajustar ARIMA para order={order}: {e}")
            continue
    return best_res, best_order, best_aic

# ========================================================================================
# FASE 1 - CARREGAMENTO
# ========================================================================================
t0 = time.time()
logger.info("[FASE 1] Carregando dados de access_merge.acessar_dados_merge_lat_long()")
timeseries = access_merge.acessar_dados_merge_lat_long()  # Series univariada: chuva diária da estação
logger.info(f"Registros carregados: {len(timeseries)} | Período: {timeseries.index.min()} a {timeseries.index.max()}")

# ========================================================================================
# FASE 2 - FEATURES (18)
# ========================================================================================
inicio = time.time()
logger.info("[FASE 2] Criando features temporais e estatísticas...")

#timeseries, colunas_normalizar = utilDataset.criar_data_frame_chuva(df=timeseries, tmax_col='Tmax', tmin_col='Tmin', W=30,wet_thr=1.0)

logger.info(f"Engenharia de features concluída. Total de colunas: {timeseries.shape[1]}")
logger.info(f"Colunas criadas: {list(timeseries.columns)}")
logger.info(f"Tempo total da Fase 2: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 3 - SPLIT E PADRONIZAÇÃO DAS EXÓGENAS
# ========================================================================================
inicio3 = time.time()
logger.info("[FASE 3] Normalizando e criando sequências...")

n_test = 30
scaler = MinMaxScaler().fit(timeseries.iloc[:-n_test])
ts_scaled = scaler.transform(timeseries).astype(np.float32)
ts_scaled = pd.DataFrame(ts_scaled, index=timeseries.index, columns=timeseries.columns)
#ts_scaled = timeseries

endog = ts_scaled['chuva'].astype('float64')
#exog  = ts_scaled[colunas_normalizar].astype('float64')
# split temporal (índices permanecem alinhados)
endog_train, endog_test = endog.iloc[:-n_test], endog.iloc[-n_test:]
#exog_train,  exog_test  = exog.iloc[:-n_test],  exog.iloc[-n_test:]

logger.info(f"Tamanho treino: {len(endog_train)} | teste: {len(endog_test)}")
logger.info(f"Tempo da Fase 3: {time.time() - inicio3:.2f}s")

# ========================================================================================
# FASE 4 - BUSCA MANUAL DE HIPERPARÂMETROS (SEM AUTO_ARIMA)
# ========================================================================================
t3 = time.time()
logger.info("[FASE 4] Iniciando busca manual por hiperparâmetros (AIC).")

# Grade pequena e segura (ajuste se quiser explorar mais)
# d e D pequenos para evitar over-differencing com log1p e médias/defasagens já estabilizando
p_values = [30]
d_values = [0,1]
q_values = [0,1,2]


orders = [(p,d,q) for p in p_values for d in d_values for q in q_values]
best_res, best_order, best_aic = grid_search_aic(endog_train, orders)

if best_res is None:
    logger.info("Nenhum modelo pôde ser ajustado com a grade fornecida.")
    raise RuntimeError("Falha na busca de hiperparâmetros.")

logger.info(f"Melhor order encontrado: {best_order}")
logger.info(f"Melhor AIC (treino): {best_aic:.2f}")
logger.info(f"Tempo da Fase 4: {time.time() - t3:.2f}s")

# ========================================================================================
# FASE 5 - REFIT NO CONJUNTO COMPLETO DE TREINO
# ========================================================================================
t4 = time.time()
logger.info("[FASE 5] Reajustando melhor modelo no conjunto de treino.")
best_model = ARIMA(
    endog=endog_train,
    order=best_order,
).fit()

logger.info("Melhor modelo ajustado no treino.")
logger.info(f"Tempo da Fase 5: {time.time() - t4:.2f}s")

# ========================================================================================
# FASE 6 - PREVISÃO E AVALIAÇÃO NO TESTE
# ========================================================================================
t5 = time.time()
logger.info("[FASE 6] Gerando previsões no conjunto de teste e avaliando métricas.")

# previsões one-shot no período de teste usando exógenas de teste
y_pred = best_model.predict(
    start=endog_test.index[0],
    end=endog_test.index[-1],
)

train_size = len(timeseries) - len(y_pred)

y_pred_mm, testY_mm = util.desescalar_pred_generico(
    y_pred,
    scaler=scaler,
    ts_scaled=ts_scaled,
    timeseries=timeseries,
    target='chuva',
    start=train_size,
    index=endog_test.index
)
#y_pred_mm, testY_mm = y_pred, endog_test


util.calcular_erros(logger=logger,
                     dadoReal=testY_mm,
                     dadoPrevisao=y_pred_mm
                    )
logger.info(f"Tempo total da Fase 5: {time.time() - t5:.2f}s")

# ========================================================================================
# FASE 7 - VISUALIZAÇÃO
# ========================================================================================
logger.info("[FASE 7] Salvando gráfico de previsão vs. observado.")
plot.gerar_plot_dois_eixo(eixo_x=testY_mm, eixo_y=y_pred_mm, titulo="arimaMerge_result", xlabel="Amostra", ylabel="Chuva", legenda=['Real', 'Previsto'])

logger.info("Gráfico salvo como 'arimaMerge_result.png'.")

# ========================================================================================
# FINALIZAÇÃO
# ========================================================================================
logger.info("=" * 90)
logger.info("Execução ARIMA MERGE finalizada com sucesso.")
logger.info(f"Tempo total de execução: {time.time() - t0:.2f}s")
logger.info("=" * 90)