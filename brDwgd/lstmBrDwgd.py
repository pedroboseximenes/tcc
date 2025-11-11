import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn.functional as F
import access_br_dwgd as access_br_dwgd

# ========================================================================================
# LOGGER CONFIG
# ========================================================================================
sys.path.append(os.path.abspath(".."))
from utils.logger import Logger
from utils.lstmModel import LstmModel
import utils.utils as util
import utils.utilDataset as utilDataset
import utils.plotUtils as plot
logger = Logger.configurar_logger(nome_arquivo="lstmBrDwgd_torch.log", nome_classe="LSTM_BR_DWGD_TORCH")

logger.info("=" * 90)
logger.info("Iniciando script LSTM (PyTorch) com suporte a GPU e logs detalhados.")
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

timeseries, colunas_normalizar = utilDataset.criar_data_frame_chuva(df=timeseries, tmax_col='Tmax', tmin_col='Tmin', W=30,wet_thr=1.0)

logger.info(f"Engenharia de features concluída. Total de colunas: {timeseries.shape[1]}")
logger.info(f"Colunas criadas: {list(timeseries.columns)}")
logger.info(f"Tempo total da Fase 2: {time.time() - inicio:.2f} segundos.")
timeseries['chuva'] = np.log1p(timeseries['chuva'])
logger.info("Transformação log1p aplicada na variável 'chuva'.")

# ========================================================================================
# FASE 3 - NORMALIZAÇÃO E DIVISÃO DE DADOS
# ========================================================================================
inicio3 = time.time()
logger.info("[FASE 3] Normalizando e criando sequências...")
train_size = int(len(timeseries) * 0.92)
valid_size = int(len(timeseries) * 0.95)

y_scaler = MinMaxScaler().fit(timeseries.iloc[:train_size][['chuva']])
timeseries['chuva'] = y_scaler.transform(timeseries[['chuva']]).astype(np.float32)

scaler = MinMaxScaler().fit(timeseries.iloc[:train_size][colunas_normalizar])
timeseries.loc[:, colunas_normalizar] = scaler.transform(timeseries[colunas_normalizar]).astype(np.float32)

lookback = 30
X, y = util.create_sequence(timeseries.values, lookback)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

logger.info(f"Shape treino: {X_train.shape}, teste: {X_test.shape}")
logger.info(f"Tempo total da Fase 3: {time.time() - inicio3:.2f}s")
# ========================================================================================
# FASE 4 - TREINAMENTO
# ========================================================================================
inicio4 = time.time()
logger.info("[FASE 4] Iniciando treinamento do modelo PyTorch...")

batch_size = 32
hidden_dim = 256
layer_dim = 2
learning_rate = 0.001
n_epochs = 1200

model = LstmModel(input_dim=X_train.shape[2], hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=batch_size)

logger.info(f"Modelo criado com input_dim={X_train.shape[2]}, hidden_dim={hidden_dim}, layers={layer_dim}")
logger.info(f"Treinando por {n_epochs} épocas com batch_size={batch_size}")

for epoch in range(1, n_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(X_batch)

        mse = F.mse_loss(outputs, y_batch, reduction='mean')
        mae = F.l1_loss(outputs, y_batch, reduction='mean')

        # pesos opcionais: alpha*MSE + beta*MAE
        alpha, beta = 1.0, 1.0
        loss = alpha*mse + beta*mae

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 100 == 0 or epoch == 1:
        logger.info(f"Época [{epoch}/{n_epochs}] - Loss: {epoch_loss / len(train_loader):.6f}")

logger.info(f"Treinamento concluído em {(time.time() - inicio4)/60:.2f} minutos")

# ========================================================================================
# FASE 5 - AVALIAÇÃO
# ========================================================================================
inicio5 = time.time()
logger.info("[FASE 5] Avaliando modelo no conjunto de teste...")

model.eval()
with torch.no_grad():
    pred, _ = model(X_test)

# Tensores -> numpy
print('y_pred raw min/max:', float(pred.min()), float(pred.max()))
print('y_TRUE raw min/max:', float(y_test.min()), float(y_test.max()))

y_pred_mm = util.desescalar_e_delogar_pred(pred.detach().cpu().numpy(), y_scaler)
testY_mm = util.desescalar_e_delogar_pred(y_test.detach().cpu().numpy(), y_scaler)
print('y_pred mm min/max:', float(y_pred_mm.min()), float(y_pred_mm.max()))
print('y_TRUE mm min/max:', float(testY_mm.min()), float(testY_mm.max()))

util.calcular_erros(logger=logger,
                     dadoReal=testY_mm,
                     dadoPrevisao=y_pred_mm
                    )

# ========================================================================================
# FASE 6 - VISUALIZAÇÃO
# ========================================================================================
logger.info("[FASE 6] Gerando gráfico de previsão...")
logger.info("[FASE 7] Gerando gráficos...")
plot.gerar_plot_dois_eixo(eixo_x=testY_mm, eixo_y=y_pred_mm, titulo="lstmTorch_gpu_br_dwgd_result", xlabel="Amostra", ylabel="Chuva", legenda=['Real', 'Previsto'])


# ========================================================================================
# FINALIZAÇÃO
# ========================================================================================
logger.info("=" * 90)
logger.info("Execução finalizada com sucesso.")
logger.info(f"Dispositivo utilizado: {device}")
logger.info("=" * 90)