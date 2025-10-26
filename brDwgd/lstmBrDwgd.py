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
import access_br_dwgd as access_br_dwgd

# ========================================================================================
# LOGGER CONFIG
# ========================================================================================
sys.path.append(os.path.abspath(".."))
from utils.logger import Logger
from utils.lstmModel import LstmModel
import utils.utils as util
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
timeseries['chuva'] = np.log1p(timeseries['chuva'])
logger.info(f"Dados carregados com {len(timeseries)} registros.")
logger.info(f"Período: {timeseries.index.min()} → {timeseries.index.max()}")
logger.info(f"Primeiras linhas:\n{timeseries.head()}")

# ========================================================================================
# FASE 2 - ENGENHARIA DE FEATURES
# ========================================================================================
inicio2 = time.time()
logger.info("[FASE 2] Criando features temporais e estatísticas...")

timeseries = util.criar_data_frame_chuva(timeseries)

logger.info(f"Engenharia de features concluída. Total de colunas: {timeseries.shape[1]}")
logger.info(f"Colunas criadas: {list(timeseries.columns)}")
logger.info(f"Tempo total da Fase 2: {time.time() - inicio:.2f} segundos.")

# ========================================================================================
# FASE 3 - NORMALIZAÇÃO E DIVISÃO DE DADOS
# ========================================================================================
inicio3 = time.time()
logger.info("[FASE 3] Normalizando e criando sequências...")

features_dinamicas = [c for c in timeseries.columns if ('chuva' in c) or ('mediana' in c) or ('iqr_7' in c)]
scaler = MinMaxScaler()
timeseries[features_dinamicas] = scaler.fit_transform(timeseries[features_dinamicas])

lookback = 30
X, y = util.create_sequence(timeseries, lookback)
train_size = int(len(X) * 0.80)
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
learning_rate = 0.0005
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
        loss = criterion(outputs, y_batch)
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
pred_np = pred.cpu().numpy().reshape(-1, 1)
y_test_np = y_test.cpu().numpy().reshape(-1, 1)

# A MESMA ordem usada no fit:
chuva_cols = [c for c in timeseries.columns if 'chuva' in c]  # mesma lista usada no fit
col_idx = chuva_cols.index('chuva')  # posição da coluna 'chuva' dentro das "chuva*"

n_cols = scaler.n_features_in_
pred_dummy = np.zeros((len(pred_np), n_cols))
ytest_dummy = np.zeros((len(y_test_np), n_cols))

pred_dummy[:, col_idx] = pred_np.flatten()
ytest_dummy[:, col_idx] = y_test_np.flatten()

# Inverte MinMax -> volta para log1p
pred_log = scaler.inverse_transform(pred_dummy)[:, col_idx]
y_test_log = scaler.inverse_transform(ytest_dummy)[:, col_idx]

# Volta do log1p para mm
y_pred = np.expm1(pred_log)
y_true = np.expm1(y_test_log)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

logger.info(f"RMSE: {rmse:.4f}")
logger.info(f"MSE : {mse:.4f}")
logger.info(f"MAE : {mae:.4f}")
logger.info(f"Tempo total da Fase 5: {time.time() - inicio5:.2f}s")

# ========================================================================================
# FASE 6 - VISUALIZAÇÃO
# ========================================================================================
logger.info("[FASE 6] Gerando gráfico de previsão...")
plt.figure(figsize=(12, 6))
plt.plot(y_true, label="Real")
plt.plot(y_pred, label="Previsto")
plt.legend()
plt.title("Previsão de Chuva - LSTM (PyTorch)")
plt.xlabel("Amostra")
plt.ylabel("Chuva")
plt.tight_layout()
plt.savefig("pictures/lstmBrDwgd_torch_result.png")
plt.close()
logger.info("Gráfico salvo como 'lstmBrDwgd_torch_result.png'.")

# ========================================================================================
# FINALIZAÇÃO
# ========================================================================================
logger.info("=" * 90)
logger.info("Execução finalizada com sucesso.")
logger.info(f"Dispositivo utilizado: {device}")
logger.info("=" * 90)