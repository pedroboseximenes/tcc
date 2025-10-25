import time
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# --- Configuração do Ambiente e Logger ---
# Adiciona o diretório pai ao path para encontrar os módulos 'utils' e 'access_merge'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from utils.logger import Logger
    import access_merge as access_merge
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Certifique-se de que os arquivos 'utils/logger.py' e 'access_merge.py' estão no lugar correto.")
    sys.exit(1)

# Configura o logger
logger = Logger.configurar_logger(nome_arquivo="RandomForestMERGE.log", nome_classe="RandomForest MERGE")
logger.info("Iniciando script de previsão com Random Forest (Scikit-learn).")

# --- 1. Carregamento e Pré-processamento dos Dados ---
logger.info("Iniciando carregamento e pré-processamento dos dados.")
timeseries = access_merge.acessar_dados_merge()
# Aplica transformação logarítmica para estabilizar a variância
timeseries['chuva'] = np.log1p(timeseries['chuva'])
logger.info(f"Dados carregados com sucesso. Total de {len(timeseries)} registros.")

# --- 2. Engenharia de Features ---
logger.info("Iniciando engenharia de features.")
# Features Temporais
timeseries['dia_seno'] = np.sin(2 * np.pi * timeseries.index.dayofyear / 365)
timeseries['dia_cosseno'] = np.cos(2 * np.pi * timeseries.index.dayofyear / 365)
timeseries['mes_seno'] = np.sin(2 * np.pi * timeseries.index.month / 12)
timeseries['mes_cosseno'] = np.cos(2 * np.pi * timeseries.index.month / 12)
timeseries['ano'] = timeseries.index.year - timeseries.index.year.min()

# Features de Médias Móveis
timeseries['chuva_ma3'] = timeseries['chuva'].shift(1).rolling(window=3, min_periods=1).mean().fillna(0)
timeseries['chuva_ma7'] = timeseries['chuva'].shift(1).rolling(window=7, min_periods=1).mean().fillna(0)
timeseries['chuva_ma14'] = timeseries['chuva'].shift(1).rolling(window=14, min_periods=1).mean().fillna(0)
timeseries['chuva_ma30'] = timeseries['chuva'].shift(1).rolling(window=30, min_periods=1).mean().fillna(0)

# Features de Estatísticas Móveis
timeseries['chuva_std7'] = timeseries['chuva'].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
timeseries['chuva_max7'] = timeseries['chuva'].shift(1).rolling(window=7, min_periods=1).max().fillna(0)
timeseries['chuva_min7'] = timeseries['chuva'].shift(1).rolling(window=7, min_periods=1).min().fillna(0)

# Features de Lags
timeseries['chuva_lag1'] = timeseries['chuva'].shift(1).fillna(0)
timeseries['chuva_lag3'] = timeseries['chuva'].shift(3).fillna(0)
timeseries['chuva_lag7'] = timeseries['chuva'].shift(7).fillna(0)

# Flags Binários
timeseries['choveu_ontem'] = (timeseries['chuva_lag1'] > 0).astype(int)
timeseries['choveu_semana'] = (timeseries['chuva_ma7'] > 0).astype(int)

logger.info(f"Engenharia de features concluída. Total de features: {timeseries.shape[1]}")

# Normalização
features_dinamicas = [col for col in timeseries.columns if 'chuva' in col]
scaler_chuva = MinMaxScaler()
timeseries_scaled = timeseries.copy()
timeseries_scaled[features_dinamicas] = scaler_chuva.fit_transform(timeseries[features_dinamicas])
logger.info("Features normalizadas com sucesso.")

# --- 3. Visualização e Análise de Correlação ---
logger.info("Gerando gráficos de análise exploratória.")
plt.figure(figsize=(10, 5))
plt.plot(timeseries.index, timeseries['chuva'])
plt.title('Chuva Diária na Estação (Log-Normalizada)')
plt.xlabel('Data')
plt.ylabel('Chuva (log1p)')
plt.savefig('rf_chuva_diaria.png')
plt.close()

#plt.figure(figsize=(18, 14))
#corr = timeseries.corr(numeric_only=True)
#sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
#plt.title("Correlação entre as Variáveis")
#plt.savefig('rf_correlacao_variaveis.png')
#plt.close()

# --- 4. Divisão dos Dados em Treino e Teste ---
logger.info("Dividindo os dados em conjuntos de treino e teste.")
X = timeseries_scaled.drop(columns=['chuva'])
y = timeseries_scaled['chuva']
datas = timeseries.index

train_size = int(len(X) * 0.70)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
train_date, test_date = datas[:train_size], datas[train_size:]

logger.info(f"Divisão concluída. Treino: {len(X_train)} amostras, Teste: {len(X_test)} amostras.")
logger.info(f"Shape dos dados de treino (X_train): {X_train.shape}")

# --- 5. Otimização de Hiperparâmetros com GridSearchCV ---
logger.info("Iniciando GridSearchCV para otimização de hiperparâmetros do Random Forest.")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
tscv = TimeSeriesSplit(n_splits=3)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

start_grid = time.time()
grid_search.fit(X_train, y_train)
end_grid = time.time()

logger.info(f"GridSearchCV finalizado em {end_grid - start_grid:.2f} segundos.")
logger.info(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
logger.info(f"Melhor score de validação cruzada (negativo da MSE): {grid_search.best_score_:.4f}")

# --- 6. Treinamento do Modelo Final ---
logger.info("Treinando o modelo Random Forest final com os melhores parâmetros.")
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
logger.info("Treinamento do modelo final concluído.")

# --- 7. Avaliação e Previsão ---
logger.info("Realizando previsões no conjunto de teste.")
pred_scaled = best_rf.predict(X_test)

# Inversão da transformação para a escala original
pred_dummy = np.zeros((len(pred_scaled), len(features_dinamicas)))
pred_dummy[:, 0] = pred_scaled
pred_log = scaler_chuva.inverse_transform(pred_dummy)[:, 0]
pred_final = np.expm1(pred_log)

y_test_final = np.expm1(y_test) # O y_test original já está na escala log

# Cálculo das métricas
mse = mean_squared_error(y_test_final, pred_final)
mae = mean_absolute_error(y_test_final, pred_final)
r2 = r2_score(y_test_final, pred_final)

logger.info("--- Métricas de Avaliação no Conjunto de Teste ---")
logger.info(f"MSE (Erro Quadrático Médio): {mse:.4f}")
logger.info(f"MAE (Erro Absoluto Médio): {mae:.4f}")
logger.info(f"R² (Coeficiente de Determinação): {r2:.4f}")
logger.info("-------------------------------------------------")


# --- 8. Visualização Final ---
logger.info("Gerando gráfico de previsão final.")
df_results = pd.DataFrame({'Real': y_test_final, 'Previsto': pred_final}, index=test_date)

plt.figure(figsize=(15, 7))
plt.plot(df_results.index, df_results['Real'], label='Valor Real', color='blue')
plt.plot(df_results.index, df_results['Previsto'], label='Previsão', color='red', alpha=0.7)
plt.title('Previsão de Chuva com Random Forest vs. Dados Reais')
plt.xlabel('Data')
plt.ylabel('Chuva (mm)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('previsao_final_rf.png')
plt.close()

logger.info("Script Random Forest (Scikit-learn) finalizado com sucesso.")