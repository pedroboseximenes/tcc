import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pmdarima as pm
from access_br_dwgd import recuperar_dados_br_dwgd

# 1 - RECUPERANDO OS DADOS BR_DWGD
times_series_f_log = recuperar_dados_br_dwgd()
# --- 2. DIVISÃO EM TREINO/TESTE E TREINAMENTO DO MODELO SARIMA ---
# Dividir os dados (usaremos os dados transformados)
train_size = int(len(times_series_f_log) * 0.7)
train_data_log = times_series_f_log.iloc[0:train_size]
test_data_log = times_series_f_log.iloc[train_size:]

# Usar auto_arima para encontrar o melhor modelo SARIMA
print("\nIniciando a busca pelo melhor modelo SARIMA...")
auto_model = pm.auto_arima(train_data_log,
                           start_p=1, start_q=1,
                           max_p=5, max_q=5,
                           #m=5,  # Frequência da sazonalidade (7 dias)
                           d=False,
                           seasonal=True, 
                           start_P=0,
                           D=2,  # Geralmente D=1 é um bom ponto de partida para sazonalidade
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

print("\nMelhor modelo encontrado:")
print(auto_model.summary())

# --- 3. PREVISÃO E AVALIAÇÃO ---

# CORREÇÃO: Usar o modelo encontrado pelo auto_arima para prever
forecast_log, conf_int_log = auto_model.predict(n_periods=len(test_data_log),
                                                return_conf_int=True)

# MELHORIA: Reverter a transformação da previsão e do intervalo de confiança para a escala original
forecast_revertido = np.expm1(forecast_log)
conf_int_revertido = np.expm1(conf_int_log)

# Para calcular o erro, usamos os dados de teste originais (sem log)
original_test_data = times_series_f.iloc[train_size:]
rmse = np.sqrt(mean_squared_error(original_test_data, forecast_revertido))
print(f"\nRMSE (Root Mean Squared Error): {rmse:.4f}")

# --- 4. VISUALIZAÇÃO ---

# Preparar dados para o gráfico
predicted_mean = pd.Series(forecast_revertido, index=original_test_data.index)
conf_int_df = pd.DataFrame(conf_int_revertido, index=original_test_data.index, columns=['lower', 'upper'])

# Plotar os resultados
plt.figure(figsize=(15, 7))
# Usamos os dados originais (sem log) para o plot
plt.plot(times_series_f.iloc[0:train_size], label='Dados de Treino')
plt.plot(original_test_data, label='Dados Reais (Teste)', color='orange')
plt.plot(predicted_mean, label='Previsão SARIMA', color='green')
plt.fill_between(conf_int_df.index,
                 conf_int_df['lower'],
                 conf_int_df['upper'], color='k', alpha=.15, label='Intervalo de Confiança')

plt.title(f'Previsão do Modelo SARIMA vs. Dados Reais para {station_to_model}')
plt.xlabel('Data')
plt.ylabel('Precipitação')
plt.legend()
plt.grid(True)
plt.show()