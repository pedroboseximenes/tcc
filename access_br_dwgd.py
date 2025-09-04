import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def recuperar_dados_br_dwgd():
    # Carregar o arquivo .npz
    # Substitua pelo seu caminho se for diferente
    caminho_completo = os.path.join("/home/pbose/tcc/dataset/", "pr.npz")

    npzfile = np.load(caminho_completo, allow_pickle=True)
    var = npzfile['data']
    id_station = npzfile['id']
    days = pd.date_range("1961-01-01", "2024-03-20")
    df = pd.DataFrame(data=var, index=days, columns=id_station)

    # --- 1. SELEÇÃO E PREPARAÇÃO DOS DADOS ---
    # Encontrar a estação com a maior quantidade de dados válidos
    contagem_valores = df.count()
    melhor_estacao_id = contagem_valores.idxmax()

    station_to_model = melhor_estacao_id
    times_series_f = df[[station_to_model]].dropna().values

    #times_series_f = [time_series['2010-01-01':'2024-03-20']]
    #plt.plot(times_series_f)
    #plt.show()
    #times_series_f_log = np.log1p(times_series_f)
    #plt.plot(times_series_f_log)
    #plt.show()
    print(f"Dados carregados com sucesso para a estação: {station_to_model}")
    print(f"Total de {len(times_series_f)} dias válidos no período selecionado.")
    return times_series_f
