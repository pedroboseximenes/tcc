import numpy as np
import pandas as pd
import os

def recuperar_dados_br_dwgd(isLstm):
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
    df_escolhido = df[station_to_model].fillna(0)

    if(isLstm):
        times_series_f = df[[station_to_model]].fillna(0).values 
        #times_series_f = np.log1p(times_series_f)
    else:
        times_series_f = df_escolhido
        times_series_f = times_series_f['2010-01-01':'2024-03-20']
        #times_series_f = np.log1p(times_series_f)

    #times_series_f = [time_series['2010-01-01':'2024-03-20']]
    print(f"Dados carregados com sucesso para a estação: {station_to_model}")
    print(f"Total de {len(times_series_f)} dias válidos no período selecionado.")
    return times_series_f
