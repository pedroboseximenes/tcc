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
    df_escolhido = df[station_to_model].dropna()

    if(isLstm):
        times_series_f = df[[station_to_model]].fillna(0).values 
        #times_series_f = np.log1p(times_series_f)
    else:
        times_series_f = df_escolhido['2010-03-10':'2024-03-20']
        #times_series_f = np.log1p(times_series_f)

    print(f"Dados carregados com sucesso para a estação: {station_to_model}")
    print(f"Total de {len(times_series_f)} dias válidos no período selecionado.")
    return times_series_f


def recuperar_dados_br_dwgd_com_area():
    caminho_completo = os.path.join("/home/pbose/tcc/dataset/", "pr.npz")
    #lat_min, lat_max = -22.92, -23.01
    #lon_min, lon_max = -43.37, -43.16
    lat_min, lat_max = -12.64, -12.25
    lon_min, lon_max = -38.96, -38.59
    days = pd.date_range("1961-01-01", "2024-03-20")
    
    npzfile = np.load(caminho_completo, allow_pickle=True)
    # Assume que as chaves 'data' e 'lat_lon_alt' existem
    var = npzfile['data']
    latlon_alt = npzfile['lat_lon_alt']
    id_station = npzfile['id']
    
    # Acessa as colunas de latitude e longitude (assumindo a ordem)
    latitudes = latlon_alt[:, 0]
    longitudes = latlon_alt[:, 1]
    # Cria a máscara booleana para encontrar as estações dentro da caixa de coordenadas
    lat_mask = (latitudes >= lat_min) & (latitudes <= lat_max)
    lon_mask = (longitudes >= lon_min) & (longitudes <= lon_max)
    
    # Combina as máscaras para obter as estações que satisfazem ambos os critérios
    combined_mask = lat_mask & lon_mask
    # Filtra o array 'var' usando a máscara.
    filtered_var = var[:, combined_mask]
    filtered_id_station = id_station[combined_mask]
    df = pd.DataFrame(data=filtered_var, index=days, columns=filtered_id_station)
    print(df.shape)

    return df