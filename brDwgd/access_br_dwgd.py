import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

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
    df_escolhido = df[station_to_model].dropna()
    df_escolhido = df_escolhido['2010-03-10':'2024-03-20']

    print(f"Dados carregados com sucesso para a estação: {station_to_model}")
    print(f"Total de {len(df_escolhido)} dias válidos no período selecionado.")
    return df_escolhido


def recuperar_dados_br_dwgd_com_area():
    caminho_completo = os.path.join("/home/pbose/tcc/dataset/", "pr.npz")
    # Coordenadas RIO DE JANEIRO:
    latRio_min, latRio_max = -23.00, -22.82
    lonRio_min, lonRio_max = -43.60, -43.15

    # Coordenadas NITEROI
    latNit_min, latNit_max = -22.95, -22.8832
    lonNit_min, lonNit_max = -43.15, -43.1034

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
    lat_mask = (latitudes >= latRio_min) & (latitudes <= latRio_max)
    lon_mask = (longitudes >= lonRio_min) & (longitudes <= lonRio_max)
    
    # Combina as máscaras para obter as estações que satisfazem ambos os critérios
    combined_mask = lat_mask & lon_mask
    # Filtra o array 'var' usando a máscara.
    filtered_latlon = latlon_alt[combined_mask]
    filtered_var = var[:, combined_mask]
    filtered_id_station = id_station[combined_mask]
   
    df = pd.DataFrame(data=filtered_var, index=days, columns=filtered_id_station)
    
    df = df.fillna(0)
    contagem_valores = (df != 0).sum()

    # Pegar o ID da estação com mais valores
    melhor_estacao_id = contagem_valores.idxmax()
   
    # Selecionar essa coluna
    df = df[melhor_estacao_id]
    df = df['1961-01-01': '2000-12-01']
    #df = df[filtered_id_station[50]]
    #coordenadas_da_estacao = filtered_latlon[50]

    return df