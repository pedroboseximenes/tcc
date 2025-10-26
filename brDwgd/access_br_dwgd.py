import numpy as np
import pandas as pd
import os

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
    df = df[melhor_estacao_id].to_frame(name='chuva')
    
    df = coletar_outras_informacoes_com_id("Tmax.npz", melhor_estacao_id, "Tmax", df)
    df = coletar_outras_informacoes_com_id("Tmin.npz", melhor_estacao_id, "Tmin", df)
    #df = coletar_outras_informacoes_com_id("RH.npz", melhor_estacao_id, "RH", df)
    #df = coletar_outras_informacoes_com_id("Rs.npz", melhor_estacao_id, "Rs", df)
    #df = coletar_outras_informacoes_com_id("u2.npz", melhor_estacao_id, "u2", df)
    df = df['1961-01-01': '2000-12-01']
    #df = df[filtered_id_station[50]]
    #coordenadas_da_estacao = filtered_latlon[50]

    return df

def coletar_outras_informacoes_com_id(arquivo, id_estacao, nome_coluna, df):
    caminho_completo = os.path.join("/home/pbose/tcc/dataset/", arquivo)
    npzfile = np.load(caminho_completo, allow_pickle=True)
    var = npzfile['data']
    latlon_alt = npzfile['lat_lon_alt']
    id_station = npzfile['id']
    days = pd.date_range("1961-01-01", "2024-03-20")    

    dfProvisorio = pd.DataFrame(data=var, index=days, columns=id_station).fillna(0)

    #print(df)
    df[nome_coluna] = dfProvisorio[id_estacao]
    return df


import xarray as xr

def abrir_empilhar(var_prefix, pasta="/home/pbose/tcc/dataset/"):
    ds = xr.open_mfdataset(
        f"{pasta}/{var_prefix}_Control_*.nc",
        combine="by_coords", parallel=True, chunks={"time": 365}
    )
    print(ds)
    # padroniza nomes comuns
    ren = {}
    if "latitude" in ds.coords: ren["latitude"] = "lat"
    if "longitude" in ds.coords: ren["longitude"] = "lon"
    if "valid_time" in ds.coords: ren["valid_time"] = "time"
    ds = ds.rename(ren)

    # identifica a variável principal
    var = var_prefix if var_prefix in ds.data_vars else list(ds.data_vars)[0]
    return ds, var

def abrindo_NETCDF(est_id):
    # Exemplo: abrir u2 e Rs
    ds_u2, var_u2 = abrir_empilhar("u2")
    ds_rs, var_rs = abrir_empilhar("Rs")

    if "station" in ds_u2.dims or "station" in ds_u2.coords:
        serie_u2 = ds_u2[var_u2].sel(station=str(est_id), drop=True)
    elif "station_id" in ds_u2:
        ds_u2 = ds_u2.assign_coords(station=ds_u2["station_id"].astype(str))
        serie_u2 = ds_u2[var_u2].sel(station=str(est_id), drop=True)

    if "station" in ds_rs.dims or "station" in ds_rs.coords:
        serie_rs = ds_rs[var_rs].sel(station=str(est_id), drop=True)
    elif "station_id" in ds_rs:
        ds_rs = ds_rs.assign_coords(station=ds_rs["station_id"].astype(str))
        serie_rs = ds_rs[var_rs].sel(station=str(est_id), drop=True)

    # Alinha no tempo e vira DataFrame
    df = xr.merge(
        [serie_u2.rename("u2").to_dataset(), serie_rs.rename("Rs").to_dataset()]
    ).to_dataframe().sort_index()
    return df
