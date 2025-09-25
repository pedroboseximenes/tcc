import xarray as xr
import cfgrib
import os
import pandas as pd

def recuperar_dados_era_5():
    # Substitua 'caminho/do/seu/arquivo.grib' pelo caminho real
    caminho_completo = os.path.join("/home/pbose/tcc/dataset/", "era5NetCDF4.nc")
    ds = xr.open_dataset(caminho_completo, chunks={"valid_time": 100})
   
    # Seleciona a variável
    tp = ds["tp"]

    # Coordenadas RIO DE JANEIRO:
    lat_estacao = -22.9068   
    lon_estacao = -43.1729

    # Coordenadas NITEROI
    lat_estacaoNiteroi = -22.8832 
    lon_estacaoNiteroi = -43.1034

    # Seleciona o ponto mais próximo
    tp_estacao = tp.sel(latitude=lat_estacao, longitude=lon_estacao, method="nearest")

    # Agregar por dia 
    tp_diario = tp_estacao.resample(valid_time="1D").sum()
    #Convertendo para mm a unidade
    tp_diario_mm = tp_diario.fillna(0)* 1000
    # Converter para DataFrame
    df = tp_diario_mm.to_dataframe().reset_index()

    return df[['valid_time', 'tp']]
