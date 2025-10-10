
import pandas as pd 
import xarray as xr
import numpy as np
from datetime import datetime
import os

def acessar_dados_merge(caminho_base="/home/pbose/tcc/dataset/merge/"):
    # Abre o arquivo GRIB2 como lista de datasets
    arquivos = sorted([
        os.path.join(caminho_base, f)
        for f in os.listdir(caminho_base)
        if f.endswith(".grib2") and f.startswith("MERGE_CPTEC_")
    ])
    latRio_min, latRio_max = -23.05, -22.75
    lonRio_min, lonRio_max = -43.79, -43.10
    lonRio_min_convertido = 360 + lonRio_min  # ~316.2
    lonRio_max_convertido = 360 + lonRio_max  # ~316.9

    registros = []
    medias_rio = []
    datas = []
    #for arq in arquivos:
    for i in range(2000):
        try:
            data_str = os.path.basename(arquivos[i]).replace("MERGE_CPTEC_", "").replace(".grib2", "")
            data = datetime.strptime(data_str, "%Y%m%d").date()

            ds = xr.open_dataset(arquivos[i], engine="cfgrib", decode_timedelta=True)
            variavel = list(ds.data_vars.keys())[0]

            # Extrai valores e coordenadas
            dados = ds[variavel].values
            lats = ds.latitude.values
            lons = ds.longitude.values
            lon_grid, lat_grid = np.meshgrid(lons, lats)
      
            # Cria máscara para área do Rio
            mask = (
                (lat_grid >= latRio_min) & (lat_grid <= latRio_max) &
                (lon_grid >= lonRio_min_convertido) & (lon_grid <= lonRio_max_convertido)
            )
            # Aplica a máscara e calcula a média (ignora NaN)
            dados_rio = dados[mask]
            registros.append(dados_rio)
            media_rio = np.nanmean(dados_rio)

            medias_rio.append(media_rio)
            datas.append(data)

            ds.close()
            print(f"Lido {data}")
        except Exception as e:
            print(f"Erro ao ler {arquivos[i]}: {e}")
    
    df = pd.DataFrame({
        "data": pd.to_datetime(datas),
        "chuva": medias_rio
    }).set_index("data")
    
    #df = pd.DataFrame(registros, index=pd.to_datetime(datas))
    #df.index.name = "data"
 
    return df