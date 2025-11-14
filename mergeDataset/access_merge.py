import pandas as pd 
import xarray as xr
import numpy as np
from datetime import datetime
import os

def acessar_dados_merge_lat_long(caminho_base="/home/pbose/tcc/dataset/merge/", logger=None, lat=-23.05 , long=-43.65):
    # Abre o arquivo GRIB2 como lista de datasets
    arquivos = sorted([
        os.path.join(caminho_base, f)
        for f in os.listdir(caminho_base)
        if f.endswith(".grib2") and f.startswith("MERGE_CPTEC_")
    ])
    # mask = (
    #     (lat_grid >= latRio_min) & (lat_grid <= latRio_max) &
    #     (lon_grid >= lonRio_min_convertido) & (lon_grid <= lonRio_max_convertido)
    # )

    chuva = []
    datas = []
    lon_conv =  360.0 + long

    for arq in arquivos:
    #for i in range(100):
        try:
            data_str = os.path.basename(arq).replace("MERGE_CPTEC_", "").replace(".grib2", "")
            data = datetime.strptime(data_str, "%Y%m%d").date()

            ds = xr.open_dataset(arq, engine="cfgrib", decode_timedelta=True)
            var = list(ds.data_vars)[0]
            da = ds[var].squeeze()
           
            # Extrai valores e coordenadas
            lat_name = "latitude" if "latitude" in da.coords else ("lat" if "lat" in da.coords else None)
            lon_name = "longitude" if "longitude" in da.coords else ("lon" if "lon" in da.coords else None)

            ponto = da.interp({lat_name: lat, lon_name: lon_conv}).squeeze(drop=True)
            valor = float(ponto.values)

            chuva.append(valor)
            datas.append(data)
            #logger.info("Lido dia " + data_str)
            ds.close()
            
        except Exception as e:
            print(f"Erro ao ler {arq}: {e}")
    
    df = pd.DataFrame({
        "data": pd.to_datetime(datas),
        "chuva": chuva
    }).set_index("data")
    
    #df = pd.DataFrame(registros, index=pd.to_datetime(datas))
    #df.index.name = "data"
 
    return df

def listar_pontos_merge(caminho_base="/home/pbose/tcc/dataset/merge/", converter_lon_180=True):
    # pega um arquivo qualquer só para ler a grade
    arq = sorted([
        os.path.join(caminho_base, f)
        for f in os.listdir(caminho_base)
        if f.endswith(".grib2") and f.startswith("MERGE_CPTEC_")
    ])[0]

    ds = xr.open_dataset(arq, engine="cfgrib")
    var = list(ds.data_vars.keys())[0]
    da = ds[var].squeeze()

    # Extrai coordenadas (normalmente 1D)
    lats = ds.latitude.values
    lons = ds.longitude.values

    # Converte lon 0..360 para -180..180 (só para visual/consulta)
    if converter_lon_180 and lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons)

    # Resolução média
    dlat = float(np.abs(np.diff(lats)).mean()) if lats.size > 1 else np.nan
    dlon = float(np.abs(np.diff(np.sort(lons))).mean()) if lons.size > 1 else np.nan

    # Cria lista de pontos (lat, lon) — cuidado: pode ficar grande
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    pontos = pd.DataFrame({
        "latitude": lat_grid.ravel(),
        "longitude": lon_grid.ravel()
    })

    latRio_min, latRio_max = -23.05, -22.75
    lonRio_min, lonRio_max = -43.79, -43.10

    pontos_rio = pontos[
        (pontos["latitude"] >= latRio_min) & (pontos["latitude"] <= latRio_max) &
        (pontos["longitude"] >= lonRio_min) & (pontos["longitude"] <= lonRio_max)
    ].reset_index(drop=True)

    print("Pontos no bbox do Rio:", len(pontos_rio))
    print(pontos_rio.head())

    ds.close()
    return pontos, dlat, dlon