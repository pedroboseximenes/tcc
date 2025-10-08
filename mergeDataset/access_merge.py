
import cfgrib
import xarray as xr
from typing import List, Optional, Union
import warnings
import os

def recuperar_dados():
    caminho_completo = os.path.join("/home/pbose/tcc/dataset/", "MERGE_CPTEC_20250101.grib2")
    # Abre o arquivo GRIB2 como lista de datasets
    datasets = cfgrib.open_datasets(caminho_completo)
    ds = datasets[0]
    latRio_min, latRio_max = -23.00, -22.82
    lonRio_min, lonRio_max = -43.60, -43.15
    
    print(f"Encontrados {len(datasets)} dataset(s)")
    
    # Imprime informações sobre cada dataset
    print(f"\nSelecionando área do Rio de Janeiro:")
    print(f"  Latitude: {latRio_min} a {latRio_max}")
    print(f"  Longitude: {lonRio_min} a {lonRio_max}")

    ds_rio = ds.sel(
        latitude=slice(latRio_max, latRio_min),  # slice invertido para lat (maior para menor)
        longitude=slice(lonRio_min, lonRio_max)
    )

    ds_ponto = ds.sel(latitude=latRio_min, longitude=lonRio_min, method='nearest')

recuperar_dados()