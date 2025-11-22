import numpy as np
import pandas as pd

def criar_data_frame_chuva_br_dwgd(df, chuva_col='chuva', tmax_col='tmax', tmin_col='tmin',
                               W=30, wet_thr=1.0):
    df["ano"] = df.index.year
    df["mes"] = df.index.month
    df["dia"] = df.index.day
    chuva = df[chuva_col].astype(float)
    wet = (chuva >= wet_thr).astype(int)


    df["Tmed"] = df[["Tmax", "Tmin"]].mean(axis=1)
    df
    colunas_normalizar = [
        'Tmed'
        'ano', 'mes', 'dia',
    ]
   
    return df, colunas_normalizar
