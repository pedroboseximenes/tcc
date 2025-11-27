import numpy as np
import pandas as pd
def criar_csv(logger, df_bruto, titulo):
    colunas_metricas = ['MSE_TRAIN','RMSE_TRAIN','MAE_TRAIN','CSI_TRAIN','MSE', 'RMSE', 'MAE', 'CSI', 'Tempo_treinamento']
    df_final = df_bruto.groupby(['Modelo', 'Configuracao'])[colunas_metricas].agg(['mean', 'std'])

    # 3. Ajustar nomes das colunas (Ex: MSE_mean, MSE_std)
    df_final.columns = ['_'.join(col).strip() for col in df_final.columns.values]
    df_final = df_final.reset_index()

    # 4. Salvar para CSV
    nome_arquivo = f'../results/{titulo}/metricas/resultados_{titulo}.csv'
    df_final.to_csv(nome_arquivo, index=False, sep=';', decimal=',',     float_format='%.3f')

    logger.info(f"Arquivo '{nome_arquivo}' gerado com sucesso!")

def criar_data_frame_chuva_br_dwgd(df, chuva_col='chuva', tmax_col='tmax', tmin_col='tmin',
                               W=30, wet_thr=1.0, janela=30):
    dia = df.index.dayofyear.values
    df['day_sin'] = np.sin(2 * np.pi * dia / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * dia / 365.25)

 
    pr = df['chuva'].astype(float)
    wet = (pr >= wet_thr)
    df["SDII_30d"] = (
        pr.where(wet)
          .rolling(window=janela, min_periods=1)
          .mean()
    ).fillna(0)

    # R10mm_30d e R20mm_30d: contagem em janela
    df["R10mm_30d"] = (pr > 10.0).rolling(window=janela, min_periods=1).sum()
    df["R20mm_30d"] = (pr > 20.0).rolling(window=janela, min_periods=1).sum()

    # Rx1day_30d: máximo da janela (na prática é o rolling max da chuva)
    df["Rx1day_30d"] = pr.rolling(window=janela, min_periods=1).max()

    # Rx5day_30d: máximo da soma em 5 dias dentro da janela dos 30 dias
    soma5 = pr.rolling(window=5, min_periods=1).sum()
    df["Rx5day_30d"] = soma5.rolling(window=janela, min_periods=1).max()

    # PRCPTOT_30d: total de chuva em dias chuvosos na janela
    df["PRCPTOT_30d"] = pr.where(wet, 0.0).rolling(window=janela, min_periods=1).sum()

    df["Tmed"] = df[["Tmax", "Tmin"]].mean(axis=1)
    df = df.drop(columns=[tmax_col, tmin_col])
    df.fillna(0)
    colunas_normalizar = [
        'Tmed',
        'chuva', 'SDII_30d', 'R10mm_30d', 'R20mm_30d', 'Rx1day_30d', 'Rx5day_30d',
        'PRCPTOT_30d',
        'RH', 'Rs', 'u2'
    ]
   
    return df, colunas_normalizar
