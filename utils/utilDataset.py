import numpy as np
import pandas as pd

def longest_run(arr_bool: np.ndarray) -> int:
    """Maior sequência consecutiva de True."""
    if arr_bool.size == 0:
        return 0
    b = arr_bool.astype(np.int8)
    # detecta inícios e fins de runs
    d = np.diff(np.concatenate(([0], b, [0])))
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    return 0 if starts.size == 0 else int(np.max(ends - starts))

def sum_top_quartile(arr: np.ndarray, wet_thr: float) -> float:
    """Soma da precipitação no quartil mais úmido (≥ Q3) entre dias úmidos."""
    x = arr[arr >= wet_thr]
    if x.size == 0:
        return 0.0
    q3 = np.quantile(x, 0.75)
    return float(x[x >= q3].sum())

def sum_bottom_quartile(arr: np.ndarray, wet_thr: float) -> float:
    """Soma da precipitação no quartil mais seco (≤ Q1) entre dias úmidos."""
    x = arr[arr >= wet_thr]
    if x.size == 0:
        return 0.0
    q1 = np.quantile(x, 0.25)
    return float(x[x <= q1].sum())

def criar_data_frame_chuva(df, chuva_col='chuva', tmax_col='tmax', tmin_col='tmin',
                               W=30, wet_thr=1.0):
    df['dia_seno']    = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['dia_cosseno'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    chuva = df[chuva_col].astype(float)
    wet = (chuva >= wet_thr).astype(int)

    roll = chuva.rolling(W, min_periods=W)  # janelas completas de W dias

    # PRCPTOT: soma da precipitação total na janela
    df['prcptot_w'] = roll.sum().fillna(0.0)

    # RX1DAY: máxima precipitação diária na janela
    df['rx1day_w'] = roll.max().fillna(0.0)

    # RX5DAYS: máximo da soma em 5 dias consecutivos dentro da janela W
    soma5 = chuva.rolling(5, min_periods=5).sum().fillna(0.0)
    df['rx5days_w'] = soma5.rolling(W, min_periods=W).max().fillna(0.0)

    # SDII: intensidade simples = precipitação nos dias úmidos / número de dias úmidos
    soma_wet = chuva.where(wet.astype(bool), 0).rolling(W, min_periods=W).sum()
    qtd_wet  = wet.rolling(W, min_periods=W).sum()
    df['sdii_w'] = (soma_wet / qtd_wet.replace(0, np.nan)).fillna(0)

    # R20mm: número de dias com precipitação >= 20mm na janela
    df['r20mm_w'] = (chuva >= 20.0).rolling(W, min_periods=W).sum().fillna(0)

    # WD / DD: contagem de dias úmidos e secos
    df['wd_w'] = wet.rolling(W, min_periods=W).sum().fillna(0)
    df['dd_w'] = ((wet == 0).astype(int)).rolling(W, min_periods=W).sum().fillna(0)

    # CWD / CDD: maior sequência consecutiva de dias úmidos / secos na janela
    df['cwd_w'] = chuva.rolling(W, min_periods=W).apply(
        lambda x: longest_run((x >= wet_thr)), raw=True
    ).fillna(0)
    df['cdd_w'] = chuva.rolling(W, min_periods=W).apply(
        lambda x: longest_run((x <  wet_thr)), raw=True
    ).fillna(0)

    # PRCWQ / PRCDQ: precipitação acumulada no quartil mais úmido / seco (entre dias úmidos)
    df['prcwq_w'] = chuva.rolling(W, min_periods=W).apply(
        lambda x: sum_top_quartile(x, wet_thr), raw=True
    ).fillna(0)
    df['prcdq_w'] = chuva.rolling(W, min_periods=W).apply(
        lambda x: sum_bottom_quartile(x, wet_thr), raw=True
    ).fillna(0)

    # --- Extras úteis com tmax/tmin ---
    tmax = df[tmax_col].astype(float)
    tmin = df[tmin_col].astype(float)
    tmean = (tmax + tmin) / 2.0
    dtr = (tmax - tmin)  # diurnal temperature range

    df['tmean_w_mean'] = tmean.rolling(W, min_periods=W).mean().fillna(0)
    df['tmean_w_std']  = tmean.rolling(W, min_periods=W).std().fillna(0)
    df['tmax_w_max']   = tmax.rolling(W, min_periods=W).max().fillna(0)
    df['tmin_w_min']   = tmin.rolling(W, min_periods=W).min().fillna(0)
    df['dtr_w_mean']   = dtr.rolling(W, min_periods=W).mean().fillna(0)
    df['dtr_w_std']    = dtr.rolling(W, min_periods=W).std().fillna(0)

    # Lags de chuva (bom pra RF): 1, 3, 7 dias
    df['chuva_lag1'] = chuva.shift(1).fillna(0)
    df['chuva_lag3'] = chuva.shift(3).fillna(0)
    df['chuva_lag7'] = chuva.shift(7).fillna(0)

    # Médias móveis de chuva (intensidade recente)
    df['chuva_ma3']  = chuva.rolling(3, min_periods=3).mean().fillna(0)
    df['chuva_ma7']  = chuva.rolling(7, min_periods=7).mean().fillna(0)
    df['chuva_ma14'] = chuva.rolling(14, min_periods=14).mean().fillna(0)
    df['chuva_ma30'] = chuva.rolling(30, min_periods=30).mean().fillna(0)

    # Evita vazamento: se for prever Y(t+1), shift no alvo e drop nas NAs
    #df['y_prox_dia'] = chuva.shift(-1).fillna(0.0)

    # Mantém também dia_seno/dia_cosseno originais (já são “futuros seguros”)
    colunas_normalizar = ['Tmax', 'Tmin',
        'prcptot_w','rx1day_w','rx5days_w','sdii_w','r20mm_w',
        'cwd_w','cdd_w','wd_w','dd_w','prcwq_w','prcdq_w',
        'tmean_w_mean','tmean_w_std','tmax_w_max','tmin_w_min','dtr_w_mean','dtr_w_std',
        'chuva_lag1','chuva_lag3','chuva_lag7','chuva_ma3','chuva_ma7','chuva_ma14','chuva_ma30'
        #'y_prox_dia'
    ]
   
    return df, colunas_normalizar

def criar_df_indices(df, chuva_col='chuva', tmax_col='tmax', tmin_col='tmin',
                          W=30, wet_thr=1.0, ref_period=None):
    df = df.copy()

    # Índice temporal
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns: df.set_index(pd.to_datetime(df['date']), inplace=True)
        else: raise ValueError("Forneça DateTimeIndex ou coluna 'date'.")
    df.sort_index(inplace=True)

    # Sazonalidade
    #df['dia_seno']    = np.sin(2*np.pi*df.index.dayofyear/365.0)
    #df['dia_cosseno'] = np.cos(2*np.pi*df.index.dayofyear/365.0)

    # Séries básicas
    p   = df[chuva_col].astype(float)
    tx  = df[tmax_col].astype(float)
    tn  = df[tmin_col].astype(float)
    wet = (p >= wet_thr).astype(int)
    dry = 1 - wet

    # Helpers (runs máximas)
    def _longest_run_bool(x_bool):
        m = cnt = 0
        for v in x_bool: cnt = cnt+1 if v else 0; m = max(m, cnt)
        return m

    # Rolagens (janelas completas)
    rollW = p.rolling(W, min_periods=W)

    # PRCPTOT, WD, DD, SDII
    df['prcptot_w'] = rollW.sum().fillna(0.0)
    df['wd_w']      = wet.rolling(W, min_periods=W).sum().fillna(0).astype(int)
    df['dd_w']      = dry.rolling(W, min_periods=W).sum().fillna(0).astype(int)
    df['sdii_w']    = (df['prcptot_w'] / df['wd_w'].replace(0, np.nan)).fillna(0.0)

    # RX1DAY, RX5DAYS
    df['rx1day_w']  = rollW.max().fillna(0.0)
    soma5 = p.rolling(5, min_periods=5).sum().fillna(0.0)
    df['rx5days_w'] = soma5.rolling(W, min_periods=W).max().fillna(0.0)

    # R20mm
    df['r20mm_w']   = (p >= 20.0).rolling(W, min_periods=W).sum().fillna(0).astype(int)

    # CWD/CDD (máx sequência em W)
    df['cwd_w'] = p.rolling(W, min_periods=W).apply(lambda x: _longest_run_bool(x >= wet_thr), raw=True).fillna(0).astype(int)
    df['cdd_w'] = p.rolling(W, min_periods=W).apply(lambda x: _longest_run_bool(x <  wet_thr), raw=True).fillna(0).astype(int)

    colunas_normalizar = [ 'chuva',
        'prcptot_w','wd_w','dd_w','sdii_w',
        'rx1day_w','rx5days_w','r20mm_w','cwd_w','cdd_w',
        'Tmax','Tmin'
    ]
    return df, colunas_normalizar

