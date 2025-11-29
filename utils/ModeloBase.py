from enum import Enum
from abc import ABC, abstractmethod

from utils.logger import Logger

class ModeloBase(ABC):
    def __init__(self,
                nome_modelo="MODELO_NULLO" ,
                timeseries=None,
                colunas_normalizar=None, 
                scaler= None, 
                num_test=30, 
                lookback=30,
                batch_size=32,
                epocas=200,
                base_dados="",
                ts_scaled_df=None,
                config=None,
                config_registrar_resultado="Padrão"
                ):
        self.nome_modelo = nome_modelo
        self.timeseries = timeseries
        self.colunas_normalizar=colunas_normalizar
        self.scaler = scaler
        self.num_test = num_test
        self.lookback = lookback
        self.base_dados = base_dados
        self.ts_scaled_df = ts_scaled_df
        self.config = config
        self.config_registrar_resultado = config_registrar_resultado
        self.batch_size = batch_size
        self.epocas = epocas


        self.logger = Logger.configurar_logger(
            nome_arquivo=f"{nome_modelo}_{base_dados}.log",
            nome_classe=f"{nome_modelo}_{base_dados}"
        )

    @abstractmethod
    def run(self, index):
        pass