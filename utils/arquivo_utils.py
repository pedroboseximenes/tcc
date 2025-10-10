from datetime import datetime
import logging
import os

import torch


class ArquivoUtils:
    @staticmethod
    def salvar_no_arquivo(nome_arquivo: str, conteudo: str):
        with open(nome_arquivo, "a", encoding="utf-8") as arquivo_saida:
            arquivo_saida.write(conteudo)

    @staticmethod
    def salvar_modelo(args, estado_modelo, estado_otimizador):
        try:
            logging.info(f"Salvando modelo executado")
            nome_arquivo = f"{args['modelo_base']}_{args['data_execucao']}.pth"

            caminho_salvamento = "checkpoints"
            os.makedirs(caminho_salvamento, exist_ok=True)

            caminho_completo = os.path.join(caminho_salvamento, nome_arquivo)

            estado = {
                'epoca': args['num_epocas'],
                'estado_modelo': estado_modelo,
                'estado_otimizador': estado_otimizador,
                'args': args,
                'data_execucao_utc': args['data_execucao']
            }

            torch.save(estado, caminho_completo)
            logging.info(f"Modelo salvo com sucesso em: {caminho_completo}")
            return caminho_completo

        except Exception as e:
            logging.error(f"Erro ao salvar o modelo: {e}")
            return None
