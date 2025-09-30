import logging
from pathlib import Path

class Logger:
    @staticmethod
    # logger_config.py
    def configurar_logger(nome_arquivo="application.log", nome_classe ="main.py", nivel=logging.DEBUG):
        caminho_arquivo = Path(f"../logs/{nome_arquivo}")
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.WARNING)
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(caminho_arquivo, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(nome_classe)
        logger.setLevel(nivel)
        return logger
