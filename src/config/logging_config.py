import logging
import os
from datetime import datetime
import sys

def configurar_logging():
    """
    Configura o sistema de logging da aplicação com suporte a Spark.
    """
    # Criar diretório de logs se não existir
    os.makedirs('logs', exist_ok=True)

    # Configurar nível de log baseado em variável de ambiente
    nivel_log = os.getenv('LOG_LEVEL', 'INFO').upper()

    # Configurar formato do log
    formato_log = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Arquivo de log com data
    arquivo_log = f'logs/aplicacao_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.FileHandler(arquivo_log)
    file_handler.setFormatter(formato_log)

    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formato_log)

    # Configurar logger root
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, nivel_log))
    
    # Remover handlers existentes
    logger.handlers = []
    
    # Adicionar novos handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Configurar loggers específicos
    loggers_externos = {
        'tensorflow': logging.WARNING,
        'mlflow': logging.WARNING,
        'py4j': logging.WARNING,
        'urllib3': logging.WARNING,
        'pyspark': logging.WARNING,
        'sparkmonitor': logging.WARNING
    }

    for logger_name, nivel in loggers_externos.items():
        logging.getLogger(logger_name).setLevel(nivel)

    logger.info(f"Logging configurado com nível: {nivel_log}")
    logger.info(f"Logs sendo salvos em: {arquivo_log}")

    return logger

def get_logger(nome_modulo: str) -> logging.Logger:
    """
    Retorna um logger configurado para um módulo específico.
    
    Args:
        nome_modulo: Nome do módulo que está solicitando o logger
        
    Returns:
        Logger configurado para o módulo
    """
    return logging.getLogger(nome_modulo)