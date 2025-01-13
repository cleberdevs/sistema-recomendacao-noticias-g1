import os
import logging
from src.config.logging_config import configurar_logging
from src.api.app import app

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def iniciar_servidor():
    """
    Inicia o servidor da API com as configurações apropriadas.
    """
    try:
        # Carregar configurações
        port = int(os.getenv('PORT', 8000))
        host = os.getenv('HOST', '0.0.0.0')
        debug = os.getenv('FLASK_ENV') == 'desenvolvimento'
        
        logger.info(f"Iniciando servidor em {host}:{port}")
        logger.info(f"Modo debug: {debug}")
        
        # Iniciar servidor
        app.run(
            host=host,
            port=port,
            debug=debug
        )
        
    except Exception as e:
        logger.error(f"Erro ao iniciar servidor: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        iniciar_servidor()
    except Exception as e:
        logger.error(f"Erro fatal ao executar servidor: {str(e)}")
        raise