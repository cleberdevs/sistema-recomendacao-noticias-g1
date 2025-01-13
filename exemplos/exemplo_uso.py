import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.modelo.recomendador import RecomendadorHibrido
from src.config.logging_config import configurar_logging
import logging

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def exemplo_completo():
    """Exemplo completo de uso do sistema."""
    try:
        # Inicializar preprocessador
        preprocessador = PreProcessadorDadosSpark(
            memoria_executor="2g",
            memoria_driver="2g"
        )
        
        # Definir caminhos dos arquivos
        arquivos_treino = ['dados/brutos/treino_parte_1.csv']
        arquivos_itens = ['dados/brutos/itens/itens-parte1.csv']
        
        # Processar dados
        logger.info("Processando dados...")
        dados_treino, dados_itens = preprocessador.processar_dados_treino(
            arquivos_treino,
            arquivos_itens
        )
        
        # Mostrar informações
        preprocessador.mostrar_info_dados(dados_treino, dados_itens)
        
        # Validar dados
        if preprocessador.validar_dados(dados_treino, dados_itens):
            logger.info("Dados validados com sucesso")
        else:
            logger.warning("Problemas encontrados nos dados")
        
        # Criar e treinar modelo
        logger.info("Treinando modelo...")
        modelo = RecomendadorHibrido()
        historia_treino = modelo.treinar(dados_treino, dados_itens)
        
        # Fazer algumas previsões
        logger.info("Testando previsões...")
        usuarios_teste = dados_treino['idUsuario'].unique()[:3]
        
        for usuario in usuarios_teste:
            recomendacoes = modelo.prever(usuario, n_recomendacoes=5)
            logger.info(f"\nRecomendações para usuário {usuario}:")
            for i, rec in enumerate(recomendacoes, 1):
                titulo = dados_itens[dados_itens['Pagina'] == rec]['Titulo'].iloc[0]
                logger.info(f"{i}. {titulo}")
        
        # Salvar modelo
        logger.info("Salvando modelo...")
        modelo.salvar_modelo('modelos/modelos_salvos/exemplo_modelo')
        
        logger.info("Exemplo concluído com sucesso")
        
    except Exception as e:
        logger.error(f"Erro durante execução do exemplo: {str(e)}")
        raise
    finally:
        if 'preprocessador' in locals():
            preprocessador.__del__()

if __name__ == "__main__":
    exemplo_completo()