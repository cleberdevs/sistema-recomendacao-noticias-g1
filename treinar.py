'''import os
import logging
from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento import PreProcessadorDados
from src.utils.helpers import carregar_arquivos_dados, criar_diretorio_se_nao_existe
from src.config.mlflow_config import MLflowConfig
from src.config.logging_config import configurar_logging

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def treinar_modelo():
    """
    Função principal para treinamento do modelo.
    Gerencia todo o processo de treinamento, incluindo:
    - Carregamento de dados
    - Preprocessamento
    - Treinamento
    - Salvamento do modelo
    - Tracking com MLflow
    """
    logger.info("Iniciando processo de treinamento")
    
    try:
        # Configurar MLflow
        mlflow_config = MLflowConfig()
        mlflow_config.setup_mlflow()
        
        with mlflow_config.iniciar_run(
            run_name="treinamento_completo",
            tags={"ambiente": os.getenv("ENVIRONMENT", "desenvolvimento")}
        ):
            # Configurar diretórios
            logger.info("Criando diretórios necessários")
            criar_diretorio_se_nao_existe('modelos/modelos_salvos')
            
            # Carregar dados
            logger.info("Carregando arquivos de dados")
            arquivos_treino = carregar_arquivos_dados('dados/brutos/treino_parte*.csv')
            arquivos_itens = carregar_arquivos_dados('dados/brutos/itens/itens-parte*.csv')
            
            if not arquivos_treino or not arquivos_itens:
                erro_msg = "Arquivos de dados não encontrados!"
                logger.error(erro_msg)
                raise ValueError(erro_msg)
            
            # Preprocessar dados
            logger.info("Iniciando preprocessamento dos dados")
            preprocessador = PreProcessadorDados()
            dados_treino, dados_itens = preprocessador.processar_dados_treino(
                arquivos_treino,
                arquivos_itens
            )
            
            # Validar dados
            logger.info("Validando dados")
            if not preprocessador.validar_dados(dados_treino, dados_itens):
                logger.warning("Dados contêm valores nulos ou inconsistências")
            
            # Preparar features de texto
            logger.info("Preparando features de texto")
            dados_itens = preprocessador.preparar_features_texto(dados_itens)
            
            # Mostrar informações dos dados
            preprocessador.mostrar_info_dados(dados_treino, dados_itens)
            
            # Criar e treinar modelo
            logger.info("Iniciando treinamento do modelo")
            modelo = RecomendadorHibrido()
            historia_treino = modelo.treinar(dados_treino, dados_itens)
            
            # Registrar métricas finais
            metricas_finais = {
                "acuracia_final": historia_treino.history['accuracy'][-1],
                "loss_final": historia_treino.history['loss'][-1],
                "val_accuracy_final": historia_treino.history['val_accuracy'][-1],
                "val_loss_final": historia_treino.history['val_loss'][-1]
            }
            mlflow_config.log_metricas(metricas_finais)
            
            # Salvar modelo
            logger.info("Salvando modelo treinado")
            caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
            modelo.salvar_modelo(caminho_modelo)
            
            logger.info("Treinamento concluído com sucesso")
            
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        if mlflow_config:
            mlflow_config.finalizar_run(status="FAILED")
        raise
        
    finally:
        if mlflow_config:
            mlflow_config.finalizar_run()

if __name__ == "__main__":
    try:
        treinar_modelo()
    except Exception as e:
        logger.error(f"Erro fatal durante execução: {str(e)}")
        raise'''

'''import os
import logging
from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento import PreProcessadorDados
from src.utils.helpers import carregar_arquivos_dados, criar_diretorio_se_nao_existe
from src.config.mlflow_config import MLflowConfig
from src.config.logging_config import configurar_logging

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def configurar_preprocessador(memoria_limite_mb=512):
    """
    Configura o preprocessador com configurações conservadoras de memória.
    """
    preprocessador = PreProcessadorDados()
    preprocessador.chunk_size = 2500  # Tamanho reduzido do chunk
    preprocessador.chunk_size_texto = 25  # Tamanho reduzido para processamento de texto
    preprocessador.limite_memoria_mb = memoria_limite_mb
    preprocessador.grupo_size_checkpoints = 2  # Número reduzido de checkpoints por grupo
    return preprocessador

def treinar_modelo():
    """
    Função principal para treinamento do modelo.
    Gerencia todo o processo de treinamento, incluindo:
    - Carregamento de dados
    - Preprocessamento
    - Treinamento
    - Salvamento do modelo
    - Tracking com MLflow
    """
    logger.info("Iniciando processo de treinamento")
    mlflow_config = None
    
    try:
        # Configurar MLflow
        mlflow_config = MLflowConfig()
        mlflow_config.setup_mlflow()
        
        with mlflow_config.iniciar_run(
            run_name="treinamento_completo",
            tags={"ambiente": os.getenv("ENVIRONMENT", "desenvolvimento")}
        ):
            # Configurar diretórios
            logger.info("Criando diretórios necessários")
            criar_diretorio_se_nao_existe('modelos/modelos_salvos')
            criar_diretorio_se_nao_existe('dados/processados')
            
            # Carregar dados
            logger.info("Carregando arquivos de dados")
            arquivos_treino = carregar_arquivos_dados('dados/brutos/treino_parte*.csv')
            arquivos_itens = carregar_arquivos_dados('dados/brutos/itens/itens-parte*.csv')
            
            logger.info(f"Arquivos de treino encontrados: {arquivos_treino}")
            logger.info(f"Arquivos de itens encontrados: {arquivos_itens}")
            
            if not arquivos_treino or not arquivos_itens:
                erro_msg = "Arquivos de dados não encontrados!"
                logger.error(erro_msg)
                raise ValueError(erro_msg)
            
            # Tentar preprocessamento com diferentes configurações de memória
            try:
                logger.info("Tentando preprocessamento com configurações iniciais")
                preprocessador = configurar_preprocessador(memoria_limite_mb=512)
                dados_treino, dados_itens = preprocessador.processar_dados_treino(
                    arquivos_treino,
                    arquivos_itens
                )
            except MemoryError:
                logger.warning("Erro de memória com configuração inicial, tentando com limite menor")
                preprocessador = configurar_preprocessador(memoria_limite_mb=256)
                dados_treino, dados_itens = preprocessador.processar_dados_treino(
                    arquivos_treino,
                    arquivos_itens
                )
            
            # Registrar configurações de preprocessamento no MLflow
            mlflow_config.log_parametros({
                "chunk_size": preprocessador.chunk_size,
                "chunk_size_texto": preprocessador.chunk_size_texto,
                "limite_memoria_mb": preprocessador.limite_memoria_mb,
                "grupo_size_checkpoints": preprocessador.grupo_size_checkpoints
            })
            
            # Validar dados
            logger.info("Validando dados")
            if not preprocessador.validar_dados(dados_treino, dados_itens):
                logger.warning("Dados contêm valores nulos ou inconsistências")
            
            # Preparar features de texto
            logger.info("Preparando features de texto")
            try:
                dados_itens = preprocessador.preparar_features_texto(dados_itens)
            except MemoryError:
                logger.warning("Erro de memória no processamento de texto, reduzindo chunk size")
                preprocessador.chunk_size_texto = 10
                dados_itens = preprocessador.preparar_features_texto(dados_itens)
            
            # Mostrar informações dos dados
            preprocessador.mostrar_info_dados(dados_treino, dados_itens)
            
            # Registrar métricas dos dados
            mlflow_config.log_parametros({
                "num_usuarios": dados_treino['idUsuario'].nunique(),
                "num_itens": len(dados_itens),
                "tamanho_dados_treino": len(dados_treino)
            })
            
            # Criar e treinar modelo
            logger.info("Iniciando treinamento do modelo")
            modelo = RecomendadorHibrido()
            historia_treino = modelo.treinar(dados_treino, dados_itens)
            
            # Registrar métricas finais
            metricas_finais = {
                "acuracia_final": historia_treino.history['accuracy'][-1],
                "loss_final": historia_treino.history['loss'][-1],
                "val_accuracy_final": historia_treino.history.get('val_accuracy', [0])[-1],
                "val_loss_final": historia_treino.history.get('val_loss', [0])[-1]
            }
            mlflow_config.log_metricas(metricas_finais)
            
            # Salvar modelo
            logger.info("Salvando modelo treinado")
            caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
            modelo.salvar_modelo(caminho_modelo)
            mlflow_config.log_artefato(caminho_modelo)
            
            logger.info("Treinamento concluído com sucesso")
            return modelo
            
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        if mlflow_config:
            mlflow_config.finalizar_run(status="FAILED")
        raise
        
    finally:
        if mlflow_config:
            mlflow_config.finalizar_run()

if __name__ == "__main__":
    try:
        modelo = treinar_modelo()
    except Exception as e:
        logger.error(f"Erro fatal durante execução: {str(e)}")
        raise'''

'''import os
import logging
from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.utils.helpers import carregar_arquivos_dados, criar_diretorio_se_nao_existe
from src.config.mlflow_config import MLflowConfig
from src.config.logging_config import configurar_logging

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def configurar_preprocessador(memoria_executor="4g", memoria_driver="4g"):
    """
    Configura o preprocessador Spark com as configurações especificadas.
    
    Args:
        memoria_executor: Quantidade de memória para executores Spark
        memoria_driver: Quantidade de memória para o driver Spark
    
    Returns:
        PreProcessadorDadosSpark configurado
    """
    try:
        return PreProcessadorDadosSpark(
            memoria_executor=memoria_executor,
            memoria_driver=memoria_driver
        )
    except Exception as e:
        logger.error(f"Erro ao configurar preprocessador: {str(e)}")
        raise

def treinar_modelo():
    """
    Função principal para treinamento do modelo.
    Gerencia todo o processo de treinamento, incluindo:
    - Carregamento de dados com Spark
    - Preprocessamento distribuído
    - Treinamento
    - Salvamento do modelo
    - Tracking com MLflow
    """
    logger.info("Iniciando processo de treinamento")
    mlflow_config = None
    preprocessador = None
    
    try:
        # Configurar MLflow
        mlflow_config = MLflowConfig()
        mlflow_config.setup_mlflow()
        
        with mlflow_config.iniciar_run(
            run_name="treinamento_completo",
            tags={
                "ambiente": os.getenv("ENVIRONMENT", "desenvolvimento"),
                "processamento": "spark"
            }
        ):
            # Configurar diretórios
            logger.info("Criando diretórios necessários")
            criar_diretorio_se_nao_existe('modelos/modelos_salvos')
            criar_diretorio_se_nao_existe('dados/processados')
            
            # Carregar dados
            logger.info("Carregando arquivos de dados")
            arquivos_treino = carregar_arquivos_dados('dados/brutos/treino_parte*.csv')
            arquivos_itens = carregar_arquivos_dados('dados/brutos/itens/itens-parte*.csv')
            
            logger.info(f"Arquivos de treino encontrados: {arquivos_treino}")
            logger.info(f"Arquivos de itens encontrados: {arquivos_itens}")
            
            if not arquivos_treino or not arquivos_itens:
                erro_msg = "Arquivos de dados não encontrados!"
                logger.error(erro_msg)
                raise ValueError(erro_msg)
            
            # Tentar processamento com diferentes configurações de memória
            try:
                logger.info("Tentando processamento com configurações iniciais")
                preprocessador = configurar_preprocessador(
                    memoria_executor="4g",
                    memoria_driver="4g"
                )
                dados_treino, dados_itens = preprocessador.processar_dados_treino(
                    arquivos_treino,
                    arquivos_itens
                )
            except Exception as e:
                logger.warning(f"Erro com configuração inicial: {str(e)}")
                logger.info("Tentando com configurações mais conservadoras")
                preprocessador = configurar_preprocessador(
                    memoria_executor="2g",
                    memoria_driver="2g"
                )
                dados_treino, dados_itens = preprocessador.processar_dados_treino(
                    arquivos_treino,
                    arquivos_itens
                )
            
            # Registrar configurações no MLflow
            mlflow_config.log_parametros({
                "memoria_executor": preprocessador.spark.conf.get("spark.executor.memory"),
                "memoria_driver": preprocessador.spark.conf.get("spark.driver.memory"),
                "num_particoes": preprocessador.spark.conf.get("spark.sql.shuffle.partitions")
            })
            
            # Validar dados
            logger.info("Validando dados")
            if not preprocessador.validar_dados(dados_treino, dados_itens):
                logger.warning("Dados contêm valores nulos ou inconsistências")
            
            # Mostrar informações dos dados
            preprocessador.mostrar_info_dados(dados_treino, dados_itens)
            
            # Registrar métricas dos dados
            mlflow_config.log_parametros({
                "num_usuarios": dados_treino['idUsuario'].nunique(),
                "num_itens": len(dados_itens),
                "tamanho_dados_treino": len(dados_treino)
            })
            
            # Criar e treinar modelo
            logger.info("Iniciando treinamento do modelo")
            modelo = RecomendadorHibrido()
            historia_treino = modelo.treinar(dados_treino, dados_itens)
            
            # Registrar métricas finais
            metricas_finais = {
                "acuracia_final": historia_treino.history['accuracy'][-1],
                "loss_final": historia_treino.history['loss'][-1],
                "val_accuracy_final": historia_treino.history.get('val_accuracy', [0])[-1],
                "val_loss_final": historia_treino.history.get('val_loss', [0])[-1]
            }
            mlflow_config.log_metricas(metricas_finais)
            
            # Salvar modelo
            logger.info("Salvando modelo treinado")
            caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
            modelo.salvar_modelo(caminho_modelo)
            mlflow_config.log_artefato(caminho_modelo)
            
            logger.info("Treinamento concluído com sucesso")
            return modelo
            
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        if mlflow_config:
            mlflow_config.finalizar_run(status="FAILED")
        raise
        
    finally:
        # Limpar recursos
        if preprocessador:
            try:
                preprocessador.__del__()
            except:
                pass
            
        if mlflow_config:
            mlflow_config.finalizar_run()

if __name__ == "__main__":
    try:
        modelo = treinar_modelo()
    except Exception as e:
        logger.error(f"Erro fatal durante execução: {str(e)}")
        raise'''
'''import os
import logging
from pyspark.sql import SparkSession
from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.utils.helpers import carregar_arquivos_dados, criar_diretorio_se_nao_existe
from src.config.mlflow_config import MLflowConfig
from src.config.logging_config import configurar_logging

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def configurar_spark(app_name="RecomendadorNoticias", master="local[*]", memoria_executor="4g", memoria_driver="4g"):
    """
    Configura e retorna uma sessão Spark.
    
    Args:
        app_name: Nome da aplicação Spark.
        master: Endereço do Spark Master (local[*] para modo local, spark://<endereço> para cluster).
        memoria_executor: Memória alocada para cada executor.
        memoria_driver: Memória alocada para o driver.
    
    Returns:
        SparkSession configurada.
    """
    try:
        spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.executor.memory", memoria_executor) \
            .config("spark.driver.memory", memoria_driver) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        logger.info("Sessão Spark inicializada")
        return spark
    except Exception as e:
        logger.error(f"Erro ao configurar Spark: {str(e)}")
        raise

def treinar_modelo():
    """
    Função principal para treinamento do modelo.
    Gerencia todo o processo de treinamento, incluindo:
    - Carregamento de dados com PySpark
    - Preprocessamento distribuído
    - Treinamento
    - Salvamento do modelo
    - Tracking com MLflow
    """
    logger.info("Iniciando processo de treinamento")
    mlflow_config = None
    spark = None
    
    try:
        # Configurar MLflow
        mlflow_config = MLflowConfig()
        mlflow_config.setup_mlflow()
        
        with mlflow_config.iniciar_run(
            run_name="treinamento_completo",
            tags={
                "ambiente": os.getenv("ENVIRONMENT", "desenvolvimento"),
                "processamento": "spark"
            }
        ):
            # Configurar diretórios
            logger.info("Criando diretórios necessários")
            criar_diretorio_se_nao_existe('modelos/modelos_salvos')
            criar_diretorio_se_nao_existe('dados/processados')
            
            # Inicializar Spark
            spark = configurar_spark(
                app_name="RecomendadorNoticias",
                master="local[*]",  # Modo local (ou use "spark://<endereço>" para cluster)
                memoria_executor="4g",
                memoria_driver="4g"
            )
            
            # Carregar dados
            logger.info("Carregando arquivos de dados")
            arquivos_treino = carregar_arquivos_dados('dados/brutos/treino_parte*.csv')
            arquivos_itens = carregar_arquivos_dados('dados/brutos/itens/itens-parte*.csv')
            
            logger.info(f"Arquivos de treino encontrados: {arquivos_treino}")
            logger.info(f"Arquivos de itens encontrados: {arquivos_itens}")
            
            if not arquivos_treino or not arquivos_itens:
                erro_msg = "Arquivos de dados não encontrados!"
                logger.error(erro_msg)
                raise ValueError(erro_msg)
            
            # Processar dados com PySpark
            preprocessador = PreProcessadorDadosSpark(spark)
            dados_treino, dados_itens = preprocessador.processar_dados_treino(
                arquivos_treino,
                arquivos_itens
            )
            
            # Validar dados
            logger.info("Validando dados")
            if not preprocessador.validar_dados(dados_treino, dados_itens):
                logger.warning("Dados contêm valores nulos ou inconsistências")
            
            # Mostrar informações dos dados
            preprocessador.mostrar_info_dados(dados_treino, dados_itens)
            
            # Registrar métricas dos dados
            mlflow_config.log_parametros({
                "num_usuarios": dados_treino['idUsuario'].nunique(),
                "num_itens": len(dados_itens),
                "tamanho_dados_treino": len(dados_treino)
            })
            
            # Criar e treinar modelo
            logger.info("Iniciando treinamento do modelo")
            modelo = RecomendadorHibrido()
            historia_treino = modelo.treinar(dados_treino, dados_itens)
            
            # Registrar métricas finais
            metricas_finais = {
                "acuracia_final": historia_treino.history['accuracy'][-1],
                "loss_final": historia_treino.history['loss'][-1],
                "val_accuracy_final": historia_treino.history.get('val_accuracy', [0])[-1],
                "val_loss_final": historia_treino.history.get('val_loss', [0])[-1]
            }
            mlflow_config.log_metricas(metricas_finais)
            
            # Salvar modelo
            logger.info("Salvando modelo treinado")
            caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
            modelo.salvar_modelo(caminho_modelo)
            mlflow_config.log_artefato(caminho_modelo)
            
            logger.info("Treinamento concluído com sucesso")
            return modelo
            
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        if mlflow_config:
            mlflow_config.finalizar_run(status="FAILED")
        raise
        
    finally:
        # Limpar recursos
        if spark:
            spark.stop()
            
        if mlflow_config:
            mlflow_config.finalizar_run()

if __name__ == "__main__":
    try:
        modelo = treinar_modelo()
    except Exception as e:
        logger.error(f"Erro fatal durante execução: {str(e)}")
        raise'''

'''import os
import logging
from pyspark.sql import SparkSession
from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.utils.helpers import carregar_arquivos_dados, criar_diretorio_se_nao_existe
from src.config.mlflow_config import MLflowConfig
from src.config.logging_config import configurar_logging

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def configurar_spark(app_name="RecomendadorNoticias", master="local[*]", memoria_executor="4g", memoria_driver="4g"):
    """
    Configura e retorna uma sessão Spark.
    
    Args:
        app_name: Nome da aplicação Spark.
        master: Endereço do Spark Master (local[*] para modo local, spark://<endereço> para cluster).
        memoria_executor: Memória alocada para cada executor.
        memoria_driver: Memória alocada para o driver.
    
    Returns:
        SparkSession configurada.
    """
    try:
        spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.executor.memory", memoria_executor) \
            .config("spark.driver.memory", memoria_driver) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        logger.info("Sessão Spark inicializada")
        return spark
    except Exception as e:
        logger.error(f"Erro ao configurar Spark: {str(e)}")
        raise

def treinar_modelo():
    """
    Função principal para treinamento do modelo.
    Gerencia todo o processo de treinamento, incluindo:
    - Carregamento de dados com PySpark
    - Preprocessamento distribuído
    - Treinamento
    - Salvamento do modelo
    - Tracking com MLflow
    """
    logger.info("Iniciando processo de treinamento")
    mlflow_config = None
    spark = None
    
    try:
        # Configurar MLflow
        mlflow_config = MLflowConfig()
        mlflow_config.setup_mlflow()
        
        with mlflow_config.iniciar_run(
            run_name="treinamento_completo",
            tags={
                "ambiente": os.getenv("ENVIRONMENT", "desenvolvimento"),
                "processamento": "spark"
            }
        ):
            # Configurar diretórios
            logger.info("Criando diretórios necessários")
            criar_diretorio_se_nao_existe('modelos/modelos_salvos')
            criar_diretorio_se_nao_existe('dados/processados')
            
            # Inicializar Spark
            spark = configurar_spark(
                app_name="RecomendadorNoticias",
                master="local[*]",  # Modo local (ou use "spark://<endereço>" para cluster)
                memoria_executor="4g",
                memoria_driver="4g"
            )
            
            # Carregar dados
            logger.info("Carregando arquivos de dados")
            arquivos_treino = carregar_arquivos_dados('dados/brutos/treino_parte*.csv')
            arquivos_itens = carregar_arquivos_dados('dados/brutos/itens/itens-parte*.csv')
            
            logger.info(f"Arquivos de treino encontrados: {arquivos_treino}")
            logger.info(f"Arquivos de itens encontrados: {arquivos_itens}")
            
            if not arquivos_treino or not arquivos_itens:
                erro_msg = "Arquivos de dados não encontrados!"
                logger.error(erro_msg)
                raise ValueError(erro_msg)
            
            # Processar dados com PySpark
            preprocessador = PreProcessadorDadosSpark(spark)
            dados_treino, dados_itens = preprocessador.processar_dados_treino(
                arquivos_treino,
                arquivos_itens
            )
            
            # Validar dados
            logger.info("Validando dados")
            if not preprocessador.validar_dados(dados_treino, dados_itens):
                logger.warning("Dados contêm valores nulos ou inconsistências")
            
            # Mostrar informações dos dados
            preprocessador.mostrar_info_dados(dados_treino, dados_itens)
            
            # Registrar métricas dos dados
            mlflow_config.log_parametros({
                "num_usuarios": dados_treino.select("idUsuario").distinct().count(),
                "num_itens": dados_itens.count(),
                "tamanho_dados_treino": dados_treino.count()
            })
            
            # Criar e treinar modelo
            logger.info("Iniciando treinamento do modelo")
            modelo = RecomendadorHibrido()
            historia_treino = modelo.treinar(dados_treino, dados_itens)
            
            # Registrar métricas finais
            metricas_finais = {
                "acuracia_final": historia_treino.history['accuracy'][-1],
                "loss_final": historia_treino.history['loss'][-1],
                "val_accuracy_final": historia_treino.history.get('val_accuracy', [0])[-1],
                "val_loss_final": historia_treino.history.get('val_loss', [0])[-1]
            }
            mlflow_config.log_metricas(metricas_finais)
            
            # Salvar modelo
            logger.info("Salvando modelo treinado")
            caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
            modelo.salvar_modelo(caminho_modelo)
            mlflow_config.log_artefato(caminho_modelo)
            
            logger.info("Treinamento concluído com sucesso")
            return modelo
            
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        if mlflow_config:
            mlflow_config.finalizar_run(status="FAILED")
        raise
        
    finally:
        # Limpar recursos
        if spark:
            spark.stop()
            
        if mlflow_config:
            mlflow_config.finalizar_run()

if __name__ == "__main__":
    try:
        modelo = treinar_modelo()
    except Exception as e:
        logger.error(f"Erro fatal durante execução: {str(e)}")
        raise'''

'''import os
import logging
from pyspark.sql import SparkSession
from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.utils.helpers import carregar_arquivos_dados, criar_diretorio_se_nao_existe
from src.config.mlflow_config import MLflowConfig
from src.config.logging_config import configurar_logging

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def configurar_spark(app_name="RecomendadorNoticias", master="local[*]", memoria_executor="4g", memoria_driver="4g"):
    """
    Configura e retorna uma sessão Spark.
    
    Args:
        app_name: Nome da aplicação Spark.
        master: Endereço do Spark Master (local[*] para modo local, spark://<endereço> para cluster).
        memoria_executor: Memória alocada para cada executor.
        memoria_driver: Memória alocada para o driver.
    
    Returns:
        SparkSession configurada.
    """
    try:
        spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.executor.memory", memoria_executor) \
            .config("spark.driver.memory", memoria_driver) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.memory.storageFraction", "0.5") \
            .getOrCreate()
        
        logger.info("Sessão Spark inicializada")
        return spark
    except Exception as e:
        logger.error(f"Erro ao configurar Spark: {str(e)}")
        raise

def treinar_modelo():
    """
    Função principal para treinamento do modelo.
    Gerencia todo o processo de treinamento, incluindo:
    - Carregamento de dados com PySpark
    - Preprocessamento distribuído
    - Treinamento
    - Salvamento do modelo
    - Tracking com MLflow
    """
    logger.info("Iniciando processo de treinamento")
    mlflow_config = None
    spark = None
    
    try:
        # Configurar MLflow
        mlflow_config = MLflowConfig()
        mlflow_config.setup_mlflow()
        
        with mlflow_config.iniciar_run(
            run_name="treinamento_completo",
            tags={
                "ambiente": os.getenv("ENVIRONMENT", "desenvolvimento"),
                "processamento": "spark"
            }
        ):
            # Configurar diretórios
            logger.info("Criando diretórios necessários")
            criar_diretorio_se_nao_existe('modelos/modelos_salvos')
            criar_diretorio_se_nao_existe('dados/processados')
            
            # Inicializar Spark
            spark = configurar_spark(
                app_name="RecomendadorNoticias",
                master="local[*]",  # Modo local (ou use "spark://<endereço>" para cluster)
                memoria_executor="8g",
                memoria_driver="8g"
            )
            
            # Carregar dados
            logger.info("Carregando arquivos de dados")
            arquivos_treino = carregar_arquivos_dados('dados/brutos/treino_parte*.csv')
            arquivos_itens = carregar_arquivos_dados('dados/brutos/itens/itens-parte*.csv')
            
            logger.info(f"Arquivos de treino encontrados: {arquivos_treino}")
            logger.info(f"Arquivos de itens encontrados: {arquivos_itens}")
            
            if not arquivos_treino or not arquivos_itens:
                erro_msg = "Arquivos de dados não encontrados!"
                logger.error(erro_msg)
                raise ValueError(erro_msg)
            
            # Processar dados com PySpark
            preprocessador = PreProcessadorDadosSpark(spark)
            dados_treino, dados_itens = preprocessador.processar_dados_treino(
                arquivos_treino,
                arquivos_itens
            )
            
            # Validar dados
            logger.info("Validando dados")
            if not preprocessador.validar_dados(dados_treino, dados_itens):
                logger.warning("Dados contêm valores nulos ou inconsistências")
            
            # Mostrar informações dos dados
            preprocessador.mostrar_info_dados(dados_treino, dados_itens)
            
            # Registrar métricas dos dados
            mlflow_config.log_parametros({
                "num_usuarios": dados_treino.select("idUsuario").distinct().count(),
                "num_itens": dados_itens.count(),
                "tamanho_dados_treino": dados_treino.count()
            })
            
            # Criar e treinar modelo
            logger.info("Iniciando treinamento do modelo")
            modelo = RecomendadorHibrido()
            historia_treino = modelo.treinar(dados_treino, dados_itens)
            
            # Registrar métricas finais
            metricas_finais = {
                "acuracia_final": historia_treino.history['accuracy'][-1],
                "loss_final": historia_treino.history['loss'][-1],
                "val_accuracy_final": historia_treino.history.get('val_accuracy', [0])[-1],
                "val_loss_final": historia_treino.history.get('val_loss', [0])[-1]
            }
            mlflow_config.log_metricas(metricas_finais)
            
            # Salvar modelo
            logger.info("Salvando modelo treinado")
            caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
            modelo.salvar_modelo(caminho_modelo)
            mlflow_config.log_artefato(caminho_modelo)
            
            logger.info("Treinamento concluído com sucesso")
            return modelo
            
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        if mlflow_config:
            mlflow_config.finalizar_run(status="FAILED")
        raise
        
    finally:
        # Limpar recursos
        if spark:
            spark.stop()
            
        if mlflow_config:
            mlflow_config.finalizar_run()

if __name__ == "__main__":
    try:
        modelo = treinar_modelo()
    except Exception as e:
        logger.error(f"Erro fatal durante execução: {str(e)}")
        raise'''


'''import os
import logging
import signal
import sys
import time
import mlflow
from pyspark.sql import SparkSession
from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.utils.helpers import carregar_arquivos_dados, criar_diretorio_se_nao_existe
from src.config.mlflow_config import MLflowConfig
from src.config.logging_config import configurar_logging
import shutil

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def limpar_diretorio_checkpoints():
    """Limpa o diretório de checkpoints do Spark se existir."""
    try:
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
        os.makedirs("checkpoints", exist_ok=True)
        logger.info("Diretório de checkpoints limpo com sucesso")
    except Exception as e:
        logger.error(f"Erro ao limpar diretório de checkpoints: {str(e)}")

def verificar_conexao_spark(spark):
    """
    Verifica se a conexão Spark está ativa e funcionando.
    
    Args:
        spark: Sessão Spark a ser verificada
    
    Returns:
        bool: True se a conexão está ativa, False caso contrário
    """
    try:
        # Tentar várias operações simples
        test_df = spark.createDataFrame([(1,)], ["test"])
        test_df.collect()
        test_df.unpersist()
        return True
    except Exception as e:
        logger.error(f"Erro na conexão Spark: {str(e)}")
        return False

def configurar_spark(app_name="RecomendadorNoticias", master="local[*]", memoria_executor="4g", memoria_driver="4g"):
    """
    Configura e retorna uma sessão Spark com configurações otimizadas.
    
    Args:
        app_name: Nome da aplicação Spark
        master: URL do master Spark
        memoria_executor: Quantidade de memória para executores
        memoria_driver: Quantidade de memória para o driver
    
    Returns:
        SparkSession configurada
    """
    try:
        # Limpar sessões anteriores de forma segura
        active_session = SparkSession._instantiatedSession
        if active_session:
            try:
                active_session.stop()
                time.sleep(2)  # Esperar a limpeza completa
            except Exception as e:
                logger.warning(f"Erro ao parar sessão anterior: {str(e)}")
        
        # Limpar diretório de checkpoints
        limpar_diretorio_checkpoints()
        
        # Configurações de proteção contra desligamento
        os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.driver.maxResultSize=2g pyspark-shell'
        os.environ['PYSPARK_PYTHON'] = sys.executable
        
        # Configurar nova sessão com proteções
        spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.executor.memory", memoria_executor) \
            .config("spark.driver.memory", memoria_driver) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.memory.storageFraction", "0.5") \
            .config("spark.network.timeout", "1200s") \
            .config("spark.executor.heartbeatInterval", "120s") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.rdd.compress", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.default.parallelism", "4") \
            .config("spark.sql.broadcastTimeout", "1200s") \
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") \
            .config("spark.python.worker.reuse", "true") \
            .config("spark.dynamicAllocation.enabled", "false") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:+HeapDumpOnOutOfMemoryError") \
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
            .config("spark.scheduler.mode", "FAIR") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
            .config("spark.sql.execution.arrow.enabled", "true") \
            .config("spark.driver.allowMultipleContexts", "false") \
            .config("spark.driver.supervise", "false") \
            .config("spark.executor.instances", "2") \
            .config("spark.executor.cores", "2") \
            .config("spark.task.maxFailures", "4") \
            .config("spark.speculation", "false") \
            .config("spark.driver.extraClassPath", ".") \
            .config("spark.executor.extraClassPath", ".") \
            .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2") \
            .config("spark.python.worker.memory", "1g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "2g") \
            .getOrCreate()
        
        # Configurações adicionais
        spark.sparkContext.setLogLevel("WARN")
        spark.sparkContext.setLocalProperty("spark.scheduler.pool", "fair_pool")
        spark.sparkContext.setCheckpointDir("checkpoints")
        
        # Verificar se a sessão está ativa
        if not verificar_conexao_spark(spark):
            raise RuntimeError("Falha ao inicializar sessão Spark")
        
        logger.info("Sessão Spark inicializada com sucesso")
        return spark
    except Exception as e:
        logger.error(f"Erro ao configurar Spark: {str(e)}")
        raise

def encerrar_spark_seguro(spark):
    """
    Encerra a sessão Spark de forma segura.
    
    Args:
        spark: Sessão Spark a ser encerrada
    """
    if spark:
        try:
            # Garantir que todos os jobs sejam finalizados
            spark.sparkContext.cancelAllJobs()
            time.sleep(2)
            
            # Limpar cache e recursos
            spark.catalog.clearCache()
            
            # Parar SparkContext e sessão de forma segura
            if not spark._jsc.sc().isStopped():
                spark.stop()
                SparkSession._instantiatedSession = None
                time.sleep(1)
            
            # Limpar checkpoints
            limpar_diretorio_checkpoints()
            
            logger.info("Sessão Spark encerrada com sucesso")
        except Exception as e:
            logger.error(f"Erro ao encerrar Spark: {str(e)}")

def treinar_modelo(spark):
    """
    Função principal para treinamento do modelo.
    
    Args:
        spark: Sessão Spark ativa
    
    Returns:
        Modelo treinado
    """
    logger.info("Iniciando processo de treinamento")
    mlflow_config = None
    dados_treino = None
    dados_itens = None
    
    try:
        # Verificar Spark
        if not verificar_conexao_spark(spark):
            raise RuntimeError("Sessão Spark não está respondendo")
        
        # Configurar MLflow
        mlflow_config = MLflowConfig()
        mlflow_config.setup_mlflow()
        
        # Finalizar qualquer run ativo do MLflow
        if mlflow.active_run():
            mlflow.end_run()
        
        with mlflow_config.iniciar_run(
            run_name="treinamento_completo",
            tags={
                "ambiente": os.getenv("ENVIRONMENT", "desenvolvimento"),
                "processamento": "spark"
            }
        ):
            # Preparar diretórios
            logger.info("Criando diretórios necessários")
            criar_diretorio_se_nao_existe('modelos/modelos_salvos')
            criar_diretorio_se_nao_existe('dados/processados')
            
            # Carregar arquivos
            logger.info("Carregando arquivos de dados")
            arquivos_treino = carregar_arquivos_dados('dados/brutos/treino_parte*.csv')
            arquivos_itens = carregar_arquivos_dados('dados/brutos/itens/itens-parte*.csv')
            
            if not arquivos_treino or not arquivos_itens:
                raise ValueError("Arquivos de dados não encontrados")
            
            logger.info(f"Arquivos de treino encontrados: {len(arquivos_treino)}")
            logger.info(f"Arquivos de itens encontrados: {len(arquivos_itens)}")
            
            # Processar dados
            preprocessador = PreProcessadorDadosSpark(spark)
            dados_treino, dados_itens = preprocessador.processar_dados_treino(
                arquivos_treino,
                arquivos_itens
            )
            
            # Verificar dados processados
            if dados_treino is None or dados_itens is None:
                raise ValueError("Erro no processamento dos dados")
            
            # Validar dados
            logger.info("Validando dados")
            if not preprocessador.validar_dados(dados_treino, dados_itens):
                logger.warning("Dados contêm valores nulos ou inconsistências")
            
            # Mostrar informações dos dados
            preprocessador.mostrar_info_dados(dados_treino, dados_itens)
            
            # Criar e treinar modelo
            logger.info("Iniciando treinamento do modelo")
            modelo = RecomendadorHibrido(mlflow_config=mlflow_config)
            
            # Treinar modelo com novo run do MLflow
            historia_treino = modelo.treinar(dados_treino, dados_itens)
            
            # Salvar modelo
            logger.info("Salvando modelo treinado")
            caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
            modelo.salvar_modelo(caminho_modelo)
            
            logger.info("Treinamento concluído com sucesso")
            return modelo
            
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        if mlflow_config:
            mlflow_config.finalizar_run(status="FAILED")
        raise
    finally:
        # Limpar recursos
        if mlflow_config:
            mlflow_config.finalizar_run()
        for df in [dados_treino, dados_itens]:
            if df and hasattr(df, 'unpersist'):
                try:
                    df.unpersist()
                except Exception as e:
                    logger.error(f"Erro ao liberar DataFrame: {str(e)}")

def signal_handler(signum, frame):
    """Handler para sinais de interrupção."""
    logger.info("Recebido sinal de interrupção. Encerrando de forma segura...")
    if 'spark' in globals() and spark:
        encerrar_spark_seguro(spark)
    sys.exit(0)

if __name__ == "__main__":
    # Registrar handlers para sinais
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    spark = None
    try:
        # Inicializar Spark com mais recursos
        spark = configurar_spark(
            app_name="RecomendadorNoticias",
            master="local[*]",
            memoria_executor="8g",
            memoria_driver="8g"
        )
        
        if not verificar_conexao_spark(spark):
            raise RuntimeError("Falha ao inicializar Spark")
        
        # Executar treinamento
        modelo = treinar_modelo(spark)
        
    except Exception as e:
        logger.error(f"Erro fatal durante execução: {str(e)}")
        raise
    finally:
        # Limpeza final
        if spark:
            encerrar_spark_seguro(spark)'''

'''import os
import logging
import signal
import sys
import time
import mlflow  
from pyspark.sql import SparkSession
from src.modelo.recomendador import RecomendadorHibrido
from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.utils.helpers import carregar_arquivos_dados, criar_diretorio_se_nao_existe
from src.config.mlflow_config import MLflowConfig
from src.config.logging_config import configurar_logging

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def verificar_conexao_spark(spark):
    """
    Verifica se a conexão Spark está ativa e funcionando.
    """
    try:
        # Tentar operação simples
        test_df = spark.createDataFrame([(1,)], ["test"])
        test_df.collect()
        test_df.unpersist()
        return True
    except Exception as e:
        logger.error(f"Erro na conexão Spark: {str(e)}")
        return False

def configurar_spark(app_name="RecomendadorNoticias", master="local[*]", memoria_executor="4g", memoria_driver="4g"):
    """
    Configura e retorna uma sessão Spark com configurações otimizadas.
    """
    try:
        # Limpar sessões anteriores
        active_session = SparkSession._instantiatedSession
        if active_session:
            active_session.stop()
            time.sleep(2)  # Aguardar limpeza completa
        
        # Configurar nova sessão
        spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.executor.memory", memoria_executor) \
            .config("spark.driver.memory", memoria_driver) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.memory.storageFraction", "0.5") \
            .config("spark.network.timeout", "800s") \
            .config("spark.executor.heartbeatInterval", "120s") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.rdd.compress", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.default.parallelism", "4") \
            .config("spark.sql.broadcastTimeout", "1200s") \
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") \
            .config("spark.python.worker.reuse", "true") \
            .config("spark.dynamicAllocation.enabled", "false") \
            .config("spark.executor.instances", "2") \
            .config("spark.executor.cores", "2") \
            .config("spark.task.maxFailures", "4") \
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:+HeapDumpOnOutOfMemoryError") \
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
            .config("spark.scheduler.mode", "FAIR") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

        # Configurar nível de log
        spark.sparkContext.setLogLevel("WARN")
        
        return spark
    except Exception as e:
        logger.error(f"Erro ao configurar Spark: {str(e)}")
        raise

def encerrar_spark_seguro(spark):
    """
    Encerra a sessão Spark de forma segura.
    """
    if spark:
        try:
            # Tentar cancelar jobs ativos
            if not spark._jsc.sc().isStopped():
                spark.sparkContext.cancelAllJobs()
                time.sleep(1)
            
            # Limpar cache
            spark.catalog.clearCache()
            
            # Parar sessão
            spark.stop()
            SparkSession._instantiatedSession = None
            
            time.sleep(2)  # Aguardar limpeza completa
            
            logger.info("Sessão Spark encerrada com sucesso")
        except Exception as e:
            logger.error(f"Erro ao encerrar Spark: {str(e)}")

def treinar_modelo(spark):
    """
    Função principal para treinamento do modelo.
    """
    logger.info("Iniciando processo de treinamento")
    mlflow_config = None
    dados_treino = None
    dados_itens = None
    
    try:
        # Verificar Spark
        if not verificar_conexao_spark(spark):
            raise RuntimeError("Sessão Spark não está respondendo")
        
        # Configurar MLflow
        mlflow_config = MLflowConfig()
        mlflow_config.setup_mlflow()
        
        # Finalizar qualquer run ativo
        if mlflow.active_run():
            mlflow.end_run()
        
        with mlflow_config.iniciar_run(
            run_name="treinamento_completo",
            tags={
                "ambiente": os.getenv("ENVIRONMENT", "desenvolvimento"),
                "processamento": "spark"
            }
        ):
            # Preparar diretórios
            logger.info("Criando diretórios necessários")
            criar_diretorio_se_nao_existe('modelos/modelos_salvos')
            criar_diretorio_se_nao_existe('dados/processados')
            
            # Carregar arquivos
            logger.info("Carregando arquivos de dados")
            arquivos_treino = carregar_arquivos_dados('dados/brutos/treino_parte*.csv')
            arquivos_itens = carregar_arquivos_dados('dados/brutos/itens/itens-parte*.csv')
            
            if not arquivos_treino or not arquivos_itens:
                raise ValueError("Arquivos de dados não encontrados")
            
            logger.info(f"Arquivos de treino encontrados: {len(arquivos_treino)}")
            logger.info(f"Arquivos de itens encontrados: {len(arquivos_itens)}")
            
            # Processar dados
            preprocessador = PreProcessadorDadosSpark(spark)
            dados_treino, dados_itens = preprocessador.processar_dados_treino(
                arquivos_treino,
                arquivos_itens
            )
            
            # Cache dos DataFrames
            dados_treino.cache()
            dados_itens.cache()
            
            # Validar dados
            logger.info("Validando dados")
            if not preprocessador.validar_dados(dados_treino, dados_itens):
                logger.warning("Dados contêm valores nulos ou inconsistências")
            
            # Mostrar informações
            preprocessador.mostrar_info_dados(dados_treino, dados_itens)
            
            # Criar e treinar modelo
            logger.info("Iniciando treinamento do modelo")
            modelo = RecomendadorHibrido()
            historia_treino = modelo.treinar(dados_treino, dados_itens)
            
            # Salvar modelo
            logger.info("Salvando modelo treinado")
            caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
            modelo.salvar_modelo(caminho_modelo)
            
            logger.info("Treinamento concluído com sucesso")
            return modelo
            
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        if mlflow_config:
            mlflow_config.finalizar_run(status="FAILED")
        raise
    finally:
        # Limpar recursos
        if mlflow_config:
            mlflow_config.finalizar_run()
        for df in [dados_treino, dados_itens]:
            if df and hasattr(df, 'unpersist'):
                try:
                    df.unpersist()
                except Exception as e:
                    logger.error(f"Erro ao liberar DataFrame: {str(e)}")

def signal_handler(signum, frame):
    """Handler para sinais de interrupção."""
    logger.info("Recebido sinal de interrupção. Encerrando de forma segura...")
    if 'spark' in globals():
        encerrar_spark_seguro(spark)
    sys.exit(0)

if __name__ == "__main__":
    # Registrar handlers para sinais
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    spark = None
    try:
        # Inicializar Spark
        spark = configurar_spark(
            app_name="RecomendadorNoticias",
            master="local[*]",
            memoria_executor="8g",
            memoria_driver="8g"
        )
        
        if not verificar_conexao_spark(spark):
            raise RuntimeError("Falha ao inicializar Spark")
        
        # Executar treinamento
        modelo = treinar_modelo(spark)
        
    except Exception as e:
        logger.error(f"Erro fatal durante execução: {str(e)}")
        raise
    finally:
        # Limpeza final
        if spark:
            encerrar_spark_seguro(spark)'''

'''import os
import sys
import logging
import signal
import time
import mlflow
import shutil
from pathlib import Path

# Adicionar diretório raiz ao PYTHONPATH
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pyspark.sql import SparkSession
from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.modelo.recomendador import RecomendadorHibrido
from src.utils.helpers import carregar_arquivos_dados, criar_diretorio_se_nao_existe
from src.config.mlflow_config import MLflowConfig
from src.config.logging_config import configurar_logging
from src.config.spark_config import configurar_ambiente_spark, get_spark_config

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def verificar_conexao_spark(spark):
    """
    Verifica se a conexão Spark está ativa e funcionando.
    
    Args:
        spark: Sessão Spark a ser verificada
        
    Returns:
        bool: True se a conexão está ativa, False caso contrário
    """
    try:
        # Tentar operação simples
        test_df = spark.createDataFrame([(1,)], ["test"])
        test_df.collect()
        test_df.unpersist()
        return True
    except Exception as e:
        logger.error(f"Erro na conexão Spark: {str(e)}")
        return False

def limpar_diretorio_checkpoints():
    """
    Limpa o diretório de checkpoints do Spark.
    """
    try:
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info("Diretório de checkpoints limpo com sucesso")
    except Exception as e:
        logger.error(f"Erro ao limpar diretório de checkpoints: {str(e)}")

def configurar_spark(app_name="RecomendadorNoticias", master="local[*]"):
    """
    Configura e retorna uma sessão Spark com configurações otimizadas.
    
    Args:
        app_name: Nome da aplicação Spark
        master: URL do master Spark
        
    Returns:
        SparkSession configurada
    """
    try:
        # Limpar sessões anteriores
        if SparkSession._instantiatedSession:
            try:
                SparkSession._instantiatedSession.stop()
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Erro ao parar sessão anterior: {str(e)}")
        
        # Limpar diretório de checkpoints
        limpar_diretorio_checkpoints()
            
        # Configurar ambiente
        configurar_ambiente_spark()
        
        # Obter configurações
        spark_config = get_spark_config()
        
        # Criar builder com configurações
        builder = SparkSession.builder \
            .appName(app_name) \
            .master(master)
            
        # Adicionar todas as configurações
        for key, value in spark_config.items():
            builder = builder.config(key, value)
        
        # Criar sessão
        # spark = builder.getOrCreate()
        spark = SparkSession.builder \
        .appName("RecomendadorNoticias") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
        .config("spark.sql.execution.arrow.timezone", "UTC") \
        .config("spark.sql.execution.arrow.pyspark.selfDestruct.enabled", "true") \
        .getOrCreate()
        
        # Configurações adicionais
        spark.sparkContext.setLogLevel("WARN")
        spark.sparkContext.setCheckpointDir("checkpoints")
        
        # Verificar se a sessão está ativa
        if not verificar_conexao_spark(spark):
            raise RuntimeError("Falha ao inicializar sessão Spark")
        
        # Aguardar inicialização completa
        time.sleep(2)
            
        logger.info("Sessão Spark inicializada com sucesso")
        return spark
        
    except Exception as e:
        logger.error(f"Erro ao configurar Spark: {str(e)}")
        raise

def encerrar_spark_seguro(spark):
    """
    Encerra a sessão Spark de forma segura.
    
    Args:
        spark: Sessão Spark a ser encerrada
    """
    if spark:
        try:
            # Limpar cache primeiro
            spark.catalog.clearCache()
            time.sleep(1)
            
            # Cancelar jobs pendentes
            if not spark._jsc.sc().isStopped():
                spark.sparkContext.cancelAllJobs()
                time.sleep(1)
            
            # Limpar checkpoints
            limpar_diretorio_checkpoints()
            
            # Parar sessão
            spark.stop()
            SparkSession._instantiatedSession = None
            
            time.sleep(2)
            
            logger.info("Sessão Spark encerrada com sucesso")
        except Exception as e:
            logger.error(f"Erro ao encerrar Spark: {str(e)}")

def treinar_modelo(spark):
    """
    Função principal para treinamento do modelo.
    
    Args:
        spark: Sessão Spark ativa
        
    Returns:
        Modelo treinado
    """
    logger.info("Iniciando processo de treinamento")
    mlflow_config = None
    dados_treino = None
    dados_itens = None
    preprocessador = None
    
    try:
        # Verificar Spark
        if not verificar_conexao_spark(spark):
            raise RuntimeError("Sessão Spark não está respondendo")
        
        # Configurar MLflow
        mlflow_config = MLflowConfig(user_name=os.getenv('USER_NAME', 'sistema-recomendacao'))
        mlflow_config.setup_mlflow()
        
        # Finalizar qualquer run ativo
        if mlflow.active_run():
            mlflow.end_run()
        
        with mlflow_config.iniciar_run(
            run_name="treinamento_completo",
            tags={
                "ambiente": os.getenv("ENVIRONMENT", "desenvolvimento"),
                "processamento": "spark",
                "mlflow.user": os.getenv("USER_NAME", "sistema-recomendacao")
            }
        ):
            # Criar todos os diretórios necessários primeiro
            diretorios = [
                'modelos/modelos_salvos',
                'dados/processados',
                'checkpoints',
                'logs',
                'mlflow-artifacts'
            ]
            for diretorio in diretorios:
                criar_diretorio_se_nao_existe(diretorio)
            
            # Carregar arquivos
            logger.info("Carregando arquivos de dados")
            arquivos_treino = carregar_arquivos_dados('dados/brutos/treino_parte*.csv')
            arquivos_itens = carregar_arquivos_dados('dados/brutos/itens/itens-parte*.csv')
            
            if not arquivos_treino or not arquivos_itens:
                raise ValueError("Arquivos de dados não encontrados")
            
            logger.info(f"Arquivos de treino encontrados: {len(arquivos_treino)}")
            logger.info(f"Arquivos de itens encontrados: {len(arquivos_itens)}")
            
            # Processar dados com melhor gestão de recursos
            preprocessador = PreProcessadorDadosSpark(spark)
            try:
                dados_treino, dados_itens = preprocessador.processar_dados_treino(
                    arquivos_treino,
                    arquivos_itens
                )
                
                # Validar dados
                if not preprocessador.validar_dados(dados_treino, dados_itens):
                    logger.warning("Dados contêm valores nulos ou inconsistências")
                
                # Mostrar informações
                preprocessador.mostrar_info_dados(dados_treino, dados_itens)
                
                # Treinar modelo
                logger.info("Iniciando treinamento do modelo")
                modelo = RecomendadorHibrido(mlflow_config=mlflow_config)
                historia_treino = modelo.treinar(dados_treino, dados_itens)
                
                # Salvar modelo
                logger.info("Salvando modelo treinado")
                caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
                modelo.salvar_modelo(caminho_modelo)
                
                return modelo
                
            finally:
                # Limpar recursos do preprocessador
                if preprocessador:
                    try:
                        preprocessador.limpar_recursos()
                    except Exception as e:
                        logger.error(f"Erro ao limpar recursos do preprocessador: {str(e)}")
    
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        if mlflow_config:
            mlflow_config.finalizar_run(status="FAILED")
        raise
        
    finally:
        # Limpar recursos
        if mlflow_config:
            mlflow_config.finalizar_run()
        
        # Liberar DataFrames
        for df in [dados_treino, dados_itens]:
            if df and hasattr(df, 'unpersist'):
                try:
                    df.unpersist()
                except Exception as e:
                    logger.error(f"Erro ao liberar DataFrame: {str(e)}")

def signal_handler(signum, frame):
    """
    Handler para sinais de interrupção.
    """
    logger.info("Recebido sinal de interrupção. Encerrando de forma segura...")
    if 'spark' in globals():
        encerrar_spark_seguro(spark)
    sys.exit(0)

if __name__ == "__main__":
    # Registrar handlers para sinais
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    spark = None
    try:
        # Inicializar Spark
        spark = configurar_spark(
            app_name="RecomendadorNoticias",
            master="local[*]"
        )
        
        # Verificar inicialização
        if not verificar_conexao_spark(spark):
            raise RuntimeError("Falha ao inicializar Spark")
        
        # Executar treinamento
        modelo = treinar_modelo(spark)
        
    except Exception as e:
        logger.error(f"Erro fatal durante execução: {str(e)}")
        raise
    finally:
        # Limpeza final
        if spark:
            encerrar_spark_seguro(spark)'''

import os
import sys
import logging
import signal
import time
import mlflow
import shutil
import gc
from pathlib import Path

# Adicionar diretório raiz ao PYTHONPATH
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.modelo.preprocessamento_spark import PreProcessadorDadosSpark
from src.modelo.recomendador import RecomendadorHibrido
from src.utils.helpers import carregar_arquivos_dados, criar_diretorio_se_nao_existe
from src.config.mlflow_config import MLflowConfig
from src.config.logging_config import configurar_logging
from src.config.spark_config import configurar_ambiente_spark, get_spark_config

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def verificar_conexao_spark(spark, max_tentativas=3):
    """
    Verifica se a conexão Spark está ativa e funcionando, com tentativas de reconexão.
    
    Args:
        spark: Sessão Spark a ser verificada
        max_tentativas: Número máximo de tentativas de reconexão
        
    Returns:
        bool: True se a conexão está ativa, False caso contrário
    """
    for tentativa in range(max_tentativas):
        try:
            # Tentar operação simples
            test_df = spark.createDataFrame([(1,)], ["test"])
            test_df.collect()
            test_df.unpersist()
            return True
        except Exception as e:
            logger.warning(f"Tentativa {tentativa + 1} falhou: {str(e)}")
            if tentativa < max_tentativas - 1:
                time.sleep(5)  # Esperar antes de tentar novamente
                try:
                    # Tentar reconectar
                    spark.sparkContext.setLogLevel("ERROR")
                    spark.catalog.clearCache()
                except:
                    pass
    
    logger.error("Todas as tentativas de conexão falharam")
    return False

def limpar_diretorio_checkpoints():
    """
    Limpa o diretório de checkpoints do Spark de forma segura.
    """
    try:
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            # Tentar remover arquivos individualmente primeiro
            for root, dirs, files in os.walk(checkpoint_dir, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except:
                        pass
            # Tentar remover diretório principal
            try:
                shutil.rmtree(checkpoint_dir)
            except:
                pass
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info("Diretório de checkpoints limpo com sucesso")
    except Exception as e:
        logger.error(f"Erro ao limpar diretório de checkpoints: {str(e)}")



def configurar_spark(app_name="RecomendadorNoticias", master="local[*]"):
    """
    Configura e retorna uma sessão Spark com configurações otimizadas.
    """
    try:
        # Forçar limpeza de memória
        gc.collect()
        
        # Limpar sessões anteriores
        if SparkSession._instantiatedSession:
            try:
                SparkSession._instantiatedSession.stop()
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Erro ao parar sessão anterior: {str(e)}")
        
        # Limpar diretório de checkpoints
        limpar_diretorio_checkpoints()
            
        # Configurar ambiente
        configurar_ambiente_spark()
        
        # Todas as configurações devem ser definidas aqui
        spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.network.timeout", "800s") \
            .config("spark.executor.heartbeatInterval", "30s") \
            .config("spark.sql.broadcastTimeout", "600s") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
            .config("spark.sql.execution.arrow.timezone", "UTC") \
            .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.default.parallelism", "200") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "2g") \
            .getOrCreate()
        
        # Configurações adicionais que podem ser modificadas após a criação
        spark.sparkContext.setLogLevel("WARN")
        spark.sparkContext.setCheckpointDir("checkpoints")
        
        # Verificar se a sessão está ativa
        if not verificar_conexao_spark(spark):
            raise RuntimeError("Falha ao inicializar sessão Spark")
        
        # Aguardar inicialização completa
        time.sleep(2)
            
        logger.info("Sessão Spark inicializada com sucesso")
        return spark
        
    except Exception as e:
        logger.error(f"Erro ao configurar Spark: {str(e)}")
        raise

def encerrar_spark_seguro(spark):
    """
    Encerra a sessão Spark de forma segura e limpa todos os recursos.
    
    Args:
        spark: Sessão Spark a ser encerrada
    """
    if spark:
        try:
            # Limpar cache e memória primeiro
            try:
                spark.catalog.clearCache()
                spark.sparkContext.emptyRDD()
                time.sleep(1)
            except:
                pass
            
            # Cancelar jobs pendentes
            try:
                if not spark._jsc.sc().isStopped():
                    spark.sparkContext.cancelAllJobs()
                    time.sleep(1)
            except:
                pass
            
            # Limpar checkpoints
            limpar_diretorio_checkpoints()
            
            # Parar sessão
            try:
                spark.stop()
                SparkSession._instantiatedSession = None
            except:
                pass
            
            # Forçar limpeza de memória
            gc.collect()
            time.sleep(2)
            
            logger.info("Sessão Spark encerrada com sucesso")
        except Exception as e:
            logger.error(f"Erro ao encerrar Spark: {str(e)}")

def treinar_modelo(spark):
    """
    Função principal para treinamento do modelo com melhor gestão de recursos.
    
    Args:
        spark: Sessão Spark ativa
        
    Returns:
        Modelo treinado
    """
    logger.info("Iniciando processo de treinamento")
    mlflow_config = None
    dados_treino = None
    dados_itens = None
    preprocessador = None
    
    try:
        # Verificar Spark
        if not verificar_conexao_spark(spark):
            raise RuntimeError("Sessão Spark não está respondendo")
        
        # Configurar MLflow
        mlflow_config = MLflowConfig(user_name=os.getenv('USER_NAME', 'sistema-recomendacao'))
        mlflow_config.setup_mlflow()
        
        # Finalizar qualquer run ativo
        if mlflow.active_run():
            mlflow.end_run()
        
        with mlflow_config.iniciar_run(
            run_name="treinamento_completo",
            tags={
                "ambiente": os.getenv("ENVIRONMENT", "desenvolvimento"),
                "processamento": "spark",
                "mlflow.user": os.getenv("USER_NAME", "sistema-recomendacao")
            }
        ):
            # Criar diretórios necessários
            diretorios = [
                'modelos/modelos_salvos',
                'dados/processados',
                'checkpoints',
                'logs',
                'mlflow-artifacts'
            ]
            for diretorio in diretorios:
                criar_diretorio_se_nao_existe(diretorio)
            
            # Inicializar preprocessador
            preprocessador = PreProcessadorDadosSpark(spark)
            
            # Verificar se existem dados processados
            dados_processados_exist = os.path.exists("dados/processados/dados_treino_processados.parquet") and \
                                    os.path.exists("dados/processados/dados_itens_processados.parquet")
            
            try:
                if dados_processados_exist:
                    logger.info("Encontrados dados processados. Tentando carregar...")
                    try:
                        # Carregar dados processados
                        dados_treino = spark.read.parquet("dados/processados/dados_treino_processados.parquet")
                        dados_itens = spark.read.parquet("dados/processados/dados_itens_processados.parquet")
                        
                        # Verificar se os dados são válidos
                        if dados_treino.count() > 0 and dados_itens.count() > 0:
                            logger.info("Dados processados carregados com sucesso")
                            
                            # Mostrar estatísticas dos dados carregados
                            logger.info(f"Registros de treino: {dados_treino.count()}")
                            logger.info(f"Registros de itens: {dados_itens.count()}")
                        else:
                            logger.warning("Dados processados vazios ou inválidos")
                            dados_processados_exist = False
                            
                    except Exception as e:
                        logger.warning(f"Erro ao carregar dados processados: {str(e)}")
                        dados_processados_exist = False
                
                if not dados_processados_exist:
                    logger.info("Processando dados brutos...")
                    # Carregar arquivos
                    logger.info("Carregando arquivos de dados")
                    arquivos_treino = carregar_arquivos_dados('dados/brutos/treino_parte*.csv')
                    arquivos_itens = carregar_arquivos_dados('dados/brutos/itens/itens-parte*.csv')
                    
                    if not arquivos_treino or not arquivos_itens:
                        raise ValueError("Arquivos de dados não encontrados")
                    
                    logger.info(f"Arquivos de treino encontrados: {len(arquivos_treino)}")
                    logger.info(f"Arquivos de itens encontrados: {len(arquivos_itens)}")
                    
                    # Processar dados
                    dados_treino, dados_itens = preprocessador.processar_dados_treino(
                        arquivos_treino,
                        arquivos_itens
                    )
                
                # Persistir DataFrames
                dados_treino.persist()
                dados_itens.persist()
                
                # Adicionar checkpoints para otimização
                dados_treino.checkpoint()
                dados_itens.checkpoint()
                
                # Validar dados
                if not preprocessador.validar_dados(dados_treino, dados_itens):
                    logger.warning("Dados contêm valores nulos ou inconsistências")
                
                # Mostrar informações e estatísticas
                preprocessador.mostrar_info_dados(dados_treino, dados_itens)
                
                # Treinar modelo
                logger.info("Iniciando treinamento do modelo")
                modelo = RecomendadorHibrido(mlflow_config=mlflow_config)
                historia_treino = modelo.treinar(dados_treino, dados_itens)
                
                # Salvar modelo
                logger.info("Salvando modelo treinado")
                caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
                modelo.salvar_modelo(caminho_modelo)
                
                return modelo
                
            finally:
                # Limpar recursos do preprocessador
                if preprocessador:
                    try:
                        preprocessador.limpar_recursos()
                    except Exception as e:
                        logger.error(f"Erro ao limpar recursos do preprocessador: {str(e)}")
                
                # Limpar DataFrames
                for df in [dados_treino, dados_itens]:
                    if df:
                        try:
                            df.unpersist()
                        except:
                            pass
    
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        if mlflow_config:
            mlflow_config.finalizar_run(status="FAILED")
        raise
        
    finally:
        # Limpar todos os recursos
        if mlflow_config:
            try:
                mlflow_config.finalizar_run()
            except:
                pass
        
        # Forçar limpeza de memória
        gc.collect()

def signal_handler(signum, frame):
    """
    Handler para sinais de interrupção com limpeza segura de recursos.
    """
    logger.info("Recebido sinal de interrupção. Encerrando de forma segura...")
    if 'spark' in globals():
        encerrar_spark_seguro(spark)
    sys.exit(0)

if __name__ == "__main__":
    # Registrar handlers para sinais
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    spark = None
    try:
        # Inicializar Spark com retry
        max_tentativas = 3
        for tentativa in range(max_tentativas):
            try:
                spark = configurar_spark(
                    app_name="RecomendadorNoticias",
                    master="local[*]"
                )
                if verificar_conexao_spark(spark):
                    break
            except Exception as e:
                if tentativa == max_tentativas - 1:
                    raise
                logger.warning(f"Tentativa {tentativa + 1} falhou: {str(e)}")
                time.sleep(5)
        
        # Executar treinamento
        modelo = treinar_modelo(spark)
        
    except Exception as e:
        logger.error(f"Erro fatal durante execução: {str(e)}")
        raise
    finally:
        # Limpeza final
        if spark:
            encerrar_spark_seguro(spark)