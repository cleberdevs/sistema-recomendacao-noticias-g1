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