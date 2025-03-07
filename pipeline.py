from datetime import datetime
import os
import sys
import logging
import signal
import time
import mlflow
import shutil
import gc
from pathlib import Path

import tensorflow as tf
from py4j.protocol import Py4JError, Py4JNetworkError

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

def verificar_conexao_spark(spark, max_tentativas=3, intervalo=5):
    """
    Verifica se a conexão Spark está ativa e funcionando, com tentativas de reconexão.
    
    Args:
        spark: Sessão Spark a ser verificada
        max_tentativas: Número máximo de tentativas de reconexão
        intervalo: Intervalo entre tentativas em segundos
        
    Returns:
        bool: True se a conexão está ativa, False caso contrário
    """
    if spark is None:
        logger.warning("Sessão Spark é None")
        return False
        
    for tentativa in range(max_tentativas):
        try:
            # Verificar se a JVM está acessível
            if not spark._jvm:
                logger.warning("JVM do Spark não está acessível")
                return False
                
            # Verificar se o contexto está parado
            if spark._jsc.sc().isStopped():
                logger.warning("Contexto Spark está parado")
                return False
                
            # Tentar operação simples
            test_df = spark.createDataFrame([(1,)], ["test"])
            count = test_df.count()  # Forçar execução
            test_df.unpersist()
            
            logger.debug(f"Verificação de conexão Spark bem-sucedida na tentativa {tentativa + 1}")
            return True
            
        except Py4JNetworkError as e:
            logger.warning(f"Erro de rede Py4J na tentativa {tentativa + 1}: {str(e)}")
        except Py4JError as e:
            logger.warning(f"Erro Py4J na tentativa {tentativa + 1}: {str(e)}")
        except Exception as e:
            logger.warning(f"Tentativa {tentativa + 1} falhou: {str(e)}")
            
        if tentativa < max_tentativas - 1:
            time.sleep(intervalo)  # Esperar antes de tentar novamente
            try:
                # Tentar recuperar a sessão
                if spark and not spark._jsc.sc().isStopped():
                    spark.sparkContext.setLogLevel("ERROR")
                    try:
                        spark.catalog.clearCache()
                    except:
                        pass
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
                    except OSError as e:
                        logger.debug(f"Não foi possível remover arquivo {name}: {str(e)}")
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except OSError as e:
                        logger.debug(f"Não foi possível remover diretório {name}: {str(e)}")
            
            # Tentar remover diretório principal
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError as e:
                logger.debug(f"Não foi possível remover diretório de checkpoints: {str(e)}")
                # Tenta usar comando do sistema para forçar remoção em sistemas Unix
                if os.name == 'posix':
                    try:
                        os.system(f"rm -rf {checkpoint_dir}")
                    except:
                        pass
        
        # Criar novo diretório
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
                logger.info("Encontrada sessão Spark anterior. Tentando encerrar...")
                SparkSession._instantiatedSession.stop()
                SparkSession._instantiatedSession = None
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
            .config("spark.driver.maxResultSize", "4g") \
            .config("spark.python.worker.reuse", "true") \
            .config("spark.python.worker.memory", "1g") \
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:+DisableExplicitGC") \
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:+DisableExplicitGC") \
            .getOrCreate()
        
        # Configurações adicionais que podem ser modificadas após a criação
        spark.sparkContext.setLogLevel("WARN")
        spark.sparkContext.setCheckpointDir("checkpoints")
        
        # Verificar se a sessão está ativa
        if not verificar_conexao_spark(spark):
            logger.error("Falha ao inicializar sessão Spark")
            raise RuntimeError("Falha ao inicializar sessão Spark. A conexão não está respondendo.")
        
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
    if not spark:
        logger.info("Nenhuma sessão Spark para encerrar")
        return

    # Verificar se a sessão está válida antes de tentar operações
    sessao_valida = False
    try:
        # Verifica apenas se o objeto JVM existe, não tenta operações
        if spark._jvm and not spark._jsc.sc().isStopped():
            sessao_valida = True
    except:
        sessao_valida = False
    
    logger.info(f"Estado da sessão Spark antes do encerramento: {'válida' if sessao_valida else 'inválida'}")
    
    try:
        # Se a sessão estiver válida, tenta operações de limpeza
        if sessao_valida:
            # Limpar cache e memória primeiro
            try:
                logger.debug("Limpando cache Spark")
                spark.catalog.clearCache()
                spark.sparkContext.emptyRDD()
                time.sleep(1)
            except Py4JNetworkError as e:
                logger.warning(f"Erro de rede Py4J ao limpar cache: {str(e)}")
            except Exception as e:
                logger.warning(f"Erro ao limpar cache: {str(e)}")
            
            # Cancelar jobs pendentes
            try:
                logger.debug("Cancelando jobs pendentes")
                spark.sparkContext.cancelAllJobs()
                time.sleep(1)
            except Py4JNetworkError as e:
                logger.warning(f"Erro de rede Py4J ao cancelar jobs: {str(e)}")
            except Exception as e:
                logger.warning(f"Erro ao cancelar jobs: {str(e)}")
        
        # Limpar checkpoints (independente do estado da sessão)
        limpar_diretorio_checkpoints()
        
        # Parar sessão
        try:
            logger.info("Tentando parar sessão Spark")
            spark.stop()
            SparkSession._instantiatedSession = None
            time.sleep(1)
        except Py4JNetworkError as e:
            logger.warning(f"Erro de rede Py4J ao parar sessão: {str(e)}")
        except Exception as e:
            logger.warning(f"Erro ao parar sessão: {str(e)}")
        
        # Forçar limpeza de memória
        gc.collect()
        time.sleep(2)
        
        logger.info("Processo de encerramento da sessão Spark concluído")
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
            run_name=f"treinamento_{datetime.now().strftime('%Y%m%d_%H%M')}",
            tags={
                "ambiente": os.getenv("ENVIRONMENT", "desenvolvimento"),
                "processamento": "spark",
                "versao_codigo": os.getenv("CODE_VERSION", "v1.0"),
                "tipo_modelo": "hibrido",
                "framework": "tensorflow",
                "mlflow.user": os.getenv("USER_NAME", "sistema-recomendacao")
            }
        ):
            # Registrar parâmetros do ambiente
            mlflow_config.log_parametros({
                "spark_version": spark.version,
                "python_version": sys.version,
                "tensorflow_version": tf.__version__,
                "mlflow_version": mlflow.__version__
            })
            
            # Criar diretórios necessários
            diretorios = [
                'modelos/modelos_salvos',
                'dados_processados',
                'checkpoints',
                'logs',
                'mlflow-artifacts'
            ]
            for diretorio in diretorios:
                criar_diretorio_se_nao_existe(diretorio)
            
            # Inicializar preprocessador
            preprocessador = PreProcessadorDadosSpark(spark)
            
            # Verificar se existem dados processados
            dados_processados_exist = os.path.exists("dados_processados/dados_treino_processados.parquet") and \
                                    os.path.exists("dados_processados/dados_itens_processados.parquet")
            
            try:
                if dados_processados_exist:
                    logger.info("Encontrados dados processados. Tentando carregar...")
                    try:
                        # Verificar conexão Spark novamente antes de operações pesadas
                        if not verificar_conexao_spark(spark):
                            raise RuntimeError("Sessão Spark não está respondendo antes de carregar dados")
                        
                        # Carregar dados processados
                        dados_treino = spark.read.parquet("dados_processados/dados_treino_processados.parquet")
                        dados_itens = spark.read.parquet("dados_processados/dados_itens_processados.parquet")
                        
                        # Verificar se os dados são válidos, mas com proteção contra erros
                        try:
                            count_treino = dados_treino.count()
                            count_itens = dados_itens.count()
                            
                            if count_treino > 0 and count_itens > 0:
                                logger.info("Dados processados carregados com sucesso")
                                logger.info(f"Registros de treino: {count_treino}")
                                logger.info(f"Registros de itens: {count_itens}")
                            else:
                                logger.warning("Dados processados vazios ou inválidos")
                                dados_processados_exist = False
                        except Exception as e:
                            logger.warning(f"Erro ao verificar contagens dos dados: {str(e)}")
                            dados_processados_exist = False
                            
                    except Exception as e:
                        logger.warning(f"Erro ao carregar dados processados: {str(e)}")
                        dados_processados_exist = False
                
                if not dados_processados_exist:
                    logger.info("Processando dados brutos...")
                    # Verificar conexão Spark novamente antes de operações pesadas
                    if not verificar_conexao_spark(spark):
                        raise RuntimeError("Sessão Spark não está respondendo antes de processar dados")
                    
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
                
                # Persistir DataFrames com verificação
                try:
                    dados_treino.persist()
                    dados_itens.persist()
                    
                    # Adicionar checkpoints para otimização
                    dados_treino.checkpoint()
                    dados_itens.checkpoint()
                except Exception as e:
                    logger.warning(f"Erro ao persistir ou checkpoint dos dados: {str(e)}")
                
                # Validar dados
                try:
                    if not preprocessador.validar_dados(dados_treino, dados_itens):
                        logger.warning("Dados contêm valores nulos ou inconsistências")
                except Exception as e:
                    logger.warning(f"Erro ao validar dados: {str(e)}")
                
                # Mostrar informações e estatísticas
                try:
                    preprocessador.mostrar_info_dados(dados_treino, dados_itens)
                except Exception as e:
                    logger.warning(f"Erro ao mostrar informações dos dados: {str(e)}")
                
                # Treinar modelo com tratamento de erros aprimorado
                logger.info("Iniciando treinamento do modelo")
                modelo = RecomendadorHibrido(mlflow_config=mlflow_config)
                
                try:
                    # Verificar conexão Spark novamente antes do treinamento
                    if not verificar_conexao_spark(spark):
                        raise RuntimeError("Sessão Spark não está respondendo antes do treinamento")
                        
                    historia_treino = modelo.treinar(dados_treino, dados_itens)
                    
                    # O modelo já foi salvo dentro do método treinar() antes da interação com MLflow
                    # Mantemos o código de salvamento aqui como backup redundante
                    logger.info("Salvando modelo treinado (backup)")
                    caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
                    modelo.salvar_modelo(caminho_modelo)
                    
                except Exception as treino_error:
                    # Verifica se o modelo foi criado, mesmo com erro
                    logger.error(f"Erro durante treinamento: {treino_error}")
                    
                    if hasattr(modelo, 'modelo') and modelo.modelo is not None:
                        logger.warning("Modelo existe apesar do erro. Tentando salvar...")
                        try:
                            # Tenta salvar o modelo mesmo com erro durante o treinamento
                            caminho_modelo = 'modelos/modelos_salvos/recomendador_hibrido'
                            modelo.salvar_modelo(caminho_modelo)
                            logger.info("Modelo salvo com sucesso (recuperação)")
                        except Exception as save_error:
                            logger.error(f"Erro ao salvar modelo após recuperação: {save_error}")
                    
                    # Mesmo após tentar salvar o modelo, propagamos o erro original
                    raise treino_error
                
                # Registrar modelo no MLflow com tratamento de erro melhorado
                try:
                    client = mlflow.tracking.MlflowClient()
                    
                    # Verificar se o modelo já está registrado
                    try:
                        registered_model = client.get_registered_model("recomendador_hibrido")
                        logger.info("Modelo já registrado no MLflow")
                    except Exception as model_error:
                        # Se não existir, criar o modelo registrado
                        logger.info(f"Modelo não encontrado no registro ({str(model_error)}). Criando novo modelo registrado no MLflow")
                        try:
                            registered_model = client.create_registered_model("recomendador_hibrido")
                        except Exception as create_error:
                            logger.error(f"Erro ao criar modelo registrado: {str(create_error)}")
                            raise

                    # Registrar nova versão do modelo com melhor tratamento de timeout
                    try:
                        if not mlflow.active_run():
                            logger.warning("Nenhum run MLflow ativo. Iniciando novo run temporário.")
                            temp_run = mlflow.start_run()
                            run_id = temp_run.info.run_id
                        else:
                            run_id = mlflow.active_run().info.run_id
                        
                        # Criar versão do modelo com timeout
                        model_version = None
                        max_tentativas = 3
                        for tentativa in range(max_tentativas):
                            try:
                                model_version = client.create_model_version(
                                    name="recomendador_hibrido",
                                    source=f"runs:/{run_id}/tensorflow-model",
                                    run_id=run_id
                                )
                                break
                            except Exception as e:
                                if tentativa == max_tentativas - 1:
                                    raise
                                logger.warning(f"Tentativa {tentativa + 1} falhou: {str(e)}")
                                time.sleep(3)
                        
                        if model_version is None:
                            raise RuntimeError("Não foi possível criar versão do modelo após múltiplas tentativas")
                        
                        # Aguardar o registro do modelo
                        timeout = 30  # segundos
                        start_time = time.time()
                        version_ready = False
                        
                        while time.time() - start_time < timeout:
                            try:
                                version = client.get_model_version(
                                    name="recomendador_hibrido",
                                    version=model_version.version
                                )
                                if version.status == "READY":
                                    version_ready = True
                                    break
                            except Exception as e:
                                logger.warning(f"Erro ao verificar status da versão: {str(e)}")
                            time.sleep(3)
                        
                        if not version_ready:
                            logger.warning(f"Timeout ao aguardar versão do modelo ficar pronta. Continuando mesmo assim.")
                        
                        # Transicionar para produção com tratamento de erros
                        try:
                            client.transition_model_version_stage(
                                name="recomendador_hibrido",
                                version=model_version.version,
                                stage="Production",
                                archive_existing_versions=True  # Arquivar versões antigas
                            )
                            logger.info(f"Modelo registrado com sucesso: versão {model_version.version}")
                        except Exception as stage_error:
                            logger.warning(f"Aviso: Não foi possível transicionar o modelo para produção: {str(stage_error)}")
                            logger.info(f"O modelo foi registrado com sucesso (versão {model_version.version}), mas não está em produção")
                        
                    except Exception as version_error:
                        logger.error(f"Erro ao criar versão do modelo: {str(version_error)}")
                        # Não propagamos o erro para não interromper o pipeline
                        
                except Exception as register_error:
                    logger.error(f"Erro ao registrar modelo no MLflow: {str(register_error)}")
                    # Não propagar o erro para permitir que o treinamento seja considerado bem-sucedido
                    # mesmo se o registro no MLflow falhar
                
                return modelo
                
            finally:
                # Limpar recursos do preprocessador
                if preprocessador:
                    try:
                        preprocessador.limpar_recursos()
                    except Py4JNetworkError as e:
                        logger.warning(f"Erro de rede Py4J ao limpar recursos do preprocessador: {str(e)}")
                    except Exception as e:
                        logger.error(f"Erro ao limpar recursos do preprocessador: {str(e)}")
                
                # Limpar DataFrames
                for df in [dados_treino, dados_itens]:
                    if df:
                        try:
                            df.unpersist()
                        except Exception as e:
                            logger.debug(f"Erro ao despersistir DataFrame: {str(e)}")
    
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        if mlflow_config:
            try:
                mlflow_config.finalizar_run(status="FAILED")
            except Exception as mlflow_error:
                logger.error(f"Erro ao finalizar run MLflow após falha: {str(mlflow_error)}")
        raise
        
    finally:
        # Limpar todos os recursos
        if mlflow_config:
            try:
                mlflow_config.finalizar_run()
            except Exception as e:
                logger.warning(f"Erro ao finalizar run MLflow: {str(e)}")
        
        # Verificar conexão Spark antes de operações finais
        if 'spark' in locals() and spark:
            try:
                verificar_conexao_spark(spark, max_tentativas=1, intervalo=1)
            except:
                pass
        
        # Forçar limpeza de memória
        gc.collect()

def signal_handler(signum, frame):
    """
    Handler para sinais de interrupção com limpeza segura de recursos.
    """
    logger.info(f"Recebido sinal de interrupção ({signum}). Encerrando de forma segura...")
    if 'spark' in globals() and globals()['spark']:
        encerrar_spark_seguro(globals()['spark'])
    sys.exit(0)

if __name__ == "__main__":
    # Registrar handlers para sinais
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    spark = None
    try:
        # Inicializar Spark com retry
        max_tentativas = 3
        falhas_consecutivas = 0
        for tentativa in range(max_tentativas):
            try:
                logger.info(f"Iniciando sessão Spark (tentativa {tentativa + 1} de {max_tentativas})")
                spark = configurar_spark(
                    app_name="RecomendadorNoticias",
                    master="local[*]"
                )
                
                if verificar_conexao_spark(spark):
                    logger.info("Sessão Spark inicializada com sucesso após verificação")
                    break
                else:
                    logger.warning(f"Verificação de sessão falhou na tentativa {tentativa + 1}")
                    falhas_consecutivas += 1
                    
                    # Limpar a sessão que falhou
                    try:
                        if spark:
                            spark.stop()
                            spark = None
                    except:
                        pass
                    
                    # Forçar GC e esperar mais tempo após falhas consecutivas
                    gc.collect()
                    time.sleep(5 + falhas_consecutivas * 5)  # Espera mais a cada falha
            except Exception as e:
                logger.warning(f"Tentativa {tentativa + 1} falhou: {str(e)}")
                falhas_consecutivas += 1
                
                if tentativa == max_tentativas - 1:
                    logger.error("Todas as tentativas de inicialização do Spark falharam")
                    raise
                
                # Limpar qualquer sessão parcial
                try:
                    if spark:
                        spark.stop()
                        spark = None
                except:
                    pass
                
                time.sleep(5 + falhas_consecutivas * 5)  # Espera mais a cada falha
        
        # Executar treinamento
        modelo = treinar_modelo(spark)
        logger.info("Treinamento concluído com sucesso")
        
    except Exception as e:
        logger.error(f"Erro fatal durante execução: {str(e)}")
        raise
    finally:
        # Limpeza final
        if spark:
            logger.info("Iniciando limpeza final")
            encerrar_spark_seguro(spark)
            # Garantir que não há referências à sessão
            spark = None
            gc.collect()
            logger.info("Limpeza final concluída")