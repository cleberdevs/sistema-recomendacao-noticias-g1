from pyspark.sql import SparkSession
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SparkSessionManager:
    _instance: Optional[SparkSession] = None

    @classmethod
    def get_session(cls, 
                   app_name: str = "RecomendadorNoticias",
                   memoria_executor: str = "2g",
                   memoria_driver: str = "2g",
                   max_result_size: str = "2g") -> SparkSession:
        """
        Obtém uma sessão Spark singleton com configurações otimizadas.
        
        Args:
            app_name: Nome da aplicação Spark
            memoria_executor: Memória para executores
            memoria_driver: Memória para driver
            max_result_size: Tamanho máximo do resultado
            
        Returns:
            SparkSession configurada
        """
        if cls._instance is None:
            logger.info("Criando nova sessão Spark")
            try:
                cls._instance = (SparkSession.builder
                    .appName(app_name)
                    .config("spark.executor.memory", memoria_executor)
                    .config("spark.driver.memory", memoria_driver)
                    .config("spark.driver.maxResultSize", max_result_size)
                    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                    .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
                    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
                    .config("spark.sql.adaptive.enabled", "true")
                    .config("spark.sql.shuffle.partitions", "200")
                    .config("spark.default.parallelism", "200")
                    .config("spark.memory.offHeap.enabled", "true")
                    .config("spark.memory.offHeap.size", "2g")
                    .config("spark.sql.broadcastTimeout", "600")
                    .config("spark.network.timeout", "800")
                    .config("spark.driver.extraClassPath", ".")
                    .getOrCreate())

                # Configurar log level
                cls._instance.sparkContext.setLogLevel("WARN")
                
            except Exception as e:
                logger.error(f"Erro ao criar sessão Spark: {str(e)}")
                raise

        return cls._instance

    @classmethod
    def stop_session(cls):
        """Encerra a sessão Spark se existir."""
        if cls._instance:
            try:
                logger.info("Encerrando sessão Spark")
                cls._instance.stop()
                cls._instance = None
            except Exception as e:
                logger.error(f"Erro ao encerrar sessão Spark: {str(e)}")