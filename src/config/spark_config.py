import os
import logging

logger = logging.getLogger(__name__)

# Configurações do Spark
SPARK_CONFIG = {
    # Memória e CPU
    "spark.executor.memory": "8g",
    "spark.driver.memory": "8g",
    "spark.executor.cores": "2",
    "spark.driver.cores": "2",
    "spark.executor.instances": "2",
    
    # Gerenciamento de memória
    "spark.memory.fraction": "0.8",
    "spark.memory.storageFraction": "0.5",
    "spark.memory.offHeap.enabled": "true",
    "spark.memory.offHeap.size": "2g",
    
    # Timeouts e rede
    "spark.network.timeout": "1200s",
    "spark.executor.heartbeatInterval": "120s",
    "spark.sql.broadcastTimeout": "1200s",
    "spark.driver.maxResultSize": "2g",
    
    # Otimizações SQL
    "spark.sql.shuffle.partitions": "200",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.sql.execution.arrow.pyspark.fallback.enabled": "true",
    "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",
    
    # Compressão e serialização
    "spark.rdd.compress": "true",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    
    # Paralelismo e execução
    "spark.default.parallelism": "4",
    "spark.task.maxFailures": "4",
    "spark.scheduler.mode": "FAIR",
    "spark.dynamicAllocation.enabled": "false",
    
    # JVM e GC
    "spark.driver.extraJavaOptions": "-XX:+UseG1GC -XX:+HeapDumpOnOutOfMemoryError",
    "spark.executor.extraJavaOptions": "-XX:+UseG1GC",
    
    # Python
    "spark.python.worker.reuse": "true",
    "spark.python.worker.memory": "1g",
    
    # Outros
    "spark.driver.allowMultipleContexts": "false",
    "spark.cleaner.referenceTracking.cleanCheckpoints": "true"
}

# Configurações de ambiente
def configurar_ambiente_spark():
    """Configura variáveis de ambiente para o Spark."""
    try:
        os.environ['PYSPARK_PYTHON'] = 'python3'
        os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'
        os.environ['SPARK_LOCAL_DIRS'] = '/tmp'
        os.environ['SPARK_WORKER_DIR'] = '/tmp'
        os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-memory 8g --executor-memory 8g pyspark-shell'
        
        logger.info("Ambiente Spark configurado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao configurar ambiente Spark: {str(e)}")
        raise

def get_spark_config():
    """Retorna as configurações do Spark."""
    return SPARK_CONFIG.copy()