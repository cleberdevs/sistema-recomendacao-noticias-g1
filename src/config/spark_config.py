import os
from pyspark.sql import SparkSession

def criar_spark_session(app_name="RecomendadorNoticias", 
                       memoria_executor="4g",
                       memoria_driver="4g"):
    """
    Cria e configura uma sessão Spark.
    
    Args:
        app_name: Nome da aplicação Spark
        memoria_executor: Quantidade de memória para executores
        memoria_driver: Quantidade de memória para o driver
    
    Returns:
        SparkSession configurada
    """
    spark = (SparkSession.builder
             .appName(app_name)
             .config("spark.executor.memory", memoria_executor)
             .config("spark.driver.memory", memoria_driver)
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
             .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.shuffle.partitions", "200")
             .config("spark.default.parallelism", "200")
             .config("spark.memory.offHeap.enabled", "true")
             .config("spark.memory.offHeap.size", "2g")
             .getOrCreate())
    
    return spark

def configurar_log_nivel(spark, nivel="WARN"):
    """Configura o nível de log do Spark."""
    spark.sparkContext.setLogLevel(nivel)