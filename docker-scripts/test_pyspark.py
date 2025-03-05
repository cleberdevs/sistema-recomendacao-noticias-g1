import os
import sys
import pyspark

# Exibe versões do Python e PySpark
print(f"Versão do Python: {sys.version}")
print(f"Versão do PySpark: {pyspark.__version__}")

# Define variáveis de ambiente
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

print(f"PYSPARK_PYTHON: {os.environ.get('PYSPARK_PYTHON')}")
print(f"PYSPARK_DRIVER_PYTHON: {os.environ.get('PYSPARK_DRIVER_PYTHON')}")
print(f"sys.executable: {sys.executable}")

# Testa funcionalidade básica do PySpark
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("PySpark-Test") \
        .config("spark.driver.extraJavaOptions", "-Dlog4j.logLevel=ERROR") \
        .config("spark.executor.extraJavaOptions", "-Dlog4j.logLevel=ERROR") \
        .config("spark.python.worker.reuse", "true") \
        .config("spark.pyspark.python", sys.executable) \
        .config("spark.pyspark.driver.python", sys.executable) \
        .getOrCreate()
    
    print("SparkSession criada com sucesso")
    
    # Cria dados de teste simples
    data = [(1, "teste")]
    df = spark.createDataFrame(data, ["id", "valor"])
    df.show()
    
    spark.stop()
    print("Teste do PySpark concluído com sucesso")
except Exception as e:
    print(f"Erro ao testar PySpark: {e}")
    sys.exit(1)