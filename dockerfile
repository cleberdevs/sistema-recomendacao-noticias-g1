# Base image: Ubuntu Focal (20.04)
FROM ubuntu:focal

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, Python 3.10, and OpenJDK 21
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    git \
    net-tools \
    procps \
    sqlite3 \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && add-apt-repository ppa:openjdk-r/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    openjdk-21-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Create symlinks for convenience
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/pip /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy the entire project first
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh

# Run the setup_environment.sh script to set up the environment properly
RUN ./scripts/setup_environment.sh

# Create necessary directories (if they don't already exist from setup_environment.sh)
RUN mkdir -p dados/brutos/itens
RUN mkdir -p modelos/modelos_salvos
RUN mkdir -p mlruns artifacts

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    FLASK_APP=src/api/app.py \
    FLASK_DEBUG=0 \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    PYTHONPATH=/app \
    PYSPARK_PYTHON=python3.10 \
    PYSPARK_DRIVER_PYTHON=python3.10

# Expose ports
# 8000: Flask API
# 5000: MLflow server
EXPOSE 8000 5000

# Create a helper script to initialize MLflow db
RUN echo 'import mlflow\n\
import os\n\
import sys\n\
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore\n\
\n\
# Remove existing db file if it exists\n\
if os.path.exists("mlflow.db"):\n\
    os.remove("mlflow.db")\n\
\n\
# Create a new store and initialize it\n\
try:\n\
    store = SqlAlchemyStore("sqlite:///mlflow.db")\n\
    # Force initialization of tables\n\
    store.get_experiment_by_name("Default")\n\
    print("MLflow database initialized successfully")\n\
except Exception as e:\n\
    print(f"Error initializing MLflow database: {e}")\n\
    sys.exit(1)\n\
' > init_mlflow_db.py

# Create a pyspark configuration script
RUN echo 'import os\n\
import sys\n\
import pyspark\n\
\n\
# Display Python and PySpark versions\n\
print(f"Python version: {sys.version}")\n\
print(f"PySpark version: {pyspark.__version__}")\n\
\n\
# Set environment variables\n\
os.environ["PYSPARK_PYTHON"] = sys.executable\n\
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable\n\
\n\
print(f"PYSPARK_PYTHON: {os.environ.get(\"PYSPARK_PYTHON\")}")\n\
print(f"PYSPARK_DRIVER_PYTHON: {os.environ.get(\"PYSPARK_DRIVER_PYTHON\")}")\n\
print(f"sys.executable: {sys.executable}")\n\
\n\
# Test basic PySpark functionality\n\
try:\n\
    from pyspark.sql import SparkSession\n\
    spark = SparkSession.builder \\\n\
        .appName("PySpark-Test") \\\n\
        .config("spark.driver.extraJavaOptions", "-Dlog4j.logLevel=ERROR") \\\n\
        .config("spark.executor.extraJavaOptions", "-Dlog4j.logLevel=ERROR") \\\n\
        .config("spark.python.worker.reuse", "true") \\\n\
        .config("spark.pyspark.python", sys.executable) \\\n\
        .config("spark.pyspark.driver.python", sys.executable) \\\n\
        .getOrCreate()\n\
    \n\
    print("Created SparkSession successfully")\n\
    \n\
    # Create simple test data\n\
    data = [(1, "test")]\n\
    df = spark.createDataFrame(data, ["id", "value"])\n\
    df.show()\n\
    \n\
    spark.stop()\n\
    print("PySpark test successful")\n\
except Exception as e:\n\
    print(f"Error testing PySpark: {e}")\n\
    sys.exit(1)\n\
' > test_pyspark.py

# Create entrypoint script with PySpark configuration
RUN echo '#!/bin/bash\n\
echo "Verificando dependências..."\n\
pip list | grep -E "mlflow|flask|pyspark|werkzeug|nltk"\n\
\n\
# Verify Python and Java installations\n\
echo "Python path: $(which python)"\n\
echo "Python version: $(python --version)"\n\
echo "Java version:"\n\
java -version\n\
echo "JAVA_HOME=$JAVA_HOME"\n\
\n\
# Test PySpark configuration\n\
echo "Testando configuração do PySpark..."\n\
python test_pyspark.py\n\
\n\
# Ensure correct PySpark Python versions\n\
export PYSPARK_PYTHON=$(which python3.10)\n\
export PYSPARK_DRIVER_PYTHON=$(which python3.10)\n\
echo "PYSPARK_PYTHON=$PYSPARK_PYTHON"\n\
echo "PYSPARK_DRIVER_PYTHON=$PYSPARK_DRIVER_PYTHON"\n\
\n\
# Initialize MLflow database directly\n\
echo "Inicializando banco de dados MLflow..."\n\
python init_mlflow_db.py\n\
\n\
# Start MLflow in the background\n\
echo "Iniciando MLflow..."\n\
MLFLOW_TRACKING_URI=sqlite:///mlflow.db \\\n\
mlflow ui --backend-store-uri sqlite:///mlflow.db \\\n\
         --default-artifact-root ./artifacts \\\n\
         --host 0.0.0.0 \\\n\
         --port 5000 \\\n\
         > mlflow.log 2>&1 &\n\
MLFLOW_PID=$!\n\
\n\
# Wait for MLflow to start\n\
echo "Aguardando MLflow iniciar..."\n\
max_attempts=30\n\
attempt=0\n\
mlflow_ready=false\n\
\n\
while [ $attempt -lt $max_attempts ]; do\n\
    attempt=$((attempt+1))\n\
    echo "Verificando MLflow (tentativa $attempt/$max_attempts)..."\n\
    \n\
    # Check if MLflow process is still running\n\
    if ! kill -0 $MLFLOW_PID 2>/dev/null; then\n\
        echo "ERRO: Processo do MLflow encerrou prematuramente. Verificando logs:"\n\
        cat mlflow.log\n\
        exit 1\n\
    fi\n\
    \n\
    # Try to connect to MLflow API\n\
    if curl -s http://localhost:5000/ > /dev/null; then\n\
        echo "MLflow iniciado com sucesso!"\n\
        mlflow_ready=true\n\
        break\n\
    fi\n\
    \n\
    sleep 2\n\
done\n\
\n\
if [ "$mlflow_ready" = false ]; then\n\
    echo "ERRO: MLflow não iniciou corretamente após várias tentativas. Verificando logs:"\n\
    cat mlflow.log\n\
    exit 1\n\
fi\n\
\n\
# If requested, run the pipeline first\n\
if [ "$RUN_PIPELINE" = "true" ]; then\n\
    echo "Executando o pipeline..."\n\
    PYSPARK_PYTHON=$PYSPARK_PYTHON PYSPARK_DRIVER_PYTHON=$PYSPARK_DRIVER_PYTHON python pipeline.py\n\
    \n\
    # Check pipeline exit status\n\
    PIPELINE_STATUS=$?\n\
    if [ $PIPELINE_STATUS -ne 0 ]; then\n\
        echo "ERRO: Pipeline falhou com código de saída $PIPELINE_STATUS"\n\
        exit $PIPELINE_STATUS\n\
    fi\n\
    \n\
    # Check if models were created\n\
    if [ ! "$(ls -A modelos/modelos_salvos 2>/dev/null)" ]; then\n\
        echo "AVISO: Nenhum modelo foi encontrado após a execução do pipeline."\n\
        echo "A API não será iniciada sem modelos treinados."\n\
        exit 1\n\
    fi\n\
else\n\
    # Check if models exist\n\
    if [ ! "$(ls -A modelos/modelos_salvos 2>/dev/null)" ]; then\n\
        echo "ERRO: Nenhum modelo encontrado em modelos/modelos_salvos/."\n\
        echo "Execute o contêiner com -e RUN_PIPELINE=true para treinar os modelos primeiro"\n\
        echo "ou monte um volume com modelos pré-treinados."\n\
        exit 1\n\
    fi\n\
fi\n\
\n\
# Start the API server\n\
echo "Iniciando API..."\n\
./scripts/start_api.sh\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]