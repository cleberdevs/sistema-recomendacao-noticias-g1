# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jre-headless \
    curl \
    wget \
    git \
    net-tools \
    procps \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh

# Create a modified requirements file with Python 3.8 compatible versions
RUN echo "# Modified requirements for Python 3.8 compatibility" > requirements-py38.txt && \
    echo "flask==2.2.5" >> requirements-py38.txt && \
    echo "werkzeug==2.2.3" >> requirements-py38.txt && \
    echo "mlflow==2.3.0" >> requirements-py38.txt && \
    echo "alembic==1.8.1" >> requirements-py38.txt && \
    echo "sqlalchemy<2.0.0" >> requirements-py38.txt && \
    echo "pyspark==3.4.1" >> requirements-py38.txt && \
    echo "flask-restx==1.1.0" >> requirements-py38.txt && \
    echo "numpy" >> requirements-py38.txt && \
    echo "pandas" >> requirements-py38.txt && \
    echo "scikit-learn" >> requirements-py38.txt && \
    echo "scipy" >> requirements-py38.txt && \
    echo "matplotlib" >> requirements-py38.txt && \
    echo "seaborn" >> requirements-py38.txt && \
    echo "tensorflow<2.11.0" >> requirements-py38.txt && \
    echo "requests" >> requirements-py38.txt && \
    echo "nltk" >> requirements-py38.txt

# Install Python dependencies from the modified requirements file
RUN pip install --no-cache-dir -r requirements-py38.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Create necessary directories
RUN mkdir -p dados/brutos/itens
RUN mkdir -p modelos/modelos_salvos
RUN mkdir -p mlruns artifacts

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    FLASK_APP=src/api/app.py \
    FLASK_DEBUG=0 \
    JAVA_HOME=/usr/lib/jvm/default-java \
    PYTHONPATH=/app

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

# Create entrypoint script with proper MLflow initialization
RUN echo '#!/bin/bash\n\
echo "Verificando dependências..."\n\
pip list | grep -E "mlflow|flask|pyspark|werkzeug|nltk|alembic|sqlalchemy"\n\
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
    python pipeline.py\n\
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