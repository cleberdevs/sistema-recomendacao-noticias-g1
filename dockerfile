# Base image
FROM python:3.8-slim

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
    echo "mlflow==2.8.0" >> requirements-py38.txt && \
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

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    FLASK_APP=src/api/app.py \
    FLASK_ENV=production \
    JAVA_HOME=/usr/lib/jvm/default-java \
    PYTHONPATH=/app

# Expose ports
# 8000: Flask API
# 5000: MLflow server
EXPOSE 8000 5000

# Create entrypoint script with improved error handling
RUN echo '#!/bin/bash\n\
echo "Verificando dependências..."\n\
pip list | grep -E "mlflow|flask|pyspark|werkzeug|nltk"\n\
\n\
# Verificar se o MLflow DB já existe e tem o esquema correto\n\
if [ ! -f "mlflow.db" ] || [ ! -s "mlflow.db" ]; then\n\
    echo "Inicializando banco de dados MLflow..."\n\
    mlflow db upgrade sqlite:///mlflow.db\n\
fi\n\
\n\
# Start MLflow in the background\n\
echo "Iniciando MLflow..."\n\
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &\n\
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
    if ! ps -p $MLFLOW_PID > /dev/null; then\n\
        echo "ERRO: Processo do MLflow encerrou prematuramente. Verificando logs:"\n\
        cat mlflow.log\n\
        exit 1\n\
    fi\n\
    \n\
    # Try to connect to MLflow API\n\
    if curl -s http://localhost:5000/api/2.0/mlflow/experiments/list > /dev/null; then\n\
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