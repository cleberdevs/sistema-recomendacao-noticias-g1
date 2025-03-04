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
    echo "tensorflow<2.11.0" >> requirements-py38.txt

# Install Python dependencies from the modified requirements file
RUN pip install --no-cache-dir -r requirements-py38.txt

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

# Create entrypoint script
RUN echo '#!/bin/bash\n\
echo "Verificando dependências..."\n\
pip list | grep -E "mlflow|flask|pyspark|werkzeug"\n\
\n\
# Start MLflow in the background\n\
echo "Iniciando MLflow..."\n\
./scripts/start_mlflow.sh &\n\
\n\
# Wait for MLflow to start\n\
echo "Aguardando MLflow iniciar..."\n\
sleep 10\n\
\n\
# If requested, run the pipeline first\n\
if [ "$RUN_PIPELINE" = "true" ]; then\n\
    echo "Executando o pipeline..."\n\
    python pipeline.py\n\
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