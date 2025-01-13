'''# Dockerfile

# Imagem base
FROM python:3.8-slim

# Argumentos de build
ARG ENVIRONMENT=producao

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=src/api/app.py \
    FLASK_ENV=${ENVIRONMENT} \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    PORT=8000

# Diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY . .

# Criar diretórios necessários
RUN mkdir -p mlflow-artifacts logs \
    && chmod -R 777 mlflow-artifacts logs

# Script de health check
COPY scripts/healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

# Expor portas
EXPOSE 8000 5000

# Copiar e tornar executável o script de inicialização
COPY start_services.sh .
RUN chmod +x start_services.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD /healthcheck.sh

# Comando para iniciar serviços
CMD ["./start_services.sh"]'''

# Dockerfile

# Imagem base com suporte a Spark
FROM jupyter/pyspark-notebook:spark-3.3.0

USER root

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Configurar variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=src/api/app.py \
    FLASK_ENV=production \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    SPARK_LOCAL_DIRS=/tmp \
    SPARK_WORKER_DIR=/tmp \
    SPARK_WORKER_MEMORY=2g \
    SPARK_DRIVER_MEMORY=2g \
    SPARK_EXECUTOR_MEMORY=2g

# Criar diretório de trabalho
WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY . .

# Criar diretórios necessários
RUN mkdir -p mlflow-artifacts logs dados/processados modelos/modelos_salvos \
    && chmod -R 777 mlflow-artifacts logs dados modelos

# Copiar script de inicialização
COPY scripts/start_services.sh .
RUN chmod +x start_services.sh

# Expor portas
EXPOSE 8000 5000

# Configurar usuário não-root
RUN chown -R jovyan:users /app
USER jovyan

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/saude || exit 1

# Comando para iniciar serviços
CMD ["./start_services.sh"]
```

```bash
# scripts/start_services.sh

#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Função para log
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error_log() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Criar diretórios necessários
log "Criando diretórios necessários..."
mkdir -p mlflow-artifacts logs dados/processados modelos/modelos_salvos

# Verificar espaço em disco
SPACE_AVAILABLE=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$SPACE_AVAILABLE" -lt 10 ]; then
    error_log "Espaço em disco insuficiente (menor que 10GB)"
    exit 1
fi

# Iniciar MLflow
log "Iniciando servidor MLflow..."
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts \
    > logs/mlflow.log 2>&1 &

# Aguardar MLflow iniciar
sleep 5
if ! netstat -tuln | grep -q ':5000 '; then
    error_log "Falha ao iniciar MLflow"
    exit 1
fi
log "MLflow iniciado com sucesso"

# Iniciar servidor Flask
log "Iniciando servidor Flask..."
python -m flask run --host=0.0.0.0 --port=8000 >> logs/api.log 2>&1