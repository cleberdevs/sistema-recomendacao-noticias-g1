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

# Obter diretório pai
PARENT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

# Criar diretórios necessários no diretório pai
log "Criando diretórios necessários em $PARENT_DIR..."
mkdir -p "$PARENT_DIR/mlflow-artifacts"
mkdir -p "$PARENT_DIR/logs"
mkdir -p "$PARENT_DIR/spark-logs"
mkdir -p "$PARENT_DIR/backups"  # Diretório para backups



# Iniciar servidor MLflow em background
log "Iniciando servidor MLflow..."
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "sqlite:///$PARENT_DIR/mlflow.db" \
    --default-artifact-root "$PARENT_DIR/mlflow-artifacts" \
    > "$PARENT_DIR/logs/mlflow.log" 2>&1 &

# Aguardar MLflow iniciar
sleep 5
if ! netstat -tuln | grep -q ':5000 '; then
    error_log "Falha ao iniciar MLflow"
    exit 1
fi
log "MLflow iniciado com sucesso"