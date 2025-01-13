#!/bin/bash

# Configurar cores para output
VERDE='\033[0;32m'
VERMELHO='\033[0;31m'
SEM_COR='\033[0m'

# Função para log
log() {
    echo -e "${VERDE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${SEM_COR}"
}

error_log() {
    echo -e "${VERMELHO}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${SEM_COR}"
}

# Criar diretórios necessários
log "Criando diretórios necessários..."
mkdir -p mlflow-artifacts
mkdir -p logs

# Verificar se o MLflow está instalado
if ! command -v mlflow &> /dev/null; then
    error_log "MLflow não está instalado. Instalando..."
    pip install mlflow
fi

# Iniciar servidor MLflow
log "Iniciando servidor MLflow..."
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts \
    > logs/mlflow.log 2>&1 &

# Verificar se MLflow iniciou corretamente
sleep 5
if ! netstat -tuln | grep -q ':5000 '; then
    error_log "Falha ao iniciar MLflow"
    exit 1
fi
log "MLflow iniciado com sucesso"

# Exportar variáveis de ambiente
export FLASK_APP=src/api/app.py
export FLASK_ENV=producao
export MLFLOW_TRACKING_URI=http://localhost:5000

# Iniciar servidor Flask
log "Iniciando servidor Flask..."
python executar.py >> logs/api.log 2>&1