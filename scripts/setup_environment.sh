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

# Verificar Java
log "Verificando Java..."
if ! command -v java &> /dev/null; then
    error_log "Java não encontrado. Instalando..."
    sudo apt-get update
    sudo apt-get install -y openjdk-21-jdk
fi


# Diretório pai (um nível acima)
PARENT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

# Instalar dependências
#log "Instalando dependências..."
#pip install --upgrade pip
#pip install -r ${PARENT_DIR}/requirements.txt



# Criar diretórios
log "Criando estrutura de diretórios..."
mkdir -p ${PARENT_DIR}/dados/{brutos,processados}
mkdir -p ${PARENT_DIR}/dados/brutos/itens
mkdir -p ${PARENT_DIR}/modelos/modelos_salvos
mkdir -p ${PARENT_DIR}/logs
mkdir -p ${PARENT_DIR}/mlflow-artifacts
mkdir -p ${PARENT_DIR}/spark-logs

# Configurar variáveis de ambiente
log "Configurando variáveis de ambiente..."
cat > $PARENT_DIR/.env << EOL
FLASK_APP=src/api/app.py
FLASK_ENV=desenvolvimento
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=recomendador_noticias
SPARK_LOCAL_DIRS=/tmp
SPARK_WORKER_DIR=/tmp
LOG_LEVEL=INFO
EOL

# Verificar configurações
log "Verificando configurações..."
python -c "import pyspark; print('Spark OK')"
python -c "import mlflow; print('MLflow OK')"

log "Ambiente configurado com sucesso!"