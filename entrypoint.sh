#!/bin/bash
# Configurar para sair imediatamente se algum comando falhar
set -e

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

log "=== Iniciando sequência de execução ==="

log "=== Executando setup_environment.sh ==="
/app/scripts/setup_environment.sh
if [ $? -ne 0 ]; then
    error_log "setup_environment.sh falhou com código $?"
    exit 1
else
    log "=== setup_environment.sh concluído com sucesso ==="
fi

# Nota: A descompactação de arquivos foi movida para o código Python.
log "=== A extração do arquivo ZIP será feita diretamente pelo código Python ==="

log "=== Executando start_mlflow.sh ==="
/app/scripts/start_mlflow.sh
if [ $? -ne 0 ]; then
    error_log "start_mlflow.sh falhou com código $?"
    exit 1
else
    log "=== start_mlflow.sh concluído com sucesso ==="
fi

log "=== Executando pipeline.py ==="
python /app/pipeline.py
if [ $? -ne 0 ]; then
    error_log "pipeline.py falhou com código $?"
    exit 1
else
    log "=== pipeline.py concluído com sucesso ==="
fi

# Configurar variáveis de ambiente para o Flask
export PYTHONPATH="/app:${PYTHONPATH:-}"
export FLASK_APP="src.api.app:app"
export FLASK_ENV=development
export MLFLOW_TRACKING_URI=http://localhost:5000

log "PYTHONPATH: ${PYTHONPATH}"
log "Diretório do projeto: /app"

# Verificar se o arquivo app.py existe
if [ ! -f "/app/src/api/app.py" ]; then
    error_log "Arquivo app.py não encontrado em /app/src/api/app.py"
    exit 1
fi

# Iniciar servidor Flask
log "Iniciando servidor Flask..."

# Obter o caminho do diretório de dados brutos
PARENT_DIR="/app/dados"
if [ -f "/app/scripts/setup_environment.sh" ]; then
    # Extrair PARENT_DIR do script setup_environment.sh
    PARENT_DIR=$(grep -o 'PARENT_DIR=.*' /app/scripts/setup_environment.sh | cut -d'=' -f2 | tr -d '"')
    if [ -z "$PARENT_DIR" ]; then
        PARENT_DIR="/app/dados"
    fi
fi

log "=== LOG FINAL: Estado dos diretórios ==="
find "${PARENT_DIR}/dados" -type f -name "*.csv" | sort > /app/arquivos_csv_final.log
log "Lista de todos os arquivos CSV encontrados salva em /app/arquivos_csv_final.log"
log "Total de arquivos CSV:"
wc -l /app/arquivos_csv_final.log

# Executar o Flask
exec python -m flask run --host=0.0.0.0 --port=8000