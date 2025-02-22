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

# Verificar se o PySpark está instalado
log "Verificando se o PySpark está instalado..."
if ! python -c "import pyspark" &> /dev/null; then
    error_log "PySpark não está instalado. Instalando..."
    pip install pyspark
    if [ $? -ne 0 ]; then
        error_log "Falha ao instalar o PySpark."
        exit 1
    fi
    log "PySpark instalado com sucesso."
else
    log "PySpark já está instalado."
fi

# Obter o diretório raiz do projeto (um nível acima da pasta scripts)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Configurar variáveis de ambiente para o Flask
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export FLASK_APP="src.api.app:app"
export FLASK_ENV=development
export MLFLOW_TRACKING_URI=http://localhost:5000

# Debug: mostrar informações importantes
log "PYTHONPATH: ${PYTHONPATH}"
log "Diretório do projeto: ${PROJECT_ROOT}"

# Verificar se o arquivo app.py existe
if [ ! -f "${PROJECT_ROOT}/src/api/app.py" ]; then
    error_log "Arquivo app.py não encontrado em ${PROJECT_ROOT}/src/api/app.py"
    exit 1
fi

# Mudar para o diretório raiz do projeto
cd "${PROJECT_ROOT}"

# Iniciar servidor Flask
log "Iniciando servidor Flask..."
python -m flask run --host=0.0.0.0 --port=8000

