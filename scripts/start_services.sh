'''#!/bin/bash

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
python executar.py >> logs/api.log 2>&1'''

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

# Verificar se o Spark está instalado
if [ -z "$SPARK_HOME" ]; then
    error_log "Variável SPARK_HOME não está configurada. Por favor, instale o Apache Spark e configure a variável SPARK_HOME."
    exit 1
fi

# Criar diretórios necessários
log "Criando diretórios necessários..."
mkdir -p mlflow-artifacts
mkdir -p logs
mkdir -p spark-logs

# Iniciar servidor MLflow em background
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

# Iniciar Spark Master
log "Iniciando Spark Master..."
$SPARK_HOME/sbin/start-master.sh > spark-logs/spark-master.log 2>&1 &

# Aguardar Spark Master iniciar
sleep 5
if ! netstat -tuln | grep -q ':7077 '; then
    error_log "Falha ao iniciar Spark Master"
    exit 1
fi
log "Spark Master iniciado com sucesso"

# Iniciar Spark Worker
log "Iniciando Spark Worker..."
$SPARK_HOME/sbin/start-worker.sh spark://localhost:7077 > spark-logs/spark-worker.log 2>&1 &

# Aguardar Spark Worker iniciar
sleep 5
if ! netstat -tuln | grep -q ':8081 '; then
    error_log "Falha ao iniciar Spark Worker"
    exit 1
fi
log "Spark Worker iniciado com sucesso"

# Configurar variáveis de ambiente para o Flask
export FLASK_APP=src/api/app.py
export FLASK_ENV=development
export MLFLOW_TRACKING_URI=http://localhost:5000
export SPARK_MASTER=spark://localhost:7077

# Iniciar servidor Flask
log "Iniciando servidor Flask..."
flask run --host=0.0.0.0 --port=8000