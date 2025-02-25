#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Função para log
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

debug_log() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] DEBUG: $1${NC}"
}

error_log() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

debug_log "Iniciando script start_mlflow.sh"

# Obter diretório pai
PARENT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
debug_log "PARENT_DIR = $PARENT_DIR"

# Verificar MLflow
debug_log "Verificando instalação do MLflow..."
if ! python -c "import mlflow" 2>/dev/null; then
    error_log "MLflow não está instalado!"
    exit 1
fi
MLFLOW_VERSION=$(python -c "import mlflow; print(mlflow.__version__)")
debug_log "MLflow versão $MLFLOW_VERSION está instalado"

# Criar diretórios necessários no diretório pai
log "Criando diretórios necessários em $PARENT_DIR..."
mkdir -p "$PARENT_DIR/mlflow-artifacts"
if [ $? -ne 0 ]; then
    error_log "Não foi possível criar diretório mlflow-artifacts"
    exit 1
fi
debug_log "Diretório mlflow-artifacts criado"

mkdir -p "$PARENT_DIR/logs"
if [ $? -ne 0 ]; then
    error_log "Não foi possível criar diretório logs"
    exit 1
fi
debug_log "Diretório logs criado"

mkdir -p "$PARENT_DIR/spark-logs"
if [ $? -ne 0 ]; then
    error_log "Não foi possível criar diretório spark-logs"
    exit 1
fi
debug_log "Diretório spark-logs criado"

mkdir -p "$PARENT_DIR/backups"
if [ $? -ne 0 ]; then
    error_log "Não foi possível criar diretório backups"
    exit 1
fi
debug_log "Diretório backups criado"

# Verificar se a porta 5000 já está em uso
debug_log "Verificando disponibilidade da porta 5000..."
if command -v netstat &>/dev/null; then
    if netstat -tuln | grep -q ':5000 '; then
        error_log "A porta 5000 já está em uso! Verifique se há outro processo usando esta porta."
        netstat -tuln | grep ':5000 '
        exit 1
    fi
    debug_log "Porta 5000 está disponível"
elif command -v ss &>/dev/null; then
    if ss -tuln | grep -q ':5000 '; then
        error_log "A porta 5000 já está em uso! Verifique se há outro processo usando esta porta."
        ss -tuln | grep ':5000 '
        exit 1
    fi
    debug_log "Porta 5000 está disponível"
else
    debug_log "Não foi possível verificar portas (netstat e ss não encontrados)"
fi

# Criar o arquivo de log MLflow
MLFLOW_LOG="$PARENT_DIR/logs/mlflow.log"
touch "$MLFLOW_LOG"
debug_log "Log do MLflow será salvo em: $MLFLOW_LOG"

# Verificar SQLite
debug_log "Verificando SQLite..."
SQLITE_DB_PATH="$PARENT_DIR/mlflow.db"
if [ -f "$SQLITE_DB_PATH" ]; then
    debug_log "Banco de dados SQLite já existe: $SQLITE_DB_PATH"
else
    debug_log "Banco de dados SQLite será criado: $SQLITE_DB_PATH"
fi

# Definir caminho completo do comando MLflow
MLFLOW_CMD="mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///$SQLITE_DB_PATH --default-artifact-root $PARENT_DIR/mlflow-artifacts"
debug_log "Comando MLflow: $MLFLOW_CMD"

# Iniciar servidor MLflow em background
log "Iniciando servidor MLflow..."
$MLFLOW_CMD > "$MLFLOW_LOG" 2>&1 &
MLFLOW_PID=$!
debug_log "MLflow iniciado com PID: $MLFLOW_PID"

# Salvar PID para referência
echo $MLFLOW_PID > "$PARENT_DIR/mlflow.pid"
debug_log "PID salvo em $PARENT_DIR/mlflow.pid"

# Aguardar MLflow iniciar (até 1 minuto)
debug_log "Aguardando o MLflow iniciar (timeout: 60 segundos)..."
MAX_WAIT_TIME=60
START_TIME=$(date +%s)
STARTED=false

while [ $(($(date +%s) - START_TIME)) -lt $MAX_WAIT_TIME ]; do
    # Verificar se o processo ainda está executando
    if ! ps -p $MLFLOW_PID > /dev/null; then
        error_log "O processo MLflow morreu durante a inicialização!"
        error_log "Últimas 15 linhas do log:"
        tail -n 15 "$MLFLOW_LOG"
        exit 1
    fi

    # Verificar se o serviço está na porta 5000
    if command -v netstat &>/dev/null; then
        if netstat -tuln | grep -q ':5000 '; then
            debug_log "Porta 5000 está ativa!"
            STARTED=true
            break
        fi
    elif command -v ss &>/dev/null; then
        if ss -tuln | grep -q ':5000 '; then
            debug_log "Porta 5000 está ativa!"
            STARTED=true
            break
        fi
    else
        # Se não tiver ferramentas para verificar a porta, aguarde mais um pouco
        debug_log "Não foi possível verificar a porta 5000 (netstat/ss não disponíveis)"
        # Esperar um pouco mais, pois não podemos verificar a porta
        sleep 10
        STARTED=true
        break
    fi

    # Mostre progresso a cada 10 segundos
    ELAPSED=$(($(date +%s) - START_TIME))
    if [ $((ELAPSED % 10)) -eq 0 ]; then
        debug_log "Ainda aguardando MLflow iniciar... ($ELAPSED segundos decorridos)"
    fi

    # Aguardar antes da próxima verificação
    sleep 2
done

if [ "$STARTED" = false ]; then
    error_log "Timeout! MLflow não iniciou após $MAX_WAIT_TIME segundos"
    error_log "Status do processo (PID $MLFLOW_PID):"
    ps -p $MLFLOW_PID -f
    error_log "Últimas 20 linhas do log:"
    tail -n 20 "$MLFLOW_LOG"
    kill $MLFLOW_PID
    exit 1
fi

log "MLflow iniciado com sucesso na porta 5000"
debug_log "Para verificar logs do MLflow: cat $MLFLOW_LOG"
exit 0