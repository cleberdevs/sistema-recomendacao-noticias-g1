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

# Verificar e instalar dependências
log "Verificando dependências..."

# Verificar/instalar Python e pip
if ! command -v python3 &> /dev/null; then
    log "Instalando Python3..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv
fi

# Verificar/instalar SQLite
if ! command -v sqlite3 &> /dev/null; then
    log "Instalando SQLite..."
    sudo apt-get update
    sudo apt-get install -y sqlite3 libsqlite3-dev
fi

# Criar e ativar ambiente virtual
if [ ! -d "$PARENT_DIR/venv" ]; then
    log "Criando ambiente virtual..."
    python3 -m venv "$PARENT_DIR/venv"
fi

log "Ativando ambiente virtual..."
source "$PARENT_DIR/venv/bin/activate"

# Instalar MLflow
log "Instalando MLflow..."
pip install mlflow

# Criar diretórios necessários
log "Criando diretórios necessários em $PARENT_DIR..."
mkdir -p "$PARENT_DIR/mlflow-artifacts"
mkdir -p "$PARENT_DIR/logs"
mkdir -p "$PARENT_DIR/spark-logs"
mkdir -p "$PARENT_DIR/backups"

# Iniciar servidor MLflow em background
log "Iniciando servidor MLflow..."
"$PARENT_DIR/venv/bin/mlflow" server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "sqlite:///$PARENT_DIR/mlflow.db" \
    --default-artifact-root "$PARENT_DIR/mlflow-artifacts" \
    > "$PARENT_DIR/logs/mlflow.log" 2>&1 &

# Aguardar MLflow iniciar
sleep 5
if ! netstat -tuln | grep -q ':5000 '; then
    error_log "Falha ao iniciar MLflow"
    cat "$PARENT_DIR/logs/mlflow.log"
    exit 1
fi
log "MLflow iniciado com sucesso"
```

Para uma instalação mais completa, você pode criar um script de setup separado:

```bash
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

# Instalar dependências do sistema
log "Instalando dependências do sistema..."
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    sqlite3 \
    libsqlite3-dev \
    net-tools

# Criar ambiente virtual
log "Configurando ambiente virtual..."
python3 -m venv venv
source venv/bin/activate

# Atualizar pip
log "Atualizando pip..."
pip install --upgrade pip

# Instalar dependências Python
log "Instalando dependências Python..."
pip install mlflow pandas numpy scikit-learn

# Verificar instalações
log "Verificando instalações..."
python3 -c "import mlflow; print('MLflow version:', mlflow.__version__)"
sqlite3 --version

log "Instalação concluída!"
