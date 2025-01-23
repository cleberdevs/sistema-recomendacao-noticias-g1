#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Função para log
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error_log() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warn_log() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Verificar dependências do sistema
check_system_dependencies() {
    local deps=("sqlite3" "python3" "pip3" "git")
    local missing=()

    for dep in "${deps[@]}"; do
        if ! command -v $dep &> /dev/null; then
            missing+=($dep)
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        error_log "Dependências faltando: ${missing[*]}"
        log "Instalando dependências..."
        sudo apt-get update
        sudo apt-get install -y "${missing[@]}" python3-venv libsqlite3-dev
    else
        log "Todas as dependências do sistema estão instaladas"
    fi
}



# Instalar dependências Python
install_python_dependencies() {
    log "Atualizando pip..."
    pip install --upgrade pip

    log "Instalando dependências do requirements.txt..."
    pip install -r requirements.txt
}

# Verificar instalações
verify_installations() {
    log "Verificando instalações..."
    
    # SQLite
    sqlite_version=$(sqlite3 --version)
    log "SQLite version: $sqlite_version"
    
    # Python SQLite
    python3 -c "import sqlite3; print('Python SQLite version:', sqlite3.sqlite_version)"
    
    # MLflow
    python3 -c "import mlflow; print('MLflow version:', mlflow.__version__)"
    
    # Outras dependências importantes
    python3 -c "import pandas as pd; print('Pandas version:', pd.__version__)"
    python3 -c "import numpy as np; print('NumPy version:', np.__version__)"
    python3 -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"
}

# Execução principal
main() {
    log "Iniciando configuração do ambiente..."
    
    check_system_dependencies
    setup_virtual_env
    install_python_dependencies
    verify_installations
    
    log "Ambiente configurado com sucesso!"
}

main