#!/bin/bash
set -e

echo "=== Iniciando configuração do ambiente ==="

# Criar diretórios necessários
echo "Criando diretórios..."
mkdir -p \
    dados/brutos \
    dados/processados \
    logs \
    mlflow-artifacts \
    modelos/modelos_salvos \
    checkpoints \
    spark-logs

# Configurar permissões
echo "Configurando permissões..."
chmod -R 777 dados logs mlflow-artifacts modelos checkpoints spark-logs

# Configurar PYTHONPATH
echo "Configurando PYTHONPATH..."
export PYTHONPATH=/app:${PYTHONPATH}

# Verificar dependências Python
echo "Verificando dependências Python..."
python3 -c "
import sys
import tensorflow as tf
import mlflow
import numpy as np
import pandas as pd
import sklearn
import pyspark

print('Verificação de versões:')
print('====================')
print('TensorFlow:', tf.__version__)
print('MLflow:', mlflow.__version__)
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)
print('Scikit-learn:', sklearn.__version__)
print('PySpark:', pyspark.__version__)
print('====================')
"

if [ $? -ne 0 ]; then
    echo "ERRO: Falha na verificação de dependências"
    exit 1
fi

# Verificar existência dos dados brutos
if [ ! -d "dados/brutos" ] || [ -z "$(ls -A dados/brutos)" ]; then
    echo "ERRO: Diretório de dados brutos não encontrado ou vazio"
    echo "Por favor, forneça os dados em: dados/brutos/"
    echo "Estrutura esperada:"
    echo "  dados/brutos/"
    echo "    ├── treino_parte*.csv"
    echo "    └── itens/"
    echo "         └── itens-parte*.csv"
    exit 1
fi

# Iniciar MLflow em background com logs
echo "Iniciando servidor MLflow..."
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts \
    > ./logs/mlflow.log 2>&1 &

# Aguardar MLflow iniciar
echo "Aguardando MLflow inicializar..."
max_attempts=30
attempt=1
while ! curl -s http://localhost:5000/health > /dev/null; do
    if [ $attempt -eq $max_attempts ]; then
        echo "ERRO: MLflow não iniciou após $max_attempts tentativas"
        cat ./logs/mlflow.log
        exit 1
    fi
    echo "Tentativa $attempt de $max_attempts..."
    attempt=$((attempt + 1))
    sleep 2
done
echo "MLflow iniciado com sucesso"

# Função para verificar modelo
check_model() {
    local model_path="modelos/modelos_salvos/recomendador_hibrido"
    
    # Verificar se arquivo existe
    if [ ! -f "$model_path" ]; then
        echo "Modelo não encontrado em: $model_path"
        return 1
    fi
    
    # Tentar carregar o modelo com Python
    echo "Verificando integridade do modelo..."
    if ! python3 - <<EOF
import sys
import tensorflow as tf
import mlflow

try:
    from src.modelo.recomendador import RecomendadorHibrido
    modelo = RecomendadorHibrido.carregar_modelo('modelos/modelos_salvos/recomendador_hibrido')
    print("Modelo carregado com sucesso")
    # Verificar se o modelo é válido
    if not isinstance(modelo.modelo, tf.keras.Model):
        raise ValueError("Modelo carregado não é um modelo TensorFlow válido")
    print("Verificação do modelo concluída com sucesso")
    sys.exit(0)
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    sys.exit(1)
EOF
    then
        echo "Falha na verificação do modelo"
        return 1
    fi
    
    return 0
}

# Verificar se precisamos treinar o modelo
if ! check_model; then
    echo "=== Iniciando treinamento do modelo ==="
    
    # Criar arquivo de log do pipeline
    pipeline_log="./logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
    
    # Executar pipeline com log detalhado
    if ! python3 pipeline.py 2>&1 | tee "$pipeline_log"; then
        echo "ERRO: Falha no treinamento do modelo"
        echo "Últimas linhas do log:"
        tail -n 50 "$pipeline_log"
        exit 1
    fi
    
    # Verificar novamente após treinamento
    if ! check_model; then
        echo "ERRO: Modelo não foi gerado corretamente após treinamento"
        echo "Verificando logs:"
        tail -n 50 "$pipeline_log"
        exit 1
    fi
    
    echo "=== Treinamento concluído com sucesso ==="
else
    echo "Modelo existente verificado com sucesso"
fi

# Verificação final antes de iniciar a API
if ! check_model; then
    echo "ERRO: Verificação final do modelo falhou"
    exit 1
fi

# Iniciar API Flask
echo "=== Iniciando API Flask ==="
echo "Modelo verificado e pronto para uso"

# Configurar variáveis do Flask
export FLASK_APP=src/api/app.py
export FLASK_ENV=production
export FLASK_DEBUG=0

# Iniciar Flask com log
echo "Iniciando servidor Flask..."
python3 -m flask run --host=0.0.0.0 --port=8000 2>&1 | tee ./logs/flask.log