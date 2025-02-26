#!/bin/bash

# Configurar diretórios
mkdir -p /app/dados/{brutos,processados} \
    /app/modelos/modelos_salvos \
    /app/logs \
    /app/mlflow-artifacts \
    /app/spark-logs \
    /app/checkpoints

# Iniciar MLflow
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///app/mlflow.db \
    --default-artifact-root /app/mlflow-artifacts \
    > /app/logs/mlflow.log 2>&1 &

# Aguardar MLflow iniciar
sleep 5

# Se não existir modelo treinado, treinar
if [ ! -f "/app/modelos/modelos_salvos/recomendador_hibrido" ]; then
    echo "Modelo não encontrado. Iniciando treinamento..."
    python treinar.py
fi

# Iniciar API
python -m flask run --host=0.0.0.0 --port=8000