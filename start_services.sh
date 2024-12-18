#!/bin/bash

# Criar diretórios necessários
mkdir -p mlflow-artifacts
mkdir -p logs

# Iniciar servidor MLflow em background
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts \
    > logs/mlflow.log 2>&1 &

# Aguardar MLflow iniciar
sleep 5

# Iniciar servidor Flask
flask run --host=0.0.0.0 --port=8000

